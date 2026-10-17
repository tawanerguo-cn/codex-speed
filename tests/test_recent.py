from __future__ import annotations

import hashlib
import json
import socket
import sqlite3
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from token_pulse import recent


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, UTC).isoformat()


def _turn_records(
    task_id: str,
    turn_id: str,
    response_id: str,
    item_id: str,
    *,
    base_ms: int,
    model: str | None = "gpt-6-luna",
    effort: str | None = "low",
    tokens: int = 100,
    missing_start: bool = False,
    duplicate_end_ms: int | None = None,
    corrupt_before_usage: bool = False,
) -> list[dict]:
    base_seconds = base_ms / 1000
    records = [
        {
            "type": "session_meta",
            "timestamp": _iso(base_seconds - 10),
            "payload": {"id": task_id, "source": "vscode", "model_provider": "openai"},
        },
        {
            "type": "turn_context",
            "timestamp": _iso(base_seconds - 9),
            "payload": {"turn_id": turn_id, "model": model, "effort": effort},
        },
        {
            "type": "response_item",
            "timestamp": _iso(base_seconds - 4),
            "payload": {"type": "reasoning", "id": item_id},
        },
    ]
    if corrupt_before_usage:
        records.append({"not": "a Codex record", "timestamp": "bad"})
    completion = {
        "type": "event_msg",
        "timestamp": _iso(base_seconds - 3),
        "payload": {
            "type": "item_completed",
            "thread_id": task_id,
            "turn_id": turn_id,
            "item": {"type": "Reasoning", "id": item_id},
            "completed_at_ms": base_ms,
        },
    }
    if not missing_start:
        completion["payload"]["started_at_ms"] = base_ms - 2000
    records.append(completion)
    if duplicate_end_ms is not None:
        duplicate = json.loads(json.dumps(completion))
        duplicate["payload"]["completed_at_ms"] = duplicate_end_ms
        duplicate["timestamp"] = _iso(base_seconds - 2)
        records.append(duplicate)
    records.append(
        {
            "type": "token_usage_record",
            "timestamp": _iso(base_seconds + 5),  # Must not affect generation timing.
            "payload": {
                "thread_id": task_id,
                "turn_id": turn_id,
                "response_id": response_id,
                "usage": {"output_tokens": tokens},
            },
        }
    )
    return records


def _append_records(path: Path, records: list[dict]) -> None:
    with path.open("ab") as stream:
        for record in records:
            stream.write(json.dumps(record).encode() + b"\n")


def _write_state(root: Path, entries: list[dict], *, version: int = 1) -> Path:
    db = root / f"state_{version}.sqlite"
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE threads (id TEXT, rollout_path TEXT, updated_at REAL, archived INTEGER, "
            "thread_source TEXT, source TEXT, model TEXT, reasoning_effort TEXT)"
        )
        connection.execute("CREATE INDEX threads_updated_at ON threads(updated_at DESC)")
        connection.execute("CREATE INDEX threads_source ON threads(thread_source)")
        for entry in entries:
            connection.execute(
                "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry["id"],
                    str(entry["path"]),
                    entry.get("updated", 1.0),
                    entry.get("archived", 0),
                    entry.get("thread_source", "user"),
                    entry.get("source", "vscode"),
                    entry.get("model"),
                    entry.get("effort"),
                ),
            )
    return db


def _entry(root: Path, name: str, *, end_ms: int | None = None, updated: float = 1.0, archived: int = 0):
    task_id = f"00000000-0000-0000-0000-{name:0>12}"[-36:]
    path = root / "sessions" / f"{task_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    _append_records(
        path,
        _turn_records(
            task_id,
            f"turn-{name}",
            f"response-{name}",
            f"item-{name}",
            base_ms=end_ms or int((time.time() - 5) * 1000),
        ),
    )
    return {"id": task_id, "path": path, "updated": updated, "archived": archived}


def _row(result: dict) -> dict:
    assert result["status"] == "completed", result
    assert result["mode"] == "recent"
    assert result["measurement_method"] == "local_log_estimate"
    return result["samples"][0]


def test_picks_latest_actual_item_end_and_includes_archived_user_threads(tmp_path: Path):
    old = _entry(tmp_path, "1", end_ms=int((time.time() - 20) * 1000), updated=1000, archived=0)
    latest = _entry(tmp_path, "2", end_ms=int((time.time() - 3) * 1000), updated=10, archived=1)
    _write_state(tmp_path, [old, latest])

    result = recent.query_recent(tmp_path)

    sample = _row(result)
    assert sample["model"] == "gpt-6-luna"
    assert sample["effort"] == "low"
    assert sample["output_tokens"] == 100
    assert sample["stream_seconds"] == pytest.approx(2.0)
    assert sample["tps"] == pytest.approx(50.0)
    assert sample["ended_at"] == pytest.approx(int((time.time() - 3) * 1000) / 1000, abs=1)
    assert sample["method"] == "item-event"
    assert sample["sample_key"] == hashlib.sha256(
        "\x1f".join((latest["id"], "turn-2", "response-2")).encode()
    ).hexdigest()


def test_filters_agent_and_benchmark_sources_before_recent_limit(tmp_path: Path):
    user = _entry(tmp_path, "user", updated=1)
    agents = []
    for index in range(20):
        item = _entry(tmp_path, f"agent{index}", updated=100 - index)
        item["thread_source"] = "subagent"
        item["source"] = json.dumps({"subagent": {"thread_spawn": True}})
        agents.append(item)
    _write_state(tmp_path, agents + [user])

    result = recent.query_recent(tmp_path)

    assert _row(result)["sample_key"] == hashlib.sha256(
        "\x1f".join((user["id"], "turn-user", "response-user")).encode()
    ).hexdigest()
    assert result["coverage"]["candidate_paths"] == 1


def test_uses_only_highest_state_database_version(tmp_path: Path):
    stale = _entry(tmp_path, "stale", end_ms=int((time.time() - 100) * 1000))
    current = _entry(tmp_path, "current", end_ms=int((time.time() - 4) * 1000))
    _write_state(tmp_path, [stale])
    _write_state(tmp_path, [current], version=2)

    result = recent.query_recent(tmp_path)

    assert result["coverage"]["state_databases"] == 1
    assert _row(result)["sample_key"] == hashlib.sha256(
        "\x1f".join((current["id"], "turn-current", "response-current")).encode()
    ).hexdigest()


def test_date_directory_fallback_only_runs_when_state_database_is_absent(tmp_path: Path):
    entry = _entry(tmp_path, "000000000001")
    fallback = recent.query_recent(tmp_path)
    assert _row(fallback)["sample_key"] == hashlib.sha256(
        "\x1f".join((entry["id"], "turn-000000000001", "response-000000000001")).encode()
    ).hexdigest()

    _write_state(tmp_path, [])
    no_fallback = recent.query_recent(tmp_path)
    assert no_fallback["status"] == "no_samples"
    assert no_fallback["coverage"]["candidate_paths"] == 0


def test_unreadable_state_is_reported_as_source_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_state(tmp_path, [])

    def denied(path, deadline):
        raise sqlite3.OperationalError("database access denied")

    monkeypatch.setattr(recent, "_read_threads", denied)

    result = recent.query_recent(tmp_path)

    assert result["status"] == "source_unavailable"
    assert "state_unavailable" in result["diagnostics"]


def test_large_rollout_reads_only_bounded_tail_and_repairs_prefix_at_next_turn(tmp_path: Path):
    entry = _entry(tmp_path, "large", end_ms=int((time.time() - 8) * 1000))
    entry["path"].write_bytes(b"x" * (3 * 1024 * 1024) + b"\n")
    for turn in ("prefix-turn", "trusted-turn"):
        _append_records(
            entry["path"],
            _turn_records(
                entry["id"],
                turn,
                f"response-{turn}",
                f"item-{turn}",
                base_ms=int((time.time() - (8 if turn == "prefix-turn" else 5)) * 1000),
            ),
        )
    _write_state(tmp_path, [entry])

    result = recent.query_recent(tmp_path)

    assert result["coverage"]["bytes_read"] <= 2 * 1024 * 1024
    assert _row(result)["sample_key"] == hashlib.sha256(
        "\x1f".join((entry["id"], "trusted-turn", "response-trusted-turn")).encode()
    ).hexdigest()
    assert "incomplete_prefix" in result["diagnostics"]
    assert "incomplete_prefix" not in result["coverage"]["excluded"]


def test_corrupt_record_and_missing_item_end_are_excluded(tmp_path: Path):
    bad = _entry(tmp_path, "bad", end_ms=int((time.time() - 4) * 1000))
    bad["path"].write_text("", encoding="utf-8")
    _append_records(
        bad["path"],
        _turn_records(
            bad["id"],
            "corrupt-turn",
            "corrupt-response",
            "corrupt-item",
            base_ms=int((time.time() - 4) * 1000),
            corrupt_before_usage=True,
        ),
    )
    missing = _entry(tmp_path, "missing", end_ms=int((time.time() - 4) * 1000))
    missing["path"].write_text("", encoding="utf-8")
    missing_records = _turn_records(
            missing["id"],
            "missing-turn",
            "missing-response",
            "missing-item",
            base_ms=int((time.time() - 4) * 1000),
            missing_start=True,
        )
    _append_records(
        missing["path"],
        [record for record in missing_records if record.get("payload", {}).get("type") != "item_completed"],
    )
    _write_state(tmp_path, [bad, missing])

    result = recent.query_recent(tmp_path)

    assert result["status"] == "no_samples"
    assert result["samples"] == []
    assert result["coverage"]["excluded"].get("invalid_record") == 1
    assert result["coverage"]["excluded"].get("missing_item_end") == 1


def test_conflicting_item_completion_model_filter_and_unknown_model_fail_closed(tmp_path: Path):
    conflict = _entry(tmp_path, "conflict", end_ms=int((time.time() - 4) * 1000))
    conflict["path"].write_text("", encoding="utf-8")
    _append_records(
        conflict["path"],
        _turn_records(
            conflict["id"],
            "conflict-turn",
            "conflict-response",
            "conflict-item",
            base_ms=int((time.time() - 4) * 1000),
            duplicate_end_ms=int((time.time() - 3) * 1000),
        ),
    )
    unknown = _entry(tmp_path, "unknown", end_ms=int((time.time() - 4) * 1000))
    unknown["path"].write_text("", encoding="utf-8")
    _append_records(
        unknown["path"],
        _turn_records(
            unknown["id"],
            "unknown-turn",
            "unknown-response",
            "unknown-item",
            base_ms=int((time.time() - 4) * 1000),
            model=None,
        ),
    )
    unknown["model"] = "gpt-6-luna"
    _write_state(tmp_path, [conflict, unknown])

    result = recent.query_recent(tmp_path, model="gpt-6-luna")

    assert result["status"] == "no_samples"
    assert result["coverage"]["excluded"].get("missing_item_end") == 1
    assert result["coverage"]["excluded"].get("unknown_model") == 1


def test_model_filter_is_exact_and_unknown_model_is_not_inferred(tmp_path: Path):
    entry = _entry(tmp_path, "model-filter")
    _write_state(tmp_path, [entry])

    result = recent.query_recent(tmp_path, model="gpt-6-astra")

    assert result["status"] == "no_samples"
    assert result["coverage"]["excluded"].get("model_mismatch") == 1


def test_uses_only_indexed_diagnostic_start_when_embedded_start_is_missing(tmp_path: Path):
    entry = _entry(tmp_path, "diagnostic", end_ms=int((time.time() - 4) * 1000))
    entry["path"].write_text("", encoding="utf-8")
    base_ms = int((time.time() - 4) * 1000)
    _append_records(
        entry["path"],
        _turn_records(
            entry["id"],
            "diagnostic-turn",
            "diagnostic-response",
            "diagnostic-item",
            base_ms=base_ms,
            missing_start=True,
        ),
    )
    _write_state(tmp_path, [entry])
    logs = tmp_path / "logs_1.sqlite"
    with sqlite3.connect(logs) as connection:
        connection.execute(
            "CREATE TABLE logs (id INTEGER PRIMARY KEY, ts REAL, ts_nanos INTEGER, thread_id TEXT, target TEXT, feedback_log_body TEXT)"
        )
        connection.execute("CREATE INDEX logs_thread_id ON logs(thread_id)")
        connection.execute(
            "INSERT INTO logs(ts, ts_nanos, thread_id, target, feedback_log_body) VALUES (?, ?, ?, ?, ?)",
            (
                base_ms / 1000 - 2,
                0,
                entry["id"],
                "codex_core::stream_events_utils",
                'turn.id="diagnostic-turn": Output item item_type="reasoning" item_id="diagnostic-item"',
            ),
        )

    sample = _row(recent.query_recent(tmp_path))

    assert sample["method"] == "stream-log+item-event"
    assert sample["stream_seconds"] == pytest.approx(2.0)


def test_query_is_read_only_and_never_uses_network_or_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    entry = _entry(tmp_path, "readonly")
    state = _write_state(tmp_path, [entry])
    before = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in (state, entry["path"])
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("query_recent must not use a subprocess or network")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)

    assert _row(recent.query_recent(tmp_path))["model"] == "gpt-6-luna"
    after = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in (state, entry["path"])
    }
    assert after == before


def test_budget_stops_between_bounded_rollout_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    first = _entry(tmp_path, "first")
    second = _entry(tmp_path, "second")
    _write_state(tmp_path, [first, second])
    calls = []
    original = recent._read_rollout

    def slow_once(candidate, budget, deadline, diagnostics, coverage):
        calls.append(candidate.task_id)
        time.sleep(0.25)
        return original(candidate, budget, deadline, diagnostics, coverage)

    monkeypatch.setattr(recent, "_read_rollout", slow_once)

    result = recent.query_recent(tmp_path, budget_seconds=0.2)

    assert len(calls) == 1
    assert "time_budget_exhausted" in result["diagnostics"]
    assert result["coverage"]["bytes_read"] <= 2 * 1024 * 1024

