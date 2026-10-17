"""Fast, read-only lookup of recent Codex generation samples."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from token_pulse.domain import Context
from token_pulse.logs import LogEvidence, label
from token_pulse.rollout import Response, Rollout

_STATE = re.compile(r"state_(\d+)\.sqlite\Z")
_LOGS = re.compile(r"logs_(\d+)\.sqlite\Z")
_THREAD_ID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I)
_MAX_STATE_FILES = 16
_MAX_STATE_ATTEMPTS = 3
_MAX_CANDIDATES = 12
_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_TOTAL_BYTES = 16 * 1024 * 1024
_MAX_LINE_BYTES = 1024 * 1024
_MAX_LOG_ITEMS = 48
_MAX_LOG_ROWS = 256
_STALE_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class _Candidate:
    task_id: str
    path: Path
    context: Context
    updated_at: float


@dataclass
class _Parsed:
    candidate: _Candidate
    rollout: Rollout
    # Completion boundaries are kept separately so an incomplete item event
    # cannot make Rollout fall back to a JSONL record timestamp for the end.
    ends: dict[str, tuple[str, float] | None]
    starts: dict[str, tuple[str, float] | None]
    bytes_read: int
    non_user_source: bool = False


def query_recent(
    source: Path,
    *,
    model: str | None = None,
    limit: int = 1,
    budget_seconds: float = 2.0,
) -> dict:
    """Read the latest trustworthy local rollout timing without writing or spawning.

    ``source`` is the Codex home containing ``state_*.sqlite`` and ``sessions``.
    Response timestamps and full-turn durations are deliberately never used as
    generation boundaries.
    """
    started = time.monotonic()
    source_valid = True
    try:
        root = Path(source).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        root = Path("__invalid_codex_speed_source__")
        source_valid = False
    budget = _finite_budget(budget_seconds)
    deadline = started + budget
    requested_model = label(model) if model is not None else None
    diagnostics: set[str] = set()
    coverage = {
        "state_databases": 0,
        "candidate_paths": 0,
        "files_read": 0,
        "bytes_read": 0,
        "complete_lines": 0,
        "excluded": {},
    }
    if not source_valid or not root.is_dir():
        return _result("source_unavailable", started, coverage, diagnostics)

    candidates, used_state = _discover(root, deadline, diagnostics, coverage)
    if not candidates and not used_state and "state_unavailable" not in diagnostics:
        candidates = _fallback_candidates(root, deadline, diagnostics)
    coverage["candidate_paths"] = len(candidates)
    parsed: list[_Parsed] = []
    remaining = _MAX_TOTAL_BYTES
    for candidate in candidates[:_MAX_CANDIDATES]:
        if time.monotonic() >= deadline:
            diagnostics.add("time_budget_exhausted")
            break
        if remaining <= 0:
            diagnostics.add("read_budget_exhausted")
            break
        cap = min(_MAX_FILE_BYTES, remaining)
        entry = _read_rollout(candidate, cap, deadline, diagnostics, coverage)
        if entry is None:
            continue
        parsed.append(entry)
        remaining -= entry.bytes_read
    if not used_state and not parsed and not candidates:
        diagnostics.add("no_recent_rollouts")

    # Most samples have item start/end boundaries embedded in the rollout. Only
    # seek diagnostic-log starts for responses whose start boundary is absent.
    needs_start: dict[str, list[str]] = {}
    for entry in parsed:
        responses = sorted(
            _responses(entry.rollout),
            key=lambda response: max(
                (entry.ends.get(item_id) or ("", 0.0))[1] for item_id, _ in response.items
            ) if response.items else 0.0,
            reverse=True,
        )
        for response in responses:
            if (
                response.issue is None
                and response.items
                and response.items[0][0] not in entry.starts
                and not _valid_boundary(entry.rollout.item_timings.get(response.items[0][0]), response.turn_id)
            ):
                values = needs_start.setdefault(entry.candidate.task_id, [])
                if response.items[0][0] not in values:
                    values.append(response.items[0][0])
    evidence = LogEvidence()
    if needs_start and time.monotonic() < deadline:
        _load_log_starts(root, needs_start, evidence, deadline, diagnostics, coverage)

    now = time.time()
    samples: list[dict] = []
    for entry in parsed:
        for response in _responses(entry.rollout):
            if entry.non_user_source:
                coverage["excluded"]["non_user_source"] = coverage["excluded"].get("non_user_source", 0) + 1
                continue
            row, exclusion = _sample_row(entry, response, evidence, now, requested_model)
            if row is None:
                reason = exclusion or "incomplete_boundary"
                excluded = coverage["excluded"]
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            samples.append(row)
    samples.sort(key=lambda row: (row["ended_at"], row["sample_key"]), reverse=True)
    if coverage["excluded"]:
        diagnostics.add("samples_excluded")
    if not samples:
        if not root.exists() or "state_unavailable" in diagnostics:
            status = "source_unavailable"
        elif parsed or candidates:
            status = "no_samples"
        else:
            status = "no_samples"
    else:
        status = "completed"
    coverage["valid_samples"] = len(samples)
    return _result(status, started, coverage, diagnostics, samples[: _limit(limit)])


def _finite_budget(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 2.0
    if not math.isfinite(number):
        return 2.0
    return min(10.0, max(0.01, number))


def _limit(value: int) -> int:
    return min(20, max(1, value if type(value) is int else 1))


def _result(status: str, started: float, coverage: dict, diagnostics: set[str], samples=None) -> dict:
    return {
        "status": status,
        "mode": "recent",
        "measurement_method": "local_log_estimate",
        "samples": list(samples or []),
        "elapsed_seconds": max(0.0, time.monotonic() - started),
        "coverage": coverage,
        "diagnostics": sorted(diagnostics),
    }


def _inside(root: Path, path: Path) -> Path | None:
    try:
        resolved = path.expanduser().resolve(strict=True)
        return resolved if resolved.is_relative_to(root) else None
    except (OSError, RuntimeError, ValueError):
        return None


def _state_paths(root: Path, deadline: float, diagnostics: set[str]) -> list[tuple[int, Path]]:
    found = []
    try:
        with os.scandir(root) as entries:
            for index, entry in enumerate(entries):
                if index >= 512 or time.monotonic() >= deadline:
                    break
                match = _STATE.fullmatch(entry.name)
                if not match:
                    continue
                path = _inside(root, Path(entry.path))
                if path is not None and path.is_file():
                    found.append((int(match[1]), path))
    except OSError:
        diagnostics.add("state_unavailable")
    return sorted(found, key=lambda item: item[0], reverse=True)[:_MAX_STATE_FILES]


def _discover(root: Path, deadline: float, diagnostics: set[str], coverage: dict) -> tuple[list[_Candidate], bool]:
    paths = _state_paths(root, deadline, diagnostics)
    if not paths:
        return [], False
    for _, state_path in paths[:_MAX_STATE_ATTEMPTS]:
        if time.monotonic() >= deadline:
            diagnostics.add("time_budget_exhausted")
            break
        coverage["state_databases"] += 1
        try:
            rows = _read_threads(state_path, deadline)
        except (OSError, sqlite3.Error, ValueError, TypeError):
            diagnostics.add("state_unavailable")
            continue
        candidates = []
        for task_id, raw_path, updated, thread_source, raw_source in rows:
            if not isinstance(task_id, str) or not task_id or not isinstance(raw_path, str):
                continue
            if _source_excluded(thread_source, raw_source):
                continue
            path = _inside(root, Path(raw_path))
            if path is None or path.suffix.lower() != ".jsonl" or not path.is_file():
                continue
            candidates.append(_Candidate(task_id, path, Context(), _sort_time(updated)))
        if candidates:
            # Keep the newest record for each rollout and bound the lookup.
            best: dict[tuple[str, Path], _Candidate] = {}
            for candidate in candidates:
                key = candidate.task_id, candidate.path
                previous = best.get(key)
                if previous is None or candidate.updated_at > previous.updated_at:
                    best[key] = candidate
            return sorted(
                best.values(), key=lambda item: (item.updated_at, item.task_id), reverse=True
            )[:_MAX_CANDIDATES], True
        # A readable schema with no paths is not a reason to combine stale DBs.
        return [], True
    return [], False


def _read_threads(path: Path, deadline: float) -> list[tuple]:
    remaining = max(0.01, deadline - time.monotonic())
    uri = path.as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=min(0.15, remaining))) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(threads)")}
        required = {"id", "rollout_path", "updated_at"}
        if not required.issubset(columns):
            raise sqlite3.DatabaseError("unsupported state schema")
        optional = [name if name in columns else f"NULL AS {name}" for name in ("thread_source", "source")]
        filters = []
        if "thread_source" in columns:
            filters.append("(thread_source IS NULL OR lower(thread_source)='user')")
        if "source" in columns:
            filters.append("(source IS NULL OR trim(source) NOT LIKE ?)")
        where = " WHERE " + " AND ".join(filters) if filters else ""
        parameters = ("{%\"subagent\"%", _MAX_CANDIDATES) if "source" in columns else (_MAX_CANDIDATES,)
        cursor = connection.execute(
            "SELECT id, rollout_path, updated_at, "
            + ", ".join(optional)
            + " FROM threads"
            + where
            + " ORDER BY updated_at DESC LIMIT ?",
            parameters,
        )
        return cursor.fetchall()


def _sort_time(value: object) -> float:
    if type(value) in (int, float) and math.isfinite(value):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                return parsed.timestamp()
        except (ValueError, OverflowError, OSError):
            pass
    return 0.0


def _source_excluded(thread_source: object, source: object) -> bool:
    if isinstance(thread_source, str) and thread_source.strip():
        if thread_source.strip().casefold() != "user":
            return True
    value = source
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.casefold() in {"subagent", "guardian_review", "agent_created_thread"}:
            return True
        if stripped.startswith(("{", "[")):
            try:
                value = json.loads(stripped)
            except (ValueError, RecursionError):
                return False

    def has_subagent_marker(node: object) -> bool:
        if isinstance(node, dict):
            for key, item in node.items():
                if isinstance(key, str) and key.casefold() == "subagent":
                    return True
                if isinstance(key, str) and key.casefold() in {
                    "thread_source",
                    "originator",
                    "kind",
                    "type",
                } and isinstance(item, str) and item.casefold() in {
                    "subagent",
                    "guardian_review",
                    "agent_created_thread",
                }:
                    return True
                if has_subagent_marker(item):
                    return True
        elif isinstance(node, list):
            return any(has_subagent_marker(item) for item in node)
        return False

    return has_subagent_marker(value)


def _fallback_candidates(root: Path, deadline: float, diagnostics: set[str]) -> list[_Candidate]:
    now = datetime.now().astimezone()
    directories = [root / "sessions"]
    for timezone_now in (now, datetime.now(UTC)):
        directories.extend(
            root / "sessions" / (timezone_now - timedelta(days=offset)).strftime("%Y/%m/%d")
            for offset in range(4)
        )
    directories = list(dict.fromkeys(directories))
    found: list[tuple[float, Path, str]] = []
    for directory in directories:
        if time.monotonic() >= deadline:
            diagnostics.add("time_budget_exhausted")
            break
        try:
            with os.scandir(directory) as entries:
                for index, entry in enumerate(entries):
                    if index >= 512 or time.monotonic() >= deadline:
                        break
                    if not entry.name.endswith(".jsonl"):
                        continue
                    match = _THREAD_ID.search(entry.name)
                    if not match:
                        continue
                    path = _inside(root, Path(entry.path))
                    if path is None:
                        continue
                    try:
                        found.append((entry.stat(follow_symlinks=False).st_mtime, path, match[0]))
                    except OSError:
                        continue
        except OSError:
            continue
    result = [
        _Candidate(task_id, path, Context(), mtime)
        for mtime, path, task_id in sorted(found, reverse=True)[:_MAX_CANDIDATES]
    ]
    return result


def _read_rollout(candidate: _Candidate, budget: int, deadline: float, diagnostics: set[str], coverage: dict) -> _Parsed | None:
    if time.monotonic() >= deadline:
        diagnostics.add("time_budget_exhausted")
        return None
    try:
        with candidate.path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            size = before.st_size
            # Reserve one byte of the fixed read cap to test whether the window
            # begins on a record boundary, while still reaching the file end.
            start = max(0, size - budget + int(size > budget))
            previous_byte = b"\n"
            if start > 0:
                stream.seek(start - 1)
                previous_byte = stream.read(1)
            stream.seek(start)
            window = min(max(0, budget - int(start > 0)), size - start)
            data = stream.read(window)
            after = os.fstat(stream.fileno())
            bytes_read = len(data) + int(start > 0)
            coverage["files_read"] += 1
            coverage["bytes_read"] += bytes_read
        current = candidate.path.stat()
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or (
            current.st_dev,
            current.st_ino,
        ) != (after.st_dev, after.st_ino):
            diagnostics.add("rollout_rotated")
            return None
    except OSError:
        diagnostics.add("rollout_unavailable")
        return None
    rollout = Rollout(candidate.task_id, context=Context())
    ends: dict[str, tuple[str, float] | None] = {}
    starts: dict[str, tuple[str, float] | None] = {}
    non_user_source = False
    prefix = start > 0 and previous_byte != b"\n"
    if prefix:
        diagnostics.add("incomplete_prefix")
        newline = data.find(b"\n")
        if newline < 0:
            diagnostics.add("incomplete_prefix")
            return _Parsed(candidate, rollout, ends, starts, bytes_read, False)
        data = data[newline + 1 :]
        rollout.damage("record_gap")
    lines = data.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        lines.pop()
        diagnostics.add("incomplete_tail")
    for line in lines:
        if time.monotonic() >= deadline:
            diagnostics.add("time_budget_exhausted")
            break
        coverage["complete_lines"] += 1
        if len(line) > _MAX_LINE_BYTES:
            rollout.damage("record_gap")
            diagnostics.add("oversized_line")
            continue
        try:
            record = json.loads(line)
        except (ValueError, UnicodeError, RecursionError):
            rollout.damage("record_gap")
            diagnostics.add("malformed_record")
            continue
        if not isinstance(record, dict):
            rollout.damage("record_gap")
            diagnostics.add("malformed_record")
            continue
        payload = record.get("payload")
        if record.get("type") == "session_meta" and isinstance(payload, dict):
            non_user_source |= _source_excluded(None, payload.get("source"))
        _capture_item_bounds(record, candidate.task_id, starts, ends)
        rollout.feed(record)
        del record
    return _Parsed(candidate, rollout, ends, starts, bytes_read, non_user_source)


def _capture_item_bounds(record: dict, task_id: str, starts: dict, ends: dict) -> None:
    payload = record.get("payload")
    if not isinstance(payload, dict) or record.get("type") != "event_msg":
        return
    if payload.get("type") != "item_completed" or payload.get("thread_id") != task_id:
        return
    item = payload.get("item")
    turn = payload.get("turn_id")
    if not isinstance(item, dict) or item.get("type") not in {"Reasoning", "AgentMessage"}:
        return
    item_id = item.get("id")
    end_ms = payload.get("completed_at_ms")
    if not isinstance(item_id, str) or not isinstance(turn, str) or not turn:
        return
    if type(end_ms) is not int or not 0 < end_ms <= 2**53:
        return
    end = end_ms / 1000
    previous_end = ends.get(item_id)
    value_end = (turn, end)
    ends[item_id] = value_end if previous_end in (None, value_end) else None
    start_ms = payload.get("started_at_ms")
    if type(start_ms) is int and 0 < start_ms <= end_ms:
        value_start = (turn, start_ms / 1000)
        previous_start = starts.get(item_id)
        starts[item_id] = value_start if previous_start in (None, value_start) else None


def _responses(rollout: Rollout) -> list[Response]:
    responses = list(rollout.responses) + list(rollout.corrections)
    unique = {}
    for response in responses:
        unique[(response.turn_id, response.id)] = response
    return list(unique.values())


def _valid_boundary(value, turn_id: str) -> bool:
    if value is None:
        return False
    if hasattr(value, "ambiguous"):
        return not value.ambiguous and value.turn_id == turn_id and value.start <= value.end
    return value[0] == turn_id and math.isfinite(value[1])


def _sample_row(entry: _Parsed, response: Response, evidence: LogEvidence, now: float, requested_model: str | None):
    if response.issue:
        return None, response.issue
    if not response.turn_id or not response.items:
        return None, "missing_boundary"
    tokens = response.tokens
    if type(tokens) is not int or tokens <= 0 or tokens > 2**63 - 1:
        return None, "invalid_tokens"
    model = label(response.context.model)
    effort = label(response.context.effort)
    if not model:
        return None, "unknown_model"
    if requested_model and model.casefold() != requested_model.casefold():
        return None, "model_mismatch"

    item_ids = [item_id for item_id, _ in response.items]
    completions = []
    for item_id in item_ids:
        end = entry.ends.get(item_id)
        if end is None or end[0] != response.turn_id:
            return None, "missing_item_end"
        completions.append(end[1])
    ended_at = max(completions)
    first_id = item_ids[0]
    embedded_start = entry.starts.get(first_id)
    start_at = None
    method = None
    if first_id in entry.starts and embedded_start is None:
        return None, "ambiguous_item_start"
    if _valid_boundary(embedded_start, response.turn_id):
        start_at = embedded_start[1]
        method = "item-event"
    else:
        diagnostic = evidence.starts.get((entry.candidate.task_id, first_id))
        if diagnostic is None or diagnostic.ambiguous or diagnostic.turn_id != response.turn_id:
            return None, "missing_start"
        start_at = diagnostic.at
        method = "stream-log+item-event"
    for item_id in item_ids:
        item_start = entry.starts.get(item_id)
        item_end = entry.ends.get(item_id)
        if item_end is None or item_end[0] != response.turn_id:
            return None, "ambiguous_item_end"
        if item_start is not None and (
            item_start[0] != response.turn_id or item_start[1] < start_at - 0.001 or item_start[1] > item_end[1] + 0.001
        ):
            return None, "ambiguous_item_start"
    seconds = ended_at - start_at
    if not math.isfinite(seconds) or seconds <= 0 or ended_at > now + 300:
        return None, "invalid_duration"
    age = max(0.0, now - ended_at)
    key = hashlib.sha256("\x1f".join((entry.candidate.task_id, response.turn_id, response.id)).encode()).hexdigest()
    return {
        "sample_key": key,
        "model": model,
        "effort": effort,
        "output_tokens": tokens,
        "stream_seconds": seconds,
        "tps": tokens / seconds,
        "ended_at": ended_at,
        "age_seconds": age,
        "stale": ended_at > now or age > _STALE_SECONDS,
        "method": method,
    }, None


def _load_log_starts(root: Path, needs: dict[str, list[str]], evidence: LogEvidence, deadline: float, diagnostics: set[str], coverage: dict) -> None:
    paths = []
    try:
        with os.scandir(root) as entries:
            for index, entry in enumerate(entries):
                if index >= 512 or time.monotonic() >= deadline:
                    diagnostics.add("time_budget_exhausted")
                    break
                match = _LOGS.fullmatch(entry.name)
                if match:
                    path = _inside(root, Path(entry.path))
                    if path is not None:
                        paths.append((int(match[1]), path))
    except OSError:
        diagnostics.add("log_unavailable")
        return
    if not paths:
        diagnostics.add("log_unavailable")
        return
    path = max(paths, key=lambda item: item[0])[1]
    item_ids = list(dict.fromkeys(item for values in needs.values() for item in values))[:_MAX_LOG_ITEMS]
    if not item_ids:
        return
    try:
        remaining = max(0.01, deadline - time.monotonic())
        uri = path.as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=min(0.15, remaining))) as connection:
            connection.execute("PRAGMA query_only=ON")
            connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            indexes = connection.execute("PRAGMA index_list(logs)").fetchall()
            has_thread_index = False
            for index in indexes:
                quoted_index = str(index[1]).replace("'", "''")
                columns = connection.execute(f"PRAGMA index_info('{quoted_index}')").fetchall()
                if columns and columns[0][2] == "thread_id":
                    has_thread_index = True
                    break
            if not has_thread_index:
                diagnostics.add("log_index_unavailable")
                return
            for task_id, wanted in needs.items():
                if time.monotonic() >= deadline:
                    diagnostics.add("time_budget_exhausted")
                    break
                ids = [item for item in item_ids if item in wanted]
                if not ids:
                    continue
                clauses = " OR ".join("instr(feedback_log_body, ?) > 0" for _ in ids)
                query = (
                    "SELECT ts, ts_nanos, thread_id, target, "
                    "CASE WHEN length(feedback_log_body) <= 16000 THEN feedback_log_body END "
                    "FROM logs WHERE thread_id=? AND target=? AND ("
                    + clauses
                    + ") ORDER BY id DESC LIMIT ?"
                )
                rows = connection.execute(
                    query,
                    (task_id, "codex_core::stream_events_utils", *ids, _MAX_LOG_ROWS),
                ).fetchall()
                for seconds, nanos, thread, target, body in reversed(rows):
                    if isinstance(body, str) and type(seconds) in (int, float) and type(nanos) in (int, float):
                        evidence.accept(thread, seconds + nanos / 1e9, target, body)
            coverage["diagnostic_item_ids"] = sum(len(values) for values in needs.values())
    except (OSError, sqlite3.Error, TypeError, ValueError):
        diagnostics.add("log_unavailable")
