import sqlite3
import uuid
from pathlib import Path

import pytest

from token_pulse.bench_store import BenchmarkStore, speed_data_home

RUN_ID = "b32f790a-90d3-4cb7-9d09-d98ec04b835c"


def record(**changes):
    value = {
        "run_id": RUN_ID,
        "created_at": 1_800_000_000.0,
        "started_at": 1_800_000_001.0,
        "ended_at": 1_800_000_003.0,
        "mode": "benchmark",
        "requested_model": "gpt-6.1-sol",
        "observed_model": "gpt-6.1-sol",
        "effort": "xhigh",
        "provider": "openai",
        "service_tier": None,
        "prompt_id": "codex-speed-v1",
        "prompt_sha256": "a" * 64,
        "tool_version": "0.1.2",
        "codex_version": "0.163.0",
        "status": "completed",
        "output_tokens": 1200,
        "input_tokens": 64,
        "cached_input_tokens": None,
        "reasoning_output_tokens": 300,
        "stream_seconds": 20.0,
        "end_to_end_seconds": 22.0,
        "first_output_seconds": 1.2,
        "tps": 60.0,
        "end_to_end_tps": 54.545,
        "measurement_method": "stream-log",
        "valid_samples": 1,
        "excluded_samples": 0,
        "exclusions": {},
        "diagnostics": [],
        "samples": [
            {
                "output_tokens": 1200,
                "started_at": 1_800_000_001.0,
                "ended_at": 1_800_000_003.0,
                "seconds": 20.0,
                "tps": 60.0,
                "model": "gpt-6.1-sol",
                "effort": "xhigh",
                "exclusion": None,
                "measurement_method": "stream-log",
                "timing_source": None,
            }
        ],
    }
    value.update(changes)
    return value


def test_save_get_and_reopen_preserves_full_allowlisted_run(tmp_path):
    path = tmp_path / "own" / "benchmarks.sqlite3"
    expected = record()
    with BenchmarkStore(path) as store:
        saved = store.save(expected)
        assert saved == expected
        assert store.get(RUN_ID) == expected
        assert store.get(str(uuid.uuid4())) is None

    with BenchmarkStore(path) as reopened:
        assert reopened.get(RUN_ID) == expected

    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    assert connection.execute("SELECT count(*) FROM samples").fetchone()[0] == 1
    connection.close()


def test_recent_history_mode_is_allowlisted_and_short_timeout_is_supported(tmp_path):
    path = tmp_path / "recent.sqlite3"
    recent = record(
        run_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "sample")),
        mode="recent",
        prompt_id=None,
        prompt_sha256=None,
        tool_version=None,
        codex_version=None,
        provider=None,
        service_tier=None,
        input_tokens=None,
        cached_input_tokens=None,
        reasoning_output_tokens=None,
        end_to_end_seconds=None,
        first_output_seconds=None,
        end_to_end_tps=None,
    )
    with BenchmarkStore(path, timeout=0.05) as store:
        assert store.save(recent) == recent
    with BenchmarkStore(path, read_only=True, timeout=0.05) as store:
        assert store.get(recent["run_id"]) == recent

    with pytest.raises(ValueError, match="zero and sixty"):
        BenchmarkStore(tmp_path / "invalid.sqlite3", timeout=61)


def test_read_only_reads_without_changing_database_and_rejects_writes(tmp_path):
    path = tmp_path / "existing database" / "bench marks.sqlite3"
    with BenchmarkStore(path) as store:
        store.save(record())
    original_bytes = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns

    with BenchmarkStore(path, read_only=True) as store:
        assert store.get(RUN_ID) == record()
        assert store.list_runs()[0]["run_id"] == RUN_ID
        with pytest.raises(RuntimeError, match="read-only"):
            store.save(record(run_id=str(uuid.uuid4())))
        with pytest.raises(sqlite3.OperationalError):
            store.connection.execute("CREATE TABLE forbidden(value)")

    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_mtime


def test_read_only_missing_database_does_not_create_file_or_parent(tmp_path):
    parent = tmp_path / "absent" / "database"
    path = parent / "benchmarks.sqlite3"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        BenchmarkStore(path, read_only=True)
    assert not parent.exists()
    assert not path.exists()


def test_read_only_refuses_unknown_schema_without_modifying_database(tmp_path):
    path = tmp_path / "unknown.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version=0")
    connection.close()
    original_bytes = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns

    with pytest.raises(RuntimeError, match="unsupported benchmark database schema"):
        BenchmarkStore(path, read_only=True)

    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_mtime


def test_list_filters_and_limits_without_selecting_sample_rows(tmp_path):
    path = tmp_path / "benchmarks.sqlite3"
    with BenchmarkStore(path) as store:
        store.save(record())
        store.save(
            record(run_id=str(uuid.uuid4()), created_at=1_800_000_004, observed_model="gpt-6-luna")
        )
        queries = []
        store.connection.set_trace_callback(queries.append)

        rows = store.list_runs(limit=1, model="gpt-6.1-sol")
        assert len(rows) == 1
        assert rows[0]["run_id"] == RUN_ID
        assert "samples" not in rows[0]
        assert all("samples" not in query.lower() for query in queries)
        assert [row["observed_model"] for row in store.list_runs(model="gpt-6-luna")] == [
            "gpt-6-luna"
        ]
        assert store.list_runs(limit=0) == []


def test_duplicate_run_is_idempotent_and_conflicting_run_is_rejected(tmp_path):
    with BenchmarkStore(tmp_path / "benchmarks.sqlite3") as store:
        first = record()
        assert store.save(first) == first
        assert store.save(first) == first
        with pytest.raises(ValueError, match="different data"):
            store.save(record(tps=99.0))
        assert store.get(RUN_ID)["tps"] == 60.0
        assert store.connection.execute("SELECT count(*) FROM samples").fetchone()[0] == 1


def test_store_generates_uuid4_and_supports_nested_transaction_calls(tmp_path):
    with BenchmarkStore(tmp_path / "benchmarks.sqlite3") as store:
        generated_input = record(run_id=None)
        generated_input.pop("run_id")
        generated = store.save(generated_input)
        assert uuid.UUID(generated["run_id"]).version == 4

        first_id, second_id = str(uuid.uuid4()), str(uuid.uuid4())
        store.connection.execute("BEGIN")
        store.save(record(run_id=first_id))
        store.save(record(run_id=second_id))
        assert store.connection.in_transaction
        store.connection.execute("ROLLBACK")
        assert store.get(first_id) is None
        assert store.get(second_id) is None


def test_failed_sample_insert_rolls_back_whole_run_and_next_save_succeeds(tmp_path):
    with BenchmarkStore(tmp_path / "benchmarks.sqlite3") as store:
        store.connection.execute(
            """
            CREATE TRIGGER reject_sample BEFORE INSERT ON samples
            WHEN NEW.run_id = 'b32f790a-90d3-4cb7-9d09-d98ec04b835c'
            BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END
            """
        )
        with pytest.raises(sqlite3.IntegrityError, match="synthetic failure"):
            store.save(record())
        assert store.get(RUN_ID) is None
        assert not store.connection.in_transaction

        store.connection.execute("DROP TRIGGER reject_sample")
        next_run = record(run_id=str(uuid.uuid4()))
        assert store.save(next_run) == next_run
        assert store.get(next_run["run_id"]) == next_run


@pytest.mark.parametrize(
    "change",
    [
        {"raw_thread_id": "private-thread-id"},
        {"prompt": "private benchmark prompt"},
        {"title": "private task title"},
        {"source_dump": "private log contents"},
        {"tps": float("nan")},
        {"stream_seconds": float("inf")},
        {"samples": [{"output_tokens": 3, "raw_task_id": "private-id"}]},
        {"samples": [{"tps": float("nan")}]},
        {"samples": [{"seconds": -1}]},
    ],
)
def test_unknown_or_invalid_fields_are_rejected_without_persisting_private_values(tmp_path, change):
    path = tmp_path / "benchmarks.sqlite3"
    with BenchmarkStore(path) as store:
        with pytest.raises(ValueError):
            store.save(record(**change))
        assert store.get(RUN_ID) is None
    assert b"private" not in path.read_bytes().lower()


def test_missing_token_metrics_stay_null_and_reasoning_is_not_added(tmp_path):
    run = record(output_tokens=None, reasoning_output_tokens=40, tps=None, samples=[])
    with BenchmarkStore(tmp_path / "benchmarks.sqlite3") as store:
        saved = store.save(run)
        assert saved["output_tokens"] is None
        assert saved["tps"] is None
        assert saved["reasoning_output_tokens"] == 40
        assert store.get(RUN_ID)["output_tokens"] is None


def test_safe_diagnostics_are_preserved_without_exception_text(tmp_path):
    run = record(diagnostics=["model_unconfirmed", "timing_boundary_missing"])
    with BenchmarkStore(tmp_path / "benchmarks.sqlite3") as store:
        saved = store.save(run)
        assert saved["diagnostics"] == ["model_unconfirmed", "timing_boundary_missing"]
        assert store.get(RUN_ID)["diagnostics"] == saved["diagnostics"]
        with pytest.raises(ValueError):
            store.save(record(diagnostics=["RuntimeError: private details here"]))


def test_storage_home_override_and_canonical_path_ignore_codex_home(tmp_path, monkeypatch):
    override = tmp_path / "custom speed data"
    monkeypatch.setenv("CODEX_SPEED_DATA_DIR", str(override))
    assert speed_data_home() == override.resolve()
    with BenchmarkStore() as store:
        assert store.path == (override / "benchmarks.sqlite3").resolve()

    monkeypatch.delenv("CODEX_SPEED_DATA_DIR")
    home = tmp_path / "normal-user"
    app_container = tmp_path / "app-container-localappdata"
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "alternate-codex-source"))
    monkeypatch.setenv("LOCALAPPDATA", str(app_container))
    expected = (home / ".codex" / "codex-speed").resolve()
    assert speed_data_home() == expected
    with BenchmarkStore() as store:
        assert store.path == expected / "benchmarks.sqlite3"


def test_newer_schema_and_unversioned_nonempty_database_are_refused(tmp_path):
    newer = tmp_path / "newer.sqlite3"
    connection = sqlite3.connect(newer)
    connection.execute("PRAGMA user_version=2")
    connection.close()
    with pytest.raises(RuntimeError, match="newer"):
        BenchmarkStore(newer)

    legacy = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(legacy)
    connection.execute("CREATE TABLE unrelated(value)")
    connection.close()
    with pytest.raises(RuntimeError, match="unversioned"):
        BenchmarkStore(legacy)
