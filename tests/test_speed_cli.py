import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path
from types import ModuleType

import pytest

from token_pulse import speed
from token_pulse.bench_store import BenchmarkStore


def _benchmark_module(run):
    module = ModuleType("token_pulse.benchmark")
    module.run = run
    return module


def _valid_result(**changes):
    result = {
        "status": "completed",
        "observed_model": "gpt-6-luna",
        "output_tokens": 120,
        "stream_seconds": 2.0,
        "end_to_end_seconds": 2.4,
        "first_output_seconds": 0.3,
        "tps": 60.0,
        "measurement_method": "stream-log",
        "valid_samples": 1,
        "excluded_samples": 0,
        "exclusions": {},
        "samples": [],
    }
    result.update(changes)
    return result


def _invoke(args):
    stdout, stderr = StringIO(), StringIO()
    code = speed.main(args, stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def test_help_and_version_do_not_import_data_or_gui_modules():
    script = """
import sys
from token_pulse import speed
forbidden = {
    'token_pulse.benchmark', 'token_pulse.bench_store', 'token_pulse.collector',
    'token_pulse.app', 'PySide6'
}
try:
    speed.main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
assert forbidden.isdisjoint(sys.modules)
try:
    speed.main(['--version'])
except SystemExit as exc:
    assert exc.code == 0
assert forbidden.isdisjoint(sys.modules)
"""
    env = os.environ.copy()
    source = str(
        Path(__file__).resolve().parents[1]
        / "plugins"
        / "codex-speed"
        / "skills"
        / "codex-speed"
        / "_vendor"
    )
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "codex-speed 0.2.0" in completed.stdout


@pytest.mark.parametrize(
    "args",
    [
        ["--data-dir", "{dir}", "history"],
        ["history", "--data-dir", "{dir}"],
    ],
)
def test_data_dir_is_accepted_before_or_after_subcommand_without_creating_history(tmp_path, args):
    directory = tmp_path / "missing-db"
    code, output, _ = _invoke([arg.format(dir=directory) for arg in args])
    assert code == 0
    assert json.loads(output) == []
    assert not directory.exists()


def test_write_store_keeps_legacy_constructor_call_shape(tmp_path, monkeypatch):
    class LegacyStore:
        def __init__(self, path):
            self.path = path

    monkeypatch.setattr("token_pulse.bench_store.BenchmarkStore", LegacyStore)
    monkeypatch.setattr(speed, "_database_path", lambda data_dir: data_dir / "benchmarks.sqlite3")
    assert speed._open_store(tmp_path).path == tmp_path / "benchmarks.sqlite3"


def test_benchmark_calls_runner_and_saves_one_clean_record(tmp_path, monkeypatch):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return _valid_result(requested_model="gpt-6-luna", effort="high")

    monkeypatch.setitem(sys.modules, "token_pulse.benchmark", _benchmark_module(run))
    data_dir = tmp_path / "benchmarks"
    code, output, _ = _invoke(
        [
            "--data-dir",
            str(data_dir),
            "benchmark",
            "--model",
            "gpt-6-luna",
            "--effort",
            "high",
            "--timeout",
            "17.5",
            "--codex-home",
            str(tmp_path / "codex-home"),
            "--codex-command",
            str(tmp_path / "codex.exe"),
        ]
    )
    result = json.loads(output)
    assert code == 0
    assert calls == [
        {
            "model": "gpt-6-luna",
            "effort": "high",
            "timeout": 17.5,
            "source": (tmp_path / "codex-home").resolve(),
            "codex_command": str(tmp_path / "codex.exe"),
        }
    ]
    assert result["status"] == "completed"
    assert result["tokens"] == 120
    assert result["tps"] == 60.0
    assert result["history_saved"] is True
    with BenchmarkStore(data_dir / "benchmarks.sqlite3") as store:
        saved = store.get(result["run_id"])
    assert saved["mode"] == "benchmark"
    assert saved["requested_model"] == "gpt-6-luna"
    assert saved["effort"] == "high"


def test_ultra_effort_is_accepted_forwarded_and_saved(tmp_path, monkeypatch):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return _valid_result(effort="ultra")

    monkeypatch.setitem(sys.modules, "token_pulse.benchmark", _benchmark_module(run))
    code, output, _ = _invoke(["--data-dir", str(tmp_path), "benchmark", "--effort", "ultra"])
    result = json.loads(output)
    assert code == 0
    assert calls[0]["effort"] == "ultra"
    with BenchmarkStore(tmp_path / "benchmarks.sqlite3") as store:
        assert store.get(result["run_id"])["effort"] == "ultra"


def test_runner_exception_is_redacted_and_failed_attempt_is_saved(tmp_path, monkeypatch):
    secret = "private-path-and-user-prompt"

    def run(**_kwargs):
        raise RuntimeError(secret)

    monkeypatch.setitem(sys.modules, "token_pulse.benchmark", _benchmark_module(run))
    code, output, _ = _invoke(["--data-dir", str(tmp_path), "benchmark"])
    result = json.loads(output)
    assert code == 2
    assert result["status"] == "failed"
    assert result["history_saved"] is True
    assert result["error_code"] == "benchmark_failed"
    assert secret not in output
    with BenchmarkStore(tmp_path / "benchmarks.sqlite3") as store:
        assert store.get(result["run_id"])["status"] == "failed"


def test_valid_measurement_is_reported_even_when_history_write_fails(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "token_pulse.benchmark",
        _benchmark_module(lambda **_kwargs: _valid_result()),
    )

    class BrokenStore:
        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def save(self, _record):
            raise OSError("private path")

    monkeypatch.setattr(speed, "_open_store", lambda _data_dir: BrokenStore(None))
    code, output, _ = _invoke(["benchmark"])
    result = json.loads(output)
    assert code == 2
    assert result["tokens"] == 120
    assert result["tps"] == 60.0
    assert result["history_saved"] is False
    assert result["error_code"] == "history_save_failed"
    assert "private path" not in output


def test_completed_without_valid_rate_is_marked_incomplete(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "token_pulse.benchmark",
        _benchmark_module(
            lambda **_kwargs: _valid_result(output_tokens=None, stream_seconds=None, tps=None)
        ),
    )
    code, output, _ = _invoke(["--data-dir", str(tmp_path), "benchmark"])
    result = json.loads(output)
    assert code == 2
    assert result["status"] == "incomplete"
    assert result["tokens"] is None
    assert result["tps"] is None
    assert result["error_code"] == "no_valid_measurement"
    assert result["history_saved"] is True


def test_history_show_and_export_open_the_existing_database_read_only(tmp_path, monkeypatch):
    path = tmp_path / "benchmarks.sqlite3"
    with BenchmarkStore(path) as store:
        saved = store.save(
            {
                "created_at": 1_800_000_000.0,
                "mode": "benchmark",
                "status": "completed",
                "requested_model": "gpt-6-luna",
                "observed_model": "gpt-6-luna",
                "output_tokens": 7,
                "stream_seconds": 1.0,
                "tps": 7.0,
                "measurement_method": "stream-log",
                "valid_samples": 1,
                "excluded_samples": 0,
            }
        )

    read_modes = []
    original_store = BenchmarkStore

    class RecordingStore:
        def __init__(self, store_path, *, read_only=False):
            read_modes.append(read_only)
            self.store = original_store(store_path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.store.__exit__(*args)

        def list_runs(self, **kwargs):
            return self.store.list_runs(**kwargs)

        def get(self, run_id):
            return self.store.get(run_id)

    monkeypatch.setattr("token_pulse.bench_store.BenchmarkStore", RecordingStore)
    history_code, history_output, _ = _invoke(
        ["history", "--data-dir", str(tmp_path), "--limit", "1", "--model", "gpt-6-luna"]
    )
    assert history_code == 0
    assert json.loads(history_output)[0]["run_id"] == saved["run_id"]

    show_code, show_output, _ = _invoke(["--data-dir", str(tmp_path), "show", saved["run_id"]])
    assert show_code == 0
    assert json.loads(show_output)["run_id"] == saved["run_id"]

    export_code, export_output, _ = _invoke(
        ["export", "--data-dir", str(tmp_path), "--model", "gpt-6-luna", "--limit", "1"]
    )
    assert export_code == 0
    assert json.loads(export_output.splitlines()[0])["run_id"] == saved["run_id"]
    assert read_modes == [True, True, True]


@pytest.mark.parametrize(
    "args",
    [
        ["history", "--limit", "0"],
        ["history", "--limit", "1001"],
        ["export", "--limit", "0"],
        ["benchmark", "--timeout", "nan"],
        ["benchmark", "--timeout", "0"],
        ["watch", "--seconds", "inf"],
    ],
)
def test_invalid_cli_values_fail_before_opening_a_data_source(args, monkeypatch):
    def forbidden(**_kwargs):
        raise AssertionError("benchmark runner must not be called")

    monkeypatch.setitem(sys.modules, "token_pulse.benchmark", _benchmark_module(forbidden))
    with pytest.raises(SystemExit) as exc:
        speed.main(args, stdout=StringIO(), stderr=StringIO())
    assert exc.value.code == 2


def test_watch_forwards_updates_and_saves_only_final_aggregate(tmp_path, monkeypatch):
    thread_id = "private-thread-id"

    def run(_source, *, seconds, samples, thread_id: str | None, stdout, stderr):
        assert seconds == 12.0
        assert samples == 0
        assert thread_id == "private-thread-id"
        stdout.write('{"kind":"ready","seconds":12.0}\n')
        stdout.write('{"kind":"update","models":[]}\n')
        stdout.write(
            '{"kind":"final","models":[{"model":"gpt-6-luna","tokens":50,'
            '"weighted_tps":25.0,"stream_seconds":2.0,"valid_samples":1,'
            '"excluded_samples":2}],"diagnostics":{"collector":[],"exclusions":{}},'
            '"reason":"timeout","no_samples":false}\n'
        )
        return 0

    module = ModuleType("token_pulse.watch")
    module.run = run
    monkeypatch.setitem(sys.modules, "token_pulse.watch", module)
    out, err = StringIO(), StringIO()
    code = speed.main(
        [
            "watch",
            "--seconds",
            "12",
            "--thread-id",
            thread_id,
            "--data-dir",
            str(tmp_path),
        ],
        stdout=out,
        stderr=err,
    )
    lines = out.getvalue().splitlines()
    assert code == 0
    assert [json.loads(line)["kind"] for line in lines] == ["ready", "update", "final"]
    final = json.loads(lines[-1])
    assert final["history_saved"] is True
    assert len(final["run_ids"]) == 1
    assert thread_id not in out.getvalue()
    with BenchmarkStore(tmp_path / "benchmarks.sqlite3") as store:
        record = store.get(final["run_ids"][0])
    assert record["mode"] == "watch"
    assert record["observed_model"] == "gpt-6-luna"
    assert record["output_tokens"] == 50
    assert record["tps"] == 25.0
    assert record["valid_samples"] == 1
    assert record["excluded_samples"] == 2
    assert record["samples"] == []
    assert thread_id.encode() not in (tmp_path / "benchmarks.sqlite3").read_bytes()


def test_watch_without_models_saves_one_incomplete_attempt(tmp_path, monkeypatch):
    def run(_source, *, seconds, samples, thread_id, stdout, stderr):
        stdout.write(
            '{"kind":"final","models":[],"diagnostics":{"collector":["sqlite_busy"],'
            '"exclusions":{}},"reason":"timeout","no_samples":true}\n'
        )
        return 0

    module = ModuleType("token_pulse.watch")
    module.run = run
    monkeypatch.setitem(sys.modules, "token_pulse.watch", module)
    out = StringIO()
    code = speed.main(["watch", "--data-dir", str(tmp_path)], stdout=out, stderr=StringIO())
    final = json.loads(out.getvalue().splitlines()[-1])
    assert code == 2
    assert final["status"] == "incomplete"
    assert len(final["run_ids"]) == 1
    with BenchmarkStore(tmp_path / "benchmarks.sqlite3") as store:
        record = store.get(final["run_ids"][0])
    assert record["mode"] == "watch"
    assert record["status"] == "incomplete"
    assert record["output_tokens"] is None
    assert record["tps"] is None
    assert record["diagnostics"] == ["sqlite_busy"]
