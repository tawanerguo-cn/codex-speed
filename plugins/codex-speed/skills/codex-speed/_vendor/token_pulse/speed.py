"""Small, headless command line for Codex speed queries and benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

VERSION = "0.3.0"
_MAX_LIMIT = 1000
_SAFE_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,99}\Z")
_UUID_LIKE = re.compile(r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\Z")
_SAFE_LABEL = re.compile(r"[\w .:+@-]{1,100}\Z", re.UNICODE)
_SAFE_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _positive_finite(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def _bounded_limit(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be an integer from 1 to {_MAX_LIMIT}") from exc
    if result < 1 or result > _MAX_LIMIT:
        raise argparse.ArgumentTypeError(f"must be an integer from 1 to {_MAX_LIMIT}")
    return result


def _nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return result


def _json_line(value: object, stream: TextIO = sys.stdout) -> None:
    print(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        file=stream,
    )


def _safe_number(value: object) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        finite = math.isfinite(value)
    except OverflowError:
        return None
    if not finite or value < 0:
        return None
    return value


def _local_sample_time(value: float) -> str | None:
    try:
        return datetime.fromtimestamp(value).astimezone().isoformat(timespec="seconds")
    except (OSError, OverflowError, ValueError):
        return None


def _safe_count(value: object) -> int:
    return value if type(value) is int and 0 <= value <= 2**63 - 1 else 0


def _optional_count(value: object) -> int | None:
    if value is None:
        return None
    return value if type(value) is int and 0 <= value <= 2**63 - 1 else None


def _safe_label(value: object) -> str | None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or not _SAFE_LABEL.fullmatch(value.strip())
        or _UUID_LIKE.fullmatch(value.strip())
    ):
        return None
    return value.strip()


def _safe_code(value: object) -> str | None:
    if isinstance(value, str) and _SAFE_CODE.fullmatch(value) and not _UUID_LIKE.fullmatch(value):
        return value
    return None


def _safe_exclusions(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        key: count
        for key, count in value.items()
        if isinstance(key, str)
        and _safe_code(key) is not None
        and type(count) is int
        and 0 <= count <= 2**63 - 1
    }


def _safe_codes(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [code for code in value if _safe_code(code) is not None]


def _source_path(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    configured = os.environ.get("CODEX_HOME")
    return Path(configured or Path.home() / ".codex").expanduser().resolve()


def _database_path(data_dir: Path | None) -> Path:
    from token_pulse.bench_store import DATABASE_NAME, speed_data_home

    root = data_dir.expanduser().resolve() if data_dir is not None else speed_data_home()
    return root / DATABASE_NAME


def _open_store(
    data_dir: Path | None, *, read_only: bool = False, timeout: float = 5.0
):
    from token_pulse.bench_store import BenchmarkStore

    path = _database_path(data_dir)
    if read_only:
        return BenchmarkStore(path, read_only=True)
    if timeout != 5.0:
        return BenchmarkStore(path, timeout=timeout)
    return BenchmarkStore(path)


def _base_record(
    *,
    mode: str,
    status: str,
    started_at: float,
    ended_at: float,
    requested_model: str | None = None,
    effort: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": str(uuid.uuid4()),
        "created_at": ended_at,
        "started_at": started_at,
        "ended_at": ended_at,
        "mode": mode,
        "requested_model": requested_model,
        "observed_model": None,
        "effort": effort,
        "status": status,
        "valid_samples": 0,
        "excluded_samples": 0,
        "exclusions": {},
        "samples": [],
    }


def _normalize_benchmark_result(
    raw: object,
    *,
    requested_model: str | None,
    effort: str | None,
    started_at: float,
    ended_at: float,
) -> tuple[dict[str, Any], str | None]:
    if not isinstance(raw, dict):
        return (
            _base_record(
                mode="benchmark",
                status="failed",
                started_at=started_at,
                ended_at=ended_at,
                requested_model=requested_model,
                effort=effort,
            ),
            "benchmark_failed",
        )

    allowed = {
        "run_id",
        "created_at",
        "started_at",
        "ended_at",
        "mode",
        "requested_model",
        "observed_model",
        "effort",
        "provider",
        "service_tier",
        "prompt_id",
        "prompt_sha256",
        "tool_version",
        "codex_version",
        "status",
        "output_tokens",
        "input_tokens",
        "cached_input_tokens",
        "reasoning_output_tokens",
        "stream_seconds",
        "end_to_end_seconds",
        "first_output_seconds",
        "tps",
        "end_to_end_tps",
        "measurement_method",
        "valid_samples",
        "excluded_samples",
        "exclusions",
        "diagnostics",
        "samples",
    }
    record = _base_record(
        mode="benchmark",
        status="failed",
        started_at=started_at,
        ended_at=ended_at,
        requested_model=requested_model,
        effort=effort,
    )
    record.update({key: value for key, value in raw.items() if key in allowed})
    record["mode"] = "benchmark"
    record["run_id"] = str(uuid.uuid4())
    record["created_at"] = ended_at
    record["started_at"] = _safe_number(raw.get("started_at")) or started_at
    record["ended_at"] = _safe_number(raw.get("ended_at")) or ended_at
    record["requested_model"] = _safe_label(requested_model or raw.get("requested_model"))
    record["observed_model"] = _safe_label(raw.get("observed_model"))
    record["effort"] = _safe_label(effort or raw.get("effort"))
    record["output_tokens"] = _optional_count(raw.get("output_tokens"))
    record["input_tokens"] = _optional_count(raw.get("input_tokens"))
    record["cached_input_tokens"] = _optional_count(raw.get("cached_input_tokens"))
    record["reasoning_output_tokens"] = _optional_count(raw.get("reasoning_output_tokens"))
    for key in (
        "stream_seconds",
        "end_to_end_seconds",
        "first_output_seconds",
        "tps",
        "end_to_end_tps",
    ):
        record[key] = _safe_number(raw.get(key))
    for key in ("provider", "service_tier", "tool_version", "codex_version"):
        record[key] = raw.get(key)
    record["prompt_id"] = raw.get("prompt_id")
    record["prompt_sha256"] = raw.get("prompt_sha256")
    record["valid_samples"] = _safe_count(raw.get("valid_samples"))
    record["excluded_samples"] = _safe_count(raw.get("excluded_samples"))
    record["exclusions"] = raw.get("exclusions", {})
    record["diagnostics"] = _safe_codes(raw.get("diagnostics", []))
    record["samples"] = raw.get("samples", [])
    method = raw.get("measurement_method")
    record["measurement_method"] = _safe_code(method)
    if raw.get("status") not in {"completed", "failed", "timeout", "interrupted", "incomplete"}:
        record["status"] = "incomplete"

    tokens = record.get("output_tokens")
    seconds = _safe_number(record.get("stream_seconds"))
    tps = _safe_number(record.get("tps"))
    valid = (
        record["status"] == "completed"
        and type(tokens) is int
        and tokens > 0
        and seconds is not None
        and seconds > 0
        and tps is not None
        and tps > 0
    )
    if record["status"] == "completed" and not valid:
        record["status"] = "incomplete"
        return record, "no_valid_measurement"
    return record, None


def _benchmark(args: argparse.Namespace, stdout: TextIO) -> int:
    from token_pulse.benchmark import run

    started_at = time.time()
    runner_error = False
    try:
        raw = run(
            model=args.model,
            effort=args.effort,
            timeout=args.timeout,
            source=_source_path(args.codex_home),
            codex_command=str(args.codex_command) if args.codex_command is not None else None,
        )
        ended_at = time.time()
        record, error_code = _normalize_benchmark_result(
            raw,
            requested_model=args.model,
            effort=args.effort,
            started_at=started_at,
            ended_at=ended_at,
        )
        runner_error = error_code == "benchmark_failed"
    except KeyboardInterrupt:
        ended_at = time.time()
        record = _base_record(
            mode="benchmark",
            status="interrupted",
            started_at=started_at,
            ended_at=ended_at,
            requested_model=args.model,
            effort=args.effort,
        )
        error_code = "benchmark_failed"
        runner_error = True
    except Exception:
        ended_at = time.time()
        record = _base_record(
            mode="benchmark",
            status="failed",
            started_at=started_at,
            ended_at=ended_at,
            requested_model=args.model,
            effort=args.effort,
        )
        error_code = "benchmark_failed"
        runner_error = True

    history_saved = False
    try:
        with _open_store(args.data_dir) as store:
            record = store.save(record)
        history_saved = True
    except Exception:
        error_code = "history_save_failed"

    result = {
        "run_id": record.get("run_id"),
        "status": record.get("status", "failed"),
        "model": record.get("observed_model") or record.get("requested_model"),
        "tokens": _optional_count(record.get("output_tokens")),
        "stream_seconds": _safe_number(record.get("stream_seconds")),
        "tps": _safe_number(record.get("tps")),
        "end_to_end_seconds": _safe_number(record.get("end_to_end_seconds")),
        "first_output_seconds": _safe_number(record.get("first_output_seconds")),
        "measurement_method": record.get("measurement_method"),
        "exclusions": _safe_exclusions(record.get("exclusions", {})),
        "diagnostics": _safe_codes(record.get("diagnostics", [])),
        "history_saved": history_saved,
    }
    if error_code is not None:
        result["error_code"] = error_code
    _json_line(result, stdout)
    if runner_error or not history_saved or record.get("status") != "completed":
        return 2
    return 0


class _WatchOutput:
    """Forward watch JSONL while storing only its final, aggregate summary."""

    def __init__(self, stream: TextIO, data_dir: Path | None):
        self.stream = stream
        self.data_dir = data_dir
        self.pending = ""
        self.started_at = time.time()
        self.ended_at = self.started_at
        self.final_seen = False
        self.history_saved = False
        self.run_ids: list[str] = []
        self.error_code: str | None = None
        self.status = "incomplete"

    def write(self, value: str) -> int:
        count = len(value)
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self._write_line(line + "\n")
        return count

    def flush(self) -> None:
        if self.pending:
            value, self.pending = self.pending, ""
            self._write_line(value)
        self.stream.flush()

    def _write_line(self, line: str) -> None:
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            self.stream.write(line)
            self.stream.flush()
            return
        if not isinstance(event, dict):
            self.stream.write(line)
            self.stream.flush()
            return
        if event.get("kind") == "ready":
            self.started_at = time.time()
        if event.get("kind") != "final":
            self.stream.write(line)
            self.stream.flush()
            return

        self.final_seen = True
        self.ended_at = time.time()
        self._save_final(event)
        event["status"] = self.status
        event["history_saved"] = self.history_saved
        event["run_ids"] = self.run_ids
        if self.error_code is not None:
            event["error_code"] = self.error_code
        self.stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.stream.flush()

    def _save_final(self, event: dict[str, Any]) -> None:
        rows = event.get("models")
        if not isinstance(rows, list):
            rows = []
        records: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            model = row.get("model")
            tokens = _safe_count(row.get("tokens")) if row.get("tokens") is not None else None
            stream_seconds = _safe_number(row.get("stream_seconds"))
            tps = _safe_number(row.get("weighted_tps"))
            valid_samples = _safe_count(row.get("valid_samples"))
            excluded_samples = _safe_count(row.get("excluded_samples"))
            status = "completed" if valid_samples > 0 and tps is not None else "incomplete"
            record = _base_record(
                mode="watch",
                status=status,
                started_at=self.started_at,
                ended_at=self.ended_at,
                requested_model=model if isinstance(model, str) else None,
            )
            record.update(
                {
                    "observed_model": model if isinstance(model, str) else None,
                    "output_tokens": tokens,
                    "stream_seconds": stream_seconds,
                    "tps": tps,
                    "measurement_method": "local_log_estimate",
                    "valid_samples": valid_samples,
                    "excluded_samples": excluded_samples,
                    "diagnostics": _safe_codes(
                        event.get("diagnostics", {}).get("collector", [])
                        if isinstance(event.get("diagnostics"), dict)
                        else []
                    ),
                    "samples": [],
                }
            )
            records.append(record)
        if not records:
            diagnostics = event.get("diagnostics")
            records.append(
                {
                    **_base_record(
                        mode="watch",
                        status="incomplete",
                        started_at=self.started_at,
                        ended_at=self.ended_at,
                    ),
                    "diagnostics": _safe_codes(
                        diagnostics.get("collector", []) if isinstance(diagnostics, dict) else []
                    ),
                }
            )

        try:
            with _open_store(self.data_dir) as store:
                for record in records:
                    saved = store.save(record)
                    self.run_ids.append(saved["run_id"])
            self.history_saved = len(self.run_ids) == len(records)
        except Exception:
            self.error_code = "history_save_failed"
            self.history_saved = False
        self.status = (
            "completed"
            if all(record["status"] == "completed" for record in records)
            else "incomplete"
        )
        if event.get("reason") == "collector_error":
            self.error_code = self.error_code or "watch_failed"

    def synthesize_final(self) -> None:
        if not self.final_seen:
            self.ended_at = time.time()
            self._write_line(
                json.dumps(
                    {
                        "kind": "final",
                        "models": [],
                        "diagnostics": {"collector": [], "exclusions": {}},
                        "reason": "collector_error",
                        "no_samples": True,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )


def _watch(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    from token_pulse.collector import codex_home
    from token_pulse.watch import run

    output = _WatchOutput(stdout, args.data_dir)
    try:
        return_code = run(
            args.codex_home or codex_home(),
            seconds=args.seconds,
            samples=args.samples,
            thread_id=args.thread_id,
            stdout=output,
            stderr=stderr,
        )
        output.flush()
    except KeyboardInterrupt:
        return_code = 2
    except Exception:
        return_code = 2
        print("Codex speed watch failed (watch_failed).", file=stderr)
    output.synthesize_final()
    if output.error_code is not None or output.status != "completed" or return_code != 0:
        return 2
    return 0


def _history(args: argparse.Namespace, stdout: TextIO) -> int:
    path = _database_path(args.data_dir)
    if not path.is_file():
        _json_line([], stdout)
        return 0
    try:
        with _open_store(args.data_dir, read_only=True) as store:
            runs = store.list_runs(limit=args.limit, model=args.model)
        _json_line(runs, stdout)
        return 0
    except Exception:
        _json_line({"error_code": "history_read_failed"}, stdout)
        return 2


def _show(args: argparse.Namespace, stdout: TextIO) -> int:
    path = _database_path(args.data_dir)
    if not path.is_file():
        _json_line({"found": False}, stdout)
        return 0
    try:
        with _open_store(args.data_dir, read_only=True) as store:
            record = store.get(args.run_id)
        _json_line({"found": False} if record is None else record, stdout)
        return 0
    except Exception:
        _json_line({"error_code": "history_read_failed"}, stdout)
        return 2


def _export(args: argparse.Namespace, stdout: TextIO) -> int:
    path = _database_path(args.data_dir)
    if not path.is_file():
        return 0
    try:
        with _open_store(args.data_dir, read_only=True) as store:
            summaries = store.list_runs(limit=args.limit, model=args.model)
            records = [store.get(summary["run_id"]) for summary in summaries]
        for record in records:
            if record is not None:
                _json_line(record, stdout)
        return 0
    except Exception:
        _json_line({"error_code": "history_read_failed"}, stdout)
        return 2


def _normalize_recent_result(raw: object) -> dict[str, Any]:
    """Keep recent-log output to safe measurement metadata only."""
    result: dict[str, Any] = {
        "status": "source_unavailable",
        "mode": "recent",
        "measurement_method": "local_log_estimate",
        "samples": [],
        "elapsed_seconds": None,
        "coverage": {},
        "diagnostics": [],
    }
    if not isinstance(raw, dict):
        result["diagnostics"] = ["recent_failed"]
        return result
    if raw.get("status") in {"completed", "no_samples", "source_unavailable"}:
        result["status"] = raw["status"]
    result["elapsed_seconds"] = _safe_number(raw.get("elapsed_seconds"))
    diagnostics = raw.get("diagnostics")
    result["diagnostics"] = _safe_codes(diagnostics)

    coverage = raw.get("coverage")
    if isinstance(coverage, dict):
        safe_coverage = {}
        for key, value in coverage.items():
            if not isinstance(key, str) or _safe_code(key) is None:
                continue
            number = _safe_number(value)
            if number is not None:
                safe_coverage[key] = number
            elif key == "excluded":
                safe_coverage[key] = _safe_exclusions(value)
        result["coverage"] = safe_coverage

    rows = raw.get("samples")
    if not isinstance(rows, list):
        result["status"] = "source_unavailable"
        result["diagnostics"] = result["diagnostics"] or ["recent_failed"]
        return result
    samples: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_key = row.get("sample_key")
        model = _safe_label(row.get("model"))
        effort = _safe_label(row.get("effort"))
        tokens = _optional_count(row.get("output_tokens"))
        seconds = _safe_number(row.get("stream_seconds"))
        tps = _safe_number(row.get("tps"))
        ended_at = _safe_number(row.get("ended_at"))
        age_seconds = _safe_number(row.get("age_seconds"))
        stale = row.get("stale") if type(row.get("stale")) is bool else None
        method = _safe_label(row.get("method"))
        if (
            not isinstance(sample_key, str)
            or _SAFE_HASH.fullmatch(sample_key) is None
            or tokens is None
            or tokens <= 0
            or seconds is None
            or seconds <= 0
            or tps is None
            or tps <= 0
            or ended_at is None
            or ended_at <= 0
        ):
            continue
        samples.append(
            {
                "model": model,
                "effort": effort,
                "output_tokens": tokens,
                "stream_seconds": seconds,
                "tps": tps,
                "ended_at": ended_at,
                "sample_time": _local_sample_time(ended_at),
                "age_seconds": age_seconds,
                "stale": stale,
                "sample_key": sample_key,
                "method": method,
            }
        )
    result["samples"] = samples
    if samples:
        result["status"] = "completed"
    elif result["status"] == "completed":
        result["status"] = "no_samples"
    return result


def _recent_record(sample: dict[str, Any]) -> dict[str, Any]:
    """Build a deduplicated, content-free history row for one log sample."""
    ended_at = sample["ended_at"]
    seconds = sample["stream_seconds"]
    model = sample["model"]
    effort = sample["effort"]
    tokens = sample["output_tokens"]
    tps = sample["tps"]
    identity = json.dumps(
        [sample["sample_key"], model, effort, tokens, seconds, tps, ended_at],
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    return {
        "run_id": str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"codex-speed-recent:{identity}")
        ),
        "created_at": ended_at,
        "started_at": None,
        "ended_at": ended_at,
        "mode": "recent",
        "requested_model": None,
        "observed_model": model,
        "effort": effort,
        "status": "completed",
        "output_tokens": tokens,
        "stream_seconds": seconds,
        "tps": tps,
        "measurement_method": "local_log_estimate",
        "valid_samples": 1,
        "excluded_samples": 0,
        "samples": [
            {
                "output_tokens": tokens,
                "started_at": None,
                "ended_at": ended_at,
                "seconds": seconds,
                "tps": tps,
                "model": model,
                "effort": effort,
                "measurement_method": "local_log_estimate",
                "timing_source": sample["method"],
            }
        ],
    }


def _recent(args: argparse.Namespace, stdout: TextIO) -> int:
    from token_pulse.recent import query_recent

    try:
        raw = query_recent(
            _source_path(args.codex_home),
            model=args.model,
            limit=args.limit,
            budget_seconds=args.budget_seconds,
        )
    except Exception:
        raw = {
            "status": "source_unavailable",
            "samples": [],
            "diagnostics": ["recent_failed"],
        }
    result = _normalize_recent_result(raw)
    result["samples"] = result["samples"][: args.limit]
    rows = [_recent_record(sample) for sample in result["samples"]]
    history_saved: bool | None = None
    if rows and not args.no_save:
        try:
            with _open_store(args.data_dir, timeout=0.05) as store:
                for row in rows:
                    store.save(row)
            history_saved = True
        except Exception:
            # The source logs remain authoritative; an unavailable history DB must
            # never hide a measurement or trigger a second model request.
            history_saved = False
    result["history_saved"] = history_saved
    for sample in result["samples"]:
        sample.pop("sample_key", None)
    _json_line(result, stdout)
    return 0 if result["status"] in {"completed", "no_samples"} else 2


def _add_data_dir(parser: argparse.ArgumentParser, *, default: object = argparse.SUPPRESS) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default,
        help="directory for this tool's local speed history",
    )


def _add_recent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=argparse.SUPPRESS, help="filter by model")
    parser.add_argument(
        "--limit",
        type=_bounded_limit,
        default=argparse.SUPPRESS,
        help="number of recent samples (default: 1)",
    )
    parser.add_argument(
        "--budget-seconds",
        type=_positive_finite,
        default=argparse.SUPPRESS,
        help="maximum log query time (default: 2)",
    )
    parser.add_argument(
        "--codex-home",
        type=Path,
        default=argparse.SUPPRESS,
        help="Codex data directory (defaults to CODEX_HOME or ~/.codex)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=argparse.SUPPRESS,
        help="skip saving returned samples to local history",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codex-speed",
        description="Read recent Codex output speed or run a controlled benchmark.",
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"codex-speed {VERSION}")
    _add_data_dir(parser, default=None)
    _add_recent_arguments(parser)
    commands = parser.add_subparsers(dest="command", required=False)

    recent = commands.add_parser(
        "recent", help="read the latest completed outputs from Codex logs"
    )
    _add_data_dir(recent)
    _add_recent_arguments(recent)

    benchmark = commands.add_parser("benchmark", help="run a controlled Codex benchmark")
    _add_data_dir(benchmark)
    benchmark.add_argument("--model")
    benchmark.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max", "ultra"))
    benchmark.add_argument("--timeout", type=_positive_finite, default=600.0)
    benchmark.add_argument("--codex-home", type=Path)
    benchmark.add_argument("--codex-command", type=Path)

    history = commands.add_parser("history", help="list saved runs")
    _add_data_dir(history)
    history.add_argument("--limit", type=_bounded_limit, default=20)
    history.add_argument("--model")

    show = commands.add_parser("show", help="show one saved run")
    _add_data_dir(show)
    show.add_argument("run_id")

    export = commands.add_parser("export", help="write saved runs as JSONL")
    _add_data_dir(export)
    export.add_argument("--model")
    export.add_argument("--limit", type=_bounded_limit, default=_MAX_LIMIT)

    watch = commands.add_parser("watch", help="observe completed Codex outputs for a period")
    _add_data_dir(watch)
    watch.add_argument("--seconds", type=_positive_finite, default=60.0)
    watch.add_argument("--samples", type=_nonnegative_int, default=0)
    watch.add_argument("--thread-id")
    watch.add_argument("--codex-home", type=Path)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the headless speed command and return a conventional process status."""
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    args = _parser().parse_args(argv)
    args.command = args.command or "recent"
    args.model = getattr(args, "model", None)
    args.limit = getattr(args, "limit", 1)
    args.budget_seconds = getattr(args, "budget_seconds", 2.0)
    args.codex_home = getattr(args, "codex_home", None)
    args.no_save = getattr(args, "no_save", False)
    if args.command == "recent":
        return _recent(args, out)
    if args.command == "benchmark":
        return _benchmark(args, out)
    if args.command == "watch":
        return _watch(args, out, err)
    if args.command == "history":
        return _history(args, out)
    if args.command == "show":
        return _show(args, out)
    if args.command == "export":
        return _export(args, out)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
