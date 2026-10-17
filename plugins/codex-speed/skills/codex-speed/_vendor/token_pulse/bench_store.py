"""Small, privacy-bounded SQLite storage for explicit speed benchmarks."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DATABASE_NAME = "benchmarks.sqlite3"

_RECORD_FIELDS = {
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
_SAMPLE_FIELDS = {
    "output_tokens",
    "started_at",
    "ended_at",
    "seconds",
    "tps",
    "model",
    "effort",
    "exclusion",
    "measurement_method",
    "timing_source",
}
_NULLABLE_LABELS = {
    "requested_model",
    "observed_model",
    "effort",
    "provider",
    "service_tier",
    "tool_version",
    "codex_version",
}
_NULLABLE_COUNTS = {
    "output_tokens",
    "input_tokens",
    "cached_input_tokens",
    "reasoning_output_tokens",
}
_LABEL_RE = re.compile(r"[\w .:+@-]+", re.UNICODE)
_CODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,99}\Z")
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\Z")
_HASH_RE = re.compile(r"[0-9a-fA-F]{64}\Z")


def speed_data_home() -> Path:
    """Return the directory reserved for this tool's benchmark database."""
    configured = os.environ.get("CODEX_SPEED_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".codex" / "codex-speed").resolve()


def _json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _object(value: object, allowed: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"unexpected {where} field(s): {', '.join(sorted(map(str, extra)))}")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{where} keys must be strings")
    return value


def _number(value: object, name: str, *, nullable: bool) -> float | int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return normalized


def _count(value: object, name: str, *, nullable: bool = False) -> int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 2**63 - 1:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _label(value: object, name: str, *, nullable: bool) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a short label")
    value = value.strip()
    if (
        not value
        or len(value) > 100
        or _LABEL_RE.fullmatch(value) is None
        or _UUID_RE.fullmatch(value)
    ):
        raise ValueError(f"{name} must be a short, readable label")
    return value


def _code(value: object, name: str, *, nullable: bool) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or _CODE_RE.fullmatch(value) is None or _UUID_RE.fullmatch(value):
        raise ValueError(f"{name} must be a short safe code")
    return value


def _timestamp(value: object, name: str, *, nullable: bool) -> float | None:
    number = _number(value, name, nullable=nullable)
    return None if number is None else float(number)


def _normalize_sample(value: object, index: int) -> dict[str, Any]:
    sample = _object(value, _SAMPLE_FIELDS, f"samples[{index}]")
    normalized: dict[str, Any] = {}
    normalized["output_tokens"] = _count(
        sample.get("output_tokens"), "sample.output_tokens", nullable=True
    )
    normalized["started_at"] = _timestamp(
        sample.get("started_at"), "sample.started_at", nullable=True
    )
    normalized["ended_at"] = _timestamp(sample.get("ended_at"), "sample.ended_at", nullable=True)
    normalized["seconds"] = _number(sample.get("seconds"), "sample.seconds", nullable=True)
    normalized["tps"] = _number(sample.get("tps"), "sample.tps", nullable=True)
    normalized["model"] = _label(sample.get("model"), "sample.model", nullable=True)
    normalized["effort"] = _label(sample.get("effort"), "sample.effort", nullable=True)
    normalized["exclusion"] = _code(sample.get("exclusion"), "sample.exclusion", nullable=True)
    normalized["measurement_method"] = _code(
        sample.get("measurement_method"), "sample.measurement_method", nullable=True
    )
    normalized["timing_source"] = _label(
        sample.get("timing_source"), "sample.timing_source", nullable=True
    )
    return normalized


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    value = _object(record, _RECORD_FIELDS, "record")
    run_id = value.get("run_id")
    if run_id is None:
        run_id = str(uuid.uuid4())
    elif not isinstance(run_id, str):
        raise ValueError("run_id must be a UUID string")
    else:
        try:
            run_id = str(uuid.UUID(run_id))
        except (ValueError, AttributeError) as exc:
            raise ValueError("run_id must be a UUID string") from exc

    mode = value.get("mode")
    if not isinstance(mode, str) or mode not in {"benchmark", "watch", "import", "recent"}:
        raise ValueError("mode must be benchmark, watch, import, or recent")
    status = value.get("status")
    if not isinstance(status, str) or status not in {
        "completed",
        "failed",
        "timeout",
        "interrupted",
        "incomplete",
    }:
        raise ValueError("status must describe the run outcome")
    if "created_at" not in value:
        raise ValueError("created_at is required")

    result: dict[str, Any] = {"run_id": run_id}
    result["created_at"] = _timestamp(value["created_at"], "created_at", nullable=False)
    result["started_at"] = _timestamp(value.get("started_at"), "started_at", nullable=True)
    result["ended_at"] = _timestamp(value.get("ended_at"), "ended_at", nullable=True)
    result["mode"] = mode
    for name in _NULLABLE_LABELS:
        result[name] = _label(value.get(name), name, nullable=True)
    prompt_id = _code(value.get("prompt_id"), "prompt_id", nullable=True)
    prompt_sha256 = value.get("prompt_sha256")
    if prompt_sha256 is not None:
        if not isinstance(prompt_sha256, str) or _HASH_RE.fullmatch(prompt_sha256) is None:
            raise ValueError("prompt_sha256 must be a 64-character hexadecimal digest")
        prompt_sha256 = prompt_sha256.lower()
    result["prompt_id"] = prompt_id
    result["prompt_sha256"] = prompt_sha256
    result["status"] = status
    for name in _NULLABLE_COUNTS:
        result[name] = _count(value.get(name), name, nullable=True)
    for name in (
        "stream_seconds",
        "end_to_end_seconds",
        "first_output_seconds",
        "tps",
        "end_to_end_tps",
    ):
        result[name] = _number(value.get(name), name, nullable=True)
    result["measurement_method"] = _code(
        value.get("measurement_method"), "measurement_method", nullable=True
    )
    result["valid_samples"] = _count(value.get("valid_samples"), "valid_samples")
    result["excluded_samples"] = _count(value.get("excluded_samples"), "excluded_samples")
    exclusions = value.get("exclusions", {})
    if not isinstance(exclusions, dict):
        raise ValueError("exclusions must map safe reason codes to counts")
    result["exclusions"] = {
        _code(key, "exclusion reason", nullable=False): _count(count, "exclusion count")
        for key, count in exclusions.items()
    }
    diagnostics = value.get("diagnostics", [])
    if not isinstance(diagnostics, list):
        raise ValueError("diagnostics must be an array of safe codes")
    result["diagnostics"] = [
        _code(diagnostic, "diagnostic", nullable=False) for diagnostic in diagnostics
    ]
    samples = value.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("samples must be an array")
    result["samples"] = [_normalize_sample(sample, index) for index, sample in enumerate(samples)]
    return result


class BenchmarkStore:
    """Append-only benchmark storage using only allowlisted JSON fields.

    ``path`` names the SQLite file. Leaving it unset writes
    ``speed_data_home()/benchmarks.sqlite3``.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        read_only: bool = False,
        timeout: float = 5.0,
    ):
        self.path = Path(path if path is not None else speed_data_home() / DATABASE_NAME)
        self.path = self.path.expanduser().resolve()
        self.read_only = read_only
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be between zero and sixty seconds")
        if not math.isfinite(timeout) or timeout < 0 or timeout > 60:
            raise ValueError("timeout must be between zero and sixty seconds")
        timeout = float(timeout)
        if read_only:
            if not self.path.exists():
                raise FileNotFoundError(f"benchmark database does not exist: {self.path}")
            if self.path.is_dir():
                raise ValueError("database path must name a file")
            self.connection = sqlite3.connect(
                f"{self.path.as_uri()}?mode=ro",
                uri=True,
                timeout=timeout,
                isolation_level=None,
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.is_dir():
                raise ValueError("database path must name a file")
            self.connection = sqlite3.connect(self.path, timeout=timeout, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
        self.connection.execute("PRAGMA foreign_keys=ON")
        try:
            if read_only:
                self._validate_existing_schema()
            else:
                self._initialize()
        except Exception:
            self.connection.close()
            raise
        self._closed = False
        self._savepoint_sequence = 0

    def _validate_existing_schema(self) -> None:
        """Check the supported schema without creating or changing anything."""
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"benchmark database schema {version} is newer than supported {SCHEMA_VERSION}"
            )
        if version != SCHEMA_VERSION:
            raise RuntimeError(f"unsupported benchmark database schema {version}")
        tables = {
            row[0]
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        expected = {"runs", "samples"}
        if not expected.issubset(tables):
            raise RuntimeError("benchmark database is missing required tables")
        expected_columns = {
            "runs": {"run_id", "created_at", "model", "status", "payload"},
            "samples": {"run_id", "index", "payload"},
        }
        for table, columns in expected_columns.items():
            actual = {row[1] for row in self.connection.execute(f'PRAGMA table_info("{table}")')}
            if actual != columns:
                raise RuntimeError(f"benchmark database has an unsupported {table} table schema")

    def _initialize(self) -> None:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"benchmark database schema {version} is newer than supported {SCHEMA_VERSION}"
                )
            tables = {
                row[0]
                for row in self.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if version == 0 and tables:
                raise RuntimeError("refusing to use an unversioned database with existing tables")
            if version == SCHEMA_VERSION:
                expected = {"runs", "samples"}
                if not expected.issubset(tables):
                    raise RuntimeError("benchmark database is missing required tables")
            else:
                self.connection.execute(
                    """
                    CREATE TABLE runs (
                        run_id TEXT PRIMARY KEY,
                        created_at REAL NOT NULL,
                        model TEXT,
                        status TEXT NOT NULL,
                        payload TEXT NOT NULL
                    )
                    """
                )
                self.connection.execute(
                    """
                    CREATE TABLE samples (
                        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                        "index" INTEGER NOT NULL,
                        payload TEXT NOT NULL,
                        PRIMARY KEY (run_id, "index")
                    )
                    """
                )
                self.connection.execute(
                    "CREATE INDEX runs_created_at_idx ON runs(created_at DESC, run_id)"
                )
                self.connection.execute(
                    "CREATE INDEX runs_model_created_at_idx ON runs(model, created_at DESC, run_id)"
                )
                self.connection.execute("PRAGMA user_version=1")
            self.connection.execute("COMMIT")
        except Exception:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise

    def __enter__(self) -> BenchmarkStore:
        if self._closed:
            raise RuntimeError("benchmark store is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("benchmark store is closed")

    def save(self, record: dict[str, Any]) -> dict[str, Any]:
        """Save a complete run atomically and return its normalized representation."""
        self._ensure_open()
        if self.read_only:
            raise RuntimeError("benchmark store is read-only")
        normalized = _normalize_record(record)
        run_id = normalized["run_id"]
        summary = {key: value for key, value in normalized.items() if key != "samples"}
        run_payload = _json(summary)
        sample_payloads = [_json(sample) for sample in normalized["samples"]]
        model = normalized["observed_model"] or normalized["requested_model"]
        nested = self.connection.in_transaction
        if nested:
            self._savepoint_sequence += 1
            savepoint = f"benchmark_store_save_{self._savepoint_sequence}"
            self.connection.execute(f"SAVEPOINT {savepoint}")
        else:
            savepoint = None
            self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.connection.execute(
                "SELECT payload FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if existing is not None:
                existing_samples = [
                    row[0]
                    for row in self.connection.execute(
                        'SELECT payload FROM samples WHERE run_id=? ORDER BY "index"',
                        (run_id,),
                    )
                ]
                if existing["payload"] != run_payload or existing_samples != sample_payloads:
                    raise ValueError(f"run_id {run_id} already exists with different data")
                if nested:
                    self.connection.execute(f"RELEASE {savepoint}")
                else:
                    self.connection.execute("COMMIT")
                return normalized
            self.connection.execute(
                "INSERT INTO runs(run_id, created_at, model, status, payload) VALUES(?,?,?,?,?)",
                (run_id, normalized["created_at"], model, normalized["status"], run_payload),
            )
            self.connection.executemany(
                'INSERT INTO samples(run_id, "index", payload) VALUES(?,?,?)',
                [(run_id, index, payload) for index, payload in enumerate(sample_payloads)],
            )
            if nested:
                self.connection.execute(f"RELEASE {savepoint}")
            else:
                self.connection.execute("COMMIT")
        except Exception:
            if nested:
                self.connection.execute(f"ROLLBACK TO {savepoint}")
                self.connection.execute(f"RELEASE {savepoint}")
            elif self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise
        return normalized

    def list_runs(self, limit: int = 20, model: str | None = None) -> list[dict[str, Any]]:
        """List run summaries; sample payload rows are not read by this query."""
        self._ensure_open()
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        if model is not None:
            model = _label(model, "model", nullable=False)
        if model is None:
            rows = self.connection.execute(
                "SELECT payload FROM runs ORDER BY created_at DESC, run_id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT payload FROM runs WHERE model=? "
                "ORDER BY created_at DESC, run_id DESC LIMIT ?",
                (model, limit),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def get(self, run_id: str) -> dict[str, Any] | None:
        """Return a complete run and its ordered samples, if present."""
        self._ensure_open()
        if not isinstance(run_id, str):
            raise ValueError("run_id must be a UUID string")
        try:
            run_id = str(uuid.UUID(run_id))
        except (ValueError, AttributeError) as exc:
            raise ValueError("run_id must be a UUID string") from exc
        row = self.connection.execute(
            "SELECT payload FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        result = json.loads(row["payload"])
        result["samples"] = [
            json.loads(sample["payload"])
            for sample in self.connection.execute(
                'SELECT payload FROM samples WHERE run_id=? ORDER BY "index"', (run_id,)
            )
        ]
        return result

    def close(self) -> None:
        if not self._closed:
            self.connection.close()
            self._closed = True
