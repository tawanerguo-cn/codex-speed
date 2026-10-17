"""Bounded discovery and polling, suitable for a background worker."""

from __future__ import annotations

import os
import re
import sqlite3
import time
import tomllib
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from token_pulse.backfill import Backfill
from token_pulse.domain import Context, Sample, Task
from token_pulse.logs import LogEvidence, SqliteLogs, label, read_only
from token_pulse.rollout import REJECTIONS, Rollout
from token_pulse.stats import unique
from token_pulse.tailer import JsonlTail


def resolved_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    # Windows' extended-length spelling can identify the same local file but fail
    # pathlib's containment comparison against the ordinary source-root spelling.
    value = str(path)
    if os.name == "nt" and value.startswith("\\\\?\\"):
        value = "\\\\" + value[8:] if value.startswith("\\\\?\\UNC\\") else value[4:]
        return Path(value)
    return path


def codex_home() -> Path:
    return resolved_path(Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")))


def data_home() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "TokenPulse"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "token-pulse"


def newest_database(root: Path, stem: str) -> Path | None:
    candidates = []
    for path in root.glob(f"{stem}_*.sqlite"):
        match = re.fullmatch(rf"{stem}_(\d+)\.sqlite", path.name)
        if match:
            candidates.append((int(match[1]), path))
    return max(candidates)[1] if candidates else None


@dataclass(frozen=True)
class Snapshot:
    tasks: tuple[Task, ...] = ()
    samples: tuple[Sample, ...] = ()
    diagnostics: tuple[str, ...] = ()
    at: float = 0
    backfill_pending: int = 0
    backfill_completed: int = 0
    rejected_samples: tuple[Sample, ...] = ()
    backfill_reused: int = 0


class Collector:
    def __init__(
        self,
        root: Path,
        *,
        max_tasks: int = 32,
        live_only: bool = False,
        started_at: float | None = None,
    ):
        self.root = resolved_path(root)
        self.max_tasks = max_tasks
        self.live_only = live_only
        self.live_started_at = time.time() if started_at is None else started_at
        self.evidence = LogEvidence()
        self.logs: SqliteLogs | None = None
        self.watched: dict[str, tuple[JsonlTail, Rollout]] = {}
        self.live_tails: dict[str, tuple[Path, JsonlTail]] = {}
        self.live_baseline_tasks: set[str] = set()
        self.live_baseline_ready = False
        self.next_discovery = 0.0
        self.configured_tier: str | None = None
        self.discovery_issues: set[str] = set()
        self.backfill = Backfill()
        self.next_backfill_discovery = 0.0
        if self.live_only:
            self._baseline_live_sources()

    def _baseline_live_sources(self) -> None:
        """Record current source ends without reading historical rollout content."""
        state = newest_database(self.root, "state")
        state_loaded = False
        if state is not None:
            try:
                with closing(read_only(state)) as connection:
                    rows = connection.execute("SELECT id, rollout_path FROM threads").fetchall()
                self.live_baseline_ready = True
                state_loaded = True
                for task_id, raw_path in rows:
                    self.live_baseline_tasks.add(task_id)
                    try:
                        candidate = Path(raw_path).expanduser()
                        if candidate.suffix != ".jsonl" or not candidate.is_file():
                            continue
                        path = resolved_path(candidate)
                        if not path.is_relative_to(self.root) or path.suffix != ".jsonl":
                            continue
                        tail = JsonlTail(path, backfill=None, live_only=True)
                        tail.baseline()
                        self.live_tails[task_id] = (path, tail)
                    except (OSError, TypeError, ValueError):
                        continue
            except (OSError, sqlite3.Error, TypeError, ValueError):
                self.discovery_issues.add("state_unavailable")
        if not state_loaded:
            self._baseline_recent_files()
        logs_path = newest_database(self.root, "logs")
        if logs_path is not None:
            self.logs = SqliteLogs(logs_path, self.evidence, live_only=True)
            issue = self.logs.baseline()
            if issue:
                self.discovery_issues.add(issue)

    def _baseline_recent_files(self) -> None:
        """Fallback baseline for the same bounded directories used by discovery."""
        recent = datetime.fromtimestamp(self.live_started_at, UTC)
        directories = [self.root / "sessions"]
        directories += [
            self.root / "sessions" / (recent - timedelta(days=i)).strftime("%Y/%m/%d")
            for i in range(3)
        ]
        for directory in directories:
            try:
                paths = tuple(directory.glob("*.jsonl"))
            except OSError:
                continue
            for path in paths:
                match = re.search(r"([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})", path.name)
                if not match:
                    continue
                try:
                    resolved = resolved_path(path)
                    tail = JsonlTail(resolved, backfill=None, live_only=True)
                    tail.baseline()
                    self.live_tails[match[1]] = (resolved, tail)
                    self.live_baseline_tasks.add(match[1])
                except (OSError, TypeError, ValueError):
                    continue
        self.live_baseline_ready = True

    def _discover(self, now: float) -> None:
        self.discovery_issues.clear()
        try:
            with (self.root / "config.toml").open("rb") as stream:
                config = tomllib.load(stream)
            self.configured_tier = label(config.get("service_tier"))
        except FileNotFoundError:
            self.configured_tier = None
        except (OSError, ValueError):
            self.configured_tier = None
            self.discovery_issues.add("config_unavailable")
        logs_path = newest_database(self.root, "logs")
        if logs_path and (self.logs is None or self.logs.path != logs_path):
            self.logs = SqliteLogs(logs_path, self.evidence, live_only=self.live_only)
            if self.live_only:
                issue = self.logs.baseline()
                if issue:
                    self.discovery_issues.add(issue)
        if logs_path is None:
            self.logs = None
            self.discovery_issues.add("logs_unavailable")
        found: list[tuple[str, Path, str, Context, int | None]] = []
        optional = None
        state = newest_database(self.root, "state")
        if state:
            try:
                with closing(read_only(state)) as connection:
                    columns = {r[1] for r in connection.execute("PRAGMA table_info(threads)")}
                    optional = [
                        name if name in columns else f"NULL AS {name}"
                        for name in ("model", "reasoning_effort", "model_provider")
                    ]
                    created_at = "created_at_ms" if "created_at_ms" in columns else "NULL"
                    rows = connection.execute(
                        "SELECT id, rollout_path, title, "
                        + ", ".join(optional)
                        + f", {created_at} FROM threads WHERE archived=0 "
                        + "ORDER BY updated_at DESC LIMIT ?",
                        (self.max_tasks,),
                    ).fetchall()
                for task_id, path, title, model, effort, provider, created_ms in rows:
                    path = resolved_path(Path(path))
                    if path.is_relative_to(self.root) and path.suffix == ".jsonl":
                        # Titles are for this in-memory UI only; never part of a Sample.
                        found.append(
                            (
                                task_id,
                                path,
                                str(title)[:160],
                                Context(label(model), label(effort), label(provider)),
                                created_ms,
                            )
                        )
            except (OSError, sqlite3.Error, TypeError, ValueError):
                self.discovery_issues.add("state_unavailable")
        if not found:
            # Fallback probes recent date directories, never a recursive full-history scan.
            recent = datetime.fromtimestamp(now, UTC)
            directories = [self.root / "sessions"]
            directories += [
                self.root / "sessions" / (recent - timedelta(days=i)).strftime("%Y/%m/%d")
                for i in range(3)
            ]
            candidates = []
            for directory in directories:
                for path in directory.glob("*.jsonl"):
                    match = re.search(r"([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})", path.name)
                    if match:
                        candidates.append((path.stat().st_mtime, match[1], path))
            for _, task_id, path in sorted(candidates, reverse=True)[: self.max_tasks]:
                found.append((task_id, path, "", Context(), None))
        selected = {item[0] for item in found}
        self.watched = {key: value for key, value in self.watched.items() if key in selected}
        for task_id, path, title, context, created_ms in found:
            if task_id not in self.watched or self.watched[task_id][0].path != path:
                if self.live_only:
                    baseline = self.live_tails.get(task_id)
                    if baseline is not None and baseline[0] == path:
                        tail = baseline[1]
                    else:
                        tail = JsonlTail(path, backfill=None, live_only=True)
                        created_at = (
                            created_ms / 1000 if isinstance(created_ms, (int, float)) else None
                        )
                        is_new = (
                            created_at is not None and created_at >= self.live_started_at
                        ) or (
                            created_at is None
                            and self.live_baseline_ready
                            and task_id not in self.live_baseline_tasks
                        )
                        if not is_new:
                            tail.baseline()
                        self.live_tails[task_id] = (path, tail)
                else:
                    tail = JsonlTail(path)
                rollout = (
                    Rollout(task_id, title, context, resume_live_turn=True)
                    if self.live_only
                    else Rollout(task_id, title, context)
                )
                self.watched[task_id] = (tail, rollout)
            elif title:
                self.watched[task_id][1].title = title
        if (
            not self.live_only
            and state
            and optional is not None
            and now >= self.next_backfill_discovery
        ):
            try:
                with closing(read_only(state)) as connection:
                    rows = connection.execute(
                        "SELECT id, rollout_path, "
                        + ", ".join(optional)
                        + " FROM threads WHERE updated_at >= ? ORDER BY updated_at DESC",
                        (now - 604800,),
                    )
                    for task_id, path, model, effort, provider in rows:
                        path = resolved_path(Path(path))
                        if not path.is_relative_to(self.root) or path.suffix != ".jsonl":
                            continue
                        try:
                            self.backfill.add(
                                task_id,
                                path,
                                Context(label(model), label(effort), label(provider)),
                                live=task_id in self.watched,
                                evidence=self.evidence,
                            )
                        except OSError:
                            self.discovery_issues.add("source_unavailable")
            except (OSError, sqlite3.Error, TypeError, ValueError):
                self.discovery_issues.add("state_unavailable")
            self.next_backfill_discovery = now + 60
        self.next_discovery = now + 10

    def poll(self, now: float | None = None) -> Snapshot:
        now = time.time() if now is None else now
        issues: set[str] = set()
        if now >= self.next_discovery:
            try:
                self._discover(now)
            except OSError:
                self.discovery_issues.add("source_unavailable")
                self.next_discovery = now + 10
        issues.update(self.discovery_issues)
        if self.logs:
            # Drain at most 10k rows per poll, allowing rollout/log arrival to differ.
            for _ in range(2):
                issue = self.logs.poll()
                if issue:
                    issues.add(issue)
                    break
        tasks, samples = [], []
        for tail, rollout in self.watched.values():
            batch = tail.read()
            if batch.reset or batch.skipped_prefix:
                rollout.reset()
            if batch.malformed or batch.oversized:
                issues.add("records_skipped")
            if batch.error:
                issues.add(batch.error)
            gaps = set(batch.gaps)
            for index, record in enumerate(batch.records):
                if index in gaps:
                    rollout.damage("record_gap")
                rollout.feed(record)
            if len(batch.records) in gaps:
                rollout.damage("record_gap")
            task = rollout.task(self.evidence)
            task = replace(
                task, context=replace(task.context, configured_tier=self.configured_tier)
            )
            if batch.error:
                task = replace(task, state="unknown")
            tasks.append(task)
            samples.extend(rollout.samples(self.evidence))
        if not tasks:
            issues.add("no_sessions")
        if self.live_only:
            recovered, recovery_issues = [], []
            samples = [
                sample
                for sample in samples
                if sample.started_at is not None and sample.started_at >= self.live_started_at
            ]
        else:
            recovered, recovery_issues = self.backfill.poll(self.evidence, now)
        issues.update(recovery_issues)
        observed = [s for s in (*recovered, *samples) if s.issue not in REJECTIONS]
        rejected = [s for s in (*recovered, *samples) if s.issue in REJECTIONS]
        return Snapshot(
            tuple(sorted(tasks, key=lambda t: t.updated_at, reverse=True)),
            tuple(unique(observed)),
            tuple(sorted(issues)),
            now,
            self.backfill.pending,
            len(self.backfill.done),
            tuple(rejected),
            self.backfill.reused,
        )
