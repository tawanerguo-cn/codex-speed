"""Read-only adapter for Codex's internal SQLite diagnostic log schema."""

from __future__ import annotations

import re
import sqlite3
from collections import OrderedDict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from token_pulse.submissions import SubmittedTier, submitted_tier

_START = re.compile(
    r'^Output item item_type="(reasoning|message|function_call|custom_tool_call)" '
    r'item_id="([\w-]{1,160})"$'
)
_TURN = re.compile(r'(?:turn\.id|turn_id)="?([\w-]{1,100})')


def label(value: object) -> str | None:
    """Keep identifiers, never URLs, headers or arbitrary log text."""
    if isinstance(value, str) and re.fullmatch(r"[\w. -]{1,100}", value):
        return value
    return None


def read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.2)
    connection.execute("PRAGMA query_only=ON")
    return connection


@dataclass(frozen=True)
class Start:
    at: float
    turn_id: str
    auth_mode: str | None = None
    requested_tier: str | None = None
    ambiguous: bool = False


class LogEvidence:
    def __init__(self, capacity: int = 20000):
        self.starts: OrderedDict[tuple[str, str], Start] = OrderedDict()
        self.requests: dict[tuple[str, str], tuple[float, str | None, str | None]] = {}
        self.notices: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.submissions: OrderedDict[tuple[str, str], SubmittedTier] = OrderedDict()
        self.capacity = capacity

    def accept(self, task: str, at: float, target: str, body: str) -> None:
        if not task or not body:
            return
        if target == "codex_core::session::handlers":
            result = submitted_tier(task, at, body)
            if result is not None:
                turn_id, observation = result
                key = task, turn_id
                previous = self.submissions.get(key)
                if previous is not None:
                    if (previous.tier, previous.state) != (observation.tier, observation.state):
                        observation = SubmittedTier(min(previous.at, at), None, "ambiguous")
                    else:
                        observation = SubmittedTier(
                            min(previous.at, at), previous.tier, previous.state
                        )
                self.submissions[key] = observation
                self.submissions.move_to_end(key)
                while len(self.submissions) > self.capacity:
                    self.submissions.popitem(last=False)
                return
        if "ToolCall:" in body:
            return
        # Limit pattern work; never inspect arbitrary tool bodies for markers.
        prefix, _, tail = body.rpartition(": ")
        if not prefix:
            tail = body
        turn = _TURN.search(prefix[:2000])
        turn_id = turn[1] if turn else ""
        if target == "feedback_tags" and len(body) < 16000 and "stream_request{" in prefix:
            auth = re.search(r'\bauth_mode="?(Chatgpt|ApiKey|ChatgptAuthTokens)\b', tail)
            request = re.search(r"stream_request\{([^{}]*)\}", prefix)
            tier = re.search(r'\bservice_tier="?([\w-]+)', request[1]) if request else None
            self.requests[task, turn_id] = (
                at,
                auth[1] if auth else None,
                tier[1] if tier else None,
            )
            if len(self.requests) > 1024:
                self.requests.pop(next(iter(self.requests)))
        if target == "codex_core::stream_events_utils" and "handle_output_item_done" not in prefix:
            match = _START.fullmatch(tail) if len(tail) < 400 else None
            if match and turn_id:
                key = task, match[2]
                request = self.requests.get((task, turn_id))
                auth, tier = request[1:] if request and request[0] <= at else (None, None)
                start = Start(at, turn_id, auth, tier)
                previous = self.starts.get(key)
                if previous:
                    conflict = previous.ambiguous or (previous.at, previous.turn_id) != (
                        at,
                        turn_id,
                    )
                    conflict |= any(
                        old is not None and new is not None and old != new
                        for old, new in [
                            (previous.auth_mode, auth),
                            (previous.requested_tier, tier),
                        ]
                    )
                    start = Start(
                        previous.at,
                        previous.turn_id,
                        previous.auth_mode or auth,
                        previous.requested_tier or tier,
                        ambiguous=conflict,
                    )
                self.starts[key] = start
                self.starts.move_to_end(key)
                while len(self.starts) > self.capacity:
                    self.starts.popitem(last=False)
        if target.startswith("codex_core::") and len(tail) < 1000:
            if re.fullmatch(
                r"Configured service tier `[^`]+` is not advertised as supported "
                r"for model `[^`]+` and will be omitted from requests\.?",
                tail,
            ):
                self.notices[task] = (at, "tier_ignored")
                while len(self.notices) > 128:
                    self.notices.popitem(last=False)


class SqliteLogs:
    def __init__(
        self,
        path: Path,
        evidence: LogEvidence,
        *,
        backfill: int = 50000,
        batch_size: int = 5000,
        live_only: bool = False,
    ):
        self.path, self.evidence = path, evidence
        self.backfill, self.batch_size = backfill, batch_size
        self.live_only = live_only
        self.cursor: int | None = None
        self.identity: tuple[int, int] | None = None

    def baseline(self) -> str | None:
        """Set a live cursor at the current end without replaying retained rows."""
        try:
            stat = self.path.stat()
            identity = stat.st_dev, stat.st_ino
            with closing(read_only(self.path)) as connection:
                maximum = connection.execute("SELECT MAX(id) FROM logs").fetchone()[0] or 0
            self.cursor = maximum
            self.identity = identity
        except (OSError, sqlite3.Error, TypeError, ValueError):
            self.cursor = None
            self.identity = None
            return "logs_unavailable"
        return None

    def poll(self) -> str | None:
        try:
            stat = self.path.stat()
            identity = stat.st_dev, stat.st_ino
            with closing(read_only(self.path)) as connection:
                maximum = connection.execute("SELECT MAX(id) FROM logs").fetchone()[0] or 0
                if self.cursor is None or identity != self.identity or maximum < self.cursor:
                    self.cursor = maximum if self.live_only else max(0, maximum - self.backfill)
                    self.evidence.starts.clear()
                    self.evidence.requests.clear()
                    self.evidence.submissions.clear()
                    if not self.live_only:
                        # Submission rows are sparse. The normal ID window can miss
                        # them even while they remain in the database. Backfill this
                        # one target separately, once per database reset, with a cap.
                        submissions = connection.execute(
                            "SELECT ts, ts_nanos, thread_id, target, "
                            "CASE WHEN length(feedback_log_body) <= 16000 "
                            "THEN feedback_log_body END "
                            "FROM logs WHERE target=? ORDER BY id DESC LIMIT 2000",
                            ("codex_core::session::handlers",),
                        ).fetchall()
                        for row in reversed(submissions):
                            self._accept_row(*row)
                        # Old output starts can remain in SQLite long after they leave
                        # the generic ID window. Replay bounded relevant events in ID
                        # order, with feedback preceding each start. Stream the rows
                        # so full log bodies are never accumulated in memory.
                        events = connection.execute(
                            "SELECT ts, ts_nanos, thread_id, target, "
                            "CASE WHEN length(feedback_log_body) <= 16000 "
                            "THEN feedback_log_body END "
                            "FROM logs WHERE id IN (SELECT id FROM logs WHERE id <= ? "
                            "AND target IN (?,?) ORDER BY id DESC LIMIT 40000) ORDER BY id",
                            (maximum, "feedback_tags", "codex_core::stream_events_utils"),
                        )
                        for row in events:
                            self._accept_row(*row)
                self.identity = identity
                rows = connection.execute(
                    "SELECT id, ts, ts_nanos, thread_id, target, "
                    "CASE WHEN length(feedback_log_body) <= 16000 THEN feedback_log_body END "
                    "FROM logs WHERE id > ? ORDER BY id LIMIT ?",
                    (self.cursor, self.batch_size),
                ).fetchall()
                for row_id, seconds, nanos, task, target, body in rows:
                    self._accept_row(seconds, nanos, task, target, body)
                    self.cursor = row_id
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return "logs_unavailable"
        return None

    def _accept_row(self, seconds, nanos, task, target, body):
        if (
            isinstance(body, str)
            and isinstance(task, str)
            and isinstance(target, str)
            and type(seconds) in (int, float)
            and type(nanos) in (int, float)
            and 0 <= seconds <= 2**53
            and 0 <= nanos < 1e9
        ):
            self.evidence.accept(task, seconds + nanos / 1e9, target, body)
