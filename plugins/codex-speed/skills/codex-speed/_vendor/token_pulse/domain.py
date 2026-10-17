"""UI-independent observations. Unknown evidence remains None."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Context:
    """Independent observed settings; None is unknown, never an inferred default."""

    model: str | None = None
    effort: str | None = None
    provider: str | None = None
    auth_mode: str | None = None
    configured_tier: str | None = None
    requested_tier: str | None = None
    actual_tier: str | None = None
    submitted_tier: str | None = None
    submitted_tier_state: str | None = None


@dataclass(frozen=True)
class Sample:
    """One output range, with Unix-second timestamps and an explicit token scope.

    Identifiers are source identities during parsing and HMAC pseudonyms in history.
    Incomplete records retain tokens/evidence; exclusion determines TPS eligibility.
    """

    task_id: str
    turn_id: str
    response_id: str
    ended_at: float
    output_tokens: int | None
    started_at: float | None = None
    context: Context = Context()
    source: str = "codex-local-log-v1"
    token_scope: str = "total_output"
    issue: str | None = None
    parser_revision: int = 1
    timing_source: str | None = None

    @property
    def exclusion(self) -> str | None:
        if self.issue:
            return self.issue
        if self.token_scope != "total_output":
            return "unsupported_token_scope"
        if not math.isfinite(self.ended_at):
            return "invalid_timestamp"
        if self.started_at is None:
            return "missing_start"
        if not math.isfinite(self.started_at) or self.ended_at <= self.started_at:
            return "invalid_duration"
        if type(self.output_tokens) is not int or not 0 <= self.output_tokens <= 2**63 - 1:
            return "invalid_tokens"
        return None

    @property
    def seconds(self) -> float | None:
        if self.exclusion is not None:
            return None
        assert self.started_at is not None
        return self.ended_at - self.started_at

    @property
    def tps(self) -> float | None:
        seconds = self.seconds
        return self.output_tokens / seconds if seconds is not None else None

    @property
    def key(self) -> tuple[str, str, str]:
        return self.task_id, self.turn_id, self.response_id

    @property
    def group(self) -> tuple:
        # The product summarizes each model across modes and channels. Raw context
        # remains evidence only; incompatible token scopes cannot produce a TPS.
        return (Context(model=self.context.model),)

    def is_stale(self, now: float, max_age: float = 60) -> bool:
        return now - self.ended_at > max_age or now < self.ended_at


def reconcile(previous: Sample, incoming: Sample) -> Sample:
    """Preserve complete measurements while independently refreshing tier evidence."""
    if previous.key != incoming.key:
        raise ValueError("Cannot reconcile different output samples")
    revised_measurement = (
        incoming.parser_revision > previous.parser_revision and incoming.exclusion is None
    )
    withdrawals = {"unconfirmed_tool_start", "unconfirmed_initial_span"}
    revised_boundary = (
        incoming.parser_revision > previous.parser_revision and incoming.issue in withdrawals
    )
    if previous.issue in withdrawals and previous.parser_revision > incoming.parser_revision:
        base = previous
    else:
        base = (
            incoming
            if previous.exclusion is not None or revised_measurement or revised_boundary
            else previous
        )
    evidence = (
        incoming.context if incoming.context.submitted_tier_state is not None else previous.context
    )
    return replace(
        base,
        context=replace(
            base.context,
            provider=base.context.provider or previous.context.provider,
            auth_mode=base.context.auth_mode or previous.context.auth_mode,
            requested_tier=base.context.requested_tier or previous.context.requested_tier,
            actual_tier=base.context.actual_tier or previous.context.actual_tier,
            submitted_tier=evidence.submitted_tier,
            submitted_tier_state=evidence.submitted_tier_state,
        ),
    )


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    state: str
    updated_at: float
    context: Context = Context()
    latest: Sample | None = None
    turn_seconds: float | None = None
    wait_seconds: float | None = None
    notice: str | None = None

    def has_recent_state(self, now: float, max_age: float = 120) -> bool:
        return math.isfinite(self.updated_at) and 0 <= now - self.updated_at <= max_age

    def display_state(self, now: float) -> str:
        if self.state != "unknown" and not self.has_recent_state(now):
            return "stale"
        return self.state
