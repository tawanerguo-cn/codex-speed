"""Deterministic statistics over explicit, comparable samples."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from token_pulse.domain import Sample, reconcile


@dataclass(frozen=True)
class Summary:
    count: int
    excluded: int
    tokens: int
    seconds: float
    weighted_tps: float | None
    median_tps: float | None
    low_tps: float | None
    high_tps: float | None


def unique(samples: list[Sample]) -> list[Sample]:
    """Prefer a repaired complete sample over its earlier incomplete observation."""
    result: dict[tuple, Sample] = {}
    for sample in samples:
        previous = result.get(sample.key)
        result[sample.key] = sample if previous is None else reconcile(previous, sample)
    return sorted(result.values(), key=lambda s: s.ended_at)


def summarize(samples: list[Sample], *, window: int | None = None) -> Summary:
    """Summarize one model across modes and channels."""
    samples = unique(samples)
    if len({s.group for s in samples}) > 1:
        raise ValueError("Different models must be summarized separately")
    if window is not None and window < 1:
        raise ValueError("window must be positive")
    valid = [s for s in samples if s.exclusion is None]
    excluded = len(samples) - len(valid)
    if window is not None:
        valid = valid[-window:]
    rates = [s.tps for s in valid]
    tokens = sum(s.output_tokens for s in valid)
    seconds = sum(s.seconds for s in valid)
    return Summary(
        len(valid),
        excluded,
        tokens,
        seconds,
        tokens / seconds if seconds else None,
        median(rates) if rates else None,
        min(rates) if rates else None,
        max(rates) if rates else None,
    )


def grouped(samples: list[Sample]) -> dict[tuple, Summary]:
    groups: dict[tuple, list[Sample]] = {}
    for sample in unique(samples):
        groups.setdefault(sample.group, []).append(sample)
    return {key: summarize(values) for key, values in groups.items()}


class CounterDelta:
    """A cumulative counter is not a per-response usage record."""

    def __init__(self) -> None:
        self.previous: int | None = None

    def observe(self, value: int) -> tuple[int | None, str | None]:
        if type(value) is not int or value < 0:
            return None, "invalid_tokens"
        previous, self.previous = self.previous, value
        if previous is None:
            return None, "missing_baseline"
        if value < previous:
            return None, "counter_reset"
        return value - previous, None
