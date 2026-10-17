"""On-demand JSONL measurement of completed Codex output observed this run."""

from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from threading import Event
from typing import TextIO

from token_pulse.collector import Collector
from token_pulse.domain import Sample, reconcile
from token_pulse.stats import grouped

_SAFE_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_MAX_SAMPLES = 10_000


def _safe_code(value: str) -> str:
    return value if _SAFE_CODE.fullmatch(value) else "collector_issue"


def _finite(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def _summary(samples: dict[tuple[str, str, str], Sample], diagnostics: set[str]) -> dict:
    values = list(samples.values())
    summaries = grouped(values)
    latest: dict[tuple, Sample] = {}
    excluded: Counter[str] = Counter()
    for sample in values:
        if sample.exclusion is not None:
            excluded[_safe_code(sample.exclusion)] += 1
            continue
        previous = latest.get(sample.group)
        if previous is None or sample.ended_at >= previous.ended_at:
            latest[sample.group] = sample

    rows_with_time = []
    for key, stats in summaries.items():
        model = key[0].model or "未知模型"
        recent = latest.get(key)
        rows_with_time.append(
            (
                recent.ended_at if recent is not None else float("-inf"),
                {
                    "model": model,
                    "latest_tps": _finite(recent.tps) if recent is not None else None,
                    "weighted_tps": _finite(stats.weighted_tps),
                    "tokens": stats.tokens,
                    "stream_seconds": _finite(stats.seconds),
                    "valid_samples": stats.count,
                    "excluded_samples": stats.excluded,
                },
            )
        )
    rows_with_time.sort(key=lambda item: (-item[0], item[1]["model"].casefold()))
    return {
        "models": [row for _, row in rows_with_time],
        "diagnostics": {
            "collector": sorted(_safe_code(code) for code in diagnostics),
            "exclusions": dict(sorted(excluded.items())),
        },
    }


def _emit(record: dict, stream: TextIO) -> None:
    print(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False), file=stream
    )
    stream.flush()


def _merge_snapshot(
    records: dict[tuple[str, str, str], Sample],
    observed: tuple[Sample, ...],
    rejected: tuple[Sample, ...],
    thread_id: str | None,
) -> bool:
    changed = False
    for sample in rejected:
        if thread_id is not None and sample.task_id != thread_id:
            continue
        if records.get(sample.key) != sample:
            # A parser rejection withdraws any earlier live observation for this key.
            records[sample.key] = sample
            changed = True
    for sample in observed:
        if thread_id is not None and sample.task_id != thread_id:
            continue
        previous = records.get(sample.key)
        merged = sample if previous is None else reconcile(previous, sample)
        if previous != merged:
            records[sample.key] = merged
            changed = True
    if len(records) > _MAX_SAMPLES:
        trimmed = dict(list(records.items())[-_MAX_SAMPLES:])
        changed |= trimmed != records
        records.clear()
        records.update(trimmed)
    return changed


def run(
    source: Path,
    *,
    seconds: float = 60.0,
    samples: int = 0,
    thread_id: str | None = None,
    interval: float = 2.0,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Poll live Codex sources synchronously and emit a small JSONL protocol."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    if type(samples) is not int or samples < 0:
        raise ValueError("samples must be a non-negative integer")
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("interval must be finite and positive")

    out = stdout or sys.stdout
    err = stderr or sys.stderr
    collector: Collector | None = None
    records: dict[tuple[str, str, str], Sample] = {}
    diagnostics: set[str] = set()
    reason = "timeout"
    started = False
    try:
        collector = Collector(source, live_only=True)
        # The first poll completes source discovery and establishes the live baseline.
        collector.poll()
        collector.live_started_at = time.time()
        _emit(
            {
                "kind": "ready",
                "seconds": seconds,
                "sample_target": samples,
                "thread_filter": thread_id is not None,
            },
            out,
        )
        started = True
        ready_at = time.monotonic()
        last_content: dict | None = None
        wake = Event()

        while True:
            remaining = seconds - (time.monotonic() - ready_at)
            if remaining <= 0:
                reason = "timeout"
                break
            wake.wait(min(interval, remaining))
            snapshot = collector.poll()
            changed = _merge_snapshot(
                records, snapshot.samples, snapshot.rejected_samples, thread_id
            )
            next_diagnostics = {_safe_code(code) for code in snapshot.diagnostics}
            diagnostics_changed = next_diagnostics != diagnostics
            diagnostics = next_diagnostics

            content = _summary(records, diagnostics) if changed or diagnostics_changed else None
            if content is not None and content != last_content:
                _emit({"kind": "update", **content}, out)
                last_content = content

            valid_count = sum(
                item["valid_samples"] for item in (last_content or {}).get("models", ())
            )
            if samples and valid_count >= samples:
                reason = "samples_reached"
                break
    except KeyboardInterrupt:
        reason = "interrupted"
    except Exception:
        reason = "collector_error"
        if not started:
            print("TokenPulse watch could not establish a live baseline.", file=err)
            return 1
        diagnostics.add("watch_error")

    content = _summary(records, diagnostics)
    _emit(
        {
            "kind": "final",
            **content,
            "reason": reason,
            "no_samples": not any(row["valid_samples"] for row in content["models"]),
        },
        out,
    )
    return 1 if reason == "collector_error" else 0
