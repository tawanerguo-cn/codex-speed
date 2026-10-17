---
name: codex-speed
description: Read generation speed from recent real Codex work, run a new benchmark only on explicit request, or inspect saved speed history.
---

Use the bundled standard-library launcher at `<skill-root>/scripts/codex_speed.py` with Python 3.11+. Invoke it directly once per request; do not probe Python, inspect the launcher, query model lists, or narrate routine steps.

## Choose the mode

- For ordinary requests to check, measure, or compare recent speed, read recent real-work logs. Do not start a new generation. Omit `--model` unless the user names a model; never discover models first. The default query is bounded to two seconds and one result:

  ```text
  python -S "<skill-root>/scripts/codex_speed.py" recent --budget-seconds 2
  ```

  On Codex Desktop for Windows, use the platform's normal-user execution option: its restricted sandbox may block read-only SQLite access. Do not probe that known limitation first. If another environment reports `state_unavailable`, `log_unavailable`, or `source_unavailable`, retry `recent` once through the platform's approved normal-user execution path. Never copy the source database, change permissions, or fall back to a benchmark.

- For a comparison across recent models, request more saved samples in the same query, for example `recent --limit 10 --budget-seconds 2`. Do not run benchmarks to fill missing models.
- Run a benchmark only when the user explicitly asks to run a test/benchmark or asks for a new generated sample. It performs one real model call and consumes normal model tokens:

  ```text
  python -S "<skill-root>/scripts/codex_speed.py" benchmark --timeout 600
  ```

  Pass `--model <MODEL>` or `--effort <EFFORT>` only when requested. The benchmark uses the fixed system and user prompt in the bundled runner; keep both unchanged so results remain comparable. Do not set a different reasoning effort on the user's behalf.
- For saved-history requests, use `history --limit 20`; use `show <RUN_ID>` for a record's details.

If no valid recent sample exists, say so in one short sentence. Never fall back to a benchmark.

For a recent-speed answer, write exactly one line: `本地日志：<model> <tps> token/s（<sample_time>）。` Use the returned readable sample time directly. For an explicit benchmark, give only model, TPS, and full-turn seconds. Do not add token counts, effort, exclusions, sandbox/retry details, or statements about what files/tests you did not change. Add details only when asked.

The launcher prints JSON. It saves only allowlisted measurements and excludes prompt text, answers, and raw Codex task identifiers. The history database is at `~/.codex/codex-speed/benchmarks.sqlite3` (Windows: `%USERPROFILE%\.codex\codex-speed\benchmarks.sqlite3`). `--data-dir` or `CODEX_SPEED_DATA_DIR` overrides the location.

Recent local-log estimates and active app-server benchmark estimates have different measurement sources. Label the source when comparing them. The active benchmark measures reported output tokens, including reasoning when present, over the first-to-last output-item span. It is not a per-token TTFT/ITL measurement; do not present a single sample as a stable ranking.
