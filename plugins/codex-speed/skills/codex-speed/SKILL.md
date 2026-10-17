---
name: codex-speed
description: Measure Codex generation speed on demand and inspect saved benchmark history. Use for requests such as “测一下Codex速率”, “比较模型速度”, and “查测速历史”.
---

Use the bundled standard-library launcher at `<skill-root>/scripts/codex_speed.py` with local Python 3.11+. A benchmark starts one temporary Codex app-server session and consumes normal model tokens. It does not create a user-visible test chat or persisted rollout.

Run one sample by default. Omit `--model` to use the current Codex configuration. Run multiple samples or models only when requested.

```text
python -S "<skill-root>/scripts/codex_speed.py" benchmark --timeout 600
python -S "<skill-root>/scripts/codex_speed.py" benchmark --model <MODEL> --effort <EFFORT> --timeout 600
```

Read saved results with:

```text
python -S "<skill-root>/scripts/codex_speed.py" history --limit 20
python -S "<skill-root>/scripts/codex_speed.py" history --model <MODEL> --limit 20
python -S "<skill-root>/scripts/codex_speed.py" show <RUN_ID>
python -S "<skill-root>/scripts/codex_speed.py" watch --seconds 60
```

When asked to verify a new run, read recent history and use its `run_id` with `show`.

## Reply format

- For one benchmark, report only model, generation speed, and full-turn time. If it failed, give the reason in one sentence.
- For history, use a compact table. Include the method when records use different measurement sources.
- Add token counts, effort, measurement details, resource use, tests, or setup details only when asked.
- Do not print raw JSON or narrate routine tool steps.

## Measurement and history

The launcher returns JSON. Generation TPS is reported output tokens (which may include reasoning) divided by the span from the first output-item start to the last output-item end. Full-turn time spans `turn/start` through `turn/completed` and includes waiting and generation, but not app-server startup or handshake. These are output-item estimates, not per-token TTFT/ITL measurements. If timing or usage is invalid or missing, report no valid TPS; never estimate one.

History is stored at `~/.codex/codex-speed/benchmarks.sqlite3` (Windows: `%USERPROFILE%\.codex\codex-speed\benchmarks.sqlite3`). `--data-dir` or `CODEX_SPEED_DATA_DIR` overrides this database location. The database excludes prompts, answers, and raw Codex task identifiers. Use `history` or `show`, not conversation transcripts, to retrieve prior results.

Imported `mode=import` records use local-log estimates; active `benchmark` runs use app-server output-item estimates. Label the method when comparing mixed sources. Do not present a single sample as a stable model ranking; mention this only when the user asks for a ranking or conclusion.
