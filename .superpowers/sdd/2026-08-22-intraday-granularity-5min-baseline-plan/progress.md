# SDD ledger — plan: docs/superpowers/plans/2026-08-22-intraday-granularity-5min-baseline-plan.md

## Identity
- Plan: docs/superpowers/plans/2026-08-22-intraday-granularity-5min-baseline-plan.md
- Spec: docs/superpowers/specs/2026-08-22-intraday-granularity-5min-baseline-design.md
- Base commit: 9b7704e (spec + plan committed)
- Working tree: main branch (per CLAUDE.md precedent: spec changes ship direct to main, user explicitly authorized subagent-driven execution on this session)

## Pre-flight scan (per skill rule)

| Task pair / single task | Check | Result |
|---|---|---|
| A1 → A6 | A1 takes snapshot, A6 re-checks hash; gate value flows correctly | clean |
| A2 → A3 | A2 adds PERIODS tuple; A3 uses VALID_GRANULARITIES — both come from same source-of-truth (this plan); names match | clean |
| A3 → B1 | A3 exposes C.VALID_GRANULARITIES / C.TQ_PERIOD_MAP; B1 imports them and validates `period not in C.VALID_GRANULARITIES` | clean — names match |
| A3 → A4 | A3 defines DEFAULT_INTRADAY_LOOKBACK_DAYS=60; A4 references it in `--lookback-days` default | clean |
| A4 → B1 | A4 hardcodes `choices=['daily','15m','5m','1m']`; B1 validates via `C.VALID_GRANULARITIES`; A4 changes won't affect B1 unless argparse updates | clean — minor coupling: if A4's choices diverges from C.VALID_GRANULARITIES the test for help text in B1 could fail, but B1 reads C not A4 |
| B1 → B2 | B1 adds `period` to `load_ohlcva`; B2 passes `period` to `pipeline.load_ohlcva` from `load_pair` | clean |
| B2 → B3 | B2 adds `period` to `load_pair`; B3's projection scripts call `load_pair(..., period=args.period)` | clean |
| B3 → B4 | B3 and B4 are independent scripts adding `--period` flag with same shape | clean — no shared test |
| B4 → C1 | B4 doesn't touch output paths; C1 introduces `output_subdir_for_period` helper used by C1 step 5 | clean |
| C1 → C4 | C1 creates `dynamics_granularity_compare.py` with helper functions; C4 extends the same file with `build_daily_vs_5min_report` and constants | clean — same file, sequential tasks |
| C4 → D1 | C4 defines `DELTA_*` constants and `REPORT_TABLE_COLS`; D1 imports them and tests | clean |
| D2 → D3 | D2 reads comparison report and applies verdicts; D3 writes ADR | clean |
| Plan §Global Constraints: byte-equality preserved | Tasks A1, A6 enforce this; A4 must not touch daily default branch | clean — A4 step 5 explicitly says "For daily, behavior is unchanged" |
| Plan §Global Constraints: `_projection_core` math unchanged | B2 only adds `period` to `load_pair` signature; other math functions untouched | clean |
| Plan §Global Constraints: `dynamics_forced_response.py` NO change | B4 test_dynamics_cli_period.py asserts `--period` not in help for that script | clean — test enforces exclusion |

Scan verdict: **clean, no conflicts requiring pre-flight rulings**.

## Task progress

Task 1 (Plan Task A1): DONE — snapshot taken, SHA256 gate recorded

## Phase A byte-equality gate
- Reference: `data/stocks/000001_SZ_daily.csv.snapshot_a1`
- SHA256: `8eb86c9d50d7bebe288182bc2760daf2037b567e79eed6315e0f9aaf126ab57d`
- After Phase A Task A6, recompute hash. Must match.

Task 5 (Plan Task A5): DONE — TQ 5min smoke test PASSED
- TQ client: confirmed running (TdxW.exe + 2×tdxcef.exe)
- 5 stocks fetched: 000059_SZ, 000096_SZ, 000159_SZ, 000552_SZ, 000554_SZ
- Row counts: 3409 rows each (expected ~2880, well above 1500 threshold)
- TQ depth >= 60 days confirmed — no DEFAULT_INTRADAY_LOOKBACK_DAYS adjustment needed
- Manifest: period=5m, lookback_days=60

Task 6 (Plan Task A6): pending