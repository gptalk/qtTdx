# ADR: 5min Granularity Decision — 2026-08-22

## Status
**Rejected**

## Context
Daily projection/dynamics pipeline assumes 1-day resolution. Spike #1/#2 (2026-08-22) found Nyquist deficiency on daily bar, with 5-min as candidate minimum sufficient resolution.

## Decision
**Kill 5min** — see rationale below.

## Rationale
After fixing `load_daily_prices()` to be period-aware (D2 critical bug fix), 30 factors were compared across daily vs 5min using matched stock sets (n=50 for ΔIC, n=15 for hit_rate). Results are unambiguous: 0 factors improve ΔIC ≥ +0.02, 0 factors improve ΔIC_IR ≥ +0.1. The kc-based factors (k, c, c_over_k, log_c_over_k, rho, theta, dist_to_unit) produce **identical IC values** for daily and 5min because `parameter_fit` reads the same `kc_estimates.csv` regardless of period — the (k, c) estimates are generated once and reused. OOS-based factors (hit_rate, rmse) are 2-7pp worse for 5min: hit_rate median drops 6.6pp (43.0% vs 49.6%, n=15 common stocks). The 5min pipeline introduces noise without incremental predictive signal.

## Consequences
- 5min data layer (`data/stocks_5m/`, `data/projection_5m/`, `data/dynamics_5m/`) remains dormant but intact.
- `--period` flag stays as a research affordance in all 14+ scripts — no production promotion.
- No changes to `tsfresh_config.py` defaults (DEFAULT_INTRADAY_GRANULARITY remains `5m` as a research-only setting).
- The period-aware `load_daily_prices` fix (D2) is preserved — it enables fair future comparisons if 5min is revisited with different factor definitions or longer lookback.
- Destructive cleanup (removing `--period` from CLI surfaces, archiving 5min data) is deferred to a separate user decision.

## Evidence
- Spike CSVs: `backtrace/outputs/spike_1min_nyquist/`, `backtrace/outputs/spike_granularity/`
- Comparison report: `backtrace/outputs/granularity_compare/`
- D2 final report: `.superpowers/sdd/2026-08-22-intraday-granularity-5min-baseline-plan/task-D2-final-report.md`
- Test suite: **243 passed** (D1 added 3 decision-framework tests; full suite unchanged through D3)
- Spec: `docs/superpowers/specs/2026-08-22-intraday-granularity-5min-baseline-design.md`
- Plan: `docs/superpowers/plans/2026-08-22-intraday-granularity-5min-baseline-plan.md`
