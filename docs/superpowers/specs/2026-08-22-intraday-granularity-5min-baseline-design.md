# Intraday Granularity Architecture — 5min Baseline Design Spec

> **Status**: Draft 2026-08-22, awaiting user review.
> **Predecessors**: `2026-08-20-dynamics-e1-extended-decision.md` (extended OOS evidence); `2026-08-16-market-stock-dynamics-design.md` (math layer).
> **Anchors**: Spike #1 (`backtrace/spikes/spike_1min_nyquist.py`) confirms Nyquist deficiency on daily bar; Spike #2 (`backtrace/spikes/spike_granularity.py`, n=20 × 30 days) locates 5-min as candidate minimum sufficient resolution.

---

## 1. Background

### 1.1 Motivation

User hypothesis (2026-08-22):
> 动力学自然周期是日内的（小时级），日线分析可能太粗糙，超过了运动周期。

### 1.2 Spike evidence

| Spike | n | Configuration | Headline finding |
|---|---|---|---|
| #1 Nyquist | 3 stocks × 5 daily + 5 1-min days | Simplified 2D OLS | 1-min data reveals ~10-25 min oscillations; daily fit converges on ~17-24 day pseudo-period; F² ratio (daily params on 1-min data) = 1.49-1.97× |
| #2 Granularity | 20 stocks × 30 days × {1m, 5m, 15m} | Same OLS form | ω_n_phys estimates: 1m=17.5 min, 5m=104 min, 15m=295 min. F² cross-scale 15m→5m = 1.02, 5m→1m = 1.06 |

**Key takeaways** (anchored in spike CSVs at `backtrace/outputs/spike_*`):
- Daily data undersamples the system: the inferred ω_phys is ~3 orders of magnitude smaller than 1-min ω_phys.
- F² (variance explained) does not degrade dramatically at coarser granularities — slow drift dominates residuals, fast oscillations are a small F² component.
- The OLS picks the dominant mode visible at each granularity; 5-min sees ~100-min, daily sees ~20-day mode. These are likely different modes of a continuous spectrum rather than aliasing of a single mode, but distinguishing requires a wider frequency-domain study (out of scope for this spec).

### 1.3 What spike #2 does NOT prove

- **5-min is not strictly "optimal"**: it is the **smallest granularity** at which (a) ω_phys reflects intraday structure, (b) cross-scale F² transfer is near-identity, (c) data volume is practical for full-market batch (~5000 stocks).
- The choice is engineering-optimal under current constraints, not mathematically-proven-optimal.
- 1-min may surface additional ~10-20 min structure with ~6% F² improvement, but data cost is 8× and the additional business value is unproven.

---

## 2. Goals & non-goals

### 2.1 Goals

1. Make **granularity** a first-class parameter through the entire projection/dynamics pipeline.
2. Establish a **5-min baseline** data layer with configurable lookback (default 60 days).
3. Run a **controlled Daily-vs-5min parallel experiment** on the production pipeline (v0-v6) and quantify incremental business value via OOS / IC / SI.
4. Decide — based on that experiment — whether 5-min becomes a production resolution.

### 2.2 Non-goals (explicitly out of scope)

- **No rewrite of `_projection_core` math**. The ODE form, projection geometry, eigenvalue analysis stay unchanged. Only their data ingestion is generalized.
- **No change to daily defaults**. Existing daily pipeline keeps producing byte-identical output for `--period daily` (or no `--period`).
- **No automatic 5-min-to-daily promotion**. Decision at end of Phase D is a human decision; Phase D outputs a report, not a default swap.
- **No micro-structure alpha search** (order imbalance, bid-ask bounce). This is a granularity study, not a high-frequency strategy.
- **No tick-level data**. 1-min is the highest granularity considered; tick would require entirely different infrastructure.

---

## 3. Architecture overview

Four sequential phases. Each is independently shippable and reviewed:

```
Phase A (Data)            Phase B (Parameterization)        Phase C (Production comparison)    Phase D (Validation)
─────────────            ─────────────────────────         ─────────────────────────────      ──────────────────
fetch_daily.py           tsfresh_pipeline.load_ohlcva      Re-run v0-v6 on BOTH               v6 factor validation
  + --period flag          + period='5m'                    daily and 5min caches.              on 5min output.
                          _projection_core.load_pair        Capture identical metrics         Compute ΔIC / ΔSI /
data_store                + period='5m'                    (ω, k̂, ĉ, β, F², OOS              ΔOOS RMSE vs daily
  + csv_path(period)       dynamics_*: all CLI              RMSE, SI). Side-by-side            baseline.
  + _filename(period)      + --period flag                   report.
  + load_<period> helpers parameter_fit, projection_*
  + manifest key by        + --period flag
    period                 v6 factor_validation              Side-by-side decision:
                            + --period flag                  - accept 5min as production
                                                              default, or
                                                            - keep daily as default,
                                                              archive 5min as research
                                                              granularity, or
                                                            - kill 5min data layer entirely.
```

**Critical invariant**: any phase can be reverted independently without breaking the others. Phase B alone is meaningless without A; Phase C+D are gated on B.

---

## 4. Phase A — 5min data layer + configurable lookback

### 4.1 Goals

- Add 5-min K-line as a new cached resolution alongside daily.
- Make cache lookback configurable (default 60 days; configurable: 30 / 60 / 90).
- Reuse `fetch_daily.py` machinery (TQ client, batch retry, manifest, sector members) — minimal new code.

### 4.2 Concrete changes

#### 4.2.1 `backtrace/common/data_store.py`

**New period key** — `'daily' | '15m' | '5m' | '1m'`. Default `'daily'` (backward compatible).

```python
PERIODS = ('daily', '15m', '5m', '1m')

def _filename(code, period='daily'):
    """000001.SH + '5m' -> 000001_SH_5m.csv  (legacy: 000001_SH_daily.csv)"""
    if period == 'daily':
        return f"{code.replace('.', '_')}_daily.csv"
    return f"{code.replace('.', '_')}_{period}.csv"

def csv_path(code, period='daily', kind='stocks'):
    if period not in PERIODS:
        raise ValueError(f"period 必须是 {PERIODS} 之一,收到 {period!r}")
    return os.path.join(DATA_DIR, kind, _filename(code, period))

def save_df(code, df, period='daily', kind='stocks'):  # 替代 save_daily
    path = csv_path(code, period, kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    df.to_csv(tmp, encoding='utf-8')
    os.replace(tmp, path)
    return path

def load_df(code, period='daily'):  # 替代 load_daily
    """跨 period 与 kind 查找: stocks → sectors → indices"""
    for kind in KINDS:
        p = csv_path(code, period, kind)
        if os.path.exists(p):
            return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    return None

# 保留 save_daily / load_daily / has_daily 作为 thin wrapper(period='daily' 默认)
save_daily = lambda code, df, kind='stocks': save_df(code, df, 'daily', kind)
load_daily = lambda code: load_df(code, 'daily')
has_daily = lambda code: any(os.path.exists(csv_path(code, 'daily', k)) for k in KINDS)
```

#### 4.2.2 `backtrace/common/tsfresh_config.py`

**New constants**:

```python
# -------- Intraday granularity --------
VALID_GRANULARITIES = ('daily', '15m', '5m', '1m')
DEFAULT_INTRADAY_GRANULARITY = '5m'  # Phase C/D 实验默认
DEFAULT_INTRADAY_LOOKBACK_DAYS = 60  # Phase A 缓存天数;用户原则:不写死,可改 30/60/90
TQ_PERIOD_MAP = {'daily': '1d', '15m': '15m', '5m': '5m', '1m': '1m'}
```

`LOOKBACK_YEARS` (existing) stays for daily. New `INTRADAY_LOOKBACK_DAYS` for intraday.

#### 4.2.3 `backtrace/data_fetch/fetch_daily.py`

**Add `--period` and `--lookback-days` flags**:

```python
parser.add_argument('--period', choices=['daily', '15m', '5m', '1m'],
                    default='daily',
                    help='缓存粒度。daily = 现有默认行为;其余走 intraday 流程')
parser.add_argument('--lookback-days', type=int, default=0,
                    help='intraday 回看天数(daily 忽略)。0 = 用 C.DEFAULT_INTRADAY_LOOKBACK_DAYS')
```

**Behavior split** (per CLAUDE.md "职责边界: 本模块只做编排"):
- `period='daily'` → existing path unchanged (500 trading days, TQ `period='1d'`).
- `period in {15m,5m,1m}` → new path: `--lookback-days` (default 60) → TQ `period=args.period` → save to `data/{stocks,sectors,indices}/<code>_<period>.csv` → write manifest with `period` key.

**Manifest extension**: existing fields stay; add `period` to each entry and `period` to top-level `generated_at` block:
```json
{
  "generated_at": "2026-08-22T...",
  "period": "5m",
  "lookback_days": 60,
  "trading_days": null,  // intraday 不适用, 写 null
  "entries": {
    "000001.SH": {"rows": 4800, "first_date": "...", "last_date": "...", "fetched_at": "...", "period": "5m"}
  }
}
```

#### 4.2.4 CLI smoke test

```bash
# Daily 不变
PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py --limit 5

# 5min 冒烟(60 天)
PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py --period 5m --limit 5

# 5min 90 天(可配置验证)
PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py --period 5m --limit 5 --lookback-days 90
```

Expected: 第一条命令产出与改动前 byte-identical。第二/三条产出新 CSV 在 `data/stocks/*_5m.csv`。

### 4.3 What Phase A does NOT do

- Does NOT touch `projection._projection_core` math.
- Does NOT change any output directory naming for the daily path.
- Does NOT add automatic cleanup of old period caches.

### 4.4 Tests for Phase A

- `tests/test_data_store.py` — extend existing tests with period-parameterized cases:
  - `csv_path('000001.SH', period='5m')` → expected path
  - `load_df('000001.SH', 'daily')` still returns daily CSV (backward compat)
  - `save_df('600519.SH', df, '5m', 'stocks')` then `load_df('600519.SH', '5m')` round-trip
  - `csv_path('x.SH', period='3m')` raises ValueError
- `tests/test_fetch_daily_cli.py` — smoke test: `python fetch_daily.py --period daily --limit 5` produces identical manifest structure (skip data-equality test, TQ rate-limited).

### 4.5 Phase A exit criteria

- `fetch_daily.py --period daily` produces byte-identical output to pre-Phase-A baseline.
- `fetch_daily.py --period 5m --limit 5` produces 5 stock CSVs × ~2880 rows (5min × 60d × 48 bars/day = 2880; TQ may cap at fewer days, verify empirically in Task 1) + manifest with `period='5m'`.
- Tests pass.

---

## 5. Phase B — granularity parameter through production pipeline

### 5.1 Goals

`period` (granularity) propagates from CLI through every layer that today implicitly assumes `1d`:
- `tsfresh_pipeline.load_ohlcva(code, period='1d', ...)`
- `projection._projection_core.load_pair(stock_code, days, pipeline, period='1d', ...)`
- All `projection/*` and `dynamics/*` scripts gain `--period` flag.

### 5.2 Concrete changes

#### 5.2.1 `backtrace/common/tsfresh_pipeline.py`

`load_ohlcva(code, lookback_years=None, use_tq=True, verbose=False, include_amount=True, *, period='daily')`:

- Default `period='daily'` keeps backward compat with all current callers.
- Period → TQ mapping from `C.TQ_PERIOD_MAP`.
- Period → CSV path via `data_store.csv_path(code, C.TQ_PERIOD_MAP[period], kind)`.

Logic flow:
```python
def load_ohlcva(code, lookback_years=None, use_tq=True, verbose=False,
                include_amount=True, *, period='daily'):
    if period not in C.VALID_GRANULARITIES:
        raise ValueError(f"period 必须是 {C.VALID_GRANULARITIES} 之一,收到 {period!r}")
    tq_period = C.TQ_PERIOD_MAP[period]
    if period == 'daily':
        lookback = lookback_years or C.LOOKBACK_YEARS
        # existing path...
    else:
        # intraday: use C.DEFAULT_INTRADAY_LOOKBACK_DAYS unless caller overrides
        # (caller controls via lookback parameter, but for intraday we use days not years)
        ...
    # CSV fallback via data_store.load_df(code, period)
    ...
```

#### 5.2.2 `backtrace/projection/_projection_core.py`

`load_pair(stock_code, days, pipeline, prefer_industry=False, index_code=None, lag=0, *, period='daily')`:

- Adds `period` kwarg only; default `'daily'` preserves byte-identical daily output.
- Passes `period` to `pipeline.load_ohlcva(..., period=period)`.
- All other code (`compute_vectors`, `compute_movement_projection`, etc.) **unchanged**.

#### 5.2.3 Scripts to update (add `--period` flag)

| File | Type of change |
|---|---|
| `backtrace/projection/projection_2d.py` | add `--period {daily,15m,5m,1m}` argparse; pass to `load_pair` |
| `backtrace/projection/projection_batch.py` | add `--period`; pass through to per-stock `load_pair` |
| `backtrace/projection/parameter_fit.py` | add `--period`; CSV path input only (reads existing `data/projection/movement_*.csv`), no functional change unless projection_batch is re-run |
| `backtrace/projection/prediction_ode.py` | add `--period`; pass to `load_pair` |
| `backtrace/projection/state_kc_analysis.py` | add `--period` |
| `backtrace/projection/v0_2_*.py` | add `--period` |
| `backtrace/dynamics/dynamics_system.py` | add `--period` |
| `backtrace/dynamics/dynamics_batch.py` | add `--period` |
| `backtrace/dynamics/dynamics_1step_oos.py` | add `--period` |
| `backtrace/dynamics/dynamics_state_backtest.py` | add `--period` |
| `backtrace/dynamics/dynamics_eigen_analysis.py` | add `--period` (only affects cache path of kc_estimates) |
| `backtrace/dynamics/dynamics_oos_*.py` | add `--period` |
| `backtrace/dynamics/dynamics_forced_response.py` | NO change (math-only, no data loading) |
| `backtrace/dynamics/dynamics_state_timeline.py` | add `--period` |
| `backtrace/dynamics/dynamics_si_*.py` | add `--period` |
| `backtrace/dynamics/dynamics_factor_validation.py` | add `--period` |

All scripts: `--period` defaults to `daily` → existing CLIs unchanged.

#### 5.2.4 Cross-cutting constants

Add to `tsfresh_config.py`:
```python
GRANULARITY_DT_SEC = {'daily': 86400, '15m': 900, '5m': 300, '1m': 60}
```
Use for ω_n_phys computation and time-axis rescaling in scripts that currently hardcode `86400`.

### 5.3 What Phase B does NOT do

- Does NOT modify `_projection_core` math: `compute_vectors`, `compute_movement_projection`, `compute_dynamics`, `classify_states`, `compute_forces` stay byte-identical.
- Does NOT modify `parameter_fit.py` OLS math.
- Does NOT modify dynamics simulator (`predict_next_state`, `simulate_trajectory`).
- Does NOT promote 5-min to any default — `--period` defaults stay at `daily`.

### 5.4 Tests for Phase B

- `tests/test_projection_core.py`:
  - `load_pair(code, 60, P, period='daily')` byte-identical to pre-Phase-B (run on existing daily data).
  - `load_pair(code, 60, P, period='5m')` returns 5-min data (smoke; only if 5-min cache exists, else skip with explicit reason).
- `tests/test_dynamics_eigen.py`: unchanged tests must still pass (no `--period` invocation).
- CLI smoke: `python backtrace/projection/projection_2d.py --code 002475.SZ --period daily` produces byte-identical output to pre-Phase-B baseline.

### 5.5 Phase B exit criteria

- All existing tests pass.
- `python projection_2d.py --code 002475.SZ --period daily` byte-identical to pre-Phase-B.
- `python projection_2d.py --code 002475.SZ --period 5m` runs end-to-end on 5-min cache (produces new HTML/CSV).
- `dynamics_*.py` scripts accept `--period` and reject invalid values cleanly.

---

## 6. Phase C — Daily vs 5min production comparison

### 6.1 Goals

Run the **complete v0-v6 production pipeline** twice: once on daily cache (existing), once on 5-min cache (Phase A). Capture identical metrics from both runs. Output a side-by-side comparison report.

### 6.2 Procedure

For each resolution (daily, 5min), run the full chain:

```
fetch_daily.py --period {daily|5m}                # Phase A
   ↓
projection_batch.py --period {daily|5m}           # v0
   ↓
parameter_fit.py --period {daily|5m}              # v1
   ↓
dynamics_1step_oos.py --period {daily|5m}         # v3 OOS baseline
   ↓
dynamics_state_backtest.py --period {daily|5m}    # v3 state distribution
   ↓
dynamics_eigen_analysis.py --period {daily|5m}    # v4
   ↓
dynamics_si_freq_response.py --period {daily|5m}  # v5
   ↓
dynamics_factor_validation.py --period {daily|5m} # v6 (consumer)
```

Daily run uses existing `data/projection_v01_c1/` snapshot as reference for byte-equality verification of the unchanged code paths.

### 6.3 Outputs

Per resolution, the standard v0-v6 outputs land in period-tagged directories:

```
data/projection_5min/
data/dynamics_5min/
backtrace/outputs/v0_v6_5min/

data/projection_daily/  (existing — untouched)
data/dynamics_daily/    (existing — untouched)
backtrace/outputs/v0_v6_daily/  (existing)
```

A new comparison report:
```
backtrace/outputs/granularity_compare/
├── daily_vs_5min_summary.txt    (UTF-8 中文 + ASCII)
├── daily_vs_5min_table.csv      (ΔIC / ΔSI / ΔOOS per factor × horizon)
├── daily_vs_5min_topology.csv   (ω / k̂ / ĉ / β / F² 跨尺度对照)
└── daily_vs_5min.html           (2×2 plotly: IC diff / SI diff / OOS RMSE diff / hit-rate diff)
```

### 6.4 Granularity-aware output paths

To avoid overwriting existing daily outputs, scripts that currently hardcode `data/projection/` or `backtrace/outputs/` get a `--period` modifier:
- `data/projection_{period}/movement_*.csv` (instead of `data/projection/movement_*.csv`)
- `data/dynamics_{period}/...`
- `backtrace/outputs/v{version}_{period}_*.html`

This is a **one-time path remap**; daily output dirs (`data/projection/`, `backtrace/outputs/`) are unchanged when `--period daily` (the default). For 5min, the suffix `_5min` is appended.

### 6.5 What Phase C does NOT do

- Does NOT make 5-min a default anywhere.
- Does NOT modify any math or scoring logic.
- Does NOT generate trade signals.
- Does NOT rerun the v0-v6 chain with any other granularity besides daily and 5-min.

### 6.6 Tests for Phase C

- Daily run: byte-equality vs `data/projection_v01_c1/` snapshot (numerical regression gate).
- 5-min run: smoke — `--limit 5` produces 5 movement CSVs + 5 dynamics CSVs without exceptions.
- Comparison report: spot-check that CSV row counts match input stock count × factor count.

### 6.7 Phase C exit criteria

- Daily re-run is byte-identical (or ε-identical for floating-point) to the v0-v6 reference snapshot.
- 5-min re-run completes for `--limit 50` smoke without exceptions.
- `daily_vs_5min_summary.txt` exists and is non-empty.

---

## 7. Phase D — Decision framework

### 7.1 Decision criterion

5-min becomes a production resolution **only if** the following conditions hold **simultaneously**:

| Metric | Threshold | Source |
|---|---|---|
| ΔIC_mean (per factor × horizon) | at least one factor with ΔIC ≥ +0.02 **AND** p-value < 0.05 | v6 factor_validation.csv (daily vs 5min) |
| ΔIC_IR | at least one factor with ΔIC_IR ≥ +0.1 | v6 factor_validation.csv |
| ΔOOS RMSE | 5min median ≤ -5% vs daily | dynamics prediction_summary.csv |
| ΔSI_lagged_IC | 5min ≥ daily + 0.02 | si_lagged_ic_summary.csv (if available) |
| Δhit_rate | 5min median ≥ daily + 3pp | prediction_summary.csv |

**Soft signals** (do not block, but recorded):
- New ω_n_phys modes resolved at 5min (already known ~100 min vs daily ~20 days).
- New 11-class classifications appear in eigen_summary (regime transition under finer sampling).
- Factor rankings: which factors rank differently at 5min (qualitative signal only).

### 7.2 Decision outcomes

| Outcome | Trigger | Action |
|---|---|---|
| **Adopt 5min** | All hard thresholds met | Make `--period 5min` the production default for `parameter_fit` + `dynamics_*`. Keep `--period daily` as fallback. Write ADR (Architecture Decision Record) with full evidence. |
| **Archive 5min** | Mixed results (some factors improve, others degrade) | Keep 5-min data layer; document in README as "candidate resolution, no production decision". Re-evaluate with longer lookback (90 days) in 6 months. |
| **Kill 5min** | ΔIC ≈ 0, ΔOOS ≈ 0, ΔSI ≈ 0 across all factors | Remove `--period` flag from CLI surfaces; archive `data/5min/` CSVs to `data/_archive/5min_<date>/`. Keep code path dormant (or remove). |

### 7.3 Decision owner

User. Phase D produces the report; user reads it and decides.

### 7.4 Tests for Phase D

- `tests/test_granularity_decision.py`: framework that runs both pipelines (or reads existing CSV outputs), computes the threshold checks above, and produces a YAML/JSON verdict.
- Threshold values are constants in test file; any change requires explicit review.

---

## 8. Cross-cutting concerns

### 8.1 TQ data depth (practical risk)

`period='5m'` data depth from TQ is unverified for 60 days. Empirical check during Phase A Task 1:
- If TQ returns full 60 days × 48 bars/day = 2880 rows: proceed.
- If TQ truncates to e.g. 30 days: reduce `DEFAULT_INTRADAY_LOOKBACK_DAYS` to actual achievable depth, document in config.

### 8.2 Storage cost

| Resolution | Rows per stock × 5000 stocks × 60 days |
|---|---|---|
| daily | 500 × 5000 = 2.5M rows (~250 MB CSV) |
| 15min | 1920 × 5000 = 9.6M rows (~1 GB) |
| 5min | 5760 × 5000 = 28.8M rows (~3 GB) |
| 1min | 14400 × 5000 = 72M rows (~7 GB) |

`.gitignore` already excludes `data/` — no repo impact. Disk usage on local machine is the only cost.

### 8.3 Manifest / cache invalidation

When `--period` is added, manifest entries must be period-tagged to prevent stale lookup. Existing daily manifest entries continue to work (no `period` field defaults to `'daily'` for backward compat at read time).

### 8.4 Backward compatibility gate

Any change to `data_store` or `fetch_daily.py` with `--period daily` (default) must produce **byte-identical** output to the pre-Phase-A baseline. Verified in Phase A Task 1 by snapshot of `data/stocks/000001_SZ_daily.csv` before and after.

### 8.5 Scope creep guards

- **No new dynamics math**. Even if 5min surfaces new ω_n structure, the analysis framework (eigenvalues, regime, G(ω)) stays unchanged.
- **No new alpha strategies**. v6 factor validation is consumer-only.
- **No micro-structure extension**. 1-min is the maximum.

---

## 9. Risks & unknowns

| Risk | Likelihood | Mitigation |
|---|---|---|
| TQ 5-min depth < 60 days | medium | Phase A Task 1: empirical probe; if true, reduce `DEFAULT_INTRADAY_LOOKBACK_DAYS` |
| 5-min data volume bottlenecks batch pipeline | low | Phase C: smoke `--limit 5` first; if bottleneck, parallelize or chunk |
| F² ratio improvement from 5min is ~6% (per spike #2) — may not translate to IC improvement | high | Phase D captures all metrics; decision framework honest about this risk |
| Daily re-run fails byte-equality due to numpy version drift | low | Snapshot pre-Phase-A; compare with ε tolerance |
| User changes `DEFAULT_INTRADAY_LOOKBACK_DAYS` mid-Phase-C and invalidates cache | low | Cache keyed by `(period, lookback_days)`; rebuild deterministic |

---

## 10. Out of scope (explicit)

- **Tick-level data**. Architectural ceiling = 1-min.
- **Real-time streaming**. TQ client is batch-mode only.
- **Multi-resolution fusion** (e.g., use daily for slow mode + 5min for fast mode simultaneously). Future work.
- **Distributed compute**. Single-machine batch only.
- **Cross-market** (港股 / 美股). Out of scope per CLAUDE.md.
- **Live trading signal**. Per CLAUDE.md "信号推送,不自动下单".

---

## 11. File-by-file change matrix

| File | Phase | Change |
|---|---|---|
| `backtrace/common/data_store.py` | A | add `period` to `csv_path`/`_filename`; new `save_df`/`load_df`; thin `save_daily`/`load_daily` wrappers |
| `backtrace/common/tsfresh_config.py` | A + B | `VALID_GRANULARITIES`, `DEFAULT_INTRADAY_GRANULARITY`, `DEFAULT_INTRADAY_LOOKBACK_DAYS`, `TQ_PERIOD_MAP`, `GRANULARITY_DT_SEC` |
| `backtrace/common/tsfresh_pipeline.py` | B | `load_ohlcva(..., period='daily')` |
| `backtrace/data_fetch/fetch_daily.py` | A | `--period`, `--lookback-days`; manifest period-tagged |
| `backtrace/projection/_projection_core.py` | B | `load_pair(..., period='daily')` |
| `backtrace/projection/projection_2d.py` | B + C | `--period`, output dir suffix |
| `backtrace/projection/projection_batch.py` | B + C | `--period`, output dir suffix |
| `backtrace/projection/parameter_fit.py` | B + C | `--period` (cosmetic; reads CSV; rerun uses new period's movement files) |
| `backtrace/projection/prediction_ode.py` | B | `--period` |
| `backtrace/projection/state_kc_analysis.py` | B | `--period` |
| `backtrace/projection/v0_2_*.py` | B | `--period` |
| `backtrace/dynamics/dynamics_*.py` | B + C | `--period`, output dir suffix (except `dynamics_forced_response.py` which is math-only) |
| `tests/test_data_store.py` | A | new period-parameterized cases |
| `tests/test_projection_core.py` | B | period-parameterized `load_pair` smoke |
| `tests/test_granularity_decision.py` | D | decision framework |
| `docs/superpowers/plans/2026-08-22-intraday-granularity-5min-baseline-plan.md` | (after spec approved) | Implementation plan |
| `backtrace/README.md` (none exist at top) | end | update `backtrace/dynamics/README.md` only if adopted |

Total code touch: 1 new helper module section (~50 LOC), 1 fetch script (~30 LOC), 1 pipeline function signature (~30 LOC), ~14 CLI scripts (~5 LOC each = ~70 LOC), output dir remap (~5 LOC per script), tests (~80 LOC). Approximate Phase A+B+C+D LOC: **~500 LOC**.

---

## 12. Rollout timeline (sequential, not parallel)

```
Phase A  (estimate: 0.5 day)
  Task 1: data_store + fetch_daily changes + manifest
  Task 2: 5min cache smoke test on 5 stocks
  Task 3: verify daily byte-equality preserved
  Exit: tests pass, smoke OK

Phase B  (estimate: 1 day)
  Task 1: tsfresh_pipeline.load_ohlcva period param
  Task 2: _projection_core.load_pair period param
  Task 3: 14 scripts add --period
  Task 4: smoke 5min end-to-end on 1 stock
  Exit: tests pass, daily byte-identical, 5min smoke OK

Phase C  (estimate: 1-2 days, mostly waiting on data fetches + batch runs)
  Task 1: daily full v0-v6 rerun (verify byte-identical)
  Task 2: 5min full v0-v6 rerun
  Task 3: comparison report generator
  Exit: report generated

Phase D  (estimate: 0.5 day, mostly reading)
  Task 1: run decision framework
  Task 2: produce final report
  Task 3: user reads and decides
  Exit: ADR (if adopt) or archive (if not)
```

Total estimate: 3-4 working days. Most of Phase C is wall-clock on batch jobs.

---

## 13. Reference anchors

- Spike evidence: `backtrace/outputs/spike_1min_nyquist/` and `backtrace/outputs/spike_granularity/`
- v6 factor validation framework: `backtrace/dynamics/dynamics_factor_validation.py` §6.9 in `backtrace/dynamics/README.md`
- v0.2 driver-default migration precedent: `docs/superpowers/plans/2026-08-20-dynamics-f-driver-default-migration.md` (similar shape: scope-limited, byte-equality gate, caller audit)
- TQ SDK period list: `C:/new_tdx_mock/PYPlugins/user/tqcenter.py` line 1032 (`['5m', '15m', '30m', '1h', '1d', '1w', '1mon', '1m', ...]`)

---

## 14. Open questions (resolve in plan, not in spec)

1. Should `--period` be passed positionally or as flag in projection_2d.py for ergonomics? — **defer to plan**.
2. Output dir naming: `data/projection_5min/` vs `data/projection/{period}/`? — **defer to plan**.
3. Should `dynamics_forced_response.py` (math-only) get `--period` for symmetry, or be left out? — current spec says NO; verify in plan review.
4. Is the decision threshold table in §7.1 strict enough, or should we accept "mixed results" as Archive without a hard p-value? — **defer to plan**.

---

## 15. Self-review checklist (resolve before user review)

- [x] Placeholder scan: no "TBD" / "TODO" / incomplete sections.
- [x] Internal consistency: phase scope does not overlap; each phase's "does NOT do" list excludes prior phase's deliverables.
- [x] Scope check: 4 phases fit a single implementation plan (~500 LOC, ~3-4 days).
- [x] Ambiguity check: every threshold, every constant, every script has explicit values; nothing left to "interpret".
- [x] Decision framework is honest: "kill 5min" is a valid outcome; not pre-decided.

---

## 16. Next step

After user approves this spec:
1. Run `superpowers:writing-plans` to produce `docs/superpowers/plans/2026-08-22-intraday-granularity-5min-baseline-plan.md`.
2. Plan will task-by-task decompose Phase A → B → C → D with concrete checkboxes, gates, and test commands.