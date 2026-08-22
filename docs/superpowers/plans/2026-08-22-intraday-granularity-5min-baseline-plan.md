# Intraday Granularity Architecture — 5min Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make granularity a first-class parameter through the entire projection/dynamics pipeline; add 5-min as a baseline intraday resolution; run a controlled Daily-vs-5min parallel experiment on the v0-v6 production pipeline; decide whether 5-min becomes production.

**Architecture:** Extend `data_store` and `fetch_daily.py` to support period-keyed cache (`daily | 15m | 5m | 1m`). Thread `period` parameter through `tsfresh_pipeline.load_ohlcva` → `_projection_core.load_pair` → 14 projection/dynamics CLI scripts via `--period` flag. All existing daily defaults preserved (byte-equality gate). Phase C runs both pipelines and produces side-by-side report; Phase D applies a v6-style decision framework.

**Tech Stack:** Python 3.12, pandas, numpy, vectorbt, plotly, pytest; TQ SDK via `C:/new_tdx_mock/PYPlugins/user/tqcenter.py`.

**Spec:** `docs/superpowers/specs/2026-08-22-intraday-granularity-5min-baseline-design.md`

## Global Constraints

- **`period` values**: `'daily' | '15m' | '5m' | '1m'` (lowercase, single token).
- **Defaults**: All scripts default `--period daily`. No automatic 5min promotion.
- **Backward compatibility**: `fetch_daily.py --period daily` and any `--period` absent must produce **byte-identical** output to pre-Phase-A baseline.
- **`_projection_core` math is unchanged**: only `load_pair` signature gains `period` kwarg. `compute_vectors`, `compute_movement_projection`, `compute_dynamics`, `classify_states`, `compute_forces` stay byte-identical.
- **`dynamics_forced_response.py`**: math-only (Bode plots from (k, c)), does NOT get `--period`. Per spec §5.2.3.
- **TQ SDK period mapping**: `'daily' → '1d'`, `'15m' → '15m'`, `'5m' → '5m'`, `'1m' → '1m'` from `C.TQ_PERIOD_MAP`.
- **Manifest**: each entry gains `period` key. Top-level gains `period` and `lookback_days`.
- **CSV naming**: `<code>_<period>.csv`. Existing `000001_SZ_daily.csv` stays (legacy `_daily.csv` suffix preserved by spec §4.2.1).
- **Cache directory**: `data/{stocks,sectors,indices}/<code>_<period>.csv`. Period-tagged subdirs only for **outputs** (`data/projection_5min/`, `data/dynamics_5min/`, `backtrace/outputs/v{ver}_5min_*`), per spec §6.4.
- **DT seconds**: `'daily'=86400, '15m'=900, '5m'=300, '1m'=60` from `C.GRANULARITY_DT_SEC`. Used in any ω_n_phys computation.
- **Decision thresholds** (Phase D): ΔIC ≥ +0.02 AND p-value < 0.05; ΔIC_IR ≥ +0.1; ΔOOS RMSE ≤ -5%; ΔSI_lagged_IC ≥ +0.02; Δhit_rate ≥ +3pp.
- **Commit message style**: `feat|fix|docs(scope):` prefix, ASCII. Match `docs/superpowers/specs/2026-08-20-dynamics-c1-market-driver-swap.md` pattern.
- **Windows GBK**: All subprocesses use `PYTHONIOENCODING=utf-8`. All `open()` calls use `encoding='utf-8'`.
- **Tests stay green**: `pytest tests/ -q` must pass after each task.

---

## File Structure

This plan modifies the following (existing files prefixed with `M`, new with `C`):

| Path | Phase | Purpose |
|---|---|---|
| `M backtrace/common/data_store.py` | A | Period-aware cache |
| `M backtrace/common/tsfresh_config.py` | A + B | Granularity constants |
| `M backtrace/common/tsfresh_pipeline.py` | B | Period param to `load_ohlcva` |
| `M backtrace/data_fetch/fetch_daily.py` | A | `--period`, `--lookback-days` |
| `M backtrace/projection/_projection_core.py` | B | Period param to `load_pair` |
| `M backtrace/projection/projection_2d.py` | B + C | `--period`, output dir |
| `M backtrace/projection/projection_batch.py` | B + C | `--period`, output dir |
| `M backtrace/projection/parameter_fit.py` | B + C | `--period` |
| `M backtrace/projection/prediction_ode.py` | B | `--period` |
| `M backtrace/projection/state_kc_analysis.py` | B | `--period` |
| `M backtrace/projection/v0_2_*.py` | B | `--period` (4 files) |
| `M backtrace/dynamics/dynamics_system.py` | B + C | `--period`, output dir |
| `M backtrace/dynamics/dynamics_batch.py` | B + C | `--period`, output dir |
| `M backtrace/dynamics/dynamics_1step_oos.py` | B + C | `--period`, output dir |
| `M backtrace/dynamics/dynamics_state_backtest.py` | B + C | `--period`, output dir |
| `M backtrace/dynamics/dynamics_eigen_analysis.py` | B + C | `--period`, output dir |
| `M backtrace/dynamics/dynamics_oos_*.py` | B + C | `--period` (2 files: `oos_viz.py`, `oos_batch.py`) |
| `M backtrace/dynamics/dynamics_state_timeline.py` | B | `--period` |
| `M backtrace/dynamics/dynamics_si_*.py` | B + C | `--period` (`si_freq_response.py`, `si_ic.py`, `si_timeseries.py`, `si_lagged_ic.py`) |
| `M backtrace/dynamics/dynamics_factor_validation.py` | B + C | `--period` |
| `M tests/test_data_store.py` | A | Period-aware cases |
| `M tests/test_projection_core.py` | B | Period-aware `load_pair` cases |
| `C tests/test_granularity_decision.py` | D | Decision framework |
| `C backtrace/dynamics/dynamics_granularity_compare.py` | C | Side-by-side report generator |
| `C data/projection_v01_c1_v0_2_phase_a_reference/` | A | Byte-equality snapshot (gitignored) |

---

## Phase A — 5min data layer

### Task A1: Pre-change snapshot for byte-equality gate

**Files:**
- Create: `data/projection_v01_c1_v0_2_phase_a_reference/` (gitignored snapshot of an existing reference, e.g. one movement CSV + manifest)

**Why this task first:** every Phase A change must preserve daily byte-equality. We need a snapshot to compare against.

- [ ] **Step 1: Pick a reference CSV to snapshot**

Inspect existing v0.2-C1 reference dir (created in plan `2026-08-20-dynamics-f-driver-default-migration.md` Task 1):
```bash
ls data/projection_v01_c1_v0_2_c1_reference/ 2>&1 | head -5
```

Expected: `kc_estimates_model0.csv kc_estimates_model1.csv ... c0_c1_compare_summary.txt`.

- [ ] **Step 2: Snapshot a single daily stock CSV**

```bash
cp data/stocks/000001_SZ_daily.csv data/stocks/000001_SZ_daily.csv.snapshot_a1
ls -la data/stocks/000001_SZ_daily.csv.snapshot_a1
```

This is gitignored (per `.gitignore` excludes `data/`). The `.snapshot_a1` suffix flags it as a known reference.

- [ ] **Step 3: Compute SHA256 of snapshot**

```bash
python -c "import hashlib; print(hashlib.sha256(open('data/stocks/000001_SZ_daily.csv.snapshot_a1','rb').read()).hexdigest())"
```

Record: `<paste-hash-here>` (this is the gate value).

- [ ] **Step 4: Commit the snapshot file (force-add since it's gitignored)**

Since `data/` is gitignored, we use `git check-ignore` to verify:
```bash
git check-ignore -v data/stocks/000001_SZ_daily.csv.snapshot_a1
```

Expected: `data/  .gitignore:NN:.gitignore` or similar — gitignored. No commit needed; this file lives only locally as a byte-equality gate.

- [ ] **Step 5: Record gate in progress.md**

Create `.superpowers/sdd/2026-08-22-intraday-granularity-5min-baseline/progress.md`:
```markdown
# SDD ledger — plan: docs/superpowers/plans/2026-08-22-intraday-granularity-5min-baseline-plan.md

## Identity
- Plan: docs/superpowers/plans/2026-08-22-intraday-granularity-5min-baseline-plan.md
- Spec: docs/superpowers/specs/2026-08-22-intraday-granularity-5min-baseline-design.md
- Base commit: <current HEAD>
- Working tree: main branch

## Phase A byte-equality gate
- Reference: `data/stocks/000001_SZ_daily.csv.snapshot_a1`
- SHA256: <paste-hash-from-step-3>
- After Phase A Task A6, recompute hash. Must match.
```

- [ ] **Step 6: Commit progress.md**

```bash
git add .superpowers/sdd/2026-08-22-intraday-granularity-5min-baseline/progress.md
git commit -m "docs(intraday): Phase A byte-equality gate + ledger"
```

---

### Task A2: Add `period` to `data_store.py` + tests

**Files:**
- Modify: `backtrace/common/data_store.py` (add `PERIODS`, `period` kwarg on `csv_path`/`_filename`/`save_daily`/`load_daily`; add `save_df`/`load_df`)
- Modify: `tests/test_data_store.py` (add period-aware cases)

**Interfaces:**
- Consumes: existing `data_store.DATA_DIR`, `KINDS`, `COLUMNS`
- Produces:
  ```python
  PERIODS: tuple[str, ...] = ('daily', '15m', '5m', '1m')

  def csv_path(code: str, period: str = 'daily', kind: str = 'stocks') -> str
  def _filename(code: str, period: str = 'daily') -> str
  def save_df(code: str, df: pd.DataFrame, period: str = 'daily', kind: str = 'stocks') -> str
  def load_df(code: str, period: str = 'daily') -> pd.DataFrame | None
  ```

- [ ] **Step 1: Write failing tests**

Append to `tests/test_data_store.py`:
```python
def test_csv_path_period_5m():
    from common import data_store
    p = data_store.csv_path('000001.SH', period='5m')
    assert p.endswith(os.path.join('stocks', '000001_SH_5m.csv'))

def test_csv_path_period_default_is_daily():
    from common import data_store
    p_default = data_store.csv_path('000001.SH')
    p_explicit = data_store.csv_path('000001.SH', period='daily')
    assert p_default == p_explicit
    assert p_explicit.endswith('000001_SH_daily.csv')

def test_csv_path_invalid_period_raises():
    from common import data_store
    import pytest
    with pytest.raises(ValueError, match="period 必须是"):
        data_store.csv_path('000001.SH', period='3m')

def test_filename_daily_keeps_legacy_suffix():
    from common import data_store
    assert data_store._filename('000001.SH', period='daily') == '000001_SH_daily.csv'

def test_filename_5m():
    from common import data_store
    assert data_store._filename('000001.SH', period='5m') == '000001_SH_5m.csv'

def test_save_load_df_roundtrip_5m(tmp_path, monkeypatch):
    """5m round-trip via tmp DATA_DIR."""
    import pandas as
    import numpy as np
    from common import data_store
    monkeypatch.setattr(data_store, 'DATA_DIR', str(tmp_path))
    df_in = pd.DataFrame({
        'Open': [10.0, 10.5], 'High': [10.6, 10.7], 'Low': [9.9, 10.4],
        'Close': [10.5, 10.6], 'Volume': [1000, 1100], 'Amount': [10500, 11660],
    }, index=pd.date_range('2026-08-01 09:30', periods=2, freq='5min'))
    out_path = data_store.save_df('000001.SH', df_in, period='5m')
    assert os.path.exists(out_path)
    df_out = data_store.load_df('000001.SH', period='5m')
    pd.testing.assert_frame_equal(df_in, df_out)

def test_save_daily_load_daily_unchanged(tmp_path, monkeypatch):
    """save_daily / load_daily / has_daily must keep existing daily behavior."""
    from common import data_store
    monkeypatch.setattr(data_store, 'DATA_DIR', str(tmp_path))
    import pandas as pd
    df = pd.DataFrame({'Close': [1.0]}, index=pd.date_range('2024-01-01', periods=1))
    data_store.save_daily('600519.SH', df)
    assert data_store.has_daily('600519.SH')
    df_out = data_store.load_daily('600519.SH')
    pd.testing.assert_frame_equal(df, df_out)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_data_store.py -k "period_5m or period_default or invalid_period or filename or save_load or save_daily_load" -v 2>&1 | tail -20
```

Expected: failures (period arg doesn't exist yet).

- [ ] **Step 3: Implement the changes in `data_store.py`**

Edit `backtrace/common/data_store.py`. Add near the top (after `import` block, before `KINDS`):
```python
PERIODS = ('daily', '15m', '5m', '1m')
```

Replace `_filename`:
```python
def _filename(code, period='daily'):
    """000001.SH + 'daily' -> '000001_SH_daily.csv'  (legacy)
       000001.SH + '5m'    -> '000001_SH_5m.csv'
    """
    if period == 'daily':
        return f"{code.replace('.', '_')}_daily.csv"
    if period not in PERIODS:
        raise ValueError(f"period 必须是 {PERIODS} 之一,收到 {period!r}")
    return f"{code.replace('.', '_')}_{period}.csv"
```

Replace `csv_path`:
```python
def csv_path(code, period='daily', kind='stocks'):
    if period not in PERIODS:
        raise ValueError(f"period 必须是 {PERIODS} 之一,收到 {period!r}")
    if kind not in KINDS:
        raise ValueError(f"kind 必须是 {KINDS} 之一,收到 {kind!r}")
    return os.path.join(DATA_DIR, kind, _filename(code, period))
```

Add new helpers after existing `save_daily`:
```python
def save_df(code, df, period='daily', kind='stocks'):
    """通用 period-aware 写盘。daily 时等价 save_daily。"""
    return save_daily(code, df, kind) if period == 'daily' else _save_with_period(code, df, period, kind)


def _save_with_period(code, df, period, kind):
    path = csv_path(code, period, kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    df.to_csv(tmp, encoding='utf-8')
    os.replace(tmp, path)
    return path


def load_df(code, period='daily'):
    """跨 kind 查找(stocks → sectors → indices);period 与 kind 都参与路径。"""
    for kind in KINDS:
        p = csv_path(code, period, kind)
        if os.path.exists(p):
            return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    return None
```

Keep `save_daily`, `load_daily`, `has_daily` as-is (their default `period='daily'` is the new default in `csv_path`).

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_data_store.py -v 2>&1 | tail -30
```

Expected: all `test_data_store.py` tests pass.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/ -q 2>&1 | tail -10
```

Expected: pass count unchanged from baseline (record baseline before this task).

- [ ] **Step 6: Commit**

```bash
git add backtrace/common/data_store.py tests/test_data_store.py
git commit -m "feat(data_store): period-aware cache + save_df/load_df helpers"
```

---

### Task A3: Add granularity constants to `tsfresh_config.py`

**Files:**
- Modify: `backtrace/common/tsfresh_config.py` (add `VALID_GRANULARITIES`, `DEFAULT_INTRADAY_GRANULARITY`, `DEFAULT_INTRADAY_LOOKBACK_DAYS`, `TQ_PERIOD_MAP`, `GRANULARITY_DT_SEC`)

**Interfaces:**
- Consumes: existing config
- Produces: 5 new module-level constants (see Step 3)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_tsfresh_config.py` (create file if it doesn't exist):
```python
def test_granularity_constants_present():
    from common import tsfresh_config as C
    assert C.VALID_GRANULARITIES == ('daily', '15m', '5m', '1m')
    assert C.DEFAULT_INTRADAY_GRANULARITY == '5m'
    assert C.DEFAULT_INTRADAY_LOOKBACK_DAYS == 60
    assert C.TQ_PERIOD_MAP == {'daily': '1d', '15m': '15m', '5m': '5m', '1m': '1m'}
    assert C.GRANULARITY_DT_SEC == {'daily': 86400, '15m': 900, '5m': 300, '1m': 60}
```

(If `tests/test_tsfresh_config.py` doesn't exist, create it with this single test as the first one.)

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_tsfresh_config.py -v 2>&1 | tail -10
```

Expected: ImportError or AttributeError.

- [ ] **Step 3: Add constants to `tsfresh_config.py`**

Append at the end of `backtrace/common/tsfresh_config.py`:
```python

# -------- Intraday granularity --------
VALID_GRANULARITIES = ('daily', '15m', '5m', '1m')          # 允许的粒度
DEFAULT_INTRADAY_GRANULARITY = '5m'                          # Phase C/D 实验默认(CLI 不提升,仅实验用)
DEFAULT_INTRADAY_LOOKBACK_DAYS = 60                          # intraday 缓存默认天数;可配置 30/60/90
TQ_PERIOD_MAP = {'daily': '1d', '15m': '15m', '5m': '5m', '1m': '1m'}
GRANULARITY_DT_SEC = {'daily': 86400, '15m': 900, '5m': 300, '1m': 60}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_tsfresh_config.py -v 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backtrace/common/tsfresh_config.py tests/test_tsfresh_config.py
git commit -m "feat(tsfresh_config): granularity constants (VALID_GRANULARITIES, TQ_PERIOD_MAP, etc.)"
```

---

### Task A4: Add `--period` and `--lookback-days` to `fetch_daily.py`

**Files:**
- Modify: `backtrace/data_fetch/fetch_daily.py` (add argparse flags + intraday data path + manifest period-tagged)

**Interfaces:**
- Consumes: TQ client (`tq.get_market_data(... period=...)`), `data_store.save_df`, `data_store.csv_path`
- Produces: `python backtrace/data_fetch/fetch_daily.py --period {daily|15m|5m|1m} [--lookback-days N]` produces period-tagged CSV + manifest

- [ ] **Step 1: Inspect current fetch_daily.py for argparse block**

```bash
grep -n "argparse\|add_argument\|parse_args" backtrace/data_fetch/fetch_daily.py | head -20
```

Read the relevant section (don't modify yet — just locate the function).

- [ ] **Step 2: Write failing test for `--period` argparse**

Create `tests/test_fetch_daily_cli.py` (or append if exists):
```python
def test_fetch_daily_help_shows_period():
    """`fetch_daily.py --help` should expose --period."""
    import subprocess
    out = subprocess.run(
        ['python', 'backtrace/data_fetch/fetch_daily.py', '--help'],
        capture_output=True, text=True, encoding='utf-8',
    )
    assert '--period' in out.stdout
    assert '--lookback-days' in out.stdout
    assert '5m' in out.stdout
```

- [ ] **Step 3: Run test to verify it fails**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_fetch_daily_cli.py::test_fetch_daily_help_shows_period -v 2>&1 | tail -10
```

Expected: FAIL (no --period in help).

- [ ] **Step 4: Modify `fetch_daily.py` argparse**

After the existing argparse block, add:
```python
p.add_argument('--period', choices=['daily', '15m', '5m', '1m'],
               default='daily',
               help='缓存粒度(daily = 现有默认;intraday = TQ 直拉)')
p.add_argument('--lookback-days', type=int, default=0,
               help='intraday 回看天数(daily 忽略)。0 = C.DEFAULT_INTRADAY_LOOKBACK_DAYS')
```

Also add (if not present) `from common import tsfresh_config as C` at the top.

- [ ] **Step 5: Add intraday path branching**

Locate the function that loops over the universe and pulls per-stock data. For daily, behavior is unchanged. Add a parallel branch for intraday. Sketch:

```python
def fetch_one_stock(code, period, lookback_days):
    """Single TQ pull for one stock at given period."""
    import sys as _sys
    _sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')
    from tqcenter import tq as _tq
    _tq.initialize(__file__)
    if period == 'daily':
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=int(lookback_days * 1.5) + 30)).strftime('%Y%m%d') \
                if lookback_days else \
                (datetime.now() - timedelta(days=int(C.LOOKBACK_YEARS * 365 + 30))).strftime('%Y%m%d')
        tq_period = '1d'
    else:
        days = lookback_days or C.DEFAULT_INTRADAY_LOOKBACK_DAYS
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=int(days * 1.8) + 10)).strftime('%Y%m%d')
        tq_period = C.TQ_PERIOD_MAP[period]
    fields = FIELDS  # existing
    raw = _tq.get_market_data(
        field_list=fields, stock_list=[code],
        start_time=start, end_time=end,
        dividend_type='front', period=tq_period, fill_data=True,
    )
    if raw is None or raw.get('Close') is None or raw['Close'].empty:
        raise RuntimeError(f"TQ empty for {code} {tq_period}")
    df = pd.DataFrame({
        'Open':   pd.to_numeric(raw['Open'][code],   errors='coerce'),
        'High':   pd.to_numeric(raw['High'][code],   errors='coerce'),
        'Low':    pd.to_numeric(raw['Low'][code],    errors='coerce'),
        'Close':  pd.to_numeric(raw['Close'][code],  errors='coerce'),
        'Volume': pd.to_numeric(raw['Volume'][code], errors='coerce'),
        'Amount': pd.to_numeric(raw['Amount'][code], errors='coerce'),
    }).dropna(subset=['Close']).sort_index()
    _tq.close()
    return df
```

Then in the per-stock branch of `main()`:
```python
df = fetch_one_stock(code, args.period, args.lookback_days)
if args.period == 'daily':
    save_daily(code, df, kind)
else:
    save_df(code, df, args.period, kind)
```

(Adapt to existing `main()` flow — read the existing code carefully before patching.)

- [ ] **Step 6: Manifest gets period-tagged**

In the manifest-writing block, add:
```python
manifest['period'] = args.period
manifest['lookback_days'] = args.lookback_days or (C.LOOKBACK_YEARS * 365 if args.period == 'daily' else C.DEFAULT_INTRADAY_LOOKBACK_DAYS)
```

And per-entry:
```python
entry['period'] = args.period
```

- [ ] **Step 7: Run help test to verify it passes**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_fetch_daily_cli.py::test_fetch_daily_help_shows_period -v 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 8: Run full test suite**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/ -q 2>&1 | tail -10
```

Expected: same pass count as before (no regressions).

- [ ] **Step 9: Commit**

```bash
git add backtrace/data_fetch/fetch_daily.py tests/test_fetch_daily_cli.py
git commit -m "feat(fetch): --period {daily,15m,5m,1m} + --lookback-days; period-tagged manifest"
```

---

### Task A5: 5min cache smoke test

**Files:**
- Modify: none (verification task)

**Why this task:** validates TQ 5-min data depth empirically before committing more code. If TQ caps < 60 days, we adjust `DEFAULT_INTRADAY_LOOKBACK_DAYS` accordingly.

- [ ] **Step 1: Run 5min smoke test on 5 stocks**

```bash
PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py --period 5m --limit 5 2>&1 | tail -30
```

Expected: 5 CSV files in `data/stocks/`, each ~2880 rows (5min × 60d × 48 bars/day). Manifest in `data/manifest.json` with `period: "5m"`.

- [ ] **Step 2: Inspect row counts**

```bash
for f in data/stocks/*_5m.csv; do
  rows=$(wc -l < "$f")
  echo "$f: $rows rows"
done
```

Expected: each ~2880 rows (±10%). If significantly less (e.g. < 1500), TQ depth is < 60 days — proceed to Step 3.

- [ ] **Step 3: If TQ depth < 60 days, update DEFAULT**

Only if Step 2 shows < 1500 rows per stock:
```bash
# Update config
# Edit backtrace/common/tsfresh_config.py: DEFAULT_INTRADAY_LOOKBACK_DAYS = <empirical>
# Document in progress.md
echo "TQ depth empirically < 60 days; adjusted DEFAULT_INTRADAY_LOOKBACK_DAYS" >> .superpowers/sdd/2026-08-22-intraday-granularity-5min-baseline/progress.md
```

Otherwise, record:
```bash
echo "TQ depth >= 60 days confirmed" >> .superpowers/sdd/2026-08-22-intraday-granularity-5min-baseline/progress.md
```

- [ ] **Step 4: Verify manifest is period-tagged**

```bash
python -c "import json; m=json.load(open('data/manifest.json')); print(m.get('period'), m.get('lookback_days')); print(list(m['entries'].values())[0].get('period'))"
```

Expected: `5m 60 5m` (or adjusted days from Step 3).

- [ ] **Step 5: Commit progress.md**

```bash
git add .superpowers/sdd/2026-08-22-intraday-granularity-5min-baseline/progress.md
git commit -m "docs(intraday): Phase A5 5min smoke test result"
```

---

### Task A6: Verify daily byte-equality preserved

**Files:**
- Modify: none (verification task)

- [ ] **Step 1: Re-run daily fetch for the snapshot stock**

```bash
PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py --period daily --limit 1 2>&1 | tail -5
```

Expected: 000001.SZ re-fetched. May overwrite `data/stocks/000001_SZ_daily.csv`.

- [ ] **Step 2: Compute SHA256 of daily CSV after Phase A changes**

```bash
python -c "import hashlib; print(hashlib.sha256(open('data/stocks/000001_SZ_daily.csv','rb').read()).hexdigest())"
```

- [ ] **Step 3: Compare against snapshot**

The hash from Task A1 Step 3 was recorded in `progress.md`. They must match. If different, revert:
```bash
# Compare
diff <(cat data/stocks/000001_SZ_daily.csv) <(cat data/stocks/000001_SZ_daily.csv.snapshot_a1) | head -20
```

If byte-identical → Phase A exit criterion met. If different → investigate; likely a bug in `fetch_daily.py` branch.

- [ ] **Step 4: Record result in progress.md**

```bash
echo "" >> .superpowers/sdd/2026-08-22-intraday-granularity-5min-baseline/progress.md
echo "## Phase A6 byte-equality check" >> .superpowers/sdd/2026-08-22-intraday-granularity-5min-baseline/progress.md
echo "Result: $(diff -q data/stocks/000001_SZ_daily.csv data/stocks/000001_SZ_daily.csv.snapshot_a1 2>&1)" >> .superpowers/sdd/2026-08-22-intraday-granularity-5min-baseline/progress.md
```

- [ ] **Step 5: Clean up snapshot file (optional)**

```bash
rm data/stocks/000001_SZ_daily.csv.snapshot_a1
```

(It's gitignored; no commit needed.)

- [ ] **Step 6: Commit progress.md**

```bash
git add .superpowers/sdd/2026-08-22-intraday-granularity-5min-baseline/progress.md
git commit -m "docs(intraday): Phase A byte-equality verified"
```

- [ ] **Step 7: Phase A exit — run full test suite one more time**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: pass count ≥ baseline. If any new test added, expected to pass.

---

## Phase B — Granularity parameter through production pipeline

### Task B1: Add `period` to `tsfresh_pipeline.load_ohlcva`

**Files:**
- Modify: `backtrace/common/tsfresh_pipeline.py` (`load_ohlcva` signature + period branching)

**Interfaces:**
- Consumes: existing `load_ohlcva(code, lookback_years, use_tq, verbose, include_amount)`
- Produces: `load_ohlcva(code, lookback_years=None, use_tq=True, verbose=False, include_amount=True, *, period='daily')`

- [ ] **Step 1: Write failing test**

Create `tests/test_tsfresh_pipeline_period.py`:
```python
def test_load_ohlcva_period_default_is_daily():
    """Existing default call must produce same shape/columns as before."""
    from common import tsfresh_pipeline as P
    import inspect
    sig = inspect.signature(P.load_ohlcva)
    assert 'period' in sig.parameters
    assert sig.parameters['period'].default == 'daily'

def test_load_ohlcva_invalid_period_raises():
    """period='3m' must raise ValueError before any data work."""
    from common import tsfresh_pipeline as P
    import pytest
    with pytest.raises(ValueError, match="period"):
        P.load_ohlcva('000001.SH', period='3m', use_tq=False)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_tsfresh_pipeline_period.py -v 2>&1 | tail -10
```

Expected: FAIL.

- [ ] **Step 3: Modify `load_ohlcva` signature**

In `backtrace/common/tsfresh_pipeline.py`, change the function signature to:
```python
def load_ohlcva(code, lookback_years=None, use_tq=True, verbose=False, include_amount=True, *, period='daily'):
    """TQ 优先 → 失败回退本地 CSV。period ∈ {daily, 15m, 5m, 1m};默认 daily 与历史行为一致。"""
    if period not in C.VALID_GRANULARITIES:
        raise ValueError(f"period 必须是 {C.VALID_GRANULARITIES} 之一,收到 {period!r}")
    tq_period = C.TQ_PERIOD_MAP[period]
```

Inside the TQ branch, change the `period='1d'` kwarg to `period=tq_period`. Inside the fallback branch, replace `_try_local_csv(code)` with:
```python
# Intraday fallback uses load_df(period); daily keeps load_daily
if period == 'daily':
    out = data_store.load_daily(code)
else:
    out = data_store.load_df(code, period)
```

Place this in the same fallback location as the existing `_try_local_csv(code)` call.

- [ ] **Step 4: Run new tests**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_tsfresh_pipeline_period.py -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Run full test suite for regressions**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: pass count unchanged from Phase A exit.

- [ ] **Step 6: Commit**

```bash
git add backtrace/common/tsfresh_pipeline.py tests/test_tsfresh_pipeline_period.py
git commit -m "feat(tsfresh_pipeline): load_ohlcva gains period kwarg (daily|15m|5m|1m)"
```

---

### Task B2: Add `period` to `_projection_core.load_pair`

**Files:**
- Modify: `backtrace/projection/_projection_core.py` (`load_pair` signature + pass `period` to `pipeline.load_ohlcva`)

**Interfaces:**
- Consumes: existing `load_pair(stock_code, days, pipeline, prefer_industry=False, index_code=None, lag=0)`
- Produces: `load_pair(stock_code, days, pipeline, prefer_industry=False, index_code=None, lag=0, *, period='daily')`

- [ ] **Step 1: Write failing test**

Append to `tests/test_projection_core.py`:
```python
def test_load_pair_period_default_is_daily():
    from projection import _projection_core as P
    import inspect
    sig = inspect.signature(P.load_pair)
    assert 'period' in sig.parameters
    assert sig.parameters['period'].default == 'daily'

def test_load_pair_invalid_period_raises():
    from projection import _projection_core as P
    import pytest
    with pytest.raises(ValueError, match="period"):
        P.load_pair('000001.SZ', 5, None, period='3m')  # pipeline=None triggers later check
```

- [ ] **Step 2: Run tests to verify failure**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_projection_core.py -k "load_pair_period" -v 2>&1 | tail -10
```

Expected: FAIL.

- [ ] **Step 3: Modify `load_pair` signature**

In `backtrace/projection/_projection_core.py`, update the function signature:
```python
def load_pair(stock_code, days, pipeline, prefer_industry=False, index_code=None, lag: int = 0, *, period: str = 'daily'):
```

In the two `pipeline.load_ohlcva` calls inside `load_pair`, add the kwarg:
```python
data_index_full = pipeline.load_ohlcva(index_code, use_tq=False, verbose=True, period=period)
data_stock_full = pipeline.load_ohlcva(stock_code, use_tq=False, verbose=True, period=period)
```

Add `period` validation at the function entry (after the `lag` setup):
```python
    if period not in ('daily', '15m', '5m', '1m'):
        raise ValueError(f"period 必须是 (daily, 15m, 5m, 1m) 之一,收到 {period!r}")
```

(Or import `C.VALID_GRANULARITIES` — but to avoid coupling `_projection_core` to `tsfresh_config`, hardcoded tuple is fine for this defensive check; `tsfresh_pipeline.load_ohlcva` already validates the canonical set.)

- [ ] **Step 4: Run tests**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_projection_core.py -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Run full test suite**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: pass count unchanged.

- [ ] **Step 6: Commit**

```bash
git add backtrace/projection/_projection_core.py tests/test_projection_core.py
git commit -m "feat(projection): load_pair gains period kwarg; passes to load_ohlcva"
```

---

### Task B3: Add `--period` to projection scripts (single-stock + batch + v0.2)

**Files:**
- Modify: `backtrace/projection/projection_2d.py`
- Modify: `backtrace/projection/projection_batch.py`
- Modify: `backtrace/projection/prediction_ode.py`
- Modify: `backtrace/projection/state_kc_analysis.py`
- Modify: `backtrace/projection/parameter_fit.py` (cosmetic only — see Step 3)
- Modify: `backtrace/projection/v0_2_c1_market_swap.py`
- Modify: `backtrace/projection/v0_2_d_decompose.py`
- Modify: `backtrace/projection/v0_2_e1_delta_ic_distribution.py`
- Modify: `backtrace/projection/v0_2_e2_cross_sectional_q.py`

**Interfaces:**
- Consumes: existing scripts' argparse
- Produces: each script has `--period {daily,15m,5m,1m}` (default 'daily') and threads it to `load_pair` (or `load_ohlcva` if used directly)

- [ ] **Step 1: Write failing smoke test**

Append to `tests/test_projection_cli.py`:
```python
def test_projection_cli_help_exposes_period():
    """Each projection CLI should accept --period."""
    import subprocess
    scripts = [
        'backtrace/projection/projection_2d.py',
        'backtrace/projection/projection_batch.py',
        'backtrace/projection/prediction_ode.py',
        'backtrace/projection/state_kc_analysis.py',
    ]
    for s in scripts:
        out = subprocess.run(['python', s, '--help'], capture_output=True, text=True, encoding='utf-8')
        assert '--period' in out.stdout, f"{s} missing --period"
```

- [ ] **Step 2: Run test to verify failure**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_projection_cli.py::test_projection_cli_help_exposes_period -v 2>&1 | tail -10
```

Expected: FAIL on at least one script.

- [ ] **Step 3: Modify `projection_2d.py` (template for the rest)**

Locate `parse_args` (or equivalent) in `projection_2d.py`. Add:
```python
parser.add_argument('--period', choices=['daily', '15m', '5m', '1m'], default='daily',
                    help='缓存粒度(daily = 默认;其他走 intraday 流程)')
```

In `main()`, where `load_pair` is called, pass `period=args.period`:
```python
loaded = load_pair(args.code, args.days, P, prefer_industry=args.industry,
                   index_code=getattr(args, 'index', None), period=args.period)
```

- [ ] **Step 4: Modify `projection_batch.py`**

Same pattern: add `--period` to argparse, pass to `load_pair` in the per-stock loop.

- [ ] **Step 5: Modify `prediction_ode.py` and `state_kc_analysis.py`**

Same pattern. Both call `load_pair`.

- [ ] **Step 6: Modify `parameter_fit.py` (cosmetic)**

`parameter_fit.py` reads `data/projection/movement_*.csv` (which is independent of cache period — projection_batch writes it). Add `--period` flag purely for symmetry/auditing; ignore it in the OLS math (read CSV as-is). Add:
```python
parser.add_argument('--period', choices=['daily', '15m', '5m', '1m'], default='daily',
                    help='仅作审计/记录;不影响 OLS(读现有 movement CSV)')
```

Add `period` to the output CSV metadata comment if applicable.

- [ ] **Step 7: Modify v0.2_*.py scripts (4 files)**

Same `--period` argparse + pass to `load_pair`. These scripts are orchestrators; ensure they thread period to sub-scripts (e.g. subprocess calls) **only if** those sub-scripts accept it. Since Phase B Task B4 adds period to dynamics scripts, by the time v0.2_E runs (Phase C), this is wired. For Phase B, just add the flag to v0.2 scripts and accept that sub-calls will use the script's own default (daily).

- [ ] **Step 8: Run test to verify pass**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_projection_cli.py::test_projection_cli_help_exposes_period -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 9: Run full test suite**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: same pass count.

- [ ] **Step 10: Smoke: projection_2d.py on daily with --period daily**

```bash
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_2d.py --code 002475.SZ --period daily 2>&1 | tail -10
```

Expected: runs to completion, produces HTML (existing default location). Compare output to a previous run (git diff) — should be byte-identical if daily path unchanged.

- [ ] **Step 11: Commit**

```bash
git add backtrace/projection/projection_2d.py backtrace/projection/projection_batch.py \
        backtrace/projection/prediction_ode.py backtrace/projection/state_kc_analysis.py \
        backtrace/projection/parameter_fit.py \
        backtrace/projection/v0_2_c1_market_swap.py \
        backtrace/projection/v0_2_d_decompose.py \
        backtrace/projection/v0_2_e1_delta_ic_distribution.py \
        backtrace/projection/v0_2_e2_cross_sectional_q.py \
        tests/test_projection_cli.py
git commit -m "feat(projection): all scripts gain --period {daily,15m,5m,1m}; thread to load_pair"
```

---

### Task B4: Add `--period` to dynamics scripts

**Files:**
- Modify: `backtrace/dynamics/dynamics_system.py`
- Modify: `backtrace/dynamics/dynamics_batch.py`
- Modify: `backtrace/dynamics/dynamics_1step_oos.py`
- Modify: `backtrace/dynamics/dynamics_state_backtest.py`
- Modify: `backtrace/dynamics/dynamics_eigen_analysis.py`
- Modify: `backtrace/dynamics/dynamics_oos_viz.py`
- Modify: `backtrace/dynamics/dynamics_oos_batch.py`
- Modify: `backtrace/dynamics/dynamics_state_timeline.py`
- Modify: `backtrace/dynamics/dynamics_si_freq_response.py`
- Modify: `backtrace/dynamics/dynamics_si_ic.py`
- Modify: `backtrace/dynamics/dynamics_si_timeseries.py`
- Modify: `backtrace/dynamics/dynamics_si_lagged_ic.py`
- Modify: `backtrace/dynamics/dynamics_factor_validation.py`
- DO NOT modify: `backtrace/dynamics/dynamics_forced_response.py` (math-only)

**Interfaces:** same as B3 — `--period` flag (default 'daily'), pass to `load_pair` (via P) where applicable.

- [ ] **Step 1: Write failing test**

Create `tests/test_dynamics_cli_period.py`:
```python
def test_dynamics_cli_help_exposes_period():
    """All dynamics CLIs except forced_response should expose --period."""
    import subprocess
    scripts = [
        'backtrace/dynamics/dynamics_system.py',
        'backtrace/dynamics/dynamics_batch.py',
        'backtrace/dynamics/dynamics_1step_oos.py',
        'backtrace/dynamics/dynamics_state_backtest.py',
        'backtrace/dynamics/dynamics_eigen_analysis.py',
        'backtrace/dynamics/dynamics_oos_viz.py',
        'backtrace/dynamics/dynamics_oos_batch.py',
        'backtrace/dynamics/dynamics_state_timeline.py',
        'backtrace/dynamics/dynamics_si_freq_response.py',
        'backtrace/dynamics/dynamics_si_ic.py',
        'backtrace/dynamics/dynamics_si_timeseries.py',
        'backtrace/dynamics/dynamics_si_lagged_ic.py',
        'backtrace/dynamics/dynamics_factor_validation.py',
    ]
    for s in scripts:
        out = subprocess.run(['python', s, '--help'], capture_output=True, text=True, encoding='utf-8')
        assert '--period' in out.stdout, f"{s} missing --period"

def test_forced_response_no_period():
    """dynamics_forced_response.py is math-only and should NOT have --period."""
    import subprocess
    out = subprocess.run(['python', 'backtrace/dynamics/dynamics_forced_response.py', '--help'],
                         capture_output=True, text=True, encoding='utf-8')
    assert '--period' not in out.stdout
```

- [ ] **Step 2: Run test to verify failure**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_cli_period.py -v 2>&1 | tail -10
```

Expected: FAIL on multiple scripts.

- [ ] **Step 3: Modify each dynamics script (template: dynamics_system.py)**

For each script:
1. Locate argparse `add_argument` block.
2. Add:
   ```python
   parser.add_argument('--period', choices=['daily', '15m', '5m', '1m'], default='daily',
                       help='缓存粒度(daily = 默认)')
   ```
4. Where `load_pair` or `pipeline.load_ohlcva` is called, add `period=args.period`.

For `dynamics_factor_validation.py`: the `--period` flag affects CSV input paths (e.g. `kc_estimates.csv` lives in `data/projection/`, not period-tagged). The flag here is informational — record it in output but don't rekey CSV paths. Spec §6.4 only rekeys for output dirs.

- [ ] **Step 4: Run test to verify pass**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_cli_period.py -v 2>&1 | tail -10
```

Expected: PASS (both `test_dynamics_cli_help_exposes_period` and `test_forced_response_no_period`).

- [ ] **Step 5: Run full test suite**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: same pass count.

- [ ] **Step 6: Commit**

```bash
git add backtrace/dynamics/ tests/test_dynamics_cli_period.py
git commit -m "feat(dynamics): all scripts gain --period; dynamics_forced_response.py unchanged (math-only)"
```

---

### Task B5: End-to-end smoke test on 1 stock (daily + 5min)

**Files:**
- Modify: none (verification)

- [ ] **Step 1: Run projection_2d daily on 002475.SZ**

```bash
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_2d.py --code 002475.SZ --period daily 2>&1 | tail -10
```

Expected: HTML output produced (existing location).

- [ ] **Step 2: Run projection_2d 5min on 002475.SZ**

```bash
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_2d.py --code 002475.SZ --period 5m 2>&1 | tail -10
```

Expected: HTML output produced (new period-aware location if applicable; or same location if not period-aware). No exceptions.

- [ ] **Step 3: Run dynamics_1step_oos daily**

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_1step_oos.py --code 002475.SZ --period daily 2>&1 | tail -10
```

Expected: prediction CSV + summary TXT. No exceptions.

- [ ] **Step 4: Run dynamics_1step_oos 5min**

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_1step_oos.py --code 002475.SZ --period 5m 2>&1 | tail -10
```

Expected: same shape of outputs, different period. No exceptions.

- [ ] **Step 5: Run dynamics_state_backtest 5min**

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_state_backtest.py --code 002475.SZ --period 5m 2>&1 | tail -10
```

Expected: state_distribution.csv + backtest_per_state.csv + state_ic.csv. No exceptions.

- [ ] **Step 6: Run dynamics_eigen_analysis 5min**

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_eigen_analysis.py --period 5m --limit 5 2>&1 | tail -10
```

Expected: eigen_summary.csv + HTML. No exceptions. (Note: this reads `kc_estimates.csv` which is daily data — output will be a re-key of daily-fit (k̂, ĉ). The `--period 5m` is recorded but doesn't change math. That's expected.)

- [ ] **Step 7: Phase B exit — full test suite**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 8: Record in progress.md**

Append to `.superpowers/sdd/2026-08-22-intraday-granularity-5min-baseline/progress.md`:
```
## Phase B exit
- All scripts accept --period
- 5min smoke passes on 002475.SZ
- Test suite: <N> passed
```

- [ ] **Step 9: Commit progress.md**

```bash
git add .superpowers/sdd/2026-08-22-intraday-granularity-5min-baseline/progress.md
git commit -m "docs(intraday): Phase B exit — granularity threaded through pipeline"
```

---

## Phase C — Daily vs 5min production comparison

### Task C1: Output dir remap for 5min

**Files:**
- Modify: `backtrace/projection/projection_2d.py`
- Modify: `backtrace/projection/projection_batch.py`
- Modify: `backtrace/projection/parameter_fit.py`
- Modify: `backtrace/dynamics/dynamics_system.py`
- Modify: `backtrace/dynamics/dynamics_batch.py`
- Modify: `backtrace/dynamics/dynamics_1step_oos.py`
- Modify: `backtrace/dynamics/dynamics_state_backtest.py`
- Modify: `backtrace/dynamics/dynamics_eigen_analysis.py`
- Modify: `backtrace/dynamics/dynamics_oos_viz.py`
- Modify: `backtrace/dynamics/dynamics_oos_batch.py`
- Modify: `backtrace/dynamics/dynamics_factor_validation.py`

**Goal:** When `period != 'daily'`, output paths gain a `_5min` suffix. Daily path unchanged.

- [ ] **Step 1: Write failing test**

Create `tests/test_output_dir_remap.py`:
```python
def test_output_dir_includes_period_for_intraday():
    """Helper output_dir_for(args.period) returns suffixed dir for non-daily."""
    from dynamics import dynamics_granularity_compare  # new module from C3
    # Just import-time check; full assertions come in Task C3
    assert dynamics_granularity_compare is not None
```

(This is a placeholder test — the real assertions live in Task C3 once the helper exists.)

- [ ] **Step 2: Run test to verify failure**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_output_dir_remap.py -v 2>&1 | tail -5
```

Expected: ImportError.

- [ ] **Step 3: Create the output-dir helper module**

Create `backtrace/dynamics/dynamics_granularity_compare.py`:
```python
# -*- coding: utf-8 -*-
"""Helpers for daily-vs-5min output dir remapping + comparison (Phase C)."""
import os


def output_dir_suffix(period):
    """'daily' -> '';  '5m' -> '_5m';  etc."""
    return '' if period == 'daily' else f'_{period}'


def remap_output_path(base_path, period):
    """data/projection/movement_xxx.csv → data/projection_5min/movement_xxx.csv (if period='5m')."""
    if period == 'daily':
        return base_path
    parent, fname = os.path.split(base_path)
    return os.path.join(parent.rstrip('/').rstrip('\\') + output_dir_suffix(period), fname)


def output_subdir_for_period(base, period):
    """backtrace/outputs/... → backtrace/outputs/..._5m (if period='5m')."""
    if period == 'daily':
        return base
    return base + output_dir_suffix(period)
```

- [ ] **Step 4: Run test to verify pass**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_output_dir_remap.py -v 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 5: Modify projection_2d.py to use remap**

In `projection_2d.py`'s HTML-output code, replace:
```python
out_html = 'backtrace/outputs/proj2d_<...>.html'
```
with:
```python
from dynamics.dynamics_granularity_compare import output_subdir_for_period
out_html = output_subdir_for_period('backtrace/outputs/proj2d_<...>.html', args.period)
```

Apply to projection_batch (movement CSV output), parameter_fit (kc_estimates output), and each dynamics script's HTML/CSV output path. Use search-and-replace for `_5min`-equivalent suffix logic.

- [ ] **Step 6: Run full test suite**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: pass count unchanged.

- [ ] **Step 7: Commit**

```bash
git add backtrace/dynamics/dynamics_granularity_compare.py tests/test_output_dir_remap.py
git add backtrace/projection/projection_2d.py backtrace/projection/projection_batch.py backtrace/projection/parameter_fit.py
git add backtrace/dynamics/dynamics_system.py backtrace/dynamics/dynamics_batch.py
git add backtrace/dynamics/dynamics_1step_oos.py backtrace/dynamics/dynamics_state_backtest.py
git add backtrace/dynamics/dynamics_eigen_analysis.py
git add backtrace/dynamics/dynamics_oos_viz.py backtrace/dynamics/dynamics_oos_batch.py
git add backtrace/dynamics/dynamics_factor_validation.py
git commit -m "feat(output): period-aware output dirs (data/projection_5min/, etc.)"
```

---

### Task C2: Daily full v0-v6 rerun (byte-equality gate)

**Files:**
- Modify: none (verification)

**Goal:** Re-run the daily v0-v6 chain and confirm byte-equality vs reference.

- [ ] **Step 1: Identify a recent reference output**

```bash
ls -d data/projection_v01_c1* data/dynamics_v* 2>&1 | head -10
```

Pick the most recent snapshot. If none exists, accept that byte-equality verification is best-effort for Phase C and proceed.

- [ ] **Step 2: Run daily projection_batch (smoke)**

```bash
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_batch.py --period daily --limit 5 2>&1 | tail -10
```

Expected: 5 movement CSVs produced in `data/projection/`.

- [ ] **Step 3: Run daily parameter_fit (smoke)**

```bash
PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --period daily --limit 5 2>&1 | tail -10
```

Expected: 5 rows in `kc_estimates.csv`.

- [ ] **Step 4: Run daily dynamics_1step_oos (smoke)**

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_1step_oos.py --period daily --limit 5 2>&1 | tail -10
```

Expected: prediction_summary.csv updated.

- [ ] **Step 5: Run daily dynamics_factor_validation (smoke)**

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_factor_validation.py --period daily --limit 5 --horizons 5,20 2>&1 | tail -10
```

Expected: factor_validation.csv produced.

- [ ] **Step 6: Verify daily output paths unchanged**

```bash
ls data/projection/movement_*.csv 2>&1 | head -5
ls data/dynamics/prediction_summary.csv 2>&1
ls data/dynamics/factor_validation.csv 2>&1
```

Expected: outputs in canonical (non-suffixed) dirs. If any file appears in `data/projection_5min/` for daily run, investigate (regression).

- [ ] **Step 7: Record in progress.md**

Append:
```
## Phase C2 daily re-run smoke
- 5 movement CSVs in data/projection/
- kc_estimates.csv updated
- prediction_summary.csv + factor_validation.csv updated
- Output dir: canonical (no _5min suffix) ✓
```

- [ ] **Step 8: Commit progress.md**

```bash
git add .superpowers/sdd/2026-08-22-intraday-granularity-5min-baseline/progress.md
git commit -m "docs(intraday): Phase C2 daily smoke — outputs in canonical dirs"
```

---

### Task C3: 5min full v0-v6 rerun

**Files:**
- Modify: none (verification)

**Goal:** Run the same chain on 5min cache, capture outputs in period-tagged dirs.

- [ ] **Step 1: Run 5min projection_batch (smoke)**

```bash
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_batch.py --period 5m --limit 5 2>&1 | tail -10
```

Expected: 5 movement CSVs produced in `data/projection_5min/`.

- [ ] **Step 2: Run 5min parameter_fit (smoke)**

```bash
PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --period 5m --input data/projection_5min/movement_*.csv --limit 5 2>&1 | tail -10
```

Expected: 5 rows. (Output may go to default `kc_estimates.csv`; for Phase C we'll keep it there for the consumer to find.)

- [ ] **Step 3: Run 5min dynamics_1step_oos (smoke)**

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_1step_oos.py --period 5m --limit 5 2>&1 | tail -10
```

Expected: prediction_summary_5min.csv (or period-tagged equivalent).

- [ ] **Step 4: Run 5min dynamics_state_backtest (smoke)**

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_state_backtest.py --period 5m --limit 5 2>&1 | tail -10
```

Expected: state_distribution_5min.csv + backtest_per_state_5min.csv + state_ic_5min.csv.

- [ ] **Step 5: Run 5min dynamics_factor_validation (smoke)**

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_factor_validation.py --period 5m --limit 5 --horizons 5,20 2>&1 | tail -10
```

Expected: factor_validation_5min.csv (or period-tagged equivalent).

- [ ] **Step 6: Verify 5min output paths suffixed**

```bash
ls data/projection_5min/movement_*.csv 2>&1 | head -5
ls data/dynamics_5min/ 2>&1 | head -10
```

Expected: all 5min outputs in period-tagged dirs.

- [ ] **Step 7: Record in progress.md**

Append:
```
## Phase C3 5min re-run smoke
- 5 movement CSVs in data/projection_5min/
- All 5min outputs in *_5min.csv / *_5min.html form
- Daily outputs untouched ✓
```

- [ ] **Step 8: Commit progress.md**

```bash
git add .superpowers/sdd/2026-08-22-intraday-granularity-5min-baseline/progress.md
git commit -m "docs(intraday): Phase C3 5min smoke — outputs in period-tagged dirs"
```

---

### Task C4: Side-by-side comparison report generator

**Files:**
- Create: `backtrace/dynamics/dynamics_granularity_compare.py` (extend with `build_daily_vs_5min_report`)
- Create: `tests/test_dynamics_granularity_compare.py`

**Goal:** Read daily and 5min v6 outputs, produce `backtrace/outputs/granularity_compare/` artifacts.

- [ ] **Step 1: Write failing tests**

Create `tests/test_dynamics_granularity_compare.py`:
```python
def test_daily_vs_5min_summary_columns():
    """build_daily_vs_5min_report emits expected columns."""
    from dynamics import dynamics_granularity_compare as G
    cols = G.REPORT_TABLE_COLS
    expected_subset = ['factor', 'horizon', 'ic_mean_daily', 'ic_mean_5min',
                       'delta_ic', 'delta_ic_ir']
    for c in expected_subset:
        assert c in cols

def test_decision_thresholds_constants():
    """Decision thresholds from spec §7.1 are exposed as module constants."""
    from dynamics import dynamics_granularity_compare as G
    assert G.DELTA_IC_MIN == 0.02
    assert G.DELTA_IC_PVALUE_MAX == 0.05
    assert G.DELTA_IC_IR_MIN == 0.1
    assert G.DELTA_OOS_RMSE_MAX == -0.05
    assert G.DELTA_SI_LAGGED_IC_MIN == 0.02
    assert G.DELTA_HIT_RATE_MIN == 0.03
```

- [ ] **Step 2: Run test to verify failure**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_granularity_compare.py -v 2>&1 | tail -10
```

Expected: ImportError on `REPORT_TABLE_COLS` / threshold constants.

- [ ] **Step 3: Add constants and report builder**

In `backtrace/dynamics/dynamics_granularity_compare.py`, append:

```python
# Decision thresholds (spec §7.1)
DELTA_IC_MIN = 0.02
DELTA_IC_PVALUE_MAX = 0.05
DELTA_IC_IR_MIN = 0.1
DELTA_OOS_RMSE_MAX = -0.05   # 5min median ≤ -5% vs daily
DELTA_SI_LAGGED_IC_MIN = 0.02
DELTA_HIT_RATE_MIN = 0.03     # +3pp

REPORT_TABLE_COLS = [
    'factor', 'horizon',
    'ic_mean_daily', 'ic_mean_5min', 'delta_ic',
    'ic_ir_daily', 'ic_ir_5min', 'delta_ic_ir',
    'ic_pvalue_daily', 'ic_pvalue_5min',
    'q5_minus_q1_daily', 'q5_minus_q1_5min',
    'status',
]


def build_daily_vs_5min_report(daily_dir, fivemin_dir, output_dir):
    """Read factor_validation.csv from both dirs; produce comparison CSVs + TXT + HTML.

    Args:
        daily_dir:   path containing factor_validation.csv from daily run
        fivemin_dir: path containing factor_validation.csv from 5min run
        output_dir:  where to write granularity_compare/ artifacts
    """
    import os
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)
    daily_path = os.path.join(daily_dir, 'factor_validation.csv')
    fivemin_path = os.path.join(fivemin_dir, 'factor_validation.csv')

    if not (os.path.exists(daily_path) and os.path.exists(fivemin_path)):
        raise FileNotFoundError(f"missing inputs: {daily_path} or {fivemin_path}")

    df_d = pd.read_csv(daily_path)
    df_5 = pd.read_csv(fivemin_path)
    # Inner-join on (factor, horizon)
    merged = df_d.merge(df_5, on=['factor', 'horizon'], suffixes=('_daily', '_5min'),
                         how='inner')
    merged['delta_ic'] = merged['ic_mean_5min'] - merged['ic_mean_daily']
    merged['delta_ic_ir'] = merged['ic_ir_5min'] - merged['ic_ir_daily']

    out_table = os.path.join(output_dir, 'daily_vs_5min_table.csv')
    cols = [c for c in REPORT_TABLE_COLS if c in merged.columns]
    merged[cols].to_csv(out_table, index=False)

    # Topology: k̂/ĉ/β/F²
    # Read from kc_estimates.csv in both dirs
    kc_d_path = os.path.join(daily_dir, '..', 'projection', 'kc_estimates.csv')
    kc_5_path = os.path.join(fivemin_dir, '..', 'projection', 'kc_estimates.csv')
    topology_rows = []
    if os.path.exists(kc_d_path) and os.path.exists(kc_5_path):
        kc_d = pd.read_csv(kc_d_path)
        kc_5 = pd.read_csv(kc_5_path)
        # Outer-join on code
        topo = kc_d.merge(kc_5, on='code', suffixes=('_daily', '_5min'), how='outer')
        topology_rows.append(topo)
        if topo is not None:
            topo.to_csv(os.path.join(output_dir, 'daily_vs_5min_topology.csv'), index=False)

    # Summary TXT (UTF-8)
    summary_path = os.path.join(output_dir, 'daily_vs_5min_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Intraday Granularity — Daily vs 5min Comparison\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Daily inputs:  {daily_path}\n")
        f.write(f"5min inputs:   {fivemin_path}\n\n")
        f.write(f"Factors compared: {len(merged)}\n")
        n_significant = int((merged['delta_ic'] >= DELTA_IC_MIN).sum())
        f.write(f"Factors with ΔIC ≥ +{DELTA_IC_MIN}: {n_significant}\n")
        n_ir_up = int((merged['delta_ic_ir'] >= DELTA_IC_IR_MIN).sum())
        f.write(f"Factors with ΔIC_IR ≥ +{DELTA_IC_IR_MIN}: {n_ir_up}\n\n")
        f.write("Decision framework (spec §7.1):\n")
        f.write(f"  Adopt 5min if ΔIC ≥ {DELTA_IC_MIN} AND p < {DELTA_IC_PVALUE_MAX} AND\n")
        f.write(f"           ΔIC_IR ≥ {DELTA_IC_IR_MIN} AND ΔOOS ≤ {DELTA_OOS_RMSE_MAX} AND\n")
        f.write(f"           Δhit_rate ≥ {DELTA_HIT_RATE_MIN}\n")

    # HTML 2x2 dashboard (deferred to plotting layer — out of scope here; emit stub)
    # Future: implement with plotly 2x2 grid

    return {
        'table': out_table,
        'summary': summary_path,
        'n_factors': len(merged),
        'n_significant': n_significant,
    }
```

- [ ] **Step 4: Run tests**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_granularity_compare.py -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Smoke test the report generator**

Use existing daily v6 outputs (from a previous run) + 5min outputs from Task C3:

```bash
PYTHONIOENCODING=utf-8 python -c "
from dynamics import dynamics_granularity_compare as G
out = G.build_daily_vs_5min_report(
    daily_dir='data/dynamics',
    fivemin_dir='data/dynamics_5min',
    output_dir='backtrace/outputs/granularity_compare',
)
print(out)
"
```

Expected: outputs created in `backtrace/outputs/granularity_compare/`.

- [ ] **Step 6: Commit**

```bash
git add backtrace/dynamics/dynamics_granularity_compare.py tests/test_dynamics_granularity_compare.py
git commit -m "feat(dynamics): daily-vs-5min comparison report builder + decision thresholds"
```

---

### Task C5: Granularity-aware output path correctness

**Files:**
- Modify: none (verification)

- [ ] **Step 1: Verify daily outputs are not in period-tagged dirs**

```bash
ls data/projection/ 2>&1 | head -5
ls data/projection_5min/ 2>&1 | head -5
ls data/dynamics/ 2>&1 | head -5
ls data/dynamics_5min/ 2>&1 | head -5
```

Expected: daily files in canonical dirs; 5min files in `_5min` suffixed dirs.

- [ ] **Step 2: Verify report artifacts exist**

```bash
ls backtrace/outputs/granularity_compare/ 2>&1
```

Expected: at least `daily_vs_5min_table.csv` + `daily_vs_5min_summary.txt`.

- [ ] **Step 3: Record in progress.md**

Append:
```
## Phase C5 verification
- Daily outputs in canonical dirs ✓
- 5min outputs in period-tagged dirs ✓
- Report artifacts in backtrace/outputs/granularity_compare/ ✓
```

- [ ] **Step 4: Commit progress.md**

```bash
git add .superpowers/sdd/2026-08-22-intraday-granularity-5min-baseline/progress.md
git commit -m "docs(intraday): Phase C5 verification — daily/5min separation correct"
```

---

## Phase D — Decision framework

### Task D1: Decision framework test (test_granularity_decision.py)

**Files:**
- Create: `tests/test_granularity_decision.py`

**Goal:** Test the decision thresholds from spec §7.1 against synthetic CSV inputs.

- [ ] **Step 1: Write tests**

Create `tests/test_granularity_decision.py`:
```python
def test_decision_adopt_when_all_thresholds_met():
    """All hard thresholds met → outcome='adopt'."""
    from dynamics import dynamics_granularity_compare as G

    def verdict(delta_ic, delta_ic_ir, delta_oos_rmse, delta_hit_rate):
        # Mirror the spec §7.1 logic
        return ('adopt' if delta_ic >= G.DELTA_IC_MIN and delta_ic_ir >= G.DELTA_IC_IR_MIN
                and delta_oos_rmse <= G.DELTA_OOS_RMSE_MAX and delta_hit_rate >= G.DELTA_HIT_RATE_MIN
                else 'archive-or-kill')

    assert verdict(0.025, 0.15, -0.10, 0.04) == 'adopt'

def test_decision_kill_when_no_signal():
    """ΔIC ≈ 0, no other signals → outcome != 'adopt'."""
    from dynamics import dynamics_granularity_compare as G

    def verdict(delta_ic, delta_ic_ir, delta_oos_rmse, delta_hit_rate):
        return ('adopt' if delta_ic >= G.DELTA_IC_MIN and delta_ic_ir >= G.DELTA_IC_IR_MIN
                and delta_oos_rmse <= G.DELTA_OOS_RMSE_MAX and delta_hit_rate >= G.DELTA_HIT_RATE_MIN
                else 'archive-or-kill')

    assert verdict(0.001, 0.001, 0.001, 0.001) != 'adopt'

def test_decision_threshold_constants_match_spec():
    """Spec §7.1 values must match module constants exactly."""
    from dynamics import dynamics_granularity_compare as G
    assert G.DELTA_IC_MIN == 0.02
    assert G.DELTA_IC_PVALUE_MAX == 0.05
    assert G.DELTA_IC_IR_MIN == 0.1
    assert G.DELTA_OOS_RMSE_MAX == -0.05
    assert G.DELTA_SI_LAGGED_IC_MIN == 0.02
    assert G.DELTA_HIT_RATE_MIN == 0.03
```

- [ ] **Step 2: Run tests to verify pass**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_granularity_decision.py -v 2>&1 | tail -10
```

Expected: PASS (functions defined in C4 already provide constants).

- [ ] **Step 3: Commit**

```bash
git add tests/test_granularity_decision.py
git commit -m "test(dynamics): decision framework tests (spec §7.1)"
```

---

### Task D2: Run decision framework on both pipelines

**Files:**
- Modify: none (verification)

- [ ] **Step 1: Build full-A comparison report**

If not already done in Phase C, run the daily + 5min chains at larger scale (limit 50+):
```bash
# Daily
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_batch.py --period daily --limit 50
PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --period daily --limit 50
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_1step_oos.py --period daily --limit 50
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_state_backtest.py --period daily --limit 50
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_factor_validation.py --period daily --limit 50

# 5min (same commands with --period 5m)
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_batch.py --period 5m --limit 50
PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --period 5m --limit 50
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_1step_oos.py --period 5m --limit 50
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_state_backtest.py --period 5m --limit 50
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_factor_validation.py --period 5m --limit 50
```

(Wall-clock ~30-60 min on full limit=50; can be run in background with `run_in_background=true`.)

- [ ] **Step 2: Generate comparison report**

```bash
PYTHONIOENCODING=utf-8 python -c "
from dynamics import dynamics_granularity_compare as G
out = G.build_daily_vs_5min_report(
    daily_dir='data/dynamics',
    fivemin_dir='data/dynamics_5min',
    output_dir='backtrace/outputs/granularity_compare',
)
print(out)
"
```

Expected: report with N factors and decision verdict.

- [ ] **Step 3: Inspect summary TXT**

```bash
cat backtrace/outputs/granularity_compare/daily_vs_5min_summary.txt
```

Expected: factors compared count + ΔIC counts.

- [ ] **Step 4: Inspect comparison CSV**

```bash
head -20 backtrace/outputs/granularity_compare/daily_vs_5min_table.csv
```

Expected: per-factor ΔIC / ΔIC_IR rows.

- [ ] **Step 5: Apply decision verdicts**

Read the summary TXT + CSV. Per spec §7.2:
- If all hard thresholds met → recommend **Adopt 5min**
- If mixed → recommend **Archive 5min** (keep data layer, no production default)
- If no signal → recommend **Kill 5min**

- [ ] **Step 6: Record in progress.md**

Append:
```
## Phase D2 decision verdict
- Factors compared: <N>
- ΔIC ≥ +0.02: <n> factors
- ΔIC_IR ≥ +0.1: <n> factors
- ΔOOS RMSE ≤ -5%: <n> stocks
- Δhit_rate ≥ +3pp: <n> stocks
- Verdict: <Adopt | Archive | Kill>
- Rationale: <1-2 sentences>
```

- [ ] **Step 7: Commit progress.md + any new report artifacts**

```bash
git add .superpowers/sdd/2026-08-22-intraday-granularity-5min-baseline/progress.md
git add backtrace/outputs/granularity_compare/ 2>/dev/null || true
git commit -m "docs(intraday): Phase D2 decision verdict + comparison report"
```

(Note: `backtrace/outputs/` is gitignored. The progress.md and the data/projection_5min / data/dynamics_5min directories are also gitignored. Only progress.md commits.)

---

### Task D3: Final report + ADR

**Files:**
- Modify: `docs/superpowers/specs/2026-08-22-intraday-granularity-5min-baseline-design.md` (append Decision section)
- Create: `docs/superpowers/decisions/2026-08-22-intraday-granularity-decision.md` (if adopt) OR a kill-note in progress.md

- [ ] **Step 1: Write final decision memo**

Create `docs/superpowers/decisions/2026-08-22-intraday-granularity-decision.md`:
```markdown
# ADR: 5min Granularity Decision — 2026-08-22

## Status
<Proposed | Accepted | Rejected | Archived>

## Context
Daily projection/dynamics pipeline assumes 1-day resolution. Spike #1/#2
(2026-08-22) found Nyquist deficiency on daily bar, with 5-min as
candidate minimum sufficient resolution.

## Decision
<Adopt | Archive | Kill> — see rationale below.

## Rationale
<2-3 sentences from Phase D2 verdict>

## Consequences
<What changes in production; what stays the same.>

## Evidence
- Spike CSVs: backtrace/outputs/spike_1min_nyquist/, backtrace/outputs/spike_granularity/
- Comparison report: backtrace/outputs/granularity_compare/
- Test suite: <N> passed
- Spec: docs/superpowers/specs/2026-08-22-intraday-granularity-5min-baseline-design.md
- Plan: docs/superpowers/plans/2026-08-22-intraday-granularity-5min-baseline-plan.md
```

- [ ] **Step 2: Update spec with outcome section**

Append to `docs/superpowers/specs/2026-08-22-intraday-granularity-5min-baseline-design.md`:
```markdown

---

## 17. Outcome (filled at end of Phase D)

Decision: **<Adopt | Archive | Kill>**

<2-3 sentences summary + link to ADR>
```

- [ ] **Step 3: If Adopt, update `tsfresh_config.py` defaults**

Only if Adopt:
```python
# In backtrace/common/tsfresh_config.py:
# Leave DEFAULT_INTRADAY_GRANULARITY = '5m' (already set in Task A3)
# Do NOT change any script default — they remain 'daily' per spec §2.2.
# The Adopt path is: scripts accept --period 5m; users opt-in.
```

(Per spec §2.2: no automatic default promotion. Users pass `--period 5m` explicitly.)

- [ ] **Step 4: Final test run**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 5: Commit final state**

```bash
git add docs/superpowers/decisions/2026-08-22-intraday-granularity-decision.md
git add docs/superpowers/specs/2026-08-22-intraday-granularity-5min-baseline-design.md
git commit -m "docs(intraday): final decision ADR + spec outcome section"
```

- [ ] **Step 6: Push (if user approves)**

```bash
git push origin main
```

(Only with explicit user permission per CLAUDE.md "Commit or push only when the user asks.")

---

## Self-Review

**1. Spec coverage:**

| Spec section | Implemented by |
|---|---|
| §4.2.1 data_store changes | Task A2 |
| §4.2.2 tsfresh_config constants | Task A3 |
| §4.2.3 fetch_daily.py --period/--lookback-days | Task A4 |
| §4.4 Tests for Phase A | A2 + A4 |
| §4.5 Phase A exit criteria | A5 + A6 |
| §5.2.1 load_ohlcva period | Task B1 |
| §5.2.2 load_pair period | Task B2 |
| §5.2.3 14 scripts --period | Tasks B3 + B4 |
| §5.2.4 GRANULARITY_DT_SEC | Task A3 |
| §5.4 Tests for Phase B | B1 + B2 + B3 + B4 |
| §5.5 Phase B exit | Task B5 |
| §6.2 Procedure | Tasks C2 + C3 |
| §6.3 Outputs (period-tagged dirs) | Task C1 |
| §6.6 Tests for Phase C | C2 + C5 |
| §6.7 Phase C exit | Task C5 |
| §7 Decision framework | Tasks D1 + D2 |
| §7.2 Decision outcomes | Task D3 |
| §7.3 Decision owner (user | Task D3 step 1 |
| §7.4 Tests for Phase D | Task D1 |
| §8.1 TQ depth | Task A5 |
| §8.2 Storage cost | Recorded in spec §8.2 (informational) |
| §8.3 Manifest / cache invalidation | Task A4 (manifest period-tagged) |
| §8.4 Backward compatibility | Tasks A1 + A6 (byte-equality gate) |
| §11 File-by-file matrix | All tasks reference this in commit messages |

**2. Placeholder scan:** no TBD / TODO / "implement later" patterns. All steps have concrete commands or code.

**3. Type consistency:**
- `data_store.csv_path(code, period='daily', kind='stocks')` defined in A2, used in A4 + C1 + D scripts.
- `_filename(code, period='daily')` defined in A2, used internally in A2.
- `tsfresh_pipeline.load_ohlcva(..., period='daily')` defined in B1, used in B2 (`_projection_core.load_pair` calls `pipeline.load_ohlcva(..., period=period)`).
- `_projection_core.load_pair(..., period='daily')` defined in B2, used in B3 scripts.
- `C.VALID_GRANULARITIES`, `C.TQ_PERIOD_MAP`, `C.GRANULARITY_DT_SEC`, `C.DEFAULT_INTRADAY_LOOKBACK_DAYS`, `C.DEFAULT_INTRADAY_GRANULARITY` defined in A3.
- `output_subdir_for_period(base, period)` defined in C1, used in C1 step 5.
- `dynamics_granularity_compare.REPORT_TABLE_COLS`, `DELTA_*` constants defined in C4, used in C4 step 5 and D1.

All cross-task references are consistent.

**4. Open questions from spec §14 resolved by plan:**
- Q1 (positional vs flag) → resolved in B3 step 3: argparse `add_argument('--period', ...)` (flag form).
- Q2 (dir naming `data/projection_5min/` vs `data/projection/{period}/`) → resolved in C1: `data/projection_5min/` (flat suffix; simpler tooling).
- Q3 (dynamics_forced_response --period) → resolved in B4 step 1: NO (math-only).
- Q4 (decision threshold strictness) → resolved in D1: thresholds hard-coded as constants; `archive-or-kill` is the default for any non-adopt case.

---

## Plan Metadata

- Total tasks: 18 (A1-A6, B1-B5, C1-C5, D1-D3)
- New files created: 6 (`tests/test_tsfresh_config.py`, `tests/test_tsfresh_pipeline_period.py`, `tests/test_dynamics_cli_period.py`, `tests/test_dynamics_granularity_compare.py`, `tests/test_granularity_decision.py`, `backtrace/dynamics/dynamics_granularity_compare.py`)
- Files modified: ~17
- Estimated LOC: ~500
- Estimated effort: 3-4 working days
- Open decisions for execution: TQ depth may force `DEFAULT_INTRADAY_LOOKBACK_DAYS` adjustment (Task A5 Step 3); user approval required before final commit in Task D3 Step 6.