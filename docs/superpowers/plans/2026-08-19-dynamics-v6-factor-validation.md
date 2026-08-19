# V6 Factor Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 v3-v5.11 产出的动力学变量(k̂, ĉ, λ, ρ, θ, hit_rate, RMSE, state 等)统一做 cross-sectional Spearman IC + Q1-Q5 quantile 评估,业务一次性看到"哪些因子有预测力"。

**Architecture:** 纯消费者脚本 — 读现有 CSV + daily prices,不做任何 projection/dynamics/OOS 重算。1 个新文件 + 1 个测试文件追加 + 1 个 README 更新。

**Tech Stack:** Python 3 / pandas / numpy / scipy.stats.spearmanr / plotly (可选 HTML) / 已有 P.load_ohlcva / 已有 _dynamics_core.analyze_eigenvalues

## Global Constraints

- 0 modifications to 11 protected files (`_projection_core.py` / `dynamics_forced_response.py` / `dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py` / `dynamics_si_ic.py` / `dynamics_si_timeseries.py` / `dynamics_si_lagged_ic.py` / `dynamics_eigen_analysis.py` / `parameter_fit.py` / `dynamics_oos_viz.py`)
- 1 new file: `backtrace/dynamics/dynamics_factor_validation.py`
- 1 modified test file: `tests/test_dynamics_eigen.py` (+ tests)
- 1 modified doc: `backtrace/dynamics/README.md` (新 §6 / 新 CLI 段)
- 0 new dependencies (pandas / numpy / scipy / plotly 已装)
- **Windows GBK 终端兼容**:全部 print 用 ASCII 或 `PYTHONIOENCODING=utf-8` 显式
- 因子 IC 全部 raw(不做行业/市值中性化)
- Spearman IC 用 `scipy.stats.spearmanr`,不自定义
- 缺失 CSV 显式 status='not_loaded',不抛异常(用户友好)
- 默认 limit=500,0 = 全 A 股(慢)

---

## Task 1: Core engine — 数据加载 + 因子派生 + IC/quantile 评估

**Files:**
- Create: `backtrace/dynamics/dynamics_factor_validation.py` (核心模块 + 函数,不写 CLI)
- Modify: `tests/test_dynamics_eigen.py` (+ 6 个测试)

**Interfaces (consumed by Task 2):**
- `load_kc_estimates(path: str) -> pd.DataFrame` → 必须 columns: code, k_hat, c_hat, status (string startswith 'ok')
- `load_oos_predictions_summary(path: str) -> pd.DataFrame` → 必须 columns: code, hit_rate, rmse, mae, direction_accuracy
- `load_state_distribution(path: str) -> pd.DataFrame` → 必须 columns: code, dominant_state + 7 state prop 列
- `load_kc_time_series(path: str) -> pd.DataFrame` → 必须 columns: code, asof_date, k_hat, c_hat
- `load_daily_prices(codes: list[str], repo_root: str) -> dict[str, pd.DataFrame]` → 每个 code 一份 DataFrame 含 close 列
- `load_industry_lookup(repo_root: str) -> pd.DataFrame` → 必须 columns: code, industry_l1, industry_l2
- `compute_eigen_factors(kc_df: pd.DataFrame) -> pd.DataFrame` → 添加 rho / theta / dist_to_unit / regime
- `compute_kc_evolution_factors(kc_time_df: pd.DataFrame) -> pd.DataFrame` → 添加 delta_k / delta_c / delta_rho / delta_theta (per code × asof_date)
- `compute_cross_section_ic(factor_series: pd.Series, ret_series: pd.Series) -> tuple[float, float, int]` → Spearman IC, p-value, n
- `compute_quantile_returns(factor_series: pd.Series, ret_series: pd.Series, n_quantiles: int = 5) -> dict[str, float]` → q1_ret, ..., q5_ret, q5_minus_q1
- `compute_forward_returns(daily_prices: dict[str, pd.DataFrame], dates_index: pd.DatetimeIndex, horizons: list[int]) -> pd.DataFrame` → MultiIndex (code, date) × horizons
- `build_factor_panel(kc_df, oos_df, state_df, kc_time_df, eigen_df) -> pd.DataFrame` → long format: (code, factor_name, factor_value)
- `validate_all_factors(panel: pd.DataFrame, fwd_rets: pd.DataFrame, horizons: list[int], industry_l1: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]` → (main_results, by_year_results, by_industry_results)

### Step 1.1: 写 scaffold + 数据加载层

```python
# -*- coding: utf-8 -*-
# dynamics_factor_validation.py — v6 业务验证(Cross-sectional IC + quantile)
#
# 纯消费者:读现有 CSV + daily prices,不做 projection/dynamics/OOS 重算。
# 缺失的 CSV → status='not_loaded',不抛异常。

import warnings
warnings.filterwarnings('ignore')
import sys, os
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BACKTRACE_DIR)
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger(__name__)

# === 公共 pipeline ===
from common import tsfresh_pipeline as P
from dynamics._dynamics_core import analyze_eigenvalues  # 复用,不动

__all__ = [
    'load_kc_estimates', 'load_oos_predictions_summary',
    'load_state_distribution', 'load_kc_time_series',
    'load_daily_prices', 'load_industry_lookup',
    'compute_eigen_factors', 'compute_kc_evolution_factors',
    'compute_cross_section_ic', 'compute_quantile_returns',
    'compute_forward_returns', 'build_factor_panel',
    'validate_all_factors',
]

# === 数据加载层 ===

KC_REQUIRED_COLS = ['code', 'k_hat', 'c_hat', 'status']

def load_kc_estimates(path: str) -> pd.DataFrame:
    """Load parameter_fit kc_estimates.csv.
    
    Returns: DataFrame with [code, k_hat, c_hat, status, index_code, ...]
    Filter: status.str.startswith('ok', na=False) — verbose status format.
    
    Raises: FileNotFoundError with hint to run parameter_fit.py
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f'kc_estimates.csv not found at {path}. '
            'Run: python backtrace/projection/parameter_fit.py'
        )
    df = pd.read_csv(p)
    missing = [c for c in KC_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'kc_estimates.csv missing columns: {missing}')
    df = df[df['status'].astype(str).str.startswith('ok', na=False)].copy()
    df['code'] = df['code'].astype(str).str.zfill(6)
    return df.reset_index(drop=True)


OOS_REQUIRED_COLS = ['code', 'hit_rate', 'rmse']

def load_oos_predictions_summary(path: str) -> pd.DataFrame:
    """Load v5.10 batch output prediction_summary.csv.
    
    Returns: DataFrame with [code, hit_rate, rmse, mae, direction_accuracy, ...]
    
    Raises: FileNotFoundError with hint to run dynamics_oos_batch.py
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f'prediction_summary.csv not found at {path}. '
            'Run: python backtrace/dynamics/dynamics_oos_batch.py '
            '--kc-estimates-csv data/projection/kc_estimates.csv'
        )
    df = pd.read_csv(p)
    missing = [c for c in OOS_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'prediction_summary.csv missing columns: {missing}')
    df['code'] = df['code'].astype(str).str.zfill(6)
    return df.reset_index(drop=True)


STATE_REQUIRED_COLS = ['code', 'dominant_state']

def load_state_distribution(path: str) -> pd.DataFrame:
    """Load dynamics_state_backtest.py output state_distribution.csv.
    
    Returns: DataFrame with [code, dominant_state, follow, against, ..., resonance]
    
    Raises: FileNotFoundError with hint to run dynamics_state_backtest.py
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f'state_distribution.csv not found at {path}. '
            'Run: python backtrace/dynamics/dynamics_state_backtest.py'
        )
    df = pd.read_csv(p)
    missing = [c for c in STATE_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'state_distribution.csv missing columns: {missing}')
    df['code'] = df['code'].astype(str).str.zfill(6)
    return df.reset_index(drop=True)


KC_TIME_REQUIRED_COLS = ['code', 'asof_date', 'k_hat', 'c_hat']

def load_kc_time_series(path: str) -> pd.DataFrame:
    """Load parameter_fit --rolling-time output kc_estimates_time.csv.
    
    Returns: DataFrame with [code, asof_date, k_hat, c_hat, ...]
    
    Raises: FileNotFoundError with hint to run parameter_fit.py --rolling-time
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f'kc_estimates_time.csv not found at {path}. '
            'Run: python backtrace/projection/parameter_fit.py --rolling-time'
        )
    df = pd.read_csv(p, parse_dates=['asof_date'])
    missing = [c for c in KC_TIME_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'kc_estimates_time.csv missing columns: {missing}')
    df['code'] = df['code'].astype(str).str.zfill(6)
    return df.reset_index(drop=True)


def load_daily_prices(codes: list[str], repo_root: str) -> dict[str, pd.DataFrame]:
    """Load daily close prices for given codes.
    
    Path: {repo_root}/data/stocks/{code}.csv (TQ local cache)
    
    Returns: dict[code] → DataFrame with DatetimeIndex and 'close' column.
    Missing files are silently skipped (logged warning).
    """
    out = {}
    repo_root = Path(repo_root)
    stocks_dir = repo_root / 'data' / 'stocks'
    if not stocks_dir.exists():
        raise FileNotFoundError(
            f'data/stocks not found at {stocks_dir}. '
            'Run: python backtrace/data_fetch/fetch_daily.py'
        )
    for code in codes:
        f = stocks_dir / f'{code}.csv'
        if not f.exists():
            log.warning(f'missing {code}.csv — skip')
            continue
        df = pd.read_csv(f, parse_dates=['date']).set_index('date').sort_index()
        if 'close' not in df.columns:
            log.warning(f'{code}.csv missing close — skip')
            continue
        out[code] = df[['close']]
    log.info(f'loaded daily prices for {len(out)}/{len(codes)} codes')
    return out


def load_industry_lookup(repo_root: str) -> pd.DataFrame:
    """Load industry mapping.
    
    Path: {repo_root}/data/sw2/members.csv → [code, sector_code, sector_name]
    Falls back to stock_basic.csv if sw2 missing.
    
    Returns: DataFrame with [code, industry_l1, industry_l2]
    Returns empty DataFrame if both files missing.
    """
    repo_root = Path(repo_root)
    sw2 = repo_root / 'data' / 'sw2' / 'members.csv'
    basic = repo_root / 'data' / 'stock_basic.csv'
    df = pd.DataFrame()
    if sw2.exists():
        df = pd.read_csv(sw2)
        # 申万二级 → sector_name 是 l2,industry_l1 不存在
        # 若需要 l1,后续可 join sw l1 mapping(v6 v1 不强制)
        df = df.rename(columns={'sector_name': 'industry_l2'})
        if 'sector_code' in df.columns:
            df['industry_l1'] = df['sector_code'].astype(str).str[:5]  # 简化为 l2 的前 5 位作为 l1 近似
        df['code'] = df['code'].astype(str).str.zfill(6)
    elif basic.exists():
        df = pd.read_csv(basic)
        df['code'] = df['code'].astype(str).str.zfill(6)
        df['industry_l1'] = 'unknown'
        df['industry_l2'] = 'unknown'
    else:
        log.warning('no industry lookup files found')
    return df[['code', 'industry_l1', 'industry_l2']].drop_duplicates('code').reset_index(drop=True) if not df.empty else df
```

### Step 1.2: 写派生因子函数

```python
# === 派生层 ===

def compute_eigen_factors(kc_df: pd.DataFrame) -> pd.DataFrame:
    """从 (k̂, �) 派生 eigenvalues / rho / theta / dist_to_unit / regime。
    
    复用 _dynamics_core.analyze_eigenvalues (0 修改)。
    
    Input: kc_df with [code, k_hat, c_hat]
    Output: DataFrame with [code, rho, theta, dist_to_unit, regime]
    """
    out_rows = []
    for _, row in kc_df.iterrows():
        k, c = float(row['k_hat']), float(row['c_hat'])
        try:
            eig = analyze_eigenvalues(k, c)
            lam = eig['eigenvalues']
            # 主特征值(按 |λ| 排序)
            lam_main = max(lam, key=lambda z: abs(z))
            rho = float(eig['spectral_radius'])
            theta = float(np.angle(lam_main)) if rho > 1e-10 else 0.0
            regime = eig['classification']  # 11 类
            # 简化为 4 类业务语义
            if 'stable' in regime:
                regime_4 = 'overdamped' if rho < 0.95 else 'critical'
            elif 'divergent' in regime or 'periodic' in regime:
                regime_4 = 'underdamped'
            elif 'anti' in regime:
                regime_4 = 'anti_damped'
            else:
                regime_4 = 'critical'
        except Exception as e:
            log.warning(f'analyze_eigenvalues failed for k={k}, c={c}: {e}')
            rho, theta, regime_4 = np.nan, np.nan, 'unknown'
        out_rows.append({
            'code': row['code'],
            'rho': rho,
            'theta': theta,
            'dist_to_unit': 1.0 - rho if not np.isnan(rho) else np.nan,
            'regime': regime_4,
        })
    return pd.DataFrame(out_rows)


def compute_kc_evolution_factors(kc_time_df: pd.DataFrame) -> pd.DataFrame:
    """从 rolling kc 时间序列派生 delta_k / delta_c / delta_rho / delta_theta。
    
    对每个 code,按 asof_date 排序后做 .diff()(最近 asof - 前一 asof)。
    
    Input: kc_time_df with [code, asof_date, k_hat, c_hat]
    Output: DataFrame with [code, asof_date, delta_k, delta_c, delta_rho, delta_theta]
    """
    df = kc_time_df.copy().sort_values(['code', 'asof_date'])
    # 加 rho / theta
    eigen_rows = []
    for _, row in df.iterrows():
        try:
            eig = analyze_eigenvalues(float(row['k_hat']), float(row['c_hat']))
            rho = float(eig['spectral_radius'])
            lam = eig['eigenvalues']
            lam_main = max(lam, key=lambda z: abs(z))
            theta = float(np.angle(lam_main)) if rho > 1e-10 else 0.0
        except Exception:
            rho, theta = np.nan, np.nan
        eigen_rows.append({'code': row['code'], 'asof_date': row['asof_date'], 'rho': rho, 'theta': theta})
    eig_df = pd.DataFrame(eigen_rows)
    df = df.merge(eig_df, on=['code', 'asof_date'], how='left')
    # diff
    df['delta_k'] = df.groupby('code')['k_hat'].diff()
    df['delta_c'] = df.groupby('code')['c_hat'].diff()
    df['delta_rho'] = df.groupby('code')['rho'].diff()
    df['delta_theta'] = df.groupby('code')['theta'].diff()
    return df[['code', 'asof_date', 'delta_k', 'delta_c', 'delta_rho', 'delta_theta']].reset_index(drop=True)
```

### Step 1.3: 写评估函数

```python
# === 评估层 ===

def compute_cross_section_ic(
    factor_series: pd.Series, ret_series: pd.Series
) -> tuple[float, float, int]:
    """Cross-sectional Spearman IC for one (date_t) snapshot.
    
    Returns: (ic, p_value, n_obs). NaN if n < 10 or std = 0.
    """
    df = pd.DataFrame({'f': factor_series, 'r': ret_series}).dropna()
    n = len(df)
    if n < 10:
        return (np.nan, np.nan, n)
    if df['f'].nunique() < 2 or df['r'].nunique() < 2:
        return (np.nan, np.nan, n)
    rho, pval = spearmanr(df['f'].values, df['r'].values)
    return (float(rho), float(pval), n)


def compute_quantile_returns(
    factor_series: pd.Series, ret_series: pd.Series, n_quantiles: int = 5
) -> dict[str, float]:
    """Q1-Q5 mean returns for one (date_t) snapshot.
    
    Returns: dict {q1_ret, ..., q{n}_ret, q{n}_minus_q1, n_obs}.
    NaN if n < n_quantiles * 2.
    """
    df = pd.DataFrame({'f': factor_series, 'r': ret_series}).dropna()
    n = len(df)
    if n < n_quantiles * 2:
        return {f'q{i}_ret': np.nan for i in range(1, n_quantiles + 1)} | {
            f'q{n_quantiles}_minus_q1': np.nan, 'n_obs': n,
        }
    try:
        df['quantile'] = pd.qcut(df['f'], n_quantiles, labels=False, duplicates='drop')
        q_means = df.groupby('quantile')['r'].mean()
        # 重新对齐:可能 duplicates='drop' 砍掉某些分位
        result = {f'q{i+1}_ret': float(q_means.get(i, np.nan)) for i in range(n_quantiles)}
        if n_quantiles in q_means.index and 0 in q_means.index:
            result[f'q{n_quantiles}_minus_q1'] = float(q_means[n_quantiles - 1] - q_means[0])
        else:
            result[f'q{n_quantiles}_minus_q1'] = np.nan
        result['n_obs'] = n
        return result
    except Exception as e:
        log.warning(f'compute_quantile_returns failed: {e}')
        return {f'q{i}_ret': np.nan for i in range(1, n_quantiles + 1)} | {
            f'q{n_quantiles}_minus_q1': np.nan, 'n_obs': n,
        }
```

### Step 1.4: 写主流程函数

```python
# === 主流程 ===

def compute_forward_returns(
    daily_prices: dict[str, pd.DataFrame],
    dates_index: pd.DatetimeIndex,
    horizons: list[int],
) -> pd.DataFrame:
    """Compute forward returns for each (code, date, horizon).
    
    Input: daily_prices[code] → DataFrame indexed by date with 'close'.
           dates_index → 评估日期集合(用 daily_prices 的交集).
           horizons → list of forward days.
    
    Returns: DataFrame with MultiIndex (code, date) and columns [fwd_ret_{h} for h in horizons].
    """
    rows = []
    for code, df in daily_prices.items():
        df = df.sort_index()
        for h in horizons:
            # fwd_ret_h[t] = close[t+h] / close[t] - 1
            future = df['close'].shift(-h) / df['close'] - 1.0
            for d in dates_index:
                if d in df.index and d in future.index:
                    val = future.loc[d]
                    if not np.isnan(val):
                        rows.append({'code': code, 'date': d, f'fwd_ret_{h}d': val})
    if not rows:
        return pd.DataFrame()
    df_out = pd.DataFrame(rows).set_index(['code', 'date'])
    return df_out


def build_factor_panel(
    kc_df: pd.DataFrame, oos_df: pd.DataFrame,
    state_df: pd.DataFrame, kc_time_df: pd.DataFrame | None,
    eigen_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build (code, factor_name, factor_value) long-format panel.
    
    Returns: DataFrame with columns [code, factor_name, factor_value, status].
    status='loaded' / 'not_loaded'.
    """
    rows = []
    # k̂, ĉ, c_over_k, log_c_over_k
    if not kc_df.empty:
        for _, row in kc_df.iterrows():
            k, c = row['k_hat'], row['c_hat']
            rows.append({'code': row['code'], 'factor_name': 'k', 'factor_value': k, 'status': 'loaded'})
            rows.append({'code': row['code'], 'factor_name': 'c', 'factor_value': c, 'status': 'loaded'})
            if abs(k) > 1e-12:
                rows.append({'code': row['code'], 'factor_name': 'c_over_k', 'factor_value': c / k, 'status': 'loaded'})
                rows.append({'code': row['code'], 'factor_name': 'log_c_over_k', 'factor_value': np.log10(abs(c / k) + 1e-12), 'status': 'loaded'})
            else:
                rows.append({'code': row['code'], 'factor_name': 'c_over_k', 'factor_value': np.nan, 'status': 'loaded'})
                rows.append({'code': row['code'], 'factor_name': 'log_c_over_k', 'factor_value': np.nan, 'status': 'loaded'})
    else:
        for fn in ['k', 'c', 'c_over_k', 'log_c_over_k']:
            rows.append({'code': '', 'factor_name': fn, 'factor_value': np.nan, 'status': 'not_loaded'})
    
    # eigen (rho / theta / dist_to_unit / regime)
    if eigen_df is not None and not eigen_df.empty:
        for _, row in eigen_df.iterrows():
            for fn in ['rho', 'theta', 'dist_to_unit']:
                rows.append({'code': row['code'], 'factor_name': fn, 'factor_value': row[fn], 'status': 'loaded'})
            rows.append({'code': row['code'], 'factor_name': 'regime', 'factor_value': row['regime'], 'status': 'loaded'})
    else:
        for fn in ['rho', 'theta', 'dist_to_unit', 'regime']:
            rows.append({'code': '', 'factor_name': fn, 'factor_value': np.nan, 'status': 'not_loaded'})
    
    # OOS (hit_rate / rmse / mae / direction_acc)
    if not oos_df.empty:
        for _, row in oos_df.iterrows():
            for fn in ['hit_rate', 'rmse', 'mae', 'direction_acc']:
                if fn in row and not pd.isna(row[fn]):
                    rows.append({'code': row['code'], 'factor_name': fn, 'factor_value': row[fn], 'status': 'loaded'})
    else:
        for fn in ['hit_rate', 'rmse', 'mae', 'direction_acc']:
            rows.append({'code': '', 'factor_name': fn, 'factor_value': np.nan, 'status': 'not_loaded'})
    
    # state (state_p_resonance / state_p_against / etc + dominant_state)
    if not state_df.empty:
        for _, row in state_df.iterrows():
            for fn in ['follow', 'against', 'independent', 'leading', 'lagging', 'resonance', 'orbit']:
                col = f'state_p_{fn}'
                if col in row and not pd.isna(row[col]):
                    rows.append({'code': row['code'], 'factor_name': col, 'factor_value': row[col], 'status': 'loaded'})
            rows.append({'code': row['code'], 'factor_name': 'state_dominant', 'factor_value': row['dominant_state'], 'status': 'loaded'})
    else:
        for fn in ['follow', 'against', 'independent', 'leading', 'lagging', 'resonance', 'orbit']:
            rows.append({'code': '', 'factor_name': f'state_p_{fn}', 'factor_value': np.nan, 'status': 'not_loaded'})
        rows.append({'code': '', 'factor_name': 'state_dominant', 'factor_value': np.nan, 'status': 'not_loaded'})
    
    # rolling (delta_k / delta_c / delta_rho / delta_theta) — 来自 kc_time_df
    if kc_time_df is not None and not kc_time_df.empty:
        evo_df = compute_kc_evolution_factors(kc_time_df)
        for _, row in evo_df.iterrows():
            for fn in ['delta_k', 'delta_c', 'delta_rho', 'delta_theta']:
                if not pd.isna(row[fn]):
                    rows.append({
                        'code': row['code'], 'factor_name': fn,
                        'factor_value': row[fn], 'status': 'loaded',
                        'asof_date': row['asof_date'],
                    })
    
    return pd.DataFrame(rows)


def validate_all_factors(
    panel: pd.DataFrame, fwd_rets: pd.DataFrame,
    horizons: list[int], industry_l1: pd.Series,
    min_n_per_date: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对每个 (factor, horizon) 算 IC + quantile,by-year, by-industry。
    
    Returns: (main_results, by_year_results, by_industry_results)
    """
    # panel 是 long format (code, factor_name, factor_value, status[, asof_date])
    # fwd_rets 是 wide format, MultiIndex (code, date), columns [fwd_ret_{h}d]
    
    # pivot panel 到 wide: (code, date) × factor_name
    # 但 panel 没有 date 列(除了 rolling factors)...
    # V6 v1 简化:对 non-rolling factors,假设 factor value 是"该股静态值"(from parameter_fit),
    # 对 rolling factors, 用 asof_date。
    # 
    # 这意味着 non-rolling factors 不会变 time-series,IC pool 只来自 date 维度
    # rolling factors 在 asof_date 上算 IC。
    
    main_rows = []
    by_year_rows = []
    by_industry_rows = []
    
    factor_names = panel[panel['status'] == 'loaded']['factor_name'].unique()
    for fn in factor_names:
        fpanel = panel[panel['factor_name'] == fn].copy()
        is_rolling = 'asof_date' in fpanel.columns and fpanel['asof_date'].notna().any()
        
        for h in horizons:
            ret_col = f'fwd_ret_{h}d'
            ics = []
            q_returns = {f'q{i+1}_ret': [] for i in range(5)}
            q_spread = []
            n_obs_total = 0
            
            if is_rolling:
                for asof_date, grp in fpanel.groupby('asof_date'):
                    fac_s = grp.set_index('code')['factor_value']
                    # 取 asof_date 当日的 fwd_ret
                    if asof_date not in fwd_rets.index.get_level_values('date'):
                        continue
                    ret_s = fwd_rets.xs(asof_date, level='date')[ret_col]
                    common = fac_s.index.intersection(ret_s.index)
                    if len(common) < min_n_per_date:
                        continue
                    ic, pval, n = compute_cross_section_ic(fac_s.loc[common], ret_s.loc[common])
                    if not np.isnan(ic):
                        ics.append({'ic': ic, 'date': asof_date, 'n': n, 'codes': set(common)})
                    q = compute_quantile_returns(fac_s.loc[common], ret_s.loc[common])
                    for k_ in q_returns:
                        if k_ in q and not np.isnan(q[k_]):
                            q_returns[k_].append(q[k_])
                    if f'q5_minus_q1' in q and not np.isnan(q['q5_minus_q1']):
                        q_spread.append(q['q5_minus_q1'])
                    n_obs_total += n
            else:
                # non-rolling: factor 静态,date 维度 = fwd_rets 的所有 date
                fac_s = fpanel.set_index('code')['factor_value']
                for date_t in fwd_rets.index.get_level_values('date').unique():
                    if date_t not in fwd_rets.index.get_level_values('date'):
                        continue
                    ret_s = fwd_rets.xs(date_t, level='date')[ret_col]
                    common = fac_s.index.intersection(ret_s.index)
                    if len(common) < min_n_per_date:
                        continue
                    ic, pval, n = compute_cross_section_ic(fac_s.loc[common], ret_s.loc[common])
                    if not np.isnan(ic):
                        ics.append({'ic': ic, 'date': date_t, 'n': n, 'codes': set(common)})
                    q = compute_quantile_returns(fac_s.loc[common], ret_s.loc[common])
                    for k_ in q_returns:
                        if k_ in q and not np.isnan(q[k_]):
                            q_returns[k_].append(q[k_])
                    if f'q5_minus_q1' in q and not np.isnan(q['q5_minus_q1']):
                        q_spread.append(q['q5_minus_q1'])
                    n_obs_total += n
            
            if not ics:
                main_rows.append({
                    'factor': fn, 'horizon': h, 'n_obs': 0, 'n_dates': 0,
                    'ic_mean': np.nan, 'ic_std': np.nan, 'ic_ir': np.nan, 'ic_pvalue': np.nan,
                    'q1_ret': np.nan, 'q2_ret': np.nan, 'q3_ret': np.nan, 'q4_ret': np.nan, 'q5_ret': np.nan,
                    'q5_minus_q1': np.nan, 'status': 'insufficient_data',
                })
                continue
            
            ics_arr = np.array([x['ic'] for x in ics])
            n_dates = len(ics)
            ic_mean = float(np.mean(ics_arr))
            ic_std = float(np.std(ics_arr, ddof=1)) if n_dates > 1 else np.nan
            ic_ir = ic_mean / ic_std if ic_std and not np.isnan(ic_std) and ic_std > 0 else np.nan
            # one-sample t-test: H0 mean=0
            from scipy.stats import ttest_1samp
            if n_dates > 1:
                t_stat, p_val = ttest_1samp(ics_arr, 0)
                ic_pvalue = float(p_val)
            else:
                ic_pvalue = np.nan
            
            q_means = {k_: float(np.mean(v)) if v else np.nan for k_, v in q_returns.items()}
            q_spread_mean = float(np.mean(q_spread)) if q_spread else np.nan
            
            # by-year
            years_seen = set()
            for x in ics:
                yr = pd.Timestamp(x['date']).year
                years_seen.add(yr)
            
            top_year, top_year_ic = '', np.nan
            top_industry, top_industry_ic = '', np.nan
            for yr in sorted(years_seen):
                yr_ics = [x['ic'] for x in ics if pd.Timestamp(x['date']).year == yr]
                yr_ic_mean = float(np.mean(yr_ics)) if yr_ics else np.nan
                by_year_rows.append({
                    'factor': fn, 'horizon': h, 'year': yr,
                    'n_obs': sum(x['n'] for x in ics if pd.Timestamp(x['date']).year == yr),
                    'n_dates': len(yr_ics),
                    'ic_mean': yr_ic_mean, 'ic_std': float(np.std(yr_ics, ddof=1)) if len(yr_ics) > 1 else np.nan,
                    'status': 'ok',
                })
                if not np.isnan(yr_ic_mean) and abs(yr_ic_mean) > abs(top_year_ic):
                    top_year = str(yr)
                    top_year_ic = yr_ic_mean
            
            # by-industry
            ind_l1_dict = industry_l1.to_dict() if hasattr(industry_l1, 'to_dict') else {}
            ind_ics = {}
            for x in ics:
                for c in x['codes']:
                    ind = ind_l1_dict.get(c, 'unknown')
                    ind_ics.setdefault(ind, []).append(x['ic'])
            for ind, ind_ic_list in ind_ics.items():
                if len(ind_ic_list) < 5:
                    continue
                ind_ic_mean = float(np.mean(ind_ic_list))
                by_industry_rows.append({
                    'factor': fn, 'horizon': h, 'industry_l1': ind,
                    'n_dates': len(ind_ic_list),
                    'ic_mean': ind_ic_mean,
                    'ic_std': float(np.std(ind_ic_list, ddof=1)) if len(ind_ic_list) > 1 else np.nan,
                    'status': 'ok',
                })
                if abs(ind_ic_mean) > abs(top_industry_ic):
                    top_industry = ind
                    top_industry_ic = ind_ic_mean
            
            main_rows.append({
                'factor': fn, 'horizon': h,
                'n_obs': n_obs_total, 'n_dates': n_dates,
                'ic_mean': ic_mean, 'ic_std': ic_std, 'ic_ir': ic_ir, 'ic_pvalue': ic_pvalue,
                **q_means, 'q5_minus_q1': q_spread_mean,
                'top_year': top_year, 'top_year_ic': top_year_ic,
                'top_industry': top_industry, 'top_industry_ic': top_industry_ic,
                'status': 'ok',
            })
    
    return (
        pd.DataFrame(main_rows),
        pd.DataFrame(by_year_rows),
        pd.DataFrame(by_industry_rows),
    )
```

### Step 1.5: 写测试

```python
# Append to tests/test_dynamics_eigen.py:

def test_compute_cross_section_ic_positive():
    """Perfect positive rank correlation → IC ≈ 1.0."""
    from dynamics.dynamics_factor_validation import compute_cross_section_ic
    factor = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ret = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ic, pval, n = compute_cross_section_ic(factor, ret)
    assert ic > 0.99
    assert n == 10
    assert pval < 0.001


def test_compute_cross_section_ic_negative():
    """Negative correlation → IC < 0."""
    from dynamics.dynamics_factor_validation import compute_cross_section_ic
    factor = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ret = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    ic, _, _ = compute_cross_section_ic(factor, ret)
    assert ic < -0.99


def test_compute_cross_section_ic_too_few():
    """n < 10 → NaN, n=actual."""
    from dynamics.dynamics_factor_validation import compute_cross_section_ic
    factor = pd.Series([1, 2, 3])
    ret = pd.Series([0.1, 0.2, 0.3])
    ic, _, n = compute_cross_section_ic(factor, ret)
    assert np.isnan(ic)
    assert n == 3


def test_compute_quantile_returns_monotonic():
    """Q1 < Q5 monotonic."""
    from dynamics.dynamics_factor_validation import compute_quantile_returns
    np.random.seed(42)
    factor = pd.Series(np.arange(100).astype(float))
    ret = pd.Series(factor.values * 0.01)  # 完美单调
    q = compute_quantile_returns(factor, ret, n_quantiles=5)
    assert q['q1_ret'] < q['q5_ret']
    assert q['q5_minus_q1'] > 0


def test_compute_eigen_factors():
    """(k=0.145, c=1.112) → rho ≈ 0.85, regime=overdamped."""
    from dynamics.dynamics_factor_validation import compute_eigen_factors
    kc = pd.DataFrame({'code': ['000001.SZ'], 'k_hat': [0.145], 'c_hat': [1.112]})
    out = compute_eigen_factors(kc)
    assert abs(out['rho'].iloc[0] - 0.85) < 0.01
    assert out['regime'].iloc[0] == 'overdamped'


def test_load_kc_estimates_missing():
    """missing path → FileNotFoundError with hint."""
    from dynamics.dynamics_factor_validation import load_kc_estimates
    with pytest.raises(FileNotFoundError, match='parameter_fit.py'):
        load_kc_estimates('/nonexistent/kc_estimates.csv')
```

### Step 1.6: 跑测试

```bash
cd /c/Users/yellow/mcp/qtTdx && /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py -v -k "compute_cross_section_ic or compute_quantile_returns or compute_eigen_factors or load_kc_estimates_missing"
```

Expected: 6 tests PASS

### Step 1.7: Commit Task 1

```bash
git add backtrace/dynamics/dynamics_factor_validation.py tests/test_dynamics_eigen.py
git commit -m "feat(dynamics): v6 — core engine (数据加载 + 因子派生 + IC/quantile)"
```

---

## Task 2: CLI + 输出层 (CSV / TXT / HTML)

**Files:**
- Modify: `backtrace/dynamics/dynamics_factor_validation.py` (+ CLI + output writers)
- Modify: `backtrace/dynamics/README.md` (新 §6 + 新 CLI 段)
- Modify: `tests/test_dynamics_eigen.py` (+ 1 CLI smoke test)

**Interfaces (consumed by CLI):**
- `write_main_csv(results: pd.DataFrame, path: str) -> None`
- `write_by_year_csv(results: pd.DataFrame, path: str) -> None`
- `write_by_industry_csv(results: pd.DataFrame, path: str) -> None`
- `write_summary_text(main, by_year, by_industry, path: str) -> None`
- `build_factor_validation_html(main, by_year, by_industry, path: str) -> None`
- `main() -> int` (CLI entry)

### Step 2.1: 写输出层

```python
# Append to backtrace/dynamics/dynamics_factor_validation.py:

import argparse

# === 输出层 ===

MAIN_COLS = [
    'factor', 'horizon', 'n_obs', 'n_dates',
    'ic_mean', 'ic_std', 'ic_ir', 'ic_pvalue',
    'q1_ret', 'q2_ret', 'q3_ret', 'q4_ret', 'q5_ret', 'q5_minus_q1',
    'top_year', 'top_year_ic', 'top_industry', 'top_industry_ic', 'status',
]

BY_YEAR_COLS = ['factor', 'horizon', 'year', 'n_obs', 'n_dates', 'ic_mean', 'ic_std', 'status']
BY_INDUSTRY_COLS = ['factor', 'horizon', 'industry_l1', 'n_dates', 'ic_mean', 'ic_std', 'status']


def write_main_csv(results: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = results.reindex(columns=MAIN_COLS, fill_value=np.nan)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    log.info(f'wrote {path} ({len(df)} rows)')


def write_by_year_csv(results: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = results.reindex(columns=BY_YEAR_COLS, fill_value=np.nan)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    log.info(f'wrote {path} ({len(df)} rows)')


def write_by_industry_csv(results: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = results.reindex(columns=BY_INDUSTRY_COLS, fill_value=np.nan)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    log.info(f'wrote {path} ({len(df)} rows)')


def write_summary_text(
    main: pd.DataFrame, by_year: pd.DataFrame, by_industry: pd.DataFrame, path: str,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append('=' * 80)
    lines.append('V6 Dynamics Factor Validation — Summary')
    lines.append(f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('=' * 80)
    
    # 整体概览
    ok = main[main['status'] == 'ok'] if 'status' in main.columns else pd.DataFrame()
    not_loaded = main[main['status'] == 'not_loaded']['factor'].unique() if 'status' in main.columns else []
    insufficient = main[main['status'] == 'insufficient_data']['factor'].unique() if 'status' in main.columns else []
    
    lines.append('')
    lines.append(f'Total (factor, horizon) pairs: {len(main)}')
    lines.append(f'OK: {len(ok)} pairs')
    lines.append(f'Insufficient data: {len(insufficient)} pairs')
    lines.append(f'Not loaded: {len(not_loaded)} factors')
    
    # Top 10 by |IC * IR|
    if not ok.empty and 'ic_mean' in ok.columns and 'ic_ir' in ok.columns:
        ok = ok.copy()
        ok['_score'] = ok['ic_mean'].abs() * ok['ic_ir'].abs()
        top = ok.nlargest(10, '_score')
        lines.append('')
        lines.append('--- Top 10 (factor, horizon) by |IC| * |IR| ---')
        lines.append(f'{"factor":<22} {"horizon":>8} {"ic_mean":>10} {"ic_ir":>10} {"pval":>10} {"q5-q1":>10}')
        lines.append('-' * 80)
        for _, row in top.iterrows():
            lines.append(
                f'{str(row["factor"]):<22} {int(row["horizon"]):>8} '
                f'{row["ic_mean"]:>10.4f} {row["ic_ir"]:>10.3f} '
                f'{row["ic_pvalue"]:>10.4g} {row["q5_minus_q1"]:>10.4f}'
            )
    
    # By year — 列出显著 IC(year × factor)
    if not by_year.empty:
        by_year_ok = by_year[by_year['status'] == 'ok'].copy() if 'status' in by_year.columns else by_year
        by_year_ok = by_year_ok[by_year_ok['ic_mean'].abs() > 0.05]
        if not by_year_ok.empty:
            lines.append('')
            lines.append('--- Significant yearly IC (|ic_mean| > 0.05) ---')
            lines.append(f'{"factor":<22} {"horizon":>8} {"year":>6} {"ic_mean":>10}')
            lines.append('-' * 60)
            for _, row in by_year_ok.nlargest(20, 'ic_mean', keep='all').iterrows():
                if pd.notna(row['year']):
                    lines.append(f'{str(row["factor"]):<22} {int(row["horizon"]):>8} {int(row["year"]):>6} {row["ic_mean"]:>10.4f}')
    
    # By industry
    if not by_industry.empty:
        by_ind_ok = by_industry[by_industry['status'] == 'ok'].copy() if 'status' in by_industry.columns else by_industry
        by_ind_ok = by_ind_ok[by_ind_ok['ic_mean'].abs() > 0.05]
        if not by_ind_ok.empty:
            lines.append('')
            lines.append('--- Significant industry IC (|ic_mean| > 0.05) ---')
            lines.append(f'{"factor":<22} {"horizon":>8} {"industry_l1":<20} {"ic_mean":>10}')
            lines.append('-' * 70)
            for _, row in by_ind_ok.nlargest(20, 'ic_mean', keep='all').iterrows():
                lines.append(f'{str(row["factor"]):<22} {int(row["horizon"]):>8} {str(row["industry_l1"]):<20} {row["ic_mean"]:>10.4f}')
    
    # 业务结论提示
    lines.append('')
    lines.append('--- Decision Hints ---')
    lines.append('If no |ic_mean| > 0.03 anywhere: model is descriptive not predictive. Archive v3-v5.11.')
    lines.append('If hit_rate/rmse have IC: OOS prediction is alpha-worthy (merge into tsfresh/alpha).')
    lines.append('If k/c/rho have IC but state does not: dynamic structure > state classification.')
    lines.append('If delta_* has IC but static k/c does not: regime transition is the signal (pursue rolling).')
    lines.append('=' * 80)
    
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
    log.info(f'wrote {path}')


def build_factor_validation_html(
    main: pd.DataFrame, by_year: pd.DataFrame, by_industry: pd.DataFrame, path: str,
) -> None:
    """plotly HTML — top 10 factors visualization."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.warning('plotly not available, skip HTML')
        return
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    ok = main[main['status'] == 'ok'].copy() if 'status' in main.columns else pd.DataFrame()
    if ok.empty or 'ic_mean' not in ok.columns:
        log.warning('no OK results, skip HTML')
        return
    
    # top 10 by |ic_mean|
    ok['_abs_ic'] = ok['ic_mean'].abs()
    top10 = ok.nlargest(10, '_abs_ic').copy()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Top 10 IC mean (across horizons)',
            'Top 10 Q5-Q1 spread',
            'Top factor — by-year IC',
            'Top factor — by-industry IC',
        ),
    )
    
    # (1,1) IC bar
    labels = [f'{r["factor"]}_h{int(r["horizon"])}' for _, r in top10.iterrows()]
    fig.add_trace(
        go.Bar(x=labels, y=top10['ic_mean'], name='IC mean'),
        row=1, col=1,
    )
    
    # (1,2) Q5-Q1 spread bar
    fig.add_trace(
        go.Bar(x=labels, y=top10['q5_minus_q1'], name='Q5-Q1'),
        row=1, col=2,
    )
    
    # (2,1) top factor by-year IC
    if not by_year.empty:
        top_factor = top10.iloc[0]['factor']
        top_h = top10.iloc[0]['horizon']
        by_y = by_year[(by_year['factor'] == top_factor) & (by_year['horizon'] == top_h)].copy()
        if not by_y.empty:
            fig.add_trace(
                go.Scatter(
                    x=by_y['year'].astype(int).astype(str), y=by_y['ic_mean'],
                    mode='lines+markers', name=f'{top_factor} (year)',
                ),
                row=2, col=1,
            )
    
    # (2,2) top factor by-industry IC
    if not by_industry.empty:
        top_factor = top10.iloc[0]['factor']
        top_h = top10.iloc[0]['horizon']
        by_i = by_industry[(by_industry['factor'] == top_factor) & (by_industry['horizon'] == top_h)].copy()
        if not by_i.empty:
            fig.add_trace(
                go.Bar(x=by_i['industry_l1'], y=by_i['ic_mean'], name=f'{top_factor} (industry)'),
                row=2, col=2,
            )
    
    fig.update_layout(height=900, title_text='V6 Factor Validation — Top 10 by |IC|', showlegend=False)
    fig.write_html(path, include_plotlyjs='cdn')
    log.info(f'wrote {path}')
```

### Step 2.2: 写 CLI main()

```python
# Append to backtrace/dynamics/dynamics_factor_validation.py:

def main():
    parser = argparse.ArgumentParser(description='V6 Dynamics Factor Validation')
    parser.add_argument('--kc-csv', default='data/projection/kc_estimates.csv', help='kc_estimates CSV path')
    parser.add_argument('--oos-summary', default='data/dynamics/prediction_summary.csv', help='OOS prediction summary CSV')
    parser.add_argument('--state-csv', default='data/dynamics/state_distribution.csv', help='state distribution CSV')
    parser.add_argument('--kc-time-csv', default='data/projection/kc_estimates_time.csv', help='kc rolling time series CSV')
    parser.add_argument('--horizons', default='1,5,10,20', help='comma-separated forward horizons (days)')
    parser.add_argument('--limit', type=int, default=500, help='stock limit; 0 = all')
    parser.add_argument('--factors', default='', help='comma-separated factor subset (default: all)')
    parser.add_argument('--output-dir', default='backtrace/outputs', help='output directory')
    parser.add_argument('--data-dir', default='data/dynamics', help='CSV output directory')
    parser.add_argument('--repo-root', default=REPO_ROOT, help='repo root for daily data lookup')
    args = parser.parse_args()
    
    horizons = [int(h) for h in args.horizons.split(',')]
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Load kc (required)
    try:
        kc_df = load_kc_estimates(args.kc_csv)
        log.info(f'kc_estimates: {len(kc_df)} stocks OK')
    except FileNotFoundError as e:
        log.error(str(e))
        return 1
    
    # Limit
    if args.limit > 0:
        codes = kc_df['code'].head(args.limit).tolist()
    else:
        codes = kc_df['code'].tolist()
    log.info(f'using {len(codes)} stocks')
    
    # Load OOS / state / kc_time (optional, graceful)
    oos_df = pd.DataFrame()
    try:
        oos_df = load_oos_predictions_summary(args.oos_summary)
        log.info(f'oos_summary: {len(oos_df)} stocks')
    except FileNotFoundError as e:
        log.warning(str(e))
    
    state_df = pd.DataFrame()
    try:
        state_df = load_state_distribution(args.state_csv)
        log.info(f'state_distribution: {len(state_df)} stocks')
    except FileNotFoundError as e:
        log.warning(str(e))
    
    kc_time_df = None
    try:
        kc_time_df = load_kc_time_series(args.kc_time_csv)
        log.info(f'kc_time_series: {len(kc_time_df)} rows')
    except FileNotFoundError as e:
        log.warning(str(e))
    
    # Compute eigen factors
    eigen_df = compute_eigen_factors(kc_df)
    log.info(f'eigen factors: {len(eigen_df)} rows')
    
    # Build factor panel
    panel = build_factor_panel(kc_df, oos_df, state_df, kc_time_df, eigen_df)
    log.info(f'factor panel: {len(panel)} rows, {panel["factor_name"].nunique()} factors')
    
    # Load daily prices
    daily_prices = load_daily_prices(codes, args.repo_root)
    log.info(f'daily prices loaded: {len(daily_prices)} stocks')
    
    # Build date index — 取所有 daily data 的交集,后 240 天(留 forward return 计算空间)
    if not daily_prices:
        log.error('no daily prices — exit')
        return 1
    all_dates = set(daily_prices[next(iter(daily_prices))].index)
    for code, df in list(daily_prices.items())[1:]:
        all_dates &= set(df.index)
    dates_index = pd.DatetimeIndex(sorted(all_dates))[-240:]  # 后 240 日
    log.info(f'date index: {len(dates_index)} dates ({dates_index[0]} to {dates_index[-1]})')
    
    # Compute forward returns
    fwd_rets = compute_forward_returns(daily_prices, dates_index, horizons)
    log.info(f'forward returns: {len(fwd_rets)} (code, date) pairs')
    
    # Industry lookup
    industry_df = load_industry_lookup(args.repo_root)
    industry_l1 = industry_df.set_index('code')['industry_l1'] if not industry_df.empty else pd.Series(dtype=str)
    log.info(f'industry lookup: {len(industry_l1)} stocks')
    
    # Validate
    main_results, by_year_results, by_industry_results = validate_all_factors(
        panel, fwd_rets, horizons, industry_l1,
    )
    log.info(f'main results: {len(main_results)} (factor, horizon) pairs')
    
    # Filter by user-selected factors
    if args.factors:
        wanted = set(args.factors.split(','))
        main_results = main_results[main_results['factor'].isin(wanted)]
        by_year_results = by_year_results[by_year_results['factor'].isin(wanted)]
        by_industry_results = by_industry_results[by_industry_results['factor'].isin(wanted)]
    
    # Write outputs
    write_main_csv(main_results, str(data_dir / 'factor_validation.csv'))
    write_by_year_csv(by_year_results, str(data_dir / 'factor_validation_by_year.csv'))
    write_by_industry_csv(by_industry_results, str(data_dir / 'factor_validation_by_industry.csv'))
    write_summary_text(
        main_results, by_year_results, by_industry_results,
        str(output_dir / 'dynsys_factor_validation_summary.txt'),
    )
    build_factor_validation_html(
        main_results, by_year_results, by_industry_results,
        str(output_dir / 'dynsys_factor_validation.html'),
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

### Step 2.3: 写 CLI smoke 测试

```python
# Append to tests/test_dynamics_eigen.py:

def test_cli_factor_validation_minimal(tmp_path):
    """CLI runs with required files, exits 0, writes 3 CSVs + 1 TXT."""
    import subprocess
    # 需要 kc_estimates.csv 存在 — 在 test fixture 里跳过如果缺失
    kc_path = Path('data/projection/kc_estimates.csv')
    if not kc_path.exists():
        pytest.skip('kc_estimates.csv not available — run parameter_fit.py first')
    
    out_dir = tmp_path / 'outputs'
    data_dir = tmp_path / 'data'
    result = subprocess.run(
        [
            sys.executable,
            'backtrace/dynamics/dynamics_factor_validation.py',
            '--limit', '50',
            '--horizons', '5,20',
            '--output-dir', str(out_dir),
            '--data-dir', str(data_dir),
            '--repo-root', '.',
        ],
        capture_output=True, text=True, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
    )
    assert result.returncode == 0, f'stderr: {result.stderr}'
    assert (data_dir / 'factor_validation.csv').exists()
    assert (data_dir / 'factor_validation_by_year.csv').exists()
    assert (data_dir / 'factor_validation_by_industry.csv').exists()
    assert (out_dir / 'dynsys_factor_validation_summary.txt').exists()
```

### Step 2.4: 跑 CLI smoke

```bash
cd /c/Users/yellow/mcp/qtTdx && \
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe \
    backtrace/dynamics/dynamics_factor_validation.py \
    --limit 50 --horizons 5,20 \
    --output-dir /tmp/v6_smoke --data-dir /tmp/v6_smoke_data
```

Expected: 4 files written, exit 0

### Step 2.5: 跑全测试

```bash
cd /c/Users/yellow/mcp/qtTdx && /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py -v
```

Expected: 79 tests PASS (78 old + 1 new CLI smoke; the 6 new factor_validation tests are part of the same file)

### Step 2.6: 更新 README

在 `backtrace/dynamics/README.md` §1 目录结构加 `dynamics_factor_validation.py` 行;§3 新增 §3.7"V6 因子验证";§6 新增"V6 输出 schema"。

### Step 2.7: Commit Task 2

```bash
git add backtrace/dynamics/dynamics_factor_validation.py tests/test_dynamics_eigen.py backtrace/dynamics/README.md
git commit -m "feat(dynamics): v6 — CLI + 输出层 (CSV / TXT / HTML)"
```

---

## Self-Review

**Spec coverage:**
- ✅ 数据加载 (load_*) — §3 数据来源 + §9.1
- ✅ 因子派生 (compute_*) — §4 因子列表 + §9.1
- ✅ IC + quantile 评估 — §6 评估方法 + §9.1
- ✅ Forward returns — §5
- ✅ 输出 schema — §7
- ✅ CLI — §8
- ✅ 决策标准 — §13

**Placeholder scan:** 无 TBD / TODO / "implement later"。

**Type consistency:** 所有 factor_name / horizon / status 字符串值在 main_results, by_year, by_industry 间一致。

**Risk:**
- run_in_background 推荐(全市场跑 5-10 分钟)
- daily data 文件名约定 `data/stocks/<code>.csv`,与已有 convention 一致
- scipy.stats.spearmanr 是 scipy 标准函数,无版本问题
- ttest_1samp 在 n=1 时会返回 NaN,已显式处理

**Out of scope (YAGNI):** walk-forward IC, factor correlation, industry-relative IC — 留 V6.1+。
