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


# === 派生层 ===

def compute_eigen_factors(kc_df: pd.DataFrame) -> pd.DataFrame:
    """从 (k̂, ĉ) 派生 eigenvalues / rho / theta / dist_to_unit / regime。

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
        if (n_quantiles - 1) in q_means.index and 0 in q_means.index:
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