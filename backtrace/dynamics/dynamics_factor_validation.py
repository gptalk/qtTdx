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
# (Task 1 reviewer note: P was unused — dropped.)
from dynamics._dynamics_core import analyze_eigenvalues  # 复用,不动

__all__ = [
    'load_kc_estimates', 'load_oos_predictions_summary',
    'load_state_distribution', 'load_kc_time_series',
    'load_daily_prices', 'load_industry_lookup',
    'compute_eigen_factors', 'compute_kc_evolution_factors',
    'compute_cross_section_ic', 'compute_quantile_returns',
    'compute_forward_returns', 'build_factor_panel',
    'validate_all_factors',
    # Task 2 — output layer + CLI
    'write_main_csv', 'write_by_year_csv', 'write_by_industry_csv',
    'write_summary_text', 'build_factor_validation_html',
    'main',
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
    # kc_estimates.csv 历史上偶有重复 code 行(parameter_fit 多次跑产生),按 code 去重保留首行
    n_before = len(df)
    df = df.drop_duplicates(subset=['code'], keep='first')
    if len(df) < n_before:
        log.info(f'load_kc_estimates: dropped {n_before - len(df)} duplicate code rows')
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
    # v5.10 actual CSV uses 'dir_hit_rate' (not 'hit_rate'); alias for backward compat
    if 'hit_rate' not in df.columns and 'dir_hit_rate' in df.columns:
        df = df.rename(columns={'dir_hit_rate': 'hit_rate'})
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

    Path: {repo_root}/data/stocks/{code_with_underscore}_daily.csv
    (TQ local cache convention — see backtrace/common/data_store.py:csv_path)

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
        # convention from data_store.py:csv_path — '000001.SZ' -> '000001_SZ_daily.csv'
        fname = f"{str(code).replace('.', '_')}_daily.csv"
        f = stocks_dir / fname
        if not f.exists():
            log.warning(f'missing {fname} — skip')
            continue
        df = pd.read_csv(f, index_col=0, parse_dates=True).sort_index()
        if 'Close' not in df.columns:
            log.warning(f'{fname} missing Close — skip')
            continue
        out[code] = df[['Close']].rename(columns={'Close': 'close'})
    log.info(f'loaded daily prices for {len(out)}/{len(codes)} codes')
    return out


def load_industry_lookup(repo_root: str) -> pd.DataFrame:
    """Load industry mapping.

    Path: {repo_root}/data/sw2/members.csv → [sector_code, sector_name, member_code]
    Schema 实际用 `member_code`,不是 `code`。
    Falls back to stock_basic.csv if sw2 missing.

    Returns: DataFrame with [code, industry_l1, industry_l2]
    Returns empty DataFrame if both files missing.
    """
    repo_root = Path(repo_root)
    sw2 = repo_root / 'data' / 'sw2' / 'members.csv'
    basic = repo_root / 'data' / 'stock_basic.csv'
    df = pd.DataFrame()
    if sw2.exists():
        raw = pd.read_csv(sw2)
        if 'member_code' not in raw.columns:
            log.warning(f'sw2/members.csv missing member_code — columns: {list(raw.columns)}')
            return df
        raw['code'] = raw['member_code'].astype(str).str.zfill(6)
        raw = raw.rename(columns={'sector_name': 'industry_l2'})
        if 'sector_code' in raw.columns:
            # 申万二级 sector_code 形如 881002.SH;前 5 位近似 industry_l1
            raw['industry_l1'] = raw['sector_code'].astype(str).str[:5]
        df = raw[['code', 'industry_l1', 'industry_l2']].drop_duplicates('code')
    elif basic.exists():
        raw = pd.read_csv(basic)
        if 'code' not in raw.columns:
            log.warning(f'stock_basic.csv missing code — columns: {list(raw.columns)}')
            return df
        raw['code'] = raw['code'].astype(str).str.zfill(6)
        raw['industry_l1'] = 'unknown'
        raw['industry_l2'] = 'unknown'
        df = raw[['code', 'industry_l1', 'industry_l2']].drop_duplicates('code')
    else:
        log.warning('no industry lookup files found')
    return df.reset_index(drop=True) if not df.empty else df


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
    # 非数值因子(字符串 / 分类 / bool)→ spearmanr 会按字母序排,产生 garbage IC;
    # 显式返回 NaN — 镜像 compute_quantile_returns 的 guard (v6.0.1 IMPORTANT #10)
    if not pd.api.types.is_numeric_dtype(df['f']):
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
    NaN if n < n_quantiles * 2 OR factor is non-numeric (string/categorical).

    非数值因子(例如 `regime`:overdamped / critical / underdamped / anti_damped)
    返回全部 NaN — 分位数分析只对可排序的连续因子有意义。
    """
    df = pd.DataFrame({'f': factor_series, 'r': ret_series}).dropna()
    n = len(df)
    empty = {f'q{i}_ret': np.nan for i in range(1, n_quantiles + 1)} | {
        f'q{n_quantiles}_minus_q1': np.nan, 'n_obs': n,
    }
    if n < n_quantiles * 2:
        return empty
    # 非数值因子(字符串 / 分类 / bool)→ 不适合 qcut 分位,全部 NaN
    if not pd.api.types.is_numeric_dtype(df['f']):
        return empty
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

    Returns: DataFrame with MultiIndex (code, date) and columns [fwd_ret_{h}d for h in horizons].
    """
    if not daily_prices or len(dates_index) == 0:
        return pd.DataFrame()
    # 一次性给每只票算所有 horizons 的 forward return
    # 然后取交集日 → 避免 (code, date) 重复
    all_codes = []
    all_dates = []
    all_data = {}  # (code, h) -> Series indexed by date
    for code, df in daily_prices.items():
        df = df.sort_index()
        for h in horizons:
            future = df['close'].shift(-h) / df['close'] - 1.0
            all_data[(code, h)] = future
    if not all_data:
        return pd.DataFrame()
    # 交集日:所有 (code, h) 都覆盖的日期
    common_dates = set(dates_index)
    for s in all_data.values():
        common_dates &= set(s.dropna().index)
    common_dates = sorted(common_dates)
    if not common_dates:
        return pd.DataFrame()
    # 构 wide DataFrame: rows=(code, date), cols=fwd_ret_{h}d
    rows = []
    for code in {k[0] for k in all_data.keys()}:
        for d in common_dates:
            row = {'code': code, 'date': d}
            for h in horizons:
                val = all_data[(code, h)].get(d, np.nan)
                if not (val is None or (isinstance(val, float) and np.isnan(val))):
                    row[f'fwd_ret_{h}d'] = float(val)
            if any(f'fwd_ret_{h}d' in row for h in horizons):
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    df_out = pd.DataFrame(rows).set_index(['code', 'date']).sort_index()
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

            # by-industry — v6.0.1 BLOCKER #8 fix:
            # 旧实现把市场级 IC 按"行业内每只票"append 一次,导致 n_dates 虚高、
            # 且 per-industry IC 实际就是市场级 IC,毫无区分度。
            # 新实现:每个 industry 单独算 — restrict 到 industry 内股票,
            # 在每个 date 上算该 industry 子集的 Spearman IC,跨 date 求 mean。
            # 阈值 per spec §6.4:per (date, industry) ≥ 30 stocks 才算 IC,否则 skip 该 (date, industry)。
            ind_l1_dict = industry_l1.to_dict() if hasattr(industry_l1, 'to_dict') else {}
            # 用 codes_to_industry(industry_l1) 反向索引:industry → set of codes
            industry_to_codes: dict[str, set[str]] = {}
            for c, ind in ind_l1_dict.items():
                industry_to_codes.setdefault(ind, set()).add(c)
            industries_to_eval = set(industry_to_codes.keys()) - {'unknown'}
            if not industries_to_eval:
                industries_to_eval = {'unknown'}
            # 取 fwd_rets 的 date 列
            fwd_date_col = fwd_rets.index.get_level_values('date')
            unique_dates = pd.DatetimeIndex(sorted(fwd_date_col.unique()))
            for ind in sorted(industries_to_eval):
                ind_codes = industry_to_codes[ind]
                ind_ic_list = []
                ind_n_obs_total = 0
                for date_t in unique_dates:
                    if is_rolling:
                        # rolling factors: 只考虑 asof_date == date_t 的因子行
                        grp = fpanel[fpanel['asof_date'] == date_t]
                        if grp.empty:
                            continue
                        fac_s = grp.set_index('code')['factor_value']
                    else:
                        fac_s = fpanel.set_index('code')['factor_value']
                    try:
                        ret_s = fwd_rets.xs(date_t, level='date')[ret_col]
                    except (KeyError, ValueError):
                        continue
                    # restrict 到 industry 内股票
                    common_ind = fac_s.index.intersection(ret_s.index).intersection(ind_codes)
                    if len(common_ind) < 30:  # spec §6.4 threshold
                        continue
                    ic_ind, _, n_ind = compute_cross_section_ic(
                        fac_s.loc[common_ind], ret_s.loc[common_ind]
                    )
                    if not np.isnan(ic_ind):
                        ind_ic_list.append(ic_ind)
                    ind_n_obs_total += n_ind
                if not ind_ic_list:
                    # 没有任何 date 跨过 30 阈值 → emit insufficient_data 行
                    by_industry_rows.append({
                        'factor': fn, 'horizon': h, 'industry_l1': ind,
                        'n_obs': 0, 'n_dates': 0,
                        'ic_mean': np.nan, 'ic_std': np.nan,
                        'status': 'insufficient_data',
                    })
                    continue
                ind_ic_arr = np.array(ind_ic_list)
                ind_ic_mean = float(np.mean(ind_ic_arr))
                by_industry_rows.append({
                    'factor': fn, 'horizon': h, 'industry_l1': ind,
                    'n_obs': ind_n_obs_total,
                    'n_dates': len(ind_ic_list),
                    'ic_mean': ind_ic_mean,
                    'ic_std': float(np.std(ind_ic_arr, ddof=1)) if len(ind_ic_list) > 1 else np.nan,
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


# === 输出层 (Task 2 — Step 2.1) ===

import argparse

MAIN_COLS = [
    'factor', 'horizon', 'n_obs', 'n_dates',
    'ic_mean', 'ic_std', 'ic_ir', 'ic_pvalue',
    'q1_ret', 'q2_ret', 'q3_ret', 'q4_ret', 'q5_ret', 'q5_minus_q1',
    'top_year', 'top_year_ic', 'top_industry', 'top_industry_ic', 'status',
]

BY_YEAR_COLS = ['factor', 'horizon', 'year', 'n_obs', 'n_dates', 'ic_mean', 'ic_std', 'status']
BY_INDUSTRY_COLS = ['factor', 'horizon', 'industry_l1', 'n_obs', 'n_dates', 'ic_mean', 'ic_std', 'status']


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


# === CLI (Task 2 — Step 2.2) ===

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