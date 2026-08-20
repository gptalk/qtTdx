# -*- coding: utf-8 -*-
"""V0.2-E1 E2 — cross-sectional Delta|q_drift| x features analysis (5208 stocks).

Question: does the Market-driver improvement concentrate on a specific subset of stocks?

Reads:
  data/projection_v01_c1/c0_c1_paired_compare.csv     (paired C0/C1 metrics)
  data/projection_v01_c1/kc_estimates_model2_diag.csv (per-stock fitting details)
  data/stocks/{code}_daily.csv + data/indices/*       (via _e2_features helper)

Writes to data/projection_v01_e2/:
  cross_sectional.html                (2x2 scatter matrix)
  cross_sectional_correlations.csv    (7 Spearman rho)
  cross_sectional_regression.csv      (OLS coefficients, z-scored)
  quartile_summary.csv                (Q1-Q4 mean Delta|q_drift| per feature)
  _features_cache.csv                 (per-stock beta / vol / liquidity cache)

Pure diagnostic — no model changes.
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

# Allow `import _e2_features` when run from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _e2_features import extract_features_one  # noqa: E402


PAIRED_CSV = 'data/projection_v01_c1/c0_c1_paired_compare.csv'
KC_CSV = 'data/projection_v01_c1/kc_estimates_model2_diag.csv'
OUTPUT_DIR = 'data/projection_v01_e2'
CACHE_PATH = f'{OUTPUT_DIR}/_features_cache.csv'

# Features correlated against Delta|q_drift|
FEATURES = [
    'beta_market',
    'stock_volatility',
    'liquidity',
    'q_hat',
    'r2',
    'condition_number',
    'ic_real',
]
# Quartile analysis excludes ic_real (kept to 6 features -> 24 rows)
QUARTILE_FEATURES = [f for f in FEATURES if f != 'ic_real']
# 2x2 scatter matrix panels
SCATTER_FEATURES = ['beta_market', 'stock_volatility', 'liquidity', 'q_hat']


def extract_features_cached(codes: list[str], cache_path: str = CACHE_PATH) -> pd.DataFrame:
    """Cache-aware, fault-tolerant wrapper around `_e2_features.extract_features_one`.

    Same contract as `_e2_features.extract_features_all` (cache file format is
    identical), but a single bad stock cannot abort the whole 5208-code batch.
    A handful of stocks in `data/stocks/` contain NaN closes, which makes the
    helper's stock/market return series different lengths and raises inside
    `np.cov`; those codes are skipped here instead of propagating.
    """
    cached = pd.DataFrame()
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        cached['code'] = cached['code'].astype(str)
        have = set(cached['code'])
        missing = [c for c in codes if c not in have]
        if not missing:
            return cached[cached['code'].isin(codes)].reset_index(drop=True)
    else:
        missing = list(codes)

    rows, n_fail = [], 0
    for code in missing:
        try:
            feats = extract_features_one(code)
        except Exception:
            feats = None
            n_fail += 1
        if feats is not None:
            rows.append(feats)
    if n_fail:
        print(f'  (skipped {n_fail} codes that raised during feature extraction)')

    new = pd.DataFrame(rows)
    combined = pd.concat([cached, new], ignore_index=True) if len(cached) else new
    if combined.empty:
        return combined
    combined['code'] = combined['code'].astype(str)
    combined = combined.drop_duplicates('code')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    combined.to_csv(cache_path, index=False, encoding='utf-8')
    return combined[combined['code'].isin(codes)].reset_index(drop=True)


def build_dataset() -> pd.DataFrame:
    """Merge paired metrics + daily-derived features + kc fitting diagnostics."""
    for path in (PAIRED_CSV, KC_CSV):
        if not os.path.exists(path):
            sys.exit(f'MISSING: {path} — run v0_2_c1_market_swap.py first')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    paired = pd.read_csv(PAIRED_CSV)
    paired['code'] = paired['code'].astype(str)
    print(f'Loaded {len(paired)} stocks from {PAIRED_CSV}')

    # Target: Delta|q_drift| = |q_drift_C1| - |q_drift_C0|  (negative = attenuated = better)
    paired['delta_abs_q_drift'] = (
        paired['q_drift_C1'].abs() - paired['q_drift_C0'].abs()
    )

    codes = paired['code'].tolist()
    print(f'Extracting per-stock features for {len(codes)} codes (cache: {CACHE_PATH}) ...')
    feats = extract_features_cached(codes, cache_path=CACHE_PATH)
    if feats.empty:
        sys.exit('No features extracted — check data/stocks and data/indices')
    feats['code'] = feats['code'].astype(str)
    print(f'  -> {len(feats)} stocks with valid daily-derived features')

    kc = pd.read_csv(KC_CSV)
    kc['code'] = kc['code'].astype(str)
    kc_cols = ['code', 'q_hat', 'r2', 'condition_number', 'ic_real']
    kc = kc[[c for c in kc_cols if c in kc.columns]].drop_duplicates('code')

    df = paired.merge(feats, on='code', how='inner').merge(kc, on='code', how='left')
    print(f'  -> {len(df)} stocks after merging paired + features + kc estimates')
    return df


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in FEATURES:
        if feat not in df.columns:
            rows.append({'feature': feat, 'spearman_rho': np.nan,
                         'p_value': np.nan, 'n': 0})
            continue
        sub = df[[feat, 'delta_abs_q_drift']].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 3:
            rows.append({'feature': feat, 'spearman_rho': np.nan,
                         'p_value': np.nan, 'n': len(sub)})
            continue
        rho, pval = spearmanr(sub[feat], sub['delta_abs_q_drift'])
        rows.append({'feature': feat, 'spearman_rho': float(rho),
                     'p_value': float(pval), 'n': len(sub)})
    return pd.DataFrame(rows)


def compute_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in QUARTILE_FEATURES:
        if feat not in df.columns:
            continue
        sub = df[[feat, 'delta_abs_q_drift']].replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            continue
        try:
            sub['quartile'] = pd.qcut(
                sub[feat], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop'
            )
        except ValueError:
            continue
        grouped = sub.groupby('quartile', observed=False)
        for q, g in grouped:
            if len(g) == 0:
                rows.append({'feature': feat, 'quartile': str(q), 'n': 0,
                             'mean_delta_abs_q_drift': np.nan,
                             'median_delta_abs_q_drift': np.nan,
                             'feature_mean': np.nan})
                continue
            rows.append({
                'feature': feat,
                'quartile': str(q),
                'n': int(len(g)),
                'mean_delta_abs_q_drift': float(g['delta_abs_q_drift'].mean()),
                'median_delta_abs_q_drift': float(g['delta_abs_q_drift'].median()),
                'feature_mean': float(g[feat].mean()),
            })
    return pd.DataFrame(rows)


def compute_regression(df: pd.DataFrame) -> pd.DataFrame:
    """Multivariate OLS on z-scored features (comparable coefficients)."""
    cols = [f for f in FEATURES if f in df.columns]
    sub = df[cols + ['delta_abs_q_drift']].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < len(cols) + 2:
        return pd.DataFrame({'feature': cols, 'coef_zscored': [np.nan] * len(cols),
                             'n': [len(sub)] * len(cols)})

    X = sub[cols].to_numpy(dtype=float)
    y = sub['delta_abs_q_drift'].to_numpy(dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Xz = (X - mu) / sd
    Xd = np.column_stack([np.ones(len(Xz)), Xz])

    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan

    return pd.DataFrame({
        'feature': cols,
        'coef_zscored': coef[1:],
        'intercept': coef[0],
        'model_r2': r2,
        'n': len(sub),
    })


def build_html(df: pd.DataFrame, corr: pd.DataFrame, path: str) -> None:
    rho_map = dict(zip(corr['feature'], corr['spearman_rho']))
    feats = [f for f in SCATTER_FEATURES if f in df.columns]
    titles = [f'{f} (rho={rho_map.get(f, float("nan")):+.3f})' for f in feats]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles)

    for i, feat in enumerate(feats):
        r, c = i // 2 + 1, i % 2 + 1
        sub = df[[feat, 'delta_abs_q_drift']].replace([np.inf, -np.inf], np.nan).dropna()
        fig.add_trace(
            go.Scattergl(
                x=sub[feat], y=sub['delta_abs_q_drift'], mode='markers',
                marker=dict(size=3, color='steelblue', opacity=0.4),
                name=feat, showlegend=False,
            ),
            row=r, col=c,
        )
        fig.add_hline(y=0, line_dash='dash', line_color='red', row=r, col=c)
        fig.update_xaxes(title_text=feat, row=r, col=c)
        fig.update_yaxes(title_text='Δ|q_drift|', row=r, col=c)

    fig.update_layout(
        height=850,
        title=f'V0.2-E1 E2: Δ|q_drift| (C1-C0) vs cross-sectional features — {len(df)} stocks',
    )
    fig.write_html(path, include_plotlyjs='cdn')


def main():
    df = build_dataset()

    corr = compute_correlations(df)
    quart = compute_quartiles(df)
    reg = compute_regression(df)

    corr.to_csv(f'{OUTPUT_DIR}/cross_sectional_correlations.csv',
                index=False, encoding='utf-8')
    reg.to_csv(f'{OUTPUT_DIR}/cross_sectional_regression.csv',
               index=False, encoding='utf-8')
    quart.to_csv(f'{OUTPUT_DIR}/quartile_summary.csv',
                 index=False, encoding='utf-8')
    build_html(df, corr, f'{OUTPUT_DIR}/cross_sectional.html')

    print('\n=== E2 Spearman rho (Delta|q_drift| vs feature) ===')
    for _, row in corr.iterrows():
        print(f'  {row["feature"]:20s}: rho={row["spearman_rho"]:+.4f}  '
              f'p={row["p_value"]:.3e}  n={int(row["n"])}')

    print('\n=== E2 Quartile mean Delta|q_drift| ===')
    if not quart.empty:
        pivot = quart.pivot(index='feature', columns='quartile',
                            values='mean_delta_abs_q_drift')
        print(pivot.to_string(float_format=lambda v: f'{v:+.4f}'))

    print('\n=== E2 OLS (z-scored) ===')
    for _, row in reg.iterrows():
        print(f'  {row["feature"]:20s}: coef={row["coef_zscored"]:+.4f}')
    if 'model_r2' in reg.columns and len(reg):
        print(f'  model_r2 = {reg["model_r2"].iloc[0]:.4f}  n={int(reg["n"].iloc[0])}')

    print(f'\nOutputs: {OUTPUT_DIR}/cross_sectional.html')
    print(f'         {OUTPUT_DIR}/cross_sectional_correlations.csv')
    print(f'         {OUTPUT_DIR}/cross_sectional_regression.csv')
    print(f'         {OUTPUT_DIR}/quartile_summary.csv')
    print(f'         {CACHE_PATH}')


if __name__ == '__main__':
    main()
