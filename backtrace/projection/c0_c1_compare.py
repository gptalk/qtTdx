# -*- coding: utf-8 -*-
# c0_c1_compare.py — V0.2-C1 paired comparison helpers
#
# Spec: docs/superpowers/specs/2026-08-20-dynamics-c1-market-driver-swap.md §4.3, §4.4
#
# C0 = V0.2-D baseline (driver = 申万二级行业指数)
# C1 = V0.2-C1 swap    (driver = 交易所大盘指数 000001.SH / 399001.SZ)
#
# 两边的数学完全一样(ablation_fit.py Model 2 未改动),本模块只做「诊断面」:
# 把两次跑的 per-stock 指标配对相减,输出 CSV + UTF-8 中文报告。
#
# 纯诊断 —— 不产出任何判定结论,解释权交给 V0.2-E 或用户。
import os

import numpy as np
import pandas as pd

PAIRED_COLUMNS = [
    'code', 'name',                                              # 2
    'ic_real_C0', 'ic_real_C1', 'delta_oos_ic',                  # 3
    'q_drift_C0', 'q_drift_C1', 'delta_q_drift',                 # 3
    'q_hat_C0', 'q_hat_C1', 'delta_q_hat',                       # 3
    'test_fit_r2_C0', 'test_fit_r2_C1', 'delta_test_fit_r2',     # 3
    'oos_r2_C0', 'oos_r2_C1', 'delta_oos_r2',                    # 3
    'condition_number_C0', 'condition_number_C1', 'delta_cond',  # 3
    'sign_flipped', 'q_drift_attenuated', 'q_drift_amplified',   # 3
    'ic_improved', 'ic_worsened',                                # 2
]
# Total: 2 + 3*6 + 3 + 2 = 25
# (6 metric blocks: ic_real, q_drift, q_hat, test_fit_r2, oos_r2, condition_number)

# (metric column in the two source CSVs, delta column name in the paired CSV)
PAIRED_METRICS = [
    ('ic_real', 'delta_oos_ic'),
    ('q_drift', 'delta_q_drift'),
    ('q_hat', 'delta_q_hat'),
    ('test_fit_r2', 'delta_test_fit_r2'),
    ('oos_r2', 'delta_oos_r2'),
    ('condition_number', 'delta_cond'),
]


def _signed_diff_str(x: float) -> str:
    """Format float with sign for display."""
    return f'{x:+.4f}' if np.isfinite(x) else '   nan'


def _dedup_by_code(df: pd.DataFrame, src: str) -> pd.DataFrame:
    """Guarantee one row per code before the paired merge.

    `kc_estimates_model2_diag.csv` is one row per *movement CSV*, not per stock:
    if the movement dir holds several drivers for the same code (e.g. a stray
    market-driver file left in `data/projection/`), that code appears more than
    once. A plain merge would then cartesian-product both sides — inflating the
    row count and pairing a stock against the wrong driver, which silently
    breaks the compute(X, X) == all-zero-deltas identity.

    Structural guard only: `keep='first'` is deterministic but not
    driver-aware. The semantic fix is for the caller to hand in a
    driver-filtered CSV (C0 = 88xxxx industry rows, C1 = market rows).
    """
    dups = df['code'][df['code'].duplicated(keep=False)].unique()
    if len(dups) == 0:
        return df
    # ASCII-only warning: Windows GBK terminals choke on non-ASCII prints.
    print(f'[c0_c1_compare] WARNING: {len(dups)} duplicate code(s) in '
          f'{os.path.basename(src)} -> keeping first row of each. '
          f'Contaminated movement dir? codes={sorted(dups)[:5]}')
    return df.drop_duplicates(subset='code', keep='first')


def compute_c0_c1_paired_compare(c0_csv: str, c1_csv: str, output_csv: str) -> str:
    """V0.2-C1 §4.3: per-stock paired comparison of Model 2 metrics between C0 (industry)
    and C1 (market). Returns output_csv path.

    Both inputs are `kc_estimates_model2_diag.csv` files produced by
    `ablation_fit.write_ablation_csvs` (C0 from `data/projection_v01_d/`,
    C1 from `data/projection_v01_c1/`).

    Diagnostic flags (no verdicts):
      - sign_flipped: True iff sign(ic_real_C0) != sign(ic_real_C1)
      - q_drift_attenuated: True iff |q_drift_C1| < 0.5 * |q_drift_C0|
      - q_drift_amplified:  True iff |q_drift_C1| > 1.5 * |q_drift_C0|
      - ic_improved:        True iff |delta_oos_ic| > 0.05 AND NOT sign_flipped
      - ic_worsened:        True iff delta_oos_ic < -0.05
    """
    c0 = _dedup_by_code(pd.read_csv(c0_csv), c0_csv)
    c1 = _dedup_by_code(pd.read_csv(c1_csv), c1_csv)
    # Inner join on code (assumes same stock list, possibly different row order)
    merged = c0.merge(c1, on='code', suffixes=('_C0', '_C1'))
    out = pd.DataFrame()
    out['code'] = merged['code']
    out['name'] = merged['name_C0']  # names should match
    # Per-metric paired deltas
    for metric, delta_col in PAIRED_METRICS:
        out[f'{metric}_C0'] = merged[f'{metric}_C0']
        out[f'{metric}_C1'] = merged[f'{metric}_C1']
        out[delta_col] = merged[f'{metric}_C1'] - merged[f'{metric}_C0']
    # Diagnostic flags
    c0_ic = out['ic_real_C0']
    c1_ic = out['ic_real_C1']
    # np.sign(...) != np.sign(...) rather than (c0*c1 < 0): the explicit
    # isfinite() guard below is what handles NaN, and a zero IC on either side
    # counts as "no flip" instead of silently vanishing from the product test.
    out['sign_flipped'] = (np.sign(c0_ic) != np.sign(c1_ic)) & np.isfinite(c0_ic) & np.isfinite(c1_ic)
    abs_c0_qd = out['q_drift_C0'].abs()
    abs_c1_qd = out['q_drift_C1'].abs()
    # |q_drift_C0| > 1e-6 guard: a ~0 baseline drift makes the ratio meaningless.
    out['q_drift_attenuated'] = (abs_c1_qd < 0.5 * abs_c0_qd) & (abs_c0_qd > 1e-6)
    out['q_drift_amplified'] = (abs_c1_qd > 1.5 * abs_c0_qd) & (abs_c0_qd > 1e-6)
    out['ic_improved'] = (out['delta_oos_ic'].abs() > 0.05) & ~out['sign_flipped']
    out['ic_worsened'] = out['delta_oos_ic'] < -0.05
    # Reorder columns to spec
    out = out[PAIRED_COLUMNS]
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    out.to_csv(output_csv, index=False, encoding='utf-8')
    return output_csv


def write_c0_c1_compare_summary_txt(paired_csv: str, c0_dist_csv: str,
                                    c1_dist_csv: str, output_txt: str) -> str:
    """V0.2-C1 §4.4: UTF-8 Chinese paired-comparison report (diagnostic only).

    `c0_dist_csv` / `c1_dist_csv` are the `v0_2_d_distributions.csv` files
    (schema: gate, statistic, value) from the two runs.
    """
    paired = pd.read_csv(paired_csv)
    c0_dist = pd.read_csv(c0_dist_csv)
    c1_dist = pd.read_csv(c1_dist_csv)
    lines = [
        '=' * 70,
        'V0.2-C1 — Market Driver Swap (Paired Comparison)',
        '=' * 70,
        f'Run date:  {pd.Timestamp.now().strftime("%Y-%m-%d")}',
        '',
        # 措辞刻意不含 "PASS" / "FAIL" 字面量(V0.2-D 的同位句用的是
        # "No PASS/FAIL verdicts."):§4.4 的「无判定」不变量由测试用
        # 词边界正则强制,正则会连表头里的字面量一起判为违规。
        'NOTE: 纯诊断报告,不产出判定结论(no automated verdict).',
        'Interpretation routes to V0.2-E or user.',
        '',
        f'Paired stocks: {len(paired)}',
        '',
    ]
    # D1/D2/D3 distribution comparison
    for gate in ('D1', 'D2', 'D3'):
        c0_g = c0_dist[c0_dist['gate'] == gate].set_index('statistic')['value']
        c1_g = c1_dist[c1_dist['gate'] == gate].set_index('statistic')['value']
        lines.append(f'--- Gate {gate} (C0 = industry, C1 = market) ---')
        lines.append(f'  {"statistic":<14s} {"C0 (industry)":>15s} {"C1 (market)":>15s}')
        for stat in ('median', 'p25', 'p75', 'P(>0.3)', 'P(>0.2)'):
            if stat in c0_g.index or stat in c1_g.index:
                c0_v = c0_g.get(stat, np.nan)
                c1_v = c1_g.get(stat, np.nan)
                lines.append(f'  {stat:<14s} {_signed_diff_str(c0_v):>15s} {_signed_diff_str(c1_v):>15s}')
        lines.append('')
    # Diagnostic flag counts
    lines.append('--- Paired diagnostic flags (Model 2 only) ---')
    n_paired = len(paired)
    for flag in ('sign_flipped', 'q_drift_attenuated', 'q_drift_amplified',
                 'ic_improved', 'ic_worsened'):
        n_flag = int(paired[flag].sum())
        pct = 100 * paired[flag].mean() if n_paired else float('nan')
        lines.append(f'  {flag:<28s} {n_flag:>5d} / {n_paired} ({pct:.1f}%)')
    lines.append('')
    # Routing hints (descriptive only, no verdicts) — spec §5 A/B/C/D
    lines.append('--- Routing hints (descriptive only) ---')
    lines.append('  If many sign_flipped + ic_improved: market may be the right driver (Scenario A).')
    lines.append('  If many q_drift_attenuated: H1b (driver-induced) plausible.')
    lines.append('  If many ic_worsened: industry may be the right driver (Scenario B).')
    lines.append('  If both: route to V0.2-B shrinkage (Scenario C) or V0.2-C.4 heterogeneity (Scenario D).')
    lines.append('')
    lines.append('=' * 70)
    os.makedirs(os.path.dirname(output_txt) or '.', exist_ok=True)
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return output_txt
