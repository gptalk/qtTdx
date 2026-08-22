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
