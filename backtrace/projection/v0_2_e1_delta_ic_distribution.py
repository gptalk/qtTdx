# -*- coding: utf-8 -*-
"""V0.2-E1 E1 — ΔIC distribution analysis (Market vs Industry, 5208 stocks).

Reads `data/projection_v01_c1/c0_c1_paired_compare.csv`, computes summary stats +
distribution buckets for `delta_oos_ic`, writes HTML + CSV to `data/projection_v01_e1/`.

纯诊断; — no model changes.
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def main():
    paired_csv = 'data/projection_v01_c1/c0_c1_paired_compare.csv'
    output_dir = 'data/projection_v01_e1'

    if not os.path.exists(paired_csv):
        sys.exit(f'MISSING: {paired_csv} — run v0_2_c1_market_swap.py first')

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(paired_csv)
    n_total = len(df)
    print(f'Loaded {n_total} stocks from {paired_csv}')

    # ΔIC = ic_real_C1 - ic_real_C0 (already column delta_oos_ic)
    df['delta_ic'] = df['delta_oos_ic']

    # Summary stats
    summary = {
        'n': n_total,
        'mean': float(df['delta_ic'].mean()),
        'median': float(df['delta_ic'].median()),
        'std': float(df['delta_ic'].std()),
        'p5': float(df['delta_ic'].quantile(0.05)),
        'p10': float(df['delta_ic'].quantile(0.10)),
        'p25': float(df['delta_ic'].quantile(0.25)),
        'p75': float(df['delta_ic'].quantile(0.75)),
        'p90': float(df['delta_ic'].quantile(0.90)),
        'p95': float(df['delta_ic'].quantile(0.95)),
        'min': float(df['delta_ic'].min()),
        'max': float(df['delta_ic'].max()),
        'sign_test_p_gt_0': float((df['delta_ic'] > 0).mean()),
        'sign_test_p_gt_0.05': float((df['delta_ic'] > 0.05).mean()),
        'large_movers_pct': float((df['delta_ic'].abs() > 0.1).mean() * 100),
        'very_negative_pct': float((df['delta_ic'] < -0.1).mean() * 100),
        'very_positive_pct': float((df['delta_ic'] > 0.1).mean() * 100),
    }

    # Buckets
    bins = [-np.inf, -0.1, -0.05, 0, 0.05, 0.1, np.inf]
    labels = ['(-∞,-0.1]', '(-0.1,-0.05]', '(-0.05,0]',
              '(0,0.05]', '(0.05,0.1]', '(0.1,∞)']
    df['bucket'] = pd.cut(df['delta_ic'], bins=bins, labels=labels, right=True)
    bucket_counts = df['bucket'].value_counts().reindex(labels, fill_value=0)

    # Write CSV outputs
    summary_df = pd.DataFrame([summary]).T.rename(columns={0: 'value'})
    summary_df.index.name = 'metric'
    summary_df.to_csv(f'{output_dir}/delta_ic_summary.csv', encoding='utf-8')

    bucket_df = pd.DataFrame({'bucket': labels, 'count': bucket_counts.values})
    bucket_df['pct'] = bucket_df['count'] / n_total * 100
    bucket_df.to_csv(f'{output_dir}/delta_ic_buckets.csv', index=False, encoding='utf-8')

    # HTML: histogram + summary table + bucket table
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['delta_ic'], nbinsx=50, name='ΔIC',
        marker_color='steelblue', opacity=0.75,
    ))
    fig.add_vline(x=0, line_dash='dash', line_color='red',
                  annotation_text='ΔIC=0', annotation_position='top right')
    fig.add_vline(x=summary['mean'], line_dash='dot', line_color='green',
                  annotation_text=f"mean={summary['mean']:.3f}",
                  annotation_position='top left')
    fig.update_layout(
        title=f'V0.2-E1: ΔIC = IC_C1 - IC_C0 (Market - Industry) — {n_total} stocks',
        xaxis_title='ΔIC', yaxis_title='count', bargap=0.05,
    )
    fig.write_html(f'{output_dir}/delta_ic_distribution.html', include_plotlyjs='cdn')

    # Print summary
    print('\n=== E1 ΔIC Distribution Summary ===')
    for k, v in summary.items():
        if isinstance(v, float):
            print(f'  {k:25s}: {v:+.4f}')
        else:
            print(f'  {k:25s}: {v}')
    print('\n=== E1 ΔIC Buckets ===')
    for _, row in bucket_df.iterrows():
        print(f'  {row["bucket"]:15s}: {int(row["count"]):4d} stocks ({row["pct"]:.1f}%)')
    print(f'\nOutputs: {output_dir}/delta_ic_distribution.html')
    print(f'         {output_dir}/delta_ic_summary.csv')
    print(f'         {output_dir}/delta_ic_buckets.csv')


if __name__ == '__main__':
    main()