# -*- coding: utf-8 -*-
# v0_2_d_decompose.py — V0.2-D OOS Reversal Decomposition CLI
#
# Spec: docs/superpowers/specs/2026-08-19-dynamics-oos-reversal-decomposition.md
#
# Usage:
#   PYTHONIOENCODING=utf-8 python backtrace/projection/v0_2_d_decompose.py \
#       --movement-dir data/projection --output-dir data/projection_v01_d --limit 0
import sys, os
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
PROJECT_ROOT = os.path.dirname(BACKTRACE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.stdout.reconfigure(encoding='utf-8')

import argparse
from projection.ablation_fit import (
    list_movement_csvs, write_ablation_csvs, summarize_ablation,
    build_panel5_html, compute_v0_2_d_distributions, write_v0_2_d_summary_txt,
    CSV_COLUMNS,
)


def parse_args():
    p = argparse.ArgumentParser(description='V0.2-D — OOS Reversal Decomposition')
    p.add_argument('--movement-dir', default='data/projection',
                   help='Directory containing movement_*.csv')
    p.add_argument('--output-dir', default='data/projection_v01_d',
                   help='Output directory for diagnostic CSV / HTML / TXT')
    p.add_argument('--limit', type=int, default=0,
                   help='Max stocks to process; 0 = all')
    p.add_argument(
        '--period', choices=['daily', '15m', '5m', '1m'], default='daily',
        help='缓存粒度(仅作审计/记录;不影响分析,读现有 movement CSV)',
    )
    return p.parse_args()


def main():
    args = parse_args()
    targets = list_movement_csvs(args.movement_dir)
    if args.limit > 0:
        targets = targets[:args.limit]
    print(f'输入: {args.movement_dir}/movement_*.csv')
    print(f'目标: {len(targets)} 只 (limit={args.limit})')

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: write 4-model ablation CSVs with full 36-col diagnostic schema
    write_ablation_csvs(targets, args.output_dir)

    # Step 2: regen V0.1 summary (with 3 ΔIC stats from Phase 0 audit fix)
    summary_df = summarize_ablation({m: os.path.join(args.output_dir, f'kc_estimates_model{m}.csv') for m in range(4)})
    summary_df.to_csv(os.path.join(args.output_dir, 'kc_ablation_summary.csv'),
                      encoding='utf-8')

    # Step 3: rename Model 2 CSV for downstream clarity
    src = os.path.join(args.output_dir, 'kc_estimates_model2.csv')
    dst = os.path.join(args.output_dir, 'kc_estimates_model2_diag.csv')
    if os.path.exists(src):
        os.rename(src, dst)

    # Step 4: Panel 5 (Model 2 only)
    panel5_path = build_panel5_html(dst, os.path.join(args.output_dir, 'panel5_drift_vs_collinearity.html'))

    # Step 5: distribution reports
    dist_df = compute_v0_2_d_distributions(dst)
    dist_df.to_csv(os.path.join(args.output_dir, 'v0_2_d_distributions.csv'),
                    index=False, encoding='utf-8')
    summary_txt = write_v0_2_d_summary_txt(dist_df, os.path.join(args.output_dir, 'v0_2_d_summary.txt'))

    print(f'Summary CSV: {args.output_dir}/kc_ablation_summary.csv')
    print(f'Model 2 diag: {dst}')
    print(f'Panel 5:      {panel5_path}')
    print(f'Distributions: {args.output_dir}/v0_2_d_distributions.csv')
    print(f'Summary TXT:   {summary_txt}')


if __name__ == '__main__':
    main()
