# -*- coding: utf-8 -*-
# v0_2_c1_market_swap.py — V0.2-C1 Market Driver Swap CLI orchestrator
#
# Spec: docs/superpowers/specs/2026-08-20-dynamics-c1-market-driver-swap.md
#
# Usage:
#   PYTHONIOENCODING=utf-8 python backtrace/projection/v0_2_c1_market_swap.py
#
# Pipeline:
#   1. Filter stock_basic.csv to SH / SZ subsets
#   2. Run projection_batch.py --index 000001.SH (SH stocks)
#   3. Run projection_batch.py --index 399001.SZ (SZ stocks)
#   4. Run v0_2_d_decompose.py on market-driver dir
#   5. compute_c0_c1_paired_compare + write_c0_c1_compare_summary_txt
#
# 纯诊断 —— 不产出任何 PASS/FAIL 判定,解释权交给 V0.2-E 或用户。
import sys, os, subprocess
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
PROJECT_ROOT = os.path.dirname(BACKTRACE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import pandas as pd
from projection.c0_c1_compare import (
    compute_c0_c1_paired_compare,
    write_c0_c1_compare_summary_txt,
)

# C1 的两个大盘基线。ablation_fit.list_movement_csvs() 用个股代码前缀推 index_code
# 后缀,而 C1 是「按交易所拆分」跑的(SH 票→000001,SZ 票→399001),所以推出来的
# 后缀恰好正确 —— 这两个字面量就是真实 C1 CSV 里会出现的 index_code。
MARKET_INDEX_CODES = ['000001.SH', '399001.SZ']
# C0 = 申万二级行业指数,代码形如 880xxx / 881xxx。
INDUSTRY_INDEX_PREFIX = '88'

# v0_2_d_distributions.csv 需要的列(ablation_fit.compute_v0_2_d_distributions)。
# 预填的 C1 CSV(--skip-ablation)可能缺 Group C/D 列 → 见 _safe_distributions。
DIST_SCHEMA = ['gate', 'statistic', 'value']


def parse_args():
    p = argparse.ArgumentParser(description='V0.2-C1 — Market Driver Swap')
    p.add_argument('--input', default=os.path.join(PROJECT_ROOT, 'data', 'stock_basic.csv'),
                   help='stock_basic.csv 路径。默认 data/stock_basic.csv')
    p.add_argument('--market-dir', default=os.path.join(PROJECT_ROOT, 'data', 'projection_market'),
                   help='market-driver movement 文件输出目录。默认 data/projection_market/')
    p.add_argument('--c0-dir', default=os.path.join(PROJECT_ROOT, 'data', 'projection_v01_d'),
                   help='C0 (V0.2-D industry) 输出目录。默认 data/projection_v01_d/')
    p.add_argument('--c1-output-dir', default=os.path.join(PROJECT_ROOT, 'data', 'projection_v01_c1'),
                   help='C1 输出目录。默认 data/projection_v01_c1/')
    p.add_argument('--limit', type=int, default=0,
                   help='最多处理多少只;0 = 全部。默认 0')
    p.add_argument('--days', type=int, default=240,
                   help='回看天数。默认 240')
    p.add_argument('--skip-data-gen', action='store_true',
                   help='跳过 movement 文件生成(只跑 ablation + paired compare)')
    p.add_argument('--skip-ablation', action='store_true',
                   help='跳过 v0_2_d_decompose 调用(只跑 paired compare;需要 C1 CSV 已存在)')
    return p.parse_args()


def filter_stocks(input_csv: str, exchange: str, output_csv: str) -> int:
    """Filter stock_basic.csv to exchange ('SH' or 'SZ') subset, write to output_csv."""
    df = pd.read_csv(input_csv, dtype={'code': str})
    df = df[df['market'] == exchange]
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df[['code']].to_csv(output_csv, index=False, encoding='utf-8')
    return len(df)


def run_subprocess(cmd: list, timeout: int = 600) -> int:
    """Run subprocess with UTF-8 env, return exit code.

    encoding='utf-8' on the pipe (not just PYTHONIOENCODING in the child env):
    text=True would otherwise decode the child's UTF-8 Chinese output with the
    Windows locale codec (cp936) and raise UnicodeDecodeError.
    """
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    print(f'>> {" ".join(cmd)}', flush=True)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=timeout)
    if result.returncode != 0:
        print(f'STDOUT: {result.stdout}', flush=True)
        print(f'STDERR: {result.stderr}', flush=True)
    return result.returncode


def _prune_empty_movement_files(market_dir: str) -> int:
    """Delete empty (0-row) movement_*.csv files in market_dir.

    Some newly-listed stocks (e.g. 688826.SH) have a daily_CSV with 1 valid row
    surrounded by NaN, so projection_batch.py's --movement step produces an
    empty file (header only). Those crash ablation_fit.fit_one_split at
    np.isfinite(object dtype). Prune them so v0_2_d_decompose.py never sees them.

    Returns the number of files pruned. The numeric output dir is not data:
    the empty file is regenerated next time projection_batch.py runs and is
    pruned again. No real data is lost.
    """
    if not os.path.isdir(market_dir):
        return 0
    n_pruned = 0
    for fn in os.listdir(market_dir):
        if not (fn.startswith('movement_') and fn.endswith('.csv')):
            continue
        path = os.path.join(market_dir, fn)
        try:
            n_lines = sum(1 for _ in open(path, 'rb')) - 1  # minus header
        except OSError:
            continue
        if n_lines <= 0:
            os.remove(path)
            n_pruned += 1
    return n_pruned


def _safe_distributions(model2_csv: str):
    """compute_v0_2_d_distributions() with a missing-column guard.

    D2/D3 read `corr_x_beta_d` / `corr_F_d` — Group C/D columns that only the
    real V0.2-D pipeline writes. With --skip-ablation the C1 CSV may be
    pre-populated (CI smoke, hand-made panel) and lack them; a KeyError there
    would kill the whole run at the last step, after the paired CSV is already
    computable. Degrade to an empty (schema-valid) distribution instead —
    write_c0_c1_compare_summary_txt() renders the D1/D2/D3 headers with no
    rows underneath, which is the honest reading of "not available".
    """
    from projection.ablation_fit import compute_v0_2_d_distributions
    try:
        return compute_v0_2_d_distributions(model2_csv)
    except KeyError as e:
        # ASCII-only warning: Windows GBK terminals choke on non-ASCII prints.
        print(f'[v0_2_c1] WARNING: cannot compute D1/D2/D3 distributions from '
              f'{os.path.basename(model2_csv)} (missing column {e}); '
              f'writing empty distribution.', flush=True)
        return pd.DataFrame(columns=DIST_SCHEMA)


def main():
    args = parse_args()
    print(f'输入: {args.input}')
    print(f'Market 输出目录: {args.market_dir}')
    print(f'C0 (industry) 目录: {args.c0_dir}')
    print(f'C1 输出目录: {args.c1_output_dir}')
    print(f'Limit: {args.limit} (0=全部), Days: {args.days}')

    # Step 1: Filter stocks
    if not args.skip_data_gen:
        sh_csv = os.path.join(args.market_dir, '_stocks_sh.csv')
        sz_csv = os.path.join(args.market_dir, '_stocks_sz.csv')
        n_sh = filter_stocks(args.input, 'SH', sh_csv)
        n_sz = filter_stocks(args.input, 'SZ', sz_csv)
        print(f'SH stocks: {n_sh}, SZ stocks: {n_sz}')

        # Step 2-3: Generate market-driver movement files
        # V0.2-F: --index dropped — projection_batch.py 默认 per-exchange market 已覆盖
        # SH→000001.SH / SZ→399001.SZ(原 orchestrator 显式传 --index 已 redundant)
        for label, stocks_csv in [('SH→000001', sh_csv),
                                  ('SZ→399001', sz_csv)]:
            cmd = [
                sys.executable,
                os.path.join(BACKTRACE_DIR, 'projection', 'projection_batch.py'),
                '--input', stocks_csv,
                '--output-dir', args.market_dir,
                '--movement',
                '--days', str(args.days),
            ]
            if args.limit > 0:
                cmd += ['--limit', str(args.limit)]
            rc = run_subprocess(cmd, timeout=1800)
            if rc != 0:
                sys.exit(rc)
            print(f'{label}: 完成')
    else:
        print('Movement 生成: 跳过(--skip-data-gen)')

    # Step 4: Run V0.2-D pipeline on market-driver dir
    if not args.skip_ablation:
        # Prune empty (0-row) movement files first. Newly-listed stocks (e.g.
        # 688826.SH) produce header-only files that crash fit_one_split at
        # np.isfinite(object dtype). Deleted here so v0_2_d_decompose.py never
        # sees them; the file is regenerated by projection_batch.py next run.
        n_pruned = _prune_empty_movement_files(args.market_dir)
        if n_pruned > 0:
            print(f'Pruned {n_pruned} empty movement file(s) from {args.market_dir}')
        cmd_ablation = [
            sys.executable,
            os.path.join(BACKTRACE_DIR, 'projection', 'v0_2_d_decompose.py'),
            '--movement-dir', args.market_dir,
            '--output-dir', args.c1_output_dir,
        ]
        if args.limit > 0:
            cmd_ablation += ['--limit', str(args.limit)]
        rc = run_subprocess(cmd_ablation, timeout=3600)
        if rc != 0:
            sys.exit(rc)
        print('C1 ablation: 完成')
    else:
        print('C1 ablation: 跳过(--skip-ablation)')

    # Step 5: Paired compare
    c0_csv = os.path.join(args.c0_dir, 'kc_estimates_model2_diag.csv')
    c1_csv = os.path.join(args.c1_output_dir, 'kc_estimates_model2_diag.csv')
    c0_dist = os.path.join(args.c0_dir, 'v0_2_d_distributions.csv')
    c1_dist = os.path.join(args.c1_output_dir, 'v0_2_d_distributions.csv')
    paired_path = os.path.join(args.c1_output_dir, 'c0_c1_paired_compare.csv')
    summary_path = os.path.join(args.c1_output_dir, 'c0_c1_compare_summary.txt')

    if not os.path.exists(c0_csv):
        print(f'ERROR: C0 not found at {c0_csv}; run V0.2-D first.', file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(c1_csv):
        print(f'ERROR: C1 not found at {c1_csv}', file=sys.stderr)
        sys.exit(1)

    # V0.2-C1 Task 2 concern 2 — driver-aware filtering before merge.
    # Real V0.2-D CSV may contain stray market-driver rows (data/projection_v01_d/
    # currently holds 2 × 000001.SH + 1 × 399001.SZ out of 5211). Structural dedup
    # in compute_c0_c1_paired_compare is NOT driver-aware, so we filter here by
    # index_code: C0 keeps 88xxxx industry rows; C1 keeps market index rows
    # (000001.SH / 399001.SZ).
    c0_filtered_csv = os.path.join(args.c0_dir, 'kc_estimates_model2_diag_filtered.csv')
    c1_filtered_csv = os.path.join(args.c1_output_dir, 'kc_estimates_model2_diag_filtered.csv')
    c0_df = pd.read_csv(c0_csv, dtype={'code': str})
    c1_df = pd.read_csv(c1_csv, dtype={'code': str})
    n_c0_before, n_c1_before = len(c0_df), len(c1_df)
    if 'index_code' in c0_df.columns:
        c0_df = c0_df[c0_df['index_code'].astype(str).str.startswith(INDUSTRY_INDEX_PREFIX)].copy()
    if 'index_code' in c1_df.columns:
        c1_df = c1_df[c1_df['index_code'].isin(MARKET_INDEX_CODES)].copy()
    n_c0_after, n_c1_after = len(c0_df), len(c1_df)
    print(f'C0 filter: {n_c0_before} → {n_c0_after} rows (industry-driver only)')
    print(f'C1 filter: {n_c1_before} → {n_c1_after} rows (market-driver only)')
    c0_df.to_csv(c0_filtered_csv, index=False, encoding='utf-8')
    c1_df.to_csv(c1_filtered_csv, index=False, encoding='utf-8')
    c0_csv = c0_filtered_csv
    c1_csv = c1_filtered_csv

    # If c0_dist / c1_dist are missing, create stub from the filtered CSV.
    # (The real pipeline writes them in v0_2_d_decompose Step 5; this covers
    # --skip-ablation and hand-assembled C0 dirs.)
    if not os.path.exists(c0_dist):
        _safe_distributions(c0_csv).to_csv(c0_dist, index=False, encoding='utf-8')
    if not os.path.exists(c1_dist):
        _safe_distributions(c1_csv).to_csv(c1_dist, index=False, encoding='utf-8')

    compute_c0_c1_paired_compare(c0_csv, c1_csv, paired_path)
    write_c0_c1_compare_summary_txt(paired_path, c0_dist, c1_dist, summary_path)
    print(f'C0 filtered CSV: {c0_filtered_csv}')
    print(f'C1 filtered CSV: {c1_filtered_csv}')
    print(f'Paired compare CSV: {paired_path}')
    print(f'Paired compare TXT: {summary_path}')


if __name__ == '__main__':
    main()
