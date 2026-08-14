# -*- coding: utf-8 -*-
# projection_batch.py — 批量跑 projection 2-D 投影分析(只产 CSV,不画 HTML)
#
# 基线指数(优先级从高到低):
#   1. --index <code> 显式传入 → 所有股票共用同一基线(覆盖下列自动逻辑)
#   2. 默认 → 每只票按 申万二级行业(881xxx.SH) 投影,不是大盘指数。
#      行业映射走 _projection_core.resolve_industry(data/sw2/members.csv);
#      新股/非 A 股等 sw2 缺失的代码自动回退大盘(resolve_index)。
#   3. --market-baseline → 全部回退到大盘基线(SZ→深证成指 / SH→上证综指)
#
# 参数:
#   --input              path  股票列表 CSV(列:code, 可选 name)。默认 data/projection/stocks.csv
#   --days               int   回看天数。默认 240
#   --limit              int   最多处理多少只;0 表示全部。默认 0
#   --market-baseline    flag  全部回退到大盘基线(覆盖默认行业基线)。
#   --index              str   强制指定基线指数(覆盖个股自动解析);所有股票共用。
#                              示例:--index 881427.SH(半导体)/ 000001.SH(上证综指)
#
# 用法:
#   1. 准备股票列表 CSV,至少一列 `code`,可选 `name`(例:
#        code,name
#        002475.SZ,立讯精密
#        600519.SH,贵州茅台
#      )
#      默认读取 data/projection/stocks.csv
#   2. PYTHONIOENCODING=utf-8 python backtrace/projection/projection_batch.py
#      [可选 --input PATH / --days N / --limit N / --market-baseline / --index CODE]
#
# CLI 示例:
#   python backtrace/projection/projection_batch.py                              # 默认按个股所属行业基线跑
#   python backtrace/projection/projection_batch.py --market-baseline            # 全部用大盘基线
#   python backtrace/projection/projection_batch.py --index 881427.SH            # 全部用申万二级体育指数
#   python backtrace/projection/projection_batch.py --index 000001.SH --days 120 # 上证综指,回看 120 日
#   python backtrace/projection/projection_batch.py --limit 50                   # 只跑列表前 50 只
#
# 输出:
#   - 每只股票一个 CSV:data/projection/projection_{INDEX_TAG}_{STOCK_TAG}.csv
#     (INDEX_TAG 是行业代码 881xxx 或大盘代码 000001/399001,或 --index 显式指定的代码)
#   - 批量清单:data/projection/batch_manifest.csv
#     列: code, name, index_code, index_name, rows, date_start, date_end, csv_path, status
#
# 注:本脚本不产 HTML。可视化请用 projection_2d.py 单股跑或自行读 CSV 画。
# 数学/数据载入/CSV 组装统一在 _projection_core.py。
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import pandas as pd

# 同 single-stock:把 backtrace/ 加进 path 找 common.tsfresh_pipeline
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
from common import tsfresh_pipeline as P
from _projection_core import (
    load_pair,
    compute_vectors,
    compute_projections,
    build_result_df,
)

CSV_OUT_DIR = 'data/projection'   # 每只股票 CSV + batch_manifest.csv 都落这里


def parse_args():
    parser = argparse.ArgumentParser(description='批量 projection 2-D 投影分析(只产 CSV)')
    parser.add_argument(
        '--input', default=os.path.join(CSV_OUT_DIR, 'stocks.csv'),
        help=f'股票列表 CSV 路径(列:code, 可选 name)。默认 {CSV_OUT_DIR}/stocks.csv',
    )
    parser.add_argument('--days', type=int, default=240, help='回看天数。默认 240')
    parser.add_argument('--limit', type=int, default=0, help='最多处理多少只;0 表示全部。默认 0')
    parser.add_argument(
        '--market-baseline', action='store_true',
        help='回退到大盘基线(SZ→深证成指/SH→上证综指)。默认走行业基线(申万二级)。',
    )
    parser.add_argument(
        '--index', default=None,
        help=(
            '强制指定基线指数(覆盖个股自动解析);所有股票都用同一基线。'
            '示例:--index 881427.SH(半导体)/ 000001.SH(上证综指)'
        ),
    )
    parser.add_argument(
        '--two-day-vec', action='store_true',
        help=(
            '将向量扩展为 (Vol_today, Amt_today, Vol_yesterday, Amt_yesterday) 4-D;'
            '首日丢弃。默认 2-D。'
        ),
    )
    return parser.parse_args()


def load_stock_list(path):
    """读 CSV:必须有 code 列,name 可选。返回 [(code, name|None), ...]。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"股票列表文件不存在: {path}\n"
            f"请新建该文件,最少一列 code(可选 name),例如:\n"
            f"  code,name\n"
            f"  002475.SZ,立讯精密\n"
            f"  600519.SH,贵州茅台"
        )
    df = pd.read_csv(path, dtype={'code': str})
    if 'code' not in df.columns:
        raise ValueError(f"输入文件 {path} 必须有 'code' 列(可选 'name')")
    names = df['name'] if 'name' in df.columns else [None] * len(df)
    return [
        (str(c).strip(), str(n).strip() if isinstance(n, str) else None)
        for c, n in zip(df['code'], names)
    ]


def process_one(stock_code, stock_name, days, prefer_industry, index_code, lag: int = 0):
    """处理一只股票。返回 manifest 行 dict(失败也返回,status 字段说明原因)。"""
    try:
        loaded = load_pair(stock_code, days, P, prefer_industry=prefer_industry,
                           index_code=index_code, lag=lag)
        data_stock = loaded['stock_df']
        data_index = loaded['index_df']
        common_idx = loaded['common_idx']
        index_code = loaded['index_code']
        index_name = loaded['index_name']
        index_tag = loaded['index_tag']
        stock_tag = loaded['stock_tag']

        vec_index, vec_stock, vec_index_norm, vec_stock_norm, norm_params = compute_vectors(
            data_stock, data_index, index_tag, stock_tag, lag=lag,
        )
        proj = compute_projections(vec_stock_norm, vec_index_norm)

        result_df = build_result_df(
            common_idx, vec_index, vec_stock, vec_index_norm, vec_stock_norm,
            proj['projections'], proj['residuals'], proj['dot_after'],
            proj['proj_coeffs'], proj['proj_mags'], proj['proj_prices'], proj['resi_prices'],
            norm_params, index_tag, stock_tag, lag=lag,
        )

        csv_name = f'projection_{index_tag}_{stock_tag}.csv'
        csv_path = os.path.join(CSV_OUT_DIR, csv_name)
        os.makedirs(CSV_OUT_DIR, exist_ok=True)
        result_df.to_csv(csv_path, index=False, encoding='utf-8')

        return {
            'code': stock_code,
            'name': stock_name or '',
            'index_code': index_code,
            'index_name': index_name,
            'rows': len(common_idx),
            'date_start': str(common_idx[0])[:10],
            'date_end': str(common_idx[-1])[:10],
            'csv_path': csv_path,
            'status': 'ok',
        }
    except Exception as e:
        return {
            'code': stock_code,
            'name': stock_name or '',
            'index_code': '',
            'index_name': '',
            'rows': 0,
            'date_start': '',
            'date_end': '',
            'csv_path': '',
            'status': f'failed: {type(e).__name__}: {e}',
        }


def main():
    args = parse_args()
    os.makedirs(CSV_OUT_DIR, exist_ok=True)

    stock_list = load_stock_list(args.input)
    if args.limit > 0:
        stock_list = stock_list[:args.limit]

    prefer_industry = not args.market_baseline
    if args.index:
        baseline = f'显式指定基线 {args.index}(所有股票共用)'
    elif prefer_industry:
        baseline = '申万二级行业(按个股解析;新股/非 A 股自动回退大盘)'
    else:
        baseline = '大盘指数(深证成指/上证综指)'

    print(f"输入: {args.input} ({len(stock_list)} 只)")
    print(f"回看天数: {args.days}")
    print(f"投影基线: {baseline}")
    print(f"向量维度: {'4-D (今日+前一日 Vol/Amt, --two-day-vec)' if args.two_day_vec else '2-D (今日 Vol/Amt)'}")
    print(f"输出目录: {CSV_OUT_DIR}\n")

    lag = 1 if args.two_day_vec else 0
    manifest = []
    for i, (code, name) in enumerate(stock_list, 1):
        label = f"{code} ({name})" if name else code
        print(f"[{i}/{len(stock_list)}] {label}...", end=' ', flush=True)
        row = process_one(code, name, args.days, prefer_industry, args.index, lag)
        manifest.append(row)
        if row['status'] == 'ok':
            print(f"✓ {row['rows']} 行 → {row['csv_path']}")
        else:
            print(f"✗ {row['status']}")

    manifest_df = pd.DataFrame(manifest, columns=[
        'code', 'name', 'index_code', 'index_name', 'rows',
        'date_start', 'date_end', 'csv_path', 'status',
    ])
    manifest_path = os.path.join(CSV_OUT_DIR, 'batch_manifest.csv')
    manifest_df.to_csv(manifest_path, index=False, encoding='utf-8')

    ok = sum(1 for r in manifest if r['status'] == 'ok')
    fail = len(manifest) - ok
    print(f"\n=== 汇总 ===")
    print(f"  成功: {ok}/{len(manifest)}")
    if fail:
        print(f"  失败: {fail}/{len(manifest)}")
    print(f"  清单: {manifest_path}")


if __name__ == '__main__':
    main()