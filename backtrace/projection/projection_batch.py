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
#   --two-day-vec        flag  向量扩展为 4-D(今日+前一日 Vol/Amt);首日丢弃(无前一日)。
#                              默认 2-D(仅今日 Vol/Amt)。
#   --movement           flag  运动向量投影模式:不算"当前状态"投影,而是把个股 (ΔVol, ΔAmt)
#                              投到大盘 (ΔVol, ΔAmt) 的运动方向上。首行丢弃(无前一日)。
#                              与 --two-day-vec 正交,可叠加。
#
# 用法:
#   1. 准备股票列表 CSV,至少一列 `code`,可选 `name`(例:
#        code,name
#        002475.SZ,立讯精密
#        600519.SH,贵州茅台
#      )
#      默认读取 data/projection/stocks.csv
#   2. PYTHONIOENCODING=utf-8 python backtrace/projection/projection_batch.py
#      [可选 --input PATH / --days N / --limit N / --market-baseline / --index CODE
#       / --two-day-vec / --movement]
#
# 向量维度说明:
#   默认(2-D 状态投影):每只票每个交易日产出 21 列 CSV,向量 v = (Volume, Amount)。
#   --two-day-vec(4-D 状态):v = (Volume_today, Amount_today, Volume_yesterday, Amount_yesterday),
#     产生 29 列 CSV(新增 Vol_prev / Amt_prev / prev_norm 两组共 8 列);首日无前一日数据被丢弃。
#   --movement(运动向量投影,正交于上述两种):把 Δv_s 投到 Δv_i 上,
#     产出 movement_{INDEX_TAG}_{STOCK_TAG}.csv(18 列,首行同样丢弃),文件名独立不覆盖。
#   三种模式可自由组合(--two-day-vec + --movement 同时开,会得到 29 列状态 CSV + 18 列运动 CSV)。
#   (注:2026-08-15 列名从 `Proj_*`/`Resi_*` 加 State_/Move_ 前缀,21/29 → 29 列,13 → 18 列;
#    2026-08-16 删除 `State_Resi_Price`(2-D 退化,选股无效),列数 22/30 → 21/29)
#
# CLI 示例:
#   python backtrace/projection/projection_batch.py                              # 默认 2-D,按个股所属行业基线跑
#   python backtrace/projection/projection_batch.py --market-baseline            # 全部用大盘基线
#   python backtrace/projection/projection_batch.py --index 881427.SH            # 全部用申万二级体育指数
#   python backtrace/projection/projection_batch.py --index 000001.SH --days 120 # 上证综指,回看 120 日
#   python backtrace/projection/projection_batch.py --limit 50                   # 只跑列表前 50 只
#   python backtrace/projection/projection_batch.py --two-day-vec                # 4-D 模式:含前一日 Vol/Amt
#   python backtrace/projection/projection_batch.py --two-day-vec --limit 20     # 4-D + 只跑 20 只(冒烟)
#   python backtrace/projection/projection_batch.py --two-day-vec --index 000001.SH # 4-D + 大盘基线
#   python backtrace/projection/projection_batch.py --movement                  # 运动向量投影:Δv_s → Δv_i
#   python backtrace/projection/projection_batch.py --movement --limit 20       # 运动投影 + 只跑 20 只
#   python backtrace/projection/projection_batch.py --two-day-vec --movement    # 状态 + 运动双产出
#
# 输出:
#   - 每只股票一个 CSV:data/projection/projection_{INDEX_TAG}_{STOCK_TAG}.csv
#     (INDEX_TAG 是行业代码 881xxx 或大盘代码 000001/399001,或 --index 显式指定的代码)
#     2-D:21 列 / 4-D:29 列(每个交易日一行,State_* 前缀)
#   - --movement 启用时,额外产出 data/projection/movement_{INDEX_TAG}_{STOCK_TAG}.csv(18 列,丢首行,Move_* 前缀)
#   - 批量清单:data/projection/batch_manifest.csv
#     列: code, name, index_code, index_name, rows, date_start, date_end, csv_path, status
#     rows 已扣除 4-D 模式丢弃的首日(实际写入 CSV 的行数)
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
    compute_movement_projection,
    build_movement_result_df,
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
    parser.add_argument(
        '--movement', action='store_true',
        help=(
            '运动向量投影模式:不投影当前成交状态,而是把个股 (ΔVol, ΔAmt) '
            '投影到大盘 (ΔVol, ΔAmt) 的运动方向上(首行丢弃,因 .diff 无前一日)。'
            '可与 --two-day-vec 共存,产独立 movement CSV。'
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


def process_one(stock_code, stock_name, days, prefer_industry, index_code,
                 lag: int = 0, movement: bool = False):
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
        proj = compute_projections(vec_stock, vec_index)

        result_df = build_result_df(
            common_idx, vec_index, vec_stock, vec_index_norm, vec_stock_norm,
            proj['projections'], proj['residuals'], proj['dot_after'],
            proj['proj_coeffs'], proj['proj_mags'], proj['proj_prices'],
            proj['state_stock_mag'], proj['state_index_mag'], proj['state_relative_move'],
            norm_params, index_tag, stock_tag, lag=lag,
        )

        csv_name = f'projection_{index_tag}_{stock_tag}.csv'
        csv_path = os.path.join(CSV_OUT_DIR, csv_name)
        os.makedirs(CSV_OUT_DIR, exist_ok=True)
        result_df.to_csv(csv_path, index=False, encoding='utf-8')

        # movement 模式:额外算一次运动投影,产出独立 CSV(同名 _movement 后缀)
        if movement:
            mv = compute_movement_projection(data_stock, data_index)
            mv_df = build_movement_result_df(common_idx[1:], mv, index_tag, stock_tag)
            mv_csv_name = f'movement_{index_tag}_{stock_tag}.csv'
            mv_csv_path = os.path.join(CSV_OUT_DIR, mv_csv_name)
            mv_df.to_csv(mv_csv_path, index=False, encoding='utf-8')

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
    print(f"运动投影: {'开启 (额外产出 movement_*.csv)' if args.movement else '关闭'}")
    print(f"输出目录: {CSV_OUT_DIR}\n")

    lag = 1 if args.two_day_vec else 0
    manifest = []
    for i, (code, name) in enumerate(stock_list, 1):
        label = f"{code} ({name})" if name else code
        print(f"[{i}/{len(stock_list)}] {label}...", end=' ', flush=True)
        row = process_one(code, name, args.days, prefer_industry, args.index,
                          lag, args.movement)
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