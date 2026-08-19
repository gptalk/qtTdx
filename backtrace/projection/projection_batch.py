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
#   --dynamics           flag  在 --movement 之上叠加「离散动力学层」(2026-08-16 新增):
#                              锚定强度 q_t、偏离角 θ、耦合度 R、动能 E_market/E_self、
#                              7 状态分类。自动开启 --movement(无需重复传)。
#                              额外产出 dynamics_*.csv(14 列)+ forces_*.csv(8 列)。
#   --lambda-q           float 锚定强度系数 λ_q。-1 走 median(‖ΔM‖) 自适应(默认);
#                              0 等价无阻尼 q_t=1;正值越大阻尼越强 q_t → 0。
#   --classify-thresholds str  状态分类阈值,逗号分隔 4 个浮点
#                              (R_low,R_high,theta_following_deg,theta_against_deg)。
#                              默认 0.10,0.50,30,90。
#   --k-restore          float 恢复力系数 k。F_restore = -k·d,默认 0 = 无均值回复力。
#   --c-damp             float 阻尼系数 c。F_damp = -c·u,默认 0 = 无阻尼。
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
#       / --two-day-vec / --movement / --dynamics / --lambda-q F / --classify-thresholds S
#       / --k-restore F / --c-damp F]
#
# 向量维度说明:
#   默认(2-D 状态投影):每只票每个交易日产出 21 列 CSV,向量 v = (Volume, Amount)。
#   --two-day-vec(4-D 状态):v = (Volume_today, Amount_today, Volume_yesterday, Amount_yesterday),
#     产生 29 列 CSV(新增 Vol_prev / Amt_prev / prev_norm 两组共 8 列);首日无前一日数据被丢弃。
#   --movement(运动向量投影,正交于上述两种):把 Δv_s 投到 Δv_i 上,
#     产出 movement_{INDEX_TAG}_{STOCK_TAG}.csv(18 列,首行同样丢弃),文件名独立不覆盖。
#   --dynamics(离散动力学层,叠加在 --movement 之上):
#     额外产出 dynamics_*.csv(14 列,Dyn_ 前缀)+ forces_*.csv(8 列,Frc_ 前缀)。
#     时间轴与 movement CSV 对齐(common_idx[1:],T-1 行)。
#     4 档输出 vs 3 档列数对照:
#       默认(2-D state)                → 21 列 State_*
#       --two-day-vec(4-D state)       → 29 列 State_*
#       --movement(2-D + 运动)         → 21 列 State_* + 18 列 Move_*
#       --two-day-vec --movement       → 29 列 State_* + 18 列 Move_*
#       --movement --dynamics          → + 14 列 Dyn_* + 8 列 Frc_*
#   (注:2026-08-15 列名从 `Proj_*`/`Resi_*` 加 State_/Move_ 前缀,21/29 → 29 列,13 → 18 列;
#    2026-08-16 删除 `State_Resi_Price`(2-D 退化,选股无效),列数 22/30 → 21/29;
#    2026-08-16 新增动力学层 14+8 列)
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
#   # 动力学层示例(2026-08-16 新增)
#   python backtrace/projection/projection_batch.py --movement --dynamics --limit 10 \
#       # 运动 + 动力学(自动开启 --movement);产 movement/dynamics/forces 三组 CSV
#   python backtrace/projection/projection_batch.py --dynamics --lambda-q 1e6    # 自定义 λ_q(强阻尼)
#   python backtrace/projection/projection_batch.py --dynamics --classify-thresholds 0.15,0.60,20,100  # 改阈值
#   python backtrace/projection/projection_batch.py --dynamics --k-restore 0.1 --c-damp 0.05  # 加弱回复 + 弱阻尼
#   python backtrace/projection/projection_batch.py --dynamics --k-restore 0 --c-damp 0       # 残差基线(F_self = a_S - F_market)
#   python backtrace/projection/projection_batch.py --market-baseline --dynamics --days 120 --limit 30
#       # 全市场基线 + 动力学 + 120 日 + 前 30 只冒烟
#
# 输出:
#   - 每只股票一个 CSV:data/projection/projection_{INDEX_TAG}_{STOCK_TAG}.csv
#     (INDEX_TAG 是行业代码 881xxx 或大盘代码 000001/399001,或 --index 显式指定的代码)
#     2-D:21 列 / 4-D:29 列(每个交易日一行,State_* 前缀)
#   - --movement 启用时,额外产出 data/projection/movement_{INDEX_TAG}_{STOCK_TAG}.csv(18 列,丢首行,Move_* 前缀)
#   - --dynamics 启用时(自动含 --movement),额外产出:
#       data/projection/dynamics_{INDEX_TAG}_{STOCK_TAG}.csv  (14 列,Dyn_ 前缀)
#       data/projection/forces_{INDEX_TAG}_{STOCK_TAG}.csv    (8 列,Frc_ 前缀)
#   - 批量清单:data/projection/batch_manifest.csv
#     列: code, name, index_code, index_name, rows, date_start, date_end,
#          csv_path, dyn_csv_path, frc_csv_path, status
#     rows 已扣除 4-D 模式丢弃的首日(实际写入 CSV 的行数)
#     dyn_csv_path / frc_csv_path 在 --dynamics 启用时填,否则空字符串
#     status='ok' 表示全部成功;失败时为 'failed: <ExcType>: <msg>';动力学层单独失败
#     不阻塞主路径,会被记成 'ok (dynamics failed: ...)'(movement 仍写入)
#
# 注:本脚本不产 HTML。可视化请用 projection_2d.py 单股跑或自行读 CSV 画。
# 数学/数据载入/CSV 组装统一在 _projection_core.py。
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import numpy as np
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
    # 动力学层(2026-08-16 新增)
    compute_dynamics,
    classify_states,
    build_dynamics_df,
    compute_forces,
    build_forces_df,
)

CSV_OUT_DIR = 'data/projection'   # 默认输出目录;运行时由 --output-dir 覆盖(模块全局)

# KC source dir (always default, regardless of --output-dir):
# load_kc_map() 总是从 data/projection/kc_estimates.csv 读取(parameter_fit.py 的固定输出),
# 不受 --output-dir 影响 — 这样 C0 行业基线和 C1 市场基线能共享同一份拟合表。
KC_SOURCE_DIR = 'data/projection'


def load_kc_map(status_filter: str = 'ok'):
    """从 KC_SOURCE_DIR/kc_estimates.csv 读 {(index_tag, stock_tag): (k̂, ĉ)}。

    注意:此函数**总是**从默认 `data/projection/kc_estimates.csv` 读取,
    不受 --output-dir 影响(parameter_fit.py 的固定输出位置,
    C0 / C1 共享同一份拟合表)。

    status_filter: 只取 status 以该前缀开头的行(默认 'ok');空字符串 = 全部。
    找不到 / 文件不存在 → 返回 {}。
    """
    path = os.path.join(KC_SOURCE_DIR, 'kc_estimates.csv')
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype={
        'code': str, 'index_code': str, 'index_tag': str, 'stock_tag': str,
    })
    if status_filter:
        df = df[df['status'].str.startswith(status_filter, na=False)]
    out = {}
    for _, row in df.iterrows():
        try:
            out[(row['index_tag'], row['stock_tag'])] = (
                float(row['k_hat']),
                float(row['c_hat']),
            )
        except (KeyError, ValueError):
            continue
    return out


def parse_args():
    parser = argparse.ArgumentParser(description='批量 projection 2-D 投影分析(只产 CSV)')
    parser.add_argument(
        '--input', default=os.path.join(CSV_OUT_DIR, 'stocks.csv'),
        help=f'股票列表 CSV 路径(列:code, 可选 name)。默认 {CSV_OUT_DIR}/stocks.csv',
    )
    parser.add_argument(
        '--output-dir', default='data/projection',
        help=(
            '输出目录(所有 projection / movement / dynamics / forces / batch_manifest CSV 落到这里)。'
            '默认 data/projection。注意:load_kc_map() 总是从默认 data/projection/kc_estimates.csv 读取,'
            '不受 --output-dir 影响。'
        ),
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
    # 动力学层(2026-08-16):在 --movement 之上叠加离散动力学指标 + 状态分类 + 力分解
    parser.add_argument(
        '--dynamics', action='store_true',
        help=(
            '在 --movement 之上叠加「离散动力学层」(2026-08-16 新增):'
            '锚定强度 q_t、偏离角 θ、耦合度 R、动能 E_market/E_self、'
            '7 状态分类标签。自动开启 --movement(无需重复传);'
            '额外产出 dynamics_*.csv(14 列)+ forces_*.csv(8 列)。'
            '单股 HTML 可视化请用 projection_2d.py --dynamics。'
        ),
    )
    parser.add_argument(
        '--lambda-q', type=float, default=-1.0,
        help=(
            '锚定强度系数 λ_q(浮点)。q_t = ‖ΔM‖ / (‖ΔM‖ + λ_q)。'
            '传 -1 走默认 = median(‖ΔM‖) 自适应窗口;'
            '传 0 等价无阻尼 q_t=1;正值越大阻尼越强 q_t→0。'
        ),
    )
    parser.add_argument(
        '--classify-thresholds', default='0.10,0.50,30,90',
        help=(
            '状态分类阈值,逗号分隔 4 个浮点:R_low,R_high,theta_following_deg,'
            'theta_against_deg。默认 0.10,0.50,30,90。'
            '约束:0 < R_low < R_high < 1;0 < theta_following < theta_against < 180。'
        ),
    )
    parser.add_argument(
        '--k-restore', type=float, default=0.0,
        help=(
            '恢复力系数 k(浮点)。F_restore = -k·d,默认 0 = 无均值回复力。'
            '调试时可设 0.1~1.0 看个股偏离被多大强度拉回。'
        ),
    )
    parser.add_argument(
        '--c-damp', type=float, default=0.0,
        help=(
            '阻尼系数 c(浮点)。F_damp = -c·u,默认 0 = 无阻尼。'
            '正 c 表示系统倾向于把个股与大盘的速度差消耗掉。'
        ),
    )
    parser.add_argument(
        '--k-from-fit', action='store_true',
        help=(
            '从 data/projection/kc_estimates.csv 自动加载每只票的 k̂,'
            '覆盖 --k-restore。前提:先用 parameter_fit.py 估过,且 (index_tag, stock_tag) 对得上。'
            '找不到的票用 --k-restore 默认值(0)。'
        ),
    )
    parser.add_argument(
        '--c-from-fit', action='store_true',
        help=(
            '从 data/projection/kc_estimates.csv 自动加载每只票的 ĉ,'
            '覆盖 --c-damp。找不到的票用 --c-damp 默认值(0)。'
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
                 lag: int = 0, movement: bool = False,
                 dynamics: bool = False, lambda_q=None,
                 classify_thresholds=(0.10, 0.50, np.deg2rad(30), np.deg2rad(90)),
                 k_restore: float = 0.0, c_damp: float = 0.0,
                 kc_overrides: dict | None = None):
    """处理一只股票。返回 manifest 行 dict(失败也返回,status 字段说明原因)。

    Args:
        ...(原有 state/movement 参数)...
        dynamics: bool。True 时启用离散动力学层(自动含 movement,已在 main() 内联动)。
        lambda_q: 锚定强度系数。None → median(‖ΔM‖) 自适应;float → 用户指定。
        classify_thresholds: 4 元组 (R_low, R_high, theta_following_rad, theta_against_rad)
                              度值已由 main() 转弧度传进来。
        k_restore / c_damp: 力模型系数。默认 0 = 纯残差基线。

    返回的 manifest 行多了 dyn_csv_path / frc_csv_path 两个字段(动力学启用时填;
    否则空字符串)。动力学层单独失败不阻塞主路径,status 会记成 'ok (dynamics failed: ...)'。
    """
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

        # 动力学层(2026-08-16):在 --movement 之上叠加离散动力学 + 力分解。
        # 失败不阻塞主路径(movement/state CSV 已落地),记到 status 后缀。
        dyn_csv_path = ''
        frc_csv_path = ''
        dyn_status_suffix = ''
        if dynamics:
            try:
                # 重新算 mv(运动投影在 if movement 块算了一次但没绑外层变量)。
                # 双算代价 ≈ 单只票 < 10ms,对 batch 5500 只可忽略;后续可考虑缓存。
                mv_for_dyn = compute_movement_projection(data_stock, data_index)
                dyn = compute_dynamics(mv_for_dyn, lambda_q=lambda_q)
                r_low, r_high, theta_following_rad, theta_against_rad = classify_thresholds
                states = classify_states(
                    dyn['R'], dyn['theta'], dyn['E_self'],
                    (r_low, r_high, theta_following_rad, theta_against_rad),
                )
                dyn_df = build_dynamics_df(
                    common_idx[1:], dyn, states, index_tag, stock_tag,
                )
                dyn_csv_name = f'dynamics_{index_tag}_{stock_tag}.csv'
                dyn_csv_path = os.path.join(CSV_OUT_DIR, dyn_csv_name)
                dyn_df.to_csv(dyn_csv_path, index=False, encoding='utf-8')

                # 力分解(总是跑,即便 k=c=0 也产 forces CSV 便于看 F_self 残差)
                # --k-from-fit / --c-from-fit 时按 (index_tag, stock_tag) 查表覆盖默认值
                eff_k = k_restore
                eff_c = c_damp
                if kc_overrides is not None:
                    kc = kc_overrides.get((index_tag, stock_tag))
                    if kc is not None:
                        eff_k, eff_c = kc
                frc = compute_forces(dyn, mv_for_dyn,
                                     k_restore=eff_k, c_damp=eff_c)
                frc_df = build_forces_df(
                    common_idx[1:], frc, index_tag, stock_tag,
                )
                frc_csv_name = f'forces_{index_tag}_{stock_tag}.csv'
                frc_csv_path = os.path.join(CSV_OUT_DIR, frc_csv_name)
                frc_df.to_csv(frc_csv_path, index=False, encoding='utf-8')
            except Exception as e:
                dyn_status_suffix = f'dynamics failed: {type(e).__name__}: {e}'

        status = 'ok'
        if dyn_status_suffix:
            status = f'ok ({dyn_status_suffix})'

        return {
            'code': stock_code,
            'name': stock_name or '',
            'index_code': index_code,
            'index_name': index_name,
            'rows': len(common_idx),
            'date_start': str(common_idx[0])[:10],
            'date_end': str(common_idx[-1])[:10],
            'csv_path': csv_path,
            'dyn_csv_path': dyn_csv_path,
            'frc_csv_path': frc_csv_path,
            'status': status,
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
            'dyn_csv_path': '',
            'frc_csv_path': '',
            'status': f'failed: {type(e).__name__}: {e}',
        }


def main():
    args = parse_args()
    # V0.2-C1 Task 1:把 --output-dir 绑到模块全局 CSV_OUT_DIR,
    # 让 process_one() / batch_manifest 写路径都跟着变。
    # 注意:load_kc_map() 仍走 KC_SOURCE_DIR(默认 data/projection),不受影响。
    global CSV_OUT_DIR
    CSV_OUT_DIR = args.output_dir
    os.makedirs(CSV_OUT_DIR, exist_ok=True)

    # --dynamics 自动开启 --movement(动力学层依赖 mv dict)
    if args.dynamics and not args.movement:
        args.movement = True
        print('[--dynamics] 自动开启 --movement')
    # λ_q:-1 走默认 median 自适应,其余按用户值传
    if args.lambda_q < 0:
        lambda_q = None
    else:
        lambda_q = args.lambda_q
    # 解析状态分类阈值;失败立即报错(避免后面静默错位)
    try:
        R_LOW, R_HIGH, THETA_FOLLOWING_DEG, THETA_AGAINST_DEG = (
            float(x) for x in args.classify_thresholds.split(',')
        )
    except Exception as e:
        raise SystemExit(
            f'--classify-thresholds 解析失败: {args.classify_thresholds!r}\n'
            f'需要 4 个逗号分隔浮点,例:0.10,0.50,30,90\n{e}'
        )
    if not (0 < R_LOW < R_HIGH < 1):
        raise SystemExit(f'R_low={R_LOW} / R_high={R_HIGH} 必须满足 0 < R_low < R_high < 1')
    if not (0 < THETA_FOLLOWING_DEG < THETA_AGAINST_DEG < 180):
        raise SystemExit(
            f'theta_following={THETA_FOLLOWING_DEG}° / theta_against={THETA_AGAINST_DEG}° '
            f'必须满足 0 < following < against < 180'
        )
    classify_thresholds = (
        R_LOW, R_HIGH,
        np.deg2rad(THETA_FOLLOWING_DEG), np.deg2rad(THETA_AGAINST_DEG),
    )

    # --k-from-fit / --c-from-fit:加载拟合表
    kc_overrides = None
    if args.k_from_fit or args.c_from_fit:
        kc_overrides = load_kc_map(status_filter='ok')
        if not kc_overrides:
            print('[--k-from-fit/--c-from-fit] ⚠ kc_estimates.csv 中没有 ok 记录,'
                  '将全部用 --k-restore / --c-damp 默认值')
            print('  请先跑:python backtrace/projection/parameter_fit.py')
        else:
            print(f'[--k-from-fit/--c-from-fit] 已加载 {len(kc_overrides)} 条拟合值')

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

    if args.dynamics:
        if lambda_q is None:
            lq_str = 'median 自适应'
        else:
            lq_str = f'{lambda_q:.4e}'
        dynamics_str = (
            f'开启 (λ_q={lq_str}, '
            f'阈值=R<{R_LOW}/{R_HIGH} + θ<{THETA_FOLLOWING_DEG}°/>{THETA_AGAINST_DEG}°, '
            f'k={args.k_restore}, c={args.c_damp};产 dynamics_*.csv + forces_*.csv)'
        )
    else:
        dynamics_str = '关闭'

    print(f"输入: {args.input} ({len(stock_list)} 只)")
    print(f"回看天数: {args.days}")
    print(f"投影基线: {baseline}")
    print(f"向量维度: {'4-D (今日+前一日 Vol/Amt, --two-day-vec)' if args.two_day_vec else '2-D (今日 Vol/Amt)'}")
    print(f"运动投影: {'开启 (额外产出 movement_*.csv)' if args.movement else '关闭'}")
    print(f"动力学层: {dynamics_str}")
    print(f"输出目录: {CSV_OUT_DIR}\n")

    lag = 1 if args.two_day_vec else 0
    manifest = []
    for i, (code, name) in enumerate(stock_list, 1):
        label = f"{code} ({name})" if name else code
        print(f"[{i}/{len(stock_list)}] {label}...", end=' ', flush=True)
        row = process_one(
            code, name, args.days, prefer_industry, args.index,
            lag, args.movement,
            dynamics=args.dynamics, lambda_q=lambda_q,
            classify_thresholds=classify_thresholds,
            k_restore=args.k_restore, c_damp=args.c_damp,
            kc_overrides=kc_overrides,
        )
        manifest.append(row)
        # 状态打印:动力学模式下额外列两个 CSV 路径
        if row['status'] == 'ok' or row['status'].startswith('ok ('):
            extra = ''
            if args.dynamics and row['dyn_csv_path']:
                extra = f" + dyn={os.path.basename(row['dyn_csv_path'])} frc={os.path.basename(row['frc_csv_path'])}"
            print(f"✓ {row['rows']} 行 → {row['csv_path']}{extra}")
            if row['status'].startswith('ok ('):
                print(f"    ! {row['status']}")
        else:
            print(f"✗ {row['status']}")

    manifest_df = pd.DataFrame(manifest, columns=[
        'code', 'name', 'index_code', 'index_name', 'rows',
        'date_start', 'date_end', 'csv_path',
        'dyn_csv_path', 'frc_csv_path',
        'status',
    ])
    manifest_path = os.path.join(CSV_OUT_DIR, 'batch_manifest.csv')
    manifest_df.to_csv(manifest_path, index=False, encoding='utf-8')

    ok = sum(1 for r in manifest if r['status'] == 'ok')
    ok_with_warn = sum(1 for r in manifest if r['status'].startswith('ok ('))
    fail = len(manifest) - ok - ok_with_warn
    print(f"\n=== 汇总 ===")
    print(f"  全部成功: {ok}/{len(manifest)}")
    if ok_with_warn:
        print(f"  成功(动力学层失败): {ok_with_warn}/{len(manifest)}")
    if fail:
        print(f"  失败: {fail}/{len(manifest)}")
    print(f"  清单: {manifest_path}")


if __name__ == '__main__':
    main()