# -*- coding: utf-8 -*-
# parameter_fit.py — 从已有运动投影 CSV 闭式 OLS 解出每只票的 (k, c)
#
# 动力学方程(用户 prompt §14-17):
#   a_S = β·a_M  -  k·d  -  c·u  +  F_self
#
# 模型对 (k, c) 是线性的 → 可以闭式 OLS 求解,不需网格搜索。残差即 F_self:
#
#   a_S - β·a_M = -k·d - c·u + F_self
#       ↑ Y          ↑ X·θ   ↑ noise
#
# 2-D 版本:把每个时间点的 (x, y) 分量展开堆叠成 2T 方程,2 未知数。
#   Y_{2T} = [-d_x ; -u_x ; -d_y ; -u_y]^T · [k, c]^T  +  F_self
#
# OLS 解:θ = (X^T X)^(-1) X^T Y,残差 norm 即 ‖F_self‖² 平均。
#
# 输入:data/projection/movement_*.csv(由 --dynamics batch 跑产出,含 2-D Δ 向量)
# 输出:data/projection/kc_estimates.csv
#       列:code, name, index_code, index_tag, stock_tag,
#          k_hat, c_hat, f_self_loss, n_valid_days, status
#
# 用法:
#   PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py
#   PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --limit 10  # 冒烟
#   PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --input <path.csv>
#       # 自定义股票列表(列:code;可选 name/index_code),只跑列表里的票
#
# 设计选择 — 为什么 OLS 而不是网格搜索:
#   1. 模型对 k/c 严格线性(2-D 投影后,无高阶项),OLS 是 BLUE(最优线性无偏估计)
#   2. 闭式 = 单次 np.linalg.lstsq,5500 只 × 单次解 < 5 秒
#   3. 网格搜索会强加先验范围;OLS 不强加,k̂/c 可以是任意实数
#   4. 残差直接是 ‖F_self‖² 平均,代表「模型无法解释的个股自主驱动力大小」
#
# 物理含义:
#   - k̂ > 0:个股偏离会被拉回(均值回复);k̂ 越大回复越强
#   - ĉ > 0:个股相对大盘的速度差被耗散;ĉ 越大阻尼越强
#   - k̂ < 0:个股偏离被放大(反回复,趋势强化)
#   - ĉ < 0:速度差被放大(反阻尼,助涨助跌)
#   - f_self_loss:拟合后 F_self 的均方范数;大 → 个股自主驱动力强,小 → 模型已充分解释
#
# 已知陷阱:
#   - 数据 < 3 有效行 → 跳过(2 未知数最少需要 2 个有效观测,但稳健起见 ≥ 3)
#   - 2-D 加速度末行 NaN(来自 np.diff)→ 自动 drop
#   - 奇异矩阵(X^T X 不可逆)→ np.linalg.lstsq 走伪逆,k̂/c 会很大(标记 status='singular')
#   - k̂/ĉ 可能为负(反物理);这是数据驱动的解,不代表「真实物理参数」
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import numpy as np
import pandas as pd

# 同 batch:把 backtrace/ 加进 path 找 common.tsfresh_pipeline
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
from common import tsfresh_pipeline as P  # noqa: F401  (保持导入对称,后续可扩展)

CSV_OUT_DIR = 'data/projection'
KC_OUT_NAME = 'kc_estimates.csv'


def parse_args():
    p = argparse.ArgumentParser(
        description='从 movement_*.csv 闭式 OLS 估计每只票的 (k, c)',
    )
    p.add_argument(
        '--input', default=None,
        help=(
            '可选的股票列表 CSV(列:code;可选 name/index_code)。'
            '不传则自动扫描 data/projection/movement_*.csv 全部处理。'
        ),
    )
    p.add_argument('--limit', type=int, default=0, help='最多处理多少只;0 = 全部。')
    p.add_argument(
        '--min-valid-days', type=int, default=20,
        help='最少需要的有效观测天数(默认 20)。少于则跳过,避免噪声主导。',
    )
    p.add_argument(
        '--clip-extreme', type=float, default=10.0,
        help=(
            'k̂/ĉ 的截幅范围(默认 ±10)。OLS 在病态数据下可能给出极大值,'
            '截幅到 ±10 仅作 sanity 显示,不影响原始解。'
        ),
    )
    return p.parse_args()


def fit_one(movement_csv: str, stock_tag: str, index_tag: str,
            min_valid_days: int = 20, clip_extreme: float = 10.0):
    """对一只股票做闭式 OLS,返回 (k_hat, c_hat, f_self_loss, n_valid, status)。

    Args:
        movement_csv: data/projection/movement_<idx>_<stk>.csv 路径
        stock_tag / index_tag: 用于读 Δ 向量列
        min_valid_days: 少于该有效天数的票直接返回 status='too_few_days'
        clip_extreme: 截幅阈值(仅 sanity 用)

    Returns:
        dict: k_hat, c_hat, f_self_loss, n_valid_days, status
    """
    try:
        df = pd.read_csv(movement_csv)
    except Exception as e:
        return {
            'k_hat': np.nan, 'c_hat': np.nan,
            'f_self_loss': np.nan, 'n_valid_days': 0,
            'status': f'load_failed: {type(e).__name__}: {e}',
        }

    # 2-D Δ 向量(原始量纲)
    try:
        delta_u = df[[f'Move_Delta_Vol_{stock_tag}',
                      f'Move_Delta_Amt_{stock_tag}']].to_numpy()    # (T, 2)
        delta_v = df[[f'Move_Delta_Vol_{index_tag}',
                      f'Move_Delta_Amt_{index_tag}']].to_numpy()    # (T, 2)
        beta = df['Move_Proj_Coeff'].to_numpy()                    # (T,)
    except KeyError as e:
        return {
            'k_hat': np.nan, 'c_hat': np.nan,
            'f_self_loss': np.nan, 'n_valid_days': 0,
            'status': f'missing_col: {e}',
        }

    # ---- 重建 2-D u_vec / d_vec / a_u_vec / a_v_vec(同 compute_forces 内部逻辑) ----
    u_vec = delta_u - beta[:, None] * delta_v                       # (T, 2) 速度偏离
    d_vec = np.zeros_like(delta_u)
    if len(u_vec) >= 2:
        # d[1:] = Σ_{j<i} u[j],d[0] = 0
        d_vec[1:] = np.cumsum(u_vec[:-1], axis=0)

    a_u_vec = np.full_like(delta_u, np.nan)
    a_v_vec = np.full_like(delta_v, np.nan)
    if len(delta_u) >= 2:
        # 右补 NaN(末行 NaN),与 compute_forces 对齐
        a_u_vec[:-1] = np.diff(delta_u, axis=0)
        a_v_vec[:-1] = np.diff(delta_v, axis=0)

    # ---- 取有效观测(2-D 加速度末行 NaN 自动 drop) ----
    valid = (
        np.isfinite(a_u_vec).all(axis=1)
        & np.isfinite(a_v_vec).all(axis=1)
        & np.isfinite(d_vec).all(axis=1)
        & np.isfinite(u_vec).all(axis=1)
    )
    n_valid = int(valid.sum())
    if n_valid < max(3, min_valid_days):
        return {
            'k_hat': np.nan, 'c_hat': np.nan,
            'f_self_loss': np.nan, 'n_valid_days': n_valid,
            'status': f'too_few_days ({n_valid} < {min_valid_days})',
        }

    # ---- 构造 OLS 系统 ----
    # 方程:  a_S - β·a_M  =  -k·d  -  c·u  +  F_self
    #       (2-D 向量)(T,2)
    A_full = a_u_vec[valid] - beta[valid, None] * a_v_vec[valid]    # (N, 2)
    d_full = d_vec[valid]                                            # (N, 2)
    u_full = u_vec[valid]                                            # (N, 2)

    # 堆叠成 2N 方程:
    #   前 N 行(x 分量):A_x = -k·d_x - c·u_x
    #   后 N 行(y 分量):A_y = -k·d_y - c·u_y
    Y = np.concatenate([A_full[:, 0], A_full[:, 1]])                # (2N,)
    X = np.zeros((2 * n_valid, 2))
    X[:n_valid, 0] = -d_full[:, 0]
    X[:n_valid, 1] = -u_full[:, 0]
    X[n_valid:, 0] = -d_full[:, 1]
    X[n_valid:, 1] = -u_full[:, 1]

    # ---- OLS 解 ----
    theta, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    k_hat, c_hat = float(theta[0]), float(theta[1])

    # ---- 残差 = F_self 的 2-Norm ----
    F_self_pred = Y - X @ theta                                       # (2N,)
    f_self_loss = float(np.mean(F_self_pred ** 2))

    # ---- 状态标签 ----
    # k̂/ĉ 都 < 0 → 奇异(很可能是病态数据,不要直接用)
    finite = np.isfinite(k_hat) and np.isfinite(c_hat)
    extreme = (
        abs(k_hat) > clip_extreme or abs(c_hat) > clip_extreme
    )
    if not finite:
        status = 'solve_failed'
    elif rank < 2:
        status = 'singular'
    elif extreme:
        status = f'extreme (|k| or |c| > {clip_extreme:g})'
    else:
        sign_k = 'restoring' if k_hat >= 0 else 'anti-restoring'
        sign_c = 'damping' if c_hat >= 0 else 'anti-damping'
        status = f'ok ({sign_k}, {sign_c})'

    return {
        'k_hat': k_hat, 'c_hat': c_hat,
        'f_self_loss': f_self_loss, 'n_valid_days': n_valid,
        'status': status,
    }


def list_movement_csvs(input_csv: str | None):
    """扫描或读列表,返回 [(code, name|None, mv_csv_path, index_tag, stock_tag), ...]。

    不传 --input 时:扫描 data/projection/movement_*.csv,从文件名解析 index_tag / stock_tag。
    传 --input 时:读 CSV(列:code;可选 name/index_code),按 (index_code, code) 拼路径。
    """
    if input_csv:
        df = pd.read_csv(input_csv, dtype={'code': str, 'index_code': str})
        if 'code' not in df.columns:
            raise ValueError(f'--input {input_csv} 必须含 code 列')
        out = []
        for _, row in df.iterrows():
            code = str(row['code']).strip()
            name = str(row['name']).strip() if 'name' in df.columns and pd.notna(row['name']) else None
            # 允许 --input 提供 index_code 显式覆盖;否则走默认大盘
            if 'index_code' in df.columns and pd.notna(row['index_code']):
                index_code = str(row['index_code']).strip()
            else:
                # 默认:SH→上证综指 / SZ→深证成指(只是路径拼接的占位符)
                suf = code.split('.')[-1]
                index_code = '000001.SH' if suf == 'SH' else '399001.SZ'
            stock_tag = code.split('.')[0]
            index_tag = index_code.split('.')[0]
            mv_csv = os.path.join(CSV_OUT_DIR, f'movement_{index_tag}_{stock_tag}.csv')
            out.append((code, name, mv_csv, index_tag, stock_tag, index_code))
        return out

    # 默认:扫描 data/projection/movement_*.csv
    if not os.path.isdir(CSV_OUT_DIR):
        raise FileNotFoundError(f'{CSV_OUT_DIR} 不存在;先跑 batch --movement 跑出 movement_*.csv')
    out = []
    for fn in sorted(os.listdir(CSV_OUT_DIR)):
        if not fn.startswith('movement_') or not fn.endswith('.csv'):
            continue
        # movement_<INDEX_TAG>_<STOCK_TAG>.csv
        stem = fn[len('movement_'):-len('.csv')]
        parts = stem.split('_')
        if len(parts) < 2:
            continue
        # 行业指数 tag 可能含 '881386' 但与 stock_tag 都已无 '.',简单 split 取末段当 stock_tag
        index_tag = parts[0]
        stock_tag = '_'.join(parts[1:])  # 防止未来出现含下划线的 tag
        mv_csv = os.path.join(CSV_OUT_DIR, fn)
        # 从 batch_manifest 反查 code / name / index_code / index_name(更可靠)
        # 没有 manifest 时只能从 tag 推 code
        suf = stock_tag[:6]   # 6 位数字,粗略区分 SH / SZ — 不准,只是占位
        code_guess = stock_tag + ('.SH' if suf.startswith(('6', '9', '5')) else '.SZ')
        out.append((code_guess, None, mv_csv, index_tag, stock_tag, index_tag + '.SH'))
    return out


def main():
    args = parse_args()
    os.makedirs(CSV_OUT_DIR, exist_ok=True)

    targets = list_movement_csvs(args.input)
    if args.limit > 0:
        targets = targets[:args.limit]

    print(f'输入: {args.input or f"{CSV_OUT_DIR}/movement_*.csv (扫描)"}')
    print(f'目标: {len(targets)} 只 (limit={args.limit})')
    print(f'最少有效天数: {args.min_valid_days}')
    print(f'截幅阈值: |k|,|c| ≤ {args.clip_extreme}')
    print()

    rows = []
    for i, (code, name, mv_csv, index_tag, stock_tag, index_code) in enumerate(targets, 1):
        label = f'{code} ({name})' if name else code
        print(f'[{i}/{len(targets)}] {label} ...', end=' ', flush=True)
        result = fit_one(
            mv_csv, stock_tag, index_tag,
            min_valid_days=args.min_valid_days,
            clip_extreme=args.clip_extreme,
        )
        result['code'] = code
        result['name'] = name or ''
        result['index_code'] = index_code
        result['index_tag'] = index_tag
        result['stock_tag'] = stock_tag
        rows.append(result)
        k_str = f'{result["k_hat"]:+.4f}' if np.isfinite(result['k_hat']) else 'NaN'
        c_str = f'{result["c_hat"]:+.4f}' if np.isfinite(result["c_hat"]) else 'NaN'
        print(f'k={k_str} c={c_str} F²={result["f_self_loss"]:.2e} ({result["n_valid_days"]}d) {result["status"]}')

    out_df = pd.DataFrame(rows, columns=[
        'code', 'name', 'index_code', 'index_tag', 'stock_tag',
        'k_hat', 'c_hat', 'f_self_loss', 'n_valid_days', 'status',
    ])
    out_path = os.path.join(CSV_OUT_DIR, KC_OUT_NAME)
    out_df.to_csv(out_path, index=False, encoding='utf-8')

    ok = sum(1 for r in rows if r['status'].startswith('ok'))
    singular = sum(1 for r in rows if r['status'] == 'singular')
    too_few = sum(1 for r in rows if r['status'].startswith('too_few_days'))
    fail = len(rows) - ok - singular - too_few

    print(f'\n=== 汇总 ===')
    print(f'  ok:       {ok}/{len(rows)}')
    if singular:
        print(f'  singular: {singular}/{len(rows)}')
    if too_few:
        print(f'  too_few:  {too_few}/{len(rows)}')
    if fail:
        print(f'  other:    {fail}/{len(rows)}')
    print(f'  清单: {out_path}')

    # 顺便打一下「ok 集合」的 k/c 分布
    if ok > 0:
        k_vals = np.array([r['k_hat'] for r in rows if r['status'].startswith('ok')])
        c_vals = np.array([r['c_hat'] for r in rows if r['status'].startswith('ok')])
        f_vals = np.array([r['f_self_loss'] for r in rows if r['status'].startswith('ok')])
        print(f'\n=== ok 子集分布 ===')
        print(f'  k_hat: median={np.median(k_vals):+.4f} '
              f'p25={np.percentile(k_vals, 25):+.4f} '
              f'p75={np.percentile(k_vals, 75):+.4f}')
        print(f'  c_hat: median={np.median(c_vals):+.4f} '
              f'p25={np.percentile(c_vals, 25):+.4f} '
              f'p75={np.percentile(c_vals, 75):+.4f}')
        print(f'  F²:    median={np.median(f_vals):.2e} '
              f'max={np.max(f_vals):.2e}')


if __name__ == '__main__':
    main()