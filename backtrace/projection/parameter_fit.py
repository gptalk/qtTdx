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
#
# 输出(默认全样本):
#   data/projection/kc_estimates.csv
#       列:code, name, index_code, index_tag, stock_tag,
#          k_hat, c_hat, f_self_loss, n_valid_days, status
#
# 输出(--rolling-fit 滚动拟合):
#   data/projection/kc_rolling_<idx>_<stk>.csv — 每只票 × 每窗口一行,8 列
#       列:window, window_start, window_end, k_hat, c_hat,
#          f_self_loss, n_valid_days, status
#       每个窗口 = 末 N 行(end-aligned),N 由 --rolling-windows 控制
#   data/projection/kc_rolling_summary.csv — 每只票一行汇总
#       列:code, name, index_code, index_tag, stock_tag, windows,
#          k_<w1>, c_<w1>, f2_<w1>, n_<w1>, status_<w1>, ...
#       横轴 = 各窗口的 (k̂, ĉ, F²) 直读,一眼看「窗口越长 k̂/ĉ 越漂到哪」
#
# 用法:
#   PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py
#   PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --limit 10  # 冒烟
#   PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --input <path.csv>
#       # 自定义股票列表(列:code;可选 name/index_code),只跑列表里的票
#
#   # 滚动拟合(2026-08-16 新增):每个窗口单独 OLS,看 k̂/ĉ 时序漂移
#   PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --rolling-fit --limit 10
#   PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --rolling-fit \
#       --rolling-windows 30,60,120,240 --limit 20
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
#   - 滚动拟合:窗口太小(< 30)→ 欠定,可能 rank < 2 → status='singular',慎用
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
    p.add_argument(
        '--rolling-fit', action='store_true',
        help=(
            '在每个时间窗口分别跑 OLS,产 k̂/ĉ 时序漂移。'
            '窗口由 --rolling-windows 控制(默认 60,120,240 三个 end-aligned 窗口)。'
            '输出 kc_rolling_<idx>_<stk>.csv(每只票每窗口一行)+ '
            'kc_rolling_summary.csv(每只票一行汇总)。'
        ),
    )
    p.add_argument(
        '--rolling-windows', default='60,120,240',
        help='滚动窗口大小,逗号分隔(默认 60,120,240)。如 "30,60,120,240"。',
    )
    return p.parse_args()


def _solve_ols(a_u_vec: np.ndarray, a_v_vec: np.ndarray,
               d_vec: np.ndarray, u_vec: np.ndarray,
               beta: np.ndarray, valid: np.ndarray):
    """核心 OLS 解(内部函数,fit_one 和 fit_rolling 复用)。

    输入:从 movement CSV 重建的 2-D 向量 + valid mask。
    输出:(k_hat, c_hat, f_self_loss, n_valid, rank)
    """
    n_valid = int(valid.sum())
    A_full = a_u_vec[valid] - beta[valid, None] * a_v_vec[valid]
    d_full = d_vec[valid]
    u_full = u_vec[valid]

    Y = np.concatenate([A_full[:, 0], A_full[:, 1]])
    X = np.zeros((2 * n_valid, 2))
    X[:n_valid, 0] = -d_full[:, 0]
    X[:n_valid, 1] = -u_full[:, 0]
    X[n_valid:, 0] = -d_full[:, 1]
    X[n_valid:, 1] = -u_full[:, 1]

    theta, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    k_hat, c_hat = float(theta[0]), float(theta[1])

    F_self_pred = Y - X @ theta
    f_self_loss = float(np.mean(F_self_pred ** 2))
    return k_hat, c_hat, f_self_loss, n_valid, int(rank)


def _load_movement(movement_csv: str, stock_tag: str, index_tag: str):
    """读 movement CSV → 返回 (df, delta_u, delta_v, beta) 或 None(失败)。"""
    try:
        df = pd.read_csv(movement_csv)
    except Exception as e:
        return None, f'load_failed: {type(e).__name__}: {e}'
    try:
        delta_u = df[[f'Move_Delta_Vol_{stock_tag}',
                      f'Move_Delta_Amt_{stock_tag}']].to_numpy()
        delta_v = df[[f'Move_Delta_Vol_{index_tag}',
                      f'Move_Delta_Amt_{index_tag}']].to_numpy()
        beta = df['Move_Proj_Coeff'].to_numpy()
    except KeyError as e:
        return None, f'missing_col: {e}'
    return (df, delta_u, delta_v, beta), None


def _build_kinematics(delta_u, delta_v, beta):
    """从 Δ 向量重建 u_vec / d_vec / a_u_vec / a_v_vec。"""
    u_vec = delta_u - beta[:, None] * delta_v
    d_vec = np.zeros_like(delta_u)
    if len(u_vec) >= 2:
        d_vec[1:] = np.cumsum(u_vec[:-1], axis=0)
    a_u_vec = np.full_like(delta_u, np.nan)
    a_v_vec = np.full_like(delta_v, np.nan)
    if len(delta_u) >= 2:
        a_u_vec[:-1] = np.diff(delta_u, axis=0)
        a_v_vec[:-1] = np.diff(delta_v, axis=0)
    return u_vec, d_vec, a_u_vec, a_v_vec


def fit_one(movement_csv: str, stock_tag: str, index_tag: str,
            min_valid_days: int = 20, clip_extreme: float = 10.0):
    """对一只股票的全样本做闭式 OLS,返回 (k_hat, c_hat, f_self_loss, n_valid, status)。"""
    loaded, err = _load_movement(movement_csv, stock_tag, index_tag)
    if loaded is None:
        return {
            'k_hat': np.nan, 'c_hat': np.nan,
            'f_self_loss': np.nan, 'n_valid_days': 0,
            'status': err,
        }
    df, delta_u, delta_v, beta = loaded
    u_vec, d_vec, a_u_vec, a_v_vec = _build_kinematics(delta_u, delta_v, beta)

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

    k_hat, c_hat, f_self_loss, _, rank = _solve_ols(
        a_u_vec, a_v_vec, d_vec, u_vec, beta, valid,
    )

    finite = np.isfinite(k_hat) and np.isfinite(c_hat)
    extreme = abs(k_hat) > clip_extreme or abs(c_hat) > clip_extreme
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


def fit_rolling(movement_csv: str, stock_tag: str, index_tag: str,
                windows: list[int], clip_extreme: float = 10.0):
    """对一只股票做 end-aligned 滚动 OLS,产 (k̂, ĉ) 时序。

    Args:
        movement_csv / stock_tag / index_tag: 同 fit_one
        windows: 窗口大小列表(交易日)。每个窗口 = 最后 N 行
                 (e.g. windows=[60,120,240] → 末 60 天、末 120 天、末 240 天)

    Returns:
        list[dict]:每 dict 含 'window', 'window_start', 'window_end',
                    'k_hat', 'c_hat', 'f_self_loss', 'n_valid_days', 'status'
    """
    loaded, err = _load_movement(movement_csv, stock_tag, index_tag)
    if loaded is None:
        return [{
            'window': w, 'window_start': '', 'window_end': '',
            'k_hat': np.nan, 'c_hat': np.nan,
            'f_self_loss': np.nan, 'n_valid_days': 0,
            'status': err,
        } for w in windows]
    df, delta_u, delta_v, beta = loaded
    u_vec, d_vec, a_u_vec, a_v_vec = _build_kinematics(delta_u, delta_v, beta)

    T = len(delta_u)
    out = []
    for w in windows:
        s = max(0, T - w)
        sub = slice(s, T)
        valid = (
            np.isfinite(a_u_vec[sub]).all(axis=1)
            & np.isfinite(a_v_vec[sub]).all(axis=1)
            & np.isfinite(d_vec[sub]).all(axis=1)
            & np.isfinite(u_vec[sub]).all(axis=1)
        )
        n_valid = int(valid.sum())
        if n_valid < 3:
            out.append({
                'window': w,
                'window_start': str(df['Date'].iloc[s])[:10] if s < T else '',
                'window_end': str(df['Date'].iloc[T - 1])[:10] if T > 0 else '',
                'k_hat': np.nan, 'c_hat': np.nan,
                'f_self_loss': np.nan, 'n_valid_days': n_valid,
                'status': f'too_few_days ({n_valid})',
            })
            continue
        try:
            k_hat, c_hat, f_loss, _, rank = _solve_ols(
                a_u_vec[sub], a_v_vec[sub], d_vec[sub], u_vec[sub], beta[sub], valid,
            )
        except Exception as e:
            out.append({
                'window': w,
                'window_start': str(df['Date'].iloc[s])[:10],
                'window_end': str(df['Date'].iloc[T - 1])[:10],
                'k_hat': np.nan, 'c_hat': np.nan,
                'f_self_loss': np.nan, 'n_valid_days': n_valid,
                'status': f'solve_failed: {type(e).__name__}: {e}',
            })
            continue
        finite = np.isfinite(k_hat) and np.isfinite(c_hat)
        extreme = abs(k_hat) > clip_extreme or abs(c_hat) > clip_extreme
        if not finite:
            status = 'solve_failed'
        elif rank < 2:
            status = 'singular'
        elif extreme:
            status = f'extreme (|k| or |c| > {clip_extreme:g})'
        else:
            status = 'ok'
        out.append({
            'window': w,
            'window_start': str(df['Date'].iloc[s])[:10],
            'window_end': str(df['Date'].iloc[T - 1])[:10],
            'k_hat': k_hat, 'c_hat': c_hat,
            'f_self_loss': f_loss,
            'n_valid_days': n_valid,
            'status': status,
        })
    return out


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


def _parse_windows(spec: str) -> list[int]:
    """解析 '--rolling-windows 60,120,240' → [60, 120, 240]。"""
    out = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        n = int(tok)
        if n <= 0:
            raise ValueError(f'窗口大小必须为正整数,got {n}')
        out.append(n)
    if not out:
        raise ValueError('--rolling-windows 解析后为空')
    return out


def main():
    args = parse_args()
    os.makedirs(CSV_OUT_DIR, exist_ok=True)

    targets = list_movement_csvs(args.input)
    if args.limit > 0:
        targets = targets[:args.limit]

    if args.rolling_fit:
        try:
            windows = _parse_windows(args.rolling_windows)
        except ValueError as e:
            raise SystemExit(f'--rolling-windows 解析失败: {e}')
        windows = sorted(set(windows))   # 去重 + 升序,输出顺序与写入顺序对齐

    print(f'输入: {args.input or f"{CSV_OUT_DIR}/movement_*.csv (扫描)"}')
    print(f'目标: {len(targets)} 只 (limit={args.limit})')
    print(f'截幅阈值: |k|,|c| ≤ {args.clip_extreme}')
    if args.rolling_fit:
        print(f'滚动模式: 窗口={windows}')
    else:
        print(f'最少有效天数: {args.min_valid_days}')
    print()

    if args.rolling_fit:
        main_rolling(targets, windows, clip_extreme=args.clip_extreme)
    else:
        main_fit_all(targets,
                     min_valid_days=args.min_valid_days,
                     clip_extreme=args.clip_extreme)


def main_fit_all(targets, min_valid_days: int, clip_extreme: float):
    """默认分支:对每只票的全样本跑 OLS,产 kc_estimates.csv。"""
    rows = []
    for i, (code, name, mv_csv, index_tag, stock_tag, index_code) in enumerate(targets, 1):
        label = f'{code} ({name})' if name else code
        print(f'[{i}/{len(targets)}] {label} ...', end=' ', flush=True)
        result = fit_one(
            mv_csv, stock_tag, index_tag,
            min_valid_days=min_valid_days,
            clip_extreme=clip_extreme,
        )
        result['code'] = code
        result['name'] = name or ''
        result['index_code'] = index_code
        result['index_tag'] = index_tag
        result['stock_tag'] = stock_tag
        rows.append(result)
        k_str = f'{result["k_hat"]:+.4f}' if np.isfinite(result['k_hat']) else 'NaN'
        c_str = f'{result["c_hat"]:+.4f}' if np.isfinite(result["c_hat"]) else 'NaN'
        print(f'k={k_str} c={c_str} F²={result["f_self_loss"]:.2e} '
              f'({result["n_valid_days"]}d) {result["status"]}')

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


def main_rolling(targets, windows: list[int], clip_extreme: float):
    """滚动拟合分支:对每只票产 kc_rolling_<idx>_<stk>.csv + kc_rolling_summary.csv。

    per-stock CSV:窗口大小 × {k̂, ĉ, F², n_valid, status} 多个 fit 点(end-aligned 末 N 行)。
    summary CSV:每只票一行,横轴 = 各窗口的 (k̂, ĉ, F²) 直读,便于一眼看「窗口越长 k̂/ĉ 越
    漂到哪」。
    """
    # 准备 summary 的列名:动态按 windows 展开
    base_cols = ['code', 'name', 'index_code', 'index_tag', 'stock_tag', 'windows']
    summary_cols = list(base_cols)
    for w in windows:
        summary_cols.extend([f'k_{w}', f'c_{w}', f'f2_{w}', f'n_{w}', f'status_{w}'])

    summary_rows = []
    for i, (code, name, mv_csv, index_tag, stock_tag, index_code) in enumerate(targets, 1):
        label = f'{code} ({name})' if name else code
        print(f'[{i}/{len(targets)}] {label} ...', end=' ', flush=True)
        rows = fit_rolling(
            mv_csv, stock_tag, index_tag,
            windows=windows, clip_extreme=clip_extreme,
        )

        # 落 per-stock per-window CSV
        stock_csv = os.path.join(CSV_OUT_DIR, f'kc_rolling_{index_tag}_{stock_tag}.csv')
        per_stock_df = pd.DataFrame(rows, columns=[
            'window', 'window_start', 'window_end',
            'k_hat', 'c_hat', 'f_self_loss', 'n_valid_days', 'status',
        ])
        per_stock_df.to_csv(stock_csv, index=False, encoding='utf-8')

        # 拼一行 summary
        srow = {
            'code': code,
            'name': name or '',
            'index_code': index_code,
            'index_tag': index_tag,
            'stock_tag': stock_tag,
            'windows': ','.join(str(w) for w in windows),
        }
        for r in rows:
            w = r['window']
            srow[f'k_{w}'] = r['k_hat']
            srow[f'c_{w}'] = r['c_hat']
            srow[f'f2_{w}'] = r['f_self_loss']
            srow[f'n_{w}'] = r['n_valid_days']
            srow[f'status_{w}'] = r['status']
        summary_rows.append(srow)

        # 进度行:各窗口 k/c 紧凑打
        parts = []
        for r in rows:
            k = r['k_hat']; c = r['c_hat']
            ks = f'{k:+.4f}' if np.isfinite(k) else 'NaN'
            cs = f'{c:+.4f}' if np.isfinite(c) else 'NaN'
            parts.append(f'w={r["window"]}: k={ks} c={cs}')
        print(' | '.join(parts))
        print(f'    → {stock_csv}')

    # 落 summary CSV
    summary_df = pd.DataFrame(summary_rows, columns=summary_cols)
    summary_path = os.path.join(CSV_OUT_DIR, 'kc_rolling_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')

    # 汇总统计:每个窗口的 k̂/ĉ 跨股票分布
    print(f'\n=== 滚动拟合汇总 ({len(summary_rows)} 只 × {len(windows)} 窗口) ===')
    print(f'  per-stock CSV: data/projection/kc_rolling_<idx>_<stk>.csv')
    print(f'  summary CSV:   {summary_path}')
    for w in windows:
        k_col = f'k_{w}'
        c_col = f'c_{w}'
        k_arr = pd.to_numeric(summary_df[k_col], errors='coerce').to_numpy()
        c_arr = pd.to_numeric(summary_df[c_col], errors='coerce').to_numpy()
        k_arr = k_arr[np.isfinite(k_arr)]
        c_arr = c_arr[np.isfinite(c_arr)]
        if len(k_arr) == 0:
            print(f'  w={w}: 无有效拟合')
            continue
        print(f'  w={w}: k̂ med={np.median(k_arr):+.4f} '
              f'(p25={np.percentile(k_arr,25):+.4f}, p75={np.percentile(k_arr,75):+.4f}, '
              f'n={len(k_arr)}) | ĉ med={np.median(c_arr):+.4f}')


if __name__ == '__main__':
    main()