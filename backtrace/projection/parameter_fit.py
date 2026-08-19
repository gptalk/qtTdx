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
# 输出(--rolling-fit --plot-rolling HTML 可视化,2026-08-16 新增):
#   backtrace/outputs/kc_rolling_<idx>_<stk>.html — 每只票 1 个,4 子图
#       (k̂ / ĉ / F² / n_valid 随窗口变化)
#   backtrace/outputs/kc_rolling_aggregate.html — 跨股票 1 个,2 子图
#       (k̂ / ĉ 中位数 ± p25/p75 区间)
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
#   # 滚动拟合 + HTML 可视化(2026-08-16 新增):每只票 4 子图 + 跨股票聚合
#   PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --rolling-fit \
#       --plot-rolling --limit 10
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

OUT_HTML_DIR = 'backtrace/outputs'   # rolling-fit HTML 输出目录

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
    p.add_argument(
        '--rolling-time', action='store_true',
        help='每月末用最近 N 天 OLS 估 (k̂, ĉ),产 kc_estimates_time.csv (long format)',
    )
    p.add_argument(
        '--rolling-time-window', type=int, default=240,
        help='rolling-time 模式窗口大小(交易日,默认 240)',
    )
    p.add_argument(
        '--plot-rolling', action='store_true',
        help=(
            '在 --rolling-fit 之上叠加 HTML 可视化:'
            '每只票一个 HTML,展示 (k̂, ĉ, F², n_valid) 在不同窗口下的漂移;'
            '另产一个聚合 HTML,跨股票汇总中位数分布。'
            '依赖 plotly。输出到 backtrace/outputs/。'
        ),
    )
    return p.parse_args()


def _solve_ols(a_u_vec: np.ndarray, a_v_vec: np.ndarray,
               d_vec: np.ndarray, u_vec: np.ndarray,
               beta: np.ndarray, valid: np.ndarray):
    """核心 OLS 解(内部函数,fit_one 和 fit_rolling 复用)。

    输入:从 movement CSV 重建的 2-D 向量 + valid mask。
    输出 (8-tuple):
        k_hat, c_hat, f_residual_loss, n_valid, rank,
        condition_number, regressor_corr, r2
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
    # 兜底:X 全 0 时 lstsq 返回空 theta → 强制填 0,避免下游 float() 崩
    if theta.size == 0:
        theta = np.zeros(2)
    k_hat, c_hat = float(theta[0]), float(theta[1])

    F_self_pred = Y - X @ theta
    f_residual_loss = float(np.mean(F_self_pred ** 2))

    # === v0 diagnostics (post-processing, 不动 OLS) ===

    # condition_number: cond(X), NOT cond(X.T @ X) — 后者 κ² 失真
    condition_number = float(np.linalg.cond(X)) if X.size > 0 else np.nan

    # regressor_corr: X 两列 = -d, -u 的相关系数
    if X.shape[0] >= 2 and X.shape[1] == 2:
        col0, col1 = X[:, 0], X[:, 1]
        std0, std1 = float(np.std(col0)), float(np.std(col1))
        if std0 > 1e-12 and std1 > 1e-12:
            regressor_corr = float(np.corrcoef(col0, col1)[0, 1])
        else:
            regressor_corr = np.nan
    else:
        regressor_corr = np.nan

    # R² = 1 - SS_res / SS_tot,SS_tot ≈ 0 → NaN
    y_mean = float(np.mean(Y))
    ss_tot = float(np.sum((Y - y_mean) ** 2))
    ss_res = float(np.sum(F_self_pred ** 2))
    if ss_tot <= 1e-12:
        r2 = np.nan
    else:
        r2 = 1.0 - ss_res / ss_tot

    return (k_hat, c_hat, f_residual_loss, n_valid, int(rank),
            condition_number, regressor_corr, r2)


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
    """对一只股票的全样本做闭式 OLS,返回 dict。

    新增 v0 字段: rank, condition_number, regressor_corr, r2,
                  identification_status, fit_quality, f_residual_loss
    """
    loaded, err = _load_movement(movement_csv, stock_tag, index_tag)
    if loaded is None:
        return {
            'k_hat': np.nan, 'c_hat': np.nan,
            'f_self_loss': np.nan, 'f_residual_loss': np.nan,
            'n_valid_days': 0,
            'rank': 0, 'condition_number': np.nan,
            'regressor_corr': np.nan, 'r2': np.nan,
            'identification_status': 'singular',
            'fit_quality': 'uninformative',
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
            'f_self_loss': np.nan, 'f_residual_loss': np.nan,
            'n_valid_days': n_valid,
            'rank': 0, 'condition_number': np.nan,
            'regressor_corr': np.nan, 'r2': np.nan,
            'identification_status': 'singular',
            'fit_quality': 'uninformative',
            'status': f'too_few_days ({n_valid} < {min_valid_days})',
        }

    k_hat, c_hat, f_residual_loss, _, rank, condition_number, regressor_corr, r2 = _solve_ols(
        a_u_vec, a_v_vec, d_vec, u_vec, beta, valid,
    )

    # === classification ===
    finite = np.isfinite(k_hat) and np.isfinite(c_hat)
    extreme = abs(k_hat) > clip_extreme or abs(c_hat) > clip_extreme

    # identification_status: 仅看 rank + cond
    if not finite or rank < 2:
        identification_status = 'singular'
    elif condition_number >= 1e5:
        identification_status = 'unidentifiable'
    elif condition_number >= 1e3:
        identification_status = 'ill_conditioned'
    else:
        identification_status = 'well_conditioned'

    # fit_quality: 仅看 R²
    if not np.isfinite(r2):
        fit_quality = 'uninformative'
    elif r2 < 0.01:
        fit_quality = 'poor'
    elif r2 < 0.1:
        fit_quality = 'weak'
    else:
        fit_quality = 'good'

    # 旧 status (verbose, 向后兼容)
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
        'f_self_loss': f_residual_loss,  # alias for backward compat
        'f_residual_loss': f_residual_loss,
        'n_valid_days': n_valid,
        'rank': rank,
        'condition_number': condition_number,
        'regressor_corr': regressor_corr,
        'r2': r2,
        'identification_status': identification_status,
        'fit_quality': fit_quality,
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
                    'k_hat', 'c_hat', 'f_residual_loss', 'n_valid_days', 'status',
                    + 7 新字段: rank, condition_number, regressor_corr, r2,
                                identification_status, fit_quality
    """
    loaded, err = _load_movement(movement_csv, stock_tag, index_tag)
    if loaded is None:
        return [{
            'window': w, 'window_start': '', 'window_end': '',
            'k_hat': np.nan, 'c_hat': np.nan,
            'f_self_loss': np.nan, 'f_residual_loss': np.nan,
            'n_valid_days': 0,
            'rank': 0, 'condition_number': np.nan,
            'regressor_corr': np.nan, 'r2': np.nan,
            'identification_status': 'singular',
            'fit_quality': 'uninformative',
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
                'f_self_loss': np.nan, 'f_residual_loss': np.nan,
                'n_valid_days': n_valid,
                'rank': 0, 'condition_number': np.nan,
                'regressor_corr': np.nan, 'r2': np.nan,
                'identification_status': 'singular',
                'fit_quality': 'uninformative',
                'status': f'too_few_days ({n_valid})',
            })
            continue
        try:
            k_hat, c_hat, f_residual_loss, _, rank, condition_number, regressor_corr, r2 = _solve_ols(
                a_u_vec[sub], a_v_vec[sub], d_vec[sub], u_vec[sub], beta[sub], valid,
            )
        except Exception as e:
            out.append({
                'window': w,
                'window_start': str(df['Date'].iloc[s])[:10],
                'window_end': str(df['Date'].iloc[T - 1])[:10],
                'k_hat': np.nan, 'c_hat': np.nan,
                'f_self_loss': np.nan, 'f_residual_loss': np.nan,
                'n_valid_days': n_valid,
                'rank': 0, 'condition_number': np.nan,
                'regressor_corr': np.nan, 'r2': np.nan,
                'identification_status': 'singular',
                'fit_quality': 'uninformative',
                'status': f'solve_failed: {type(e).__name__}: {e}',
            })
            continue
        finite = np.isfinite(k_hat) and np.isfinite(c_hat)
        extreme = abs(k_hat) > clip_extreme or abs(c_hat) > clip_extreme

        # identification_status: 仅看 rank + cond
        if not finite or rank < 2:
            identification_status = 'singular'
        elif condition_number >= 1e5:
            identification_status = 'unidentifiable'
        elif condition_number >= 1e3:
            identification_status = 'ill_conditioned'
        else:
            identification_status = 'well_conditioned'

        # fit_quality: 仅看 R²
        if not np.isfinite(r2):
            fit_quality = 'uninformative'
        elif r2 < 0.01:
            fit_quality = 'poor'
        elif r2 < 0.1:
            fit_quality = 'weak'
        else:
            fit_quality = 'good'

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
            'f_self_loss': f_residual_loss,  # alias for backward compat
            'f_residual_loss': f_residual_loss,
            'n_valid_days': n_valid,
            'rank': rank,
            'condition_number': condition_number,
            'regressor_corr': regressor_corr,
            'r2': r2,
            'identification_status': identification_status,
            'fit_quality': fit_quality,
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
        # 大盘 index_code 后缀按指数 tag 区分:SH 指数(000001/881xxx)→ .SH,
        # SZ 指数(399001/399xxx)→ .SZ。hardcode '.SH' 会让所有 SZ 指数变成假键,
        # 致下游按 (code, index_code) groupby 时整条 series 被丢弃。
        index_suffix = '.SZ' if index_tag.startswith(('399', '39')) else '.SH'
        index_code = index_tag + index_suffix
        out.append((code_guess, None, mv_csv, index_tag, stock_tag, index_code))
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


def plot_rolling_per_stock(srow: dict, windows: list[int],
                          code: str, name: str | None,
                          index_tag: str, stock_tag: str) -> str:
    """单只票滚动拟合 HTML:4 子图(k̂ / ĉ / F² / n_valid 随窗口变化)。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    ws = np.array(windows)
    ks = np.array([float(srow[f'k_{w}']) for w in windows])
    cs = np.array([float(srow[f'c_{w}']) for w in windows])
    f2 = np.array([float(srow[f'f2_{w}']) for w in windows])
    ns = np.array([int(srow[f'n_{w}']) for w in windows])
    sts = [srow[f'status_{w}'] for w in windows]
    title = f'{code} ({name}) → {index_tag} 滚动 OLS(k̂/ĉ 时序漂移)'
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True,
        subplot_titles=(
            'k̂ 恢复力系数',
            'ĉ 阻尼系数',
            '‖F_self‖² 平均(原始量纲)',
            '有效观测天数 n',
        ),
        vertical_spacing=0.14, horizontal_spacing=0.10,
    )
    fig.add_trace(go.Scatter(
        x=ws, y=ks, mode='lines+markers', name='k̂',
        line=dict(color='cyan', width=2), marker=dict(size=10),
        text=sts, hovertemplate='w=%{x}<br>k̂=%{y:+.4f}<br>status: %{text}<extra></extra>',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[ws[0], ws[-1]], y=[0, 0], mode='lines', name='k̂=0',
        line=dict(color='gray', dash='dash'), showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ws, y=cs, mode='lines+markers', name='ĉ',
        line=dict(color='orange', width=2), marker=dict(size=10),
        text=sts, hovertemplate='w=%{x}<br>ĉ=%{y:+.4f}<br>status: %{text}<extra></extra>',
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[ws[0], ws[-1]], y=[0, 0], mode='lines', name='ĉ=0',
        line=dict(color='gray', dash='dash'), showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=ws, y=f2, mode='lines+markers', name='‖F_self‖²',
        line=dict(color='magenta', width=2), marker=dict(size=10),
        hovertemplate='w=%{x}<br>F²=%{y:.2e}<extra></extra>',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=ws, y=ns, mode='lines+markers', name='n_valid_days',
        line=dict(color='lime', width=2), marker=dict(size=10),
        hovertemplate='w=%{x}<br>n=%{y}<extra></extra>',
    ), row=2, col=2)
    fig.update_xaxes(title_text='窗口大小(交易日)', row=2, col=1)
    fig.update_xaxes(title_text='窗口大小(交易日)', row=2, col=2)
    fig.update_yaxes(title_text='k̂', row=1, col=1)
    fig.update_yaxes(title_text='ĉ', row=1, col=2)
    fig.update_yaxes(title_text='F²(原始量纲)', type='log', row=2, col=1)
    fig.update_yaxes(title_text='天数', row=2, col=2)
    fig.update_layout(
        template='plotly_dark', height=700, width=1100,
        title_text=title,
        legend=dict(orientation='h', yanchor='bottom', y=-0.18, xanchor='right', x=1),
    )
    out = os.path.join(OUT_HTML_DIR, f'kc_rolling_{index_tag}_{stock_tag}.html')
    fig.write_html(out)
    return out


def plot_rolling_aggregate(summary_df: pd.DataFrame, windows: list[int]) -> str:
    """跨股票滚动拟合 HTML:每窗口 k̂/ĉ 中位数 ± p25/p75 区间。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    ws = np.array(windows)
    k_med, k_p25, k_p75 = [], [], []
    c_med, c_p25, c_p75 = [], [], []
    for w in windows:
        k_arr = pd.to_numeric(summary_df[f'k_{w}'], errors='coerce').to_numpy()
        c_arr = pd.to_numeric(summary_df[f'c_{w}'], errors='coerce').to_numpy()
        k_arr = k_arr[np.isfinite(k_arr)]
        c_arr = c_arr[np.isfinite(c_arr)]
        if len(k_arr) == 0:
            k_med.append(np.nan); k_p25.append(np.nan); k_p75.append(np.nan)
        else:
            k_med.append(np.median(k_arr))
            k_p25.append(np.percentile(k_arr, 25))
            k_p75.append(np.percentile(k_arr, 75))
        if len(c_arr) == 0:
            c_med.append(np.nan); c_p25.append(np.nan); c_p75.append(np.nan)
        else:
            c_med.append(np.median(c_arr))
            c_p25.append(np.percentile(c_arr, 25))
            c_p75.append(np.percentile(c_arr, 75))
    k_med, k_p25, k_p75 = map(np.array, [k_med, k_p25, k_p75])
    c_med, c_p25, c_p75 = map(np.array, [c_med, c_p25, c_p75])
    fig = make_subplots(
        rows=1, cols=2, shared_xaxes=True,
        subplot_titles=('k̂ 跨股票中位数 ± p25/p75', 'ĉ 跨股票中位数 ± p25/p75'),
        horizontal_spacing=0.10,
    )
    # k̂ panel — band + median line
    fig.add_trace(go.Scatter(
        x=list(ws) + list(ws[::-1]),
        y=list(k_p75) + list(k_p25[::-1]),
        fill='toself', fillcolor='rgba(0,255,255,0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        name='p25-p75', showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ws, y=k_med, mode='lines+markers', name='k̂ med',
        line=dict(color='cyan', width=3), marker=dict(size=10),
    ), row=1, col=1)
    # ĉ panel
    fig.add_trace(go.Scatter(
        x=list(ws) + list(ws[::-1]),
        y=list(c_p75) + list(c_p25[::-1]),
        fill='toself', fillcolor='rgba(255,165,0,0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        name='p25-p75', showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=ws, y=c_med, mode='lines+markers', name='ĉ med',
        line=dict(color='orange', width=3), marker=dict(size=10),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[ws[0], ws[-1]], y=[0, 0], mode='lines',
        line=dict(color='gray', dash='dash'), name='y=0',
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[ws[0], ws[-1]], y=[0, 0], mode='lines',
        line=dict(color='gray', dash='dash'), name='y=0',
        showlegend=False,
    ), row=1, col=2)
    fig.update_xaxes(title_text='窗口大小(交易日)', row=1, col=1)
    fig.update_xaxes(title_text='窗口大小(交易日)', row=1, col=2)
    fig.update_yaxes(title_text='k̂', row=1, col=1)
    fig.update_yaxes(title_text='ĉ', row=1, col=2)
    fig.update_layout(
        template='plotly_dark', height=500, width=1100,
        title_text=f'滚动拟合跨股票汇总({len(summary_df)} 只 × {len(windows)} 窗口)',
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='right', x=1),
    )
    out = os.path.join(OUT_HTML_DIR, 'kc_rolling_aggregate.html')
    fig.write_html(out)
    return out


def build_identifiability_distribution_html(kc_df: pd.DataFrame, output_path: str) -> str:
    """4 子图 plotly:R² 直方图 / cond 直方图 / R² vs |k̂| / (k̂, ĉ) 散点按 R² 着色。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    r2 = pd.to_numeric(kc_df['r2'], errors='coerce').to_numpy()
    cond = pd.to_numeric(kc_df['condition_number'], errors='coerce').to_numpy()
    k_abs = np.abs(pd.to_numeric(kc_df['k_hat'], errors='coerce').to_numpy())
    k = pd.to_numeric(kc_df['k_hat'], errors='coerce').to_numpy()
    c = pd.to_numeric(kc_df['c_hat'], errors='coerce').to_numpy()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'R² 直方图(模型解释力)',
            'cond(X) 直方图(数值可识别性,对数轴)',
            'R² vs |k̂| 散点(低 R² → 参数爆炸?)',
            '(k̂, ĉ) 散点(颜色 = R²)',
        ),
        vertical_spacing=0.15, horizontal_spacing=0.10,
    )

    # (1,1) R² histogram
    r2_finite = r2[np.isfinite(r2)]
    fig.add_trace(go.Histogram(
        x=r2_finite, nbinsx=50, name='R²',
        marker_color='rgba(46, 204, 113, 0.7)',
    ), row=1, col=1)

    # (1,2) cond histogram (log scale)
    cond_finite = cond[np.isfinite(cond) & (cond > 0)]
    fig.add_trace(go.Histogram(
        x=np.log10(cond_finite), nbinsx=50, name='log10(cond)',
        marker_color='rgba(52, 152, 219, 0.7)',
    ), row=1, col=2)
    # 红虚线:1e3 / 1e5
    for boundary in [3, 5]:
        fig.add_trace(go.Scatter(
            x=[boundary, boundary], y=[0, 1], mode='lines',
            line=dict(color='red', dash='dash', width=1.5),
            showlegend=False, yaxis='y2',
        ), row=1, col=2)

    # (2,1) R² vs |k̂|
    valid_mask = np.isfinite(r2) & np.isfinite(k_abs)
    fig.add_trace(go.Scatter(
        x=r2[valid_mask], y=k_abs[valid_mask],
        mode='markers', name='|k̂|',
        marker=dict(size=5, color='rgba(155, 89, 182, 0.5)'),
        showlegend=False,
    ), row=2, col=1)

    # (2,2) (k̂, ĉ) scatter, color = R²
    valid_mask2 = np.isfinite(k) & np.isfinite(c) & np.isfinite(r2)
    fig.add_trace(go.Scatter(
        x=k[valid_mask2], y=c[valid_mask2],
        mode='markers', name='(k̂, ĉ)',
        marker=dict(
            size=5, color=r2[valid_mask2],
            colorscale='RdYlGn', cmin=0, cmax=0.2,
            colorbar=dict(title='R²', x=1.02, len=0.5, y=0.2),
            showscale=True,
        ),
        showlegend=False,
    ), row=2, col=2)

    fig.update_xaxes(title_text='R²', row=1, col=1)
    fig.update_xaxes(title_text='log10(cond(X))', row=1, col=2)
    fig.update_xaxes(title_text='R²', row=2, col=1)
    fig.update_xaxes(title_text='k̂', row=2, col=2)
    fig.update_yaxes(title_text='频数', row=1, col=1)
    fig.update_yaxes(title_text='频数', row=1, col=2)
    fig.update_yaxes(title_text='|k̂|', type='log', row=2, col=1)
    fig.update_yaxes(title_text='ĉ', row=2, col=2)

    fig.update_layout(
        template='plotly_dark', height=900, width=1400,
        title_text=f'Parameter Fit Identifiability Audit (N={len(kc_df)})',
    )
    fig.write_html(output_path)
    return output_path


def write_identifiability_summary_txt(kc_df: pd.DataFrame, output_path: str) -> str:
    """UTF-8 中文汇总:分类计数 + 分布统计 + recommendation。"""
    from datetime import datetime
    n_total = len(kc_df)
    id_status = kc_df['identification_status'].fillna('singular')
    fq = kc_df['fit_quality'].fillna('uninformative')
    n_well = int((id_status == 'well_conditioned').sum())
    n_ill = int((id_status == 'ill_conditioned').sum())
    n_unid = int((id_status == 'unidentifiable').sum())
    n_sing = int((id_status == 'singular').sum())
    n_good = int((fq == 'good').sum())
    n_weak = int((fq == 'weak').sum())
    n_poor = int((fq == 'poor').sum())
    n_uninf = int((fq == 'uninformative').sum())

    r2 = pd.to_numeric(kc_df['r2'], errors='coerce').to_numpy()
    r2_finite = r2[np.isfinite(r2)]
    cond = pd.to_numeric(kc_df['condition_number'], errors='coerce').to_numpy()
    cond_finite = cond[np.isfinite(cond) & (cond > 0)]

    pct = lambda n: 100.0 * n / max(n_total, 1)
    well_pct = pct(n_well)
    if well_pct > 50:
        rec = 'well_conditioned 占比 > 50% → V6 在 well_conditioned 子集重跑 (spec v0.2)'
    elif well_pct >= 10:
        rec = 'well_conditioned 占比 10-50% → V6 因子降级,只看 well_conditioned 子集'
    else:
        rec = 'well_conditioned 占比 < 10% → 动力学模型作为方法论不可用,收口'

    lines = [
        '=' * 50,
        'Parameter Fit Identifiability Audit',
        '=' * 50,
        f'Run date:  {datetime.now().strftime("%Y-%m-%d")}',
        f'Total:           {n_total} stocks',
        '', '--- Identification Status ---',
        f'  Well conditioned:  {n_well} ({well_pct:.1f}%)',
        f'  Ill conditioned:   {n_ill} ({pct(n_ill):.1f}%)',
        f'  Unidentifiable:    {n_unid} ({pct(n_unid):.1f}%)',
        f'  Singular:          {n_sing} ({pct(n_sing):.1f}%)',
        '', '--- Fit Quality ---',
        f'  Good:              {n_good} ({pct(n_good):.1f}%)',
        f'  Weak:              {n_weak} ({pct(n_weak):.1f}%)',
        f'  Poor:              {n_poor} ({pct(n_poor):.1f}%)',
        f'  Uninformative:     {n_uninf} ({pct(n_uninf):.1f}%)',
        '', '--- R² Distribution ---',
    ]
    if len(r2_finite) > 0:
        lines.extend([
            f'  median = {np.median(r2_finite):.4f}',
            f'  p25    = {np.percentile(r2_finite, 25):.4f}',
            f'  p75    = {np.percentile(r2_finite, 75):.4f}',
        ])
    else:
        lines.append('  (no finite R²)')
    lines.extend([
        '', '--- Condition Number Distribution ---',
    ])
    if len(cond_finite) > 0:
        lines.extend([
            f'  median = {np.median(cond_finite):.2e}',
            f'  p25    = {np.percentile(cond_finite, 25):.2e}',
            f'  p75    = {np.percentile(cond_finite, 75):.2e}',
        ])
    else:
        lines.append('  (no finite condition number)')
    lines.extend([
        '', '--- Recommendation ---',
        f'  {rec}',
        '=' * 50,
    ])
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return output_path


def main():
    args = parse_args()
    os.makedirs(CSV_OUT_DIR, exist_ok=True)

    targets = list_movement_csvs(args.input)
    if args.limit > 0:
        targets = targets[:args.limit]

    if args.rolling_time:
        print(f'输入: {args.input or f"{CSV_OUT_DIR}/movement_*.csv (扫描)"}')
        print(f'目标: {len(targets)} 只 (limit={args.limit})')
        print(f'时序滚动模式: 窗口={args.rolling_time_window} 交易日,月末 asof')
        print()
        main_rolling_time(targets,
                          window=args.rolling_time_window,
                          clip_extreme=args.clip_extreme)
        return

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
        main_rolling(targets, windows,
                     clip_extreme=args.clip_extreme,
                     plot_rolling=args.plot_rolling)
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
        'k_hat', 'c_hat', 'f_self_loss', 'f_residual_loss', 'n_valid_days', 'status',
        'rank', 'condition_number', 'regressor_corr', 'r2',
        'identification_status', 'fit_quality',
    ])
    out_path = os.path.join(CSV_OUT_DIR, KC_OUT_NAME)
    out_df.to_csv(out_path, index=False, encoding='utf-8')

    # === v0: Identifiability Audit outputs ===
    HTML_OUT_DIR = 'backtrace/outputs'
    os.makedirs(HTML_OUT_DIR, exist_ok=True)
    html_path = os.path.join(HTML_OUT_DIR, 'kc_identifiability_distribution.html')
    build_identifiability_distribution_html(out_df, html_path)
    txt_path = os.path.join(CSV_OUT_DIR, 'kc_identifiability_summary.txt')
    write_identifiability_summary_txt(out_df, txt_path)
    print(f'\n  v0 audit HTML: {html_path}')
    print(f'  v0 audit TXT:  {txt_path}')

    ok = sum(1 for r in rows if r['status'].startswith('ok'))
    singular = sum(1 for r in rows if r['status'] == 'singular')
    too_few = sum(1 for r in rows if r['status'].startswith('too_few_days'))
    fail = len(rows) - ok - singular - too_few

    # v0 分类汇总
    idstatus_count = {}
    fquality_count = {}
    for r in rows:
        idstatus_count[r['identification_status']] = idstatus_count.get(r['identification_status'], 0) + 1
        fquality_count[r['fit_quality']] = fquality_count.get(r['fit_quality'], 0) + 1

    print(f'\n=== 汇总 ===')
    print(f'  ok:       {ok}/{len(rows)}')
    if singular:
        print(f'  singular: {singular}/{len(rows)}')
    if too_few:
        print(f'  too_few:  {too_few}/{len(rows)}')
    if fail:
        print(f'  other:    {fail}/{len(rows)}')
    print(f'\n=== v0 identification_status ===')
    for k, v in sorted(idstatus_count.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v}/{len(rows)}')
    print(f'\n=== v0 fit_quality ===')
    for k, v in sorted(fquality_count.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v}/{len(rows)}')
    print(f'\n  清单: {out_path}')

    if ok > 0:
        k_vals = np.array([r['k_hat'] for r in rows if r['status'].startswith('ok')])
        c_vals = np.array([r['c_hat'] for r in rows if r['status'].startswith('ok')])
        f_vals = np.array([r['f_self_loss'] for r in rows if r['status'].startswith('ok')])
        r2_vals = np.array([r['r2'] for r in rows if r['status'].startswith('ok')
                            and np.isfinite(r['r2'])])
        cond_vals = np.array([r['condition_number'] for r in rows if r['status'].startswith('ok')
                              and np.isfinite(r['condition_number'])])
        print(f'\n=== ok 子集分布 ===')
        print(f'  k_hat: median={np.median(k_vals):+.4f} '
              f'p25={np.percentile(k_vals, 25):+.4f} '
              f'p75={np.percentile(k_vals, 75):+.4f}')
        print(f'  c_hat: median={np.median(c_vals):+.4f} '
              f'p25={np.percentile(c_vals, 25):+.4f} '
              f'p75={np.percentile(c_vals, 75):+.4f}')
        print(f'  F²:    median={np.median(f_vals):.2e} '
              f'max={np.max(f_vals):.2e}')
        if len(r2_vals) > 0:
            print(f'  R²:    median={np.median(r2_vals):.4f} '
                  f'p25={np.percentile(r2_vals, 25):.4f} '
                  f'p75={np.percentile(r2_vals, 75):.4f}')
        if len(cond_vals) > 0:
            print(f'  cond:  median={np.median(cond_vals):.2e} '
                  f'max={np.max(cond_vals):.2e}')


def main_rolling(targets, windows: list[int], clip_extreme: float,
                plot_rolling: bool = False):
    """滚动拟合分支:对每只票产 kc_rolling_<idx>_<stk>.csv + kc_rolling_summary.csv。

    per-stock CSV:窗口大小 × {k̂, ĉ, F², n_valid, status} 多个 fit 点(end-aligned 末 N 行)。
    summary CSV:每只票一行,横轴 = 各窗口的 (k̂, ĉ, F²) 直读,便于一眼看「窗口越长 k̂/ĉ 越
    漂到哪」。

    plot_rolling: True 时额外产 HTML 可视化(per-stock + 跨股票聚合)。
    """
    # 准备 summary 的列名:动态按 windows 展开
    base_cols = ['code', 'name', 'index_code', 'index_tag', 'stock_tag', 'windows']
    summary_cols = list(base_cols)
    for w in windows:
        summary_cols.extend([
            f'k_{w}', f'c_{w}', f'f2_{w}', f'n_{w}', f'status_{w}',
            f'cond_{w}', f'rcorr_{w}', f'r2_{w}', f'idstatus_{w}', f'fquality_{w}',
        ])

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
            'k_hat', 'c_hat', 'f_self_loss', 'f_residual_loss', 'n_valid_days', 'status',
            'rank', 'condition_number', 'regressor_corr', 'r2',
            'identification_status', 'fit_quality',
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
            # v0 diagnostics
            srow[f'cond_{w}'] = r['condition_number']
            srow[f'rcorr_{w}'] = r['regressor_corr']
            srow[f'r2_{w}'] = r['r2']
            srow[f'idstatus_{w}'] = r['identification_status']
            srow[f'fquality_{w}'] = r['fit_quality']
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

    # --plot-rolling:产 per-stock + 跨股票 HTML
    if plot_rolling:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as e:
            raise SystemExit(
                f'--plot-rolling 需要 plotly,但导入失败: {e}\n'
                f'  pip install plotly'
            )
        os.makedirs(OUT_HTML_DIR, exist_ok=True)
        print(f'\n=== --plot-rolling HTML 可视化 ===')
        # per-stock:每只票 1 个 HTML,4 子图(k̂ / ĉ / F² / n_valid)
        for i, ((code, name, mv_csv, index_tag, stock_tag, index_code), srow) in enumerate(
            zip(targets, summary_rows), 1
        ):
            html_path = plot_rolling_per_stock(
                srow, windows, code, name, index_tag, stock_tag,
            )
            print(f'[{i}/{len(summary_rows)}] {code} → {html_path}')
        # 跨股票聚合:1 个 HTML,各窗口 k̂/ĉ 的中位数 ± p25/p75 区间
        agg_path = plot_rolling_aggregate(summary_df, windows)
        print(f'\n  跨股票聚合: {agg_path}')


def _month_ends(dates: pd.Series) -> list:
    """返回每月最后一个交易日的 Timestamp 列表(去重 + 升序)。"""
    df = pd.DataFrame({'Date': pd.to_datetime(dates)})
    df['_ym'] = df['Date'].dt.to_period('M')
    month_ends = df.groupby('_ym')['Date'].max().sort_values().tolist()
    return month_ends


def main_rolling_time(targets, window: int = 240, clip_extreme: float = 10.0):
    """每月末用最近 N 天 OLS 估 (k̂, ĉ),产 long format CSV。

    对每只票:
        1. 读 movement CSV
        2. 找月末 asof_date 列表(每月最后交易日)
        3. 对每个 asof_date:截幅到该日期 + 取最后 window 行,跑 OLS
        4. 落 1 行 (asof_date, code, k_hat, c_hat, ...)

    输出: data/projection/kc_estimates_time.csv
    """
    rows = []
    for i, (code, name, mv_csv, index_tag, stock_tag, index_code) in enumerate(targets, 1):
        label = f'{code} ({name})' if name else code
        print(f'[{i}/{len(targets)}] {label} ...', end=' ', flush=True)
        loaded, err = _load_movement(mv_csv, stock_tag, index_tag)
        if loaded is None:
            print(f'⚠ load failed: {err}')
            continue
        df, delta_u, delta_v, beta = loaded
        u_vec, d_vec, a_u_vec, a_v_vec = _build_kinematics(delta_u, delta_v, beta)
        dates = pd.to_datetime(df['Date'])
        # 防御性 assert:rolling 取末 window 行的前提是 Date 单调递增。
        # 若 movement CSV 异常(混序)→ 直接报而不是悄悄用错位置当 OLS 输入。
        assert dates.is_monotonic_increasing, (
            f'{mv_csv}: Date 非单调递增,rolling-time 模式不安全;'
            f'请先校验 movement CSV 生成逻辑。'
        )
        month_ends = _month_ends(dates)
        print(f'{len(month_ends)} asof_dates', end=' ', flush=True)
        for asof in month_ends:
            mask = (dates <= asof).values
            n_avail = int(mask.sum())
            if n_avail < max(3, window // 4):   # 至少需要 window/4 天
                rows.append({
                    'asof_date': str(asof)[:10],
                    'code': code, 'name': name or '',
                    'index_code': index_code,
                    'index_tag': index_tag, 'stock_tag': stock_tag,
                    'k_hat': np.nan, 'c_hat': np.nan,
                    'f_self_loss': np.nan, 'f_residual_loss': np.nan,
                    'n_valid_days': n_avail,   # 真实可用天数,不埋进 status 字符串
                    'rank': 0, 'condition_number': np.nan,
                    'regressor_corr': np.nan, 'r2': np.nan,
                    'identification_status': 'singular',
                    'fit_quality': 'uninformative',
                    'status': f'too_few_days ({n_avail})',
                })
                continue
            # 取最后 window 行
            idx = np.where(mask)[0][-window:]
            sub = slice(idx[0], idx[-1] + 1)
            valid = (
                np.isfinite(a_u_vec[sub]).all(axis=1)
                & np.isfinite(a_v_vec[sub]).all(axis=1)
                & np.isfinite(d_vec[sub]).all(axis=1)
                & np.isfinite(u_vec[sub]).all(axis=1)
            )
            n_valid = int(valid.sum())
            if n_valid < 3:
                rows.append({
                    'asof_date': str(asof)[:10],
                    'code': code, 'name': name or '',
                    'index_code': index_code,
                    'index_tag': index_tag, 'stock_tag': stock_tag,
                    'k_hat': np.nan, 'c_hat': np.nan,
                    'f_self_loss': np.nan, 'f_residual_loss': np.nan,
                    'n_valid_days': n_valid,
                    'rank': 0, 'condition_number': np.nan,
                    'regressor_corr': np.nan, 'r2': np.nan,
                    'identification_status': 'singular',
                    'fit_quality': 'uninformative',
                    'status': f'too_few_valid ({n_valid})',
                })
                continue
            try:
                k_hat, c_hat, f_residual_loss, _, rank, condition_number, regressor_corr, r2 = _solve_ols(
                    a_u_vec[sub], a_v_vec[sub], d_vec[sub], u_vec[sub], beta[sub], valid,
                )
            except Exception as e:
                rows.append({
                    'asof_date': str(asof)[:10],
                    'code': code, 'name': name or '',
                    'index_code': index_code,
                    'index_tag': index_tag, 'stock_tag': stock_tag,
                    'k_hat': np.nan, 'c_hat': np.nan,
                    'f_self_loss': np.nan, 'f_residual_loss': np.nan,
                    'n_valid_days': n_valid,
                    'rank': 0, 'condition_number': np.nan,
                    'regressor_corr': np.nan, 'r2': np.nan,
                    'identification_status': 'singular',
                    'fit_quality': 'uninformative',
                    'status': f'solve_failed: {type(e).__name__}: {e}',
                })
                continue
            finite = np.isfinite(k_hat) and np.isfinite(c_hat)
            extreme = abs(k_hat) > clip_extreme or abs(c_hat) > clip_extreme

            # identification_status: 仅看 rank + cond
            if not finite or rank < 2:
                identification_status = 'singular'
            elif condition_number >= 1e5:
                identification_status = 'unidentifiable'
            elif condition_number >= 1e3:
                identification_status = 'ill_conditioned'
            else:
                identification_status = 'well_conditioned'

            # fit_quality: 仅看 R²
            if not np.isfinite(r2):
                fit_quality = 'uninformative'
            elif r2 < 0.01:
                fit_quality = 'poor'
            elif r2 < 0.1:
                fit_quality = 'weak'
            else:
                fit_quality = 'good'

            if not finite:
                status = 'solve_failed'
            elif rank < 2:
                status = 'singular'
            elif extreme:
                status = f'extreme (|k| or |c| > {clip_extreme:g})'
            else:
                status = 'ok'
            rows.append({
                'asof_date': str(asof)[:10],
                'code': code, 'name': name or '',
                'index_code': index_code,
                'index_tag': index_tag, 'stock_tag': stock_tag,
                'k_hat': k_hat, 'c_hat': c_hat,
                'f_self_loss': f_residual_loss,  # alias for backward compat
                'f_residual_loss': f_residual_loss,
                'n_valid_days': n_valid,
                'rank': rank,
                'condition_number': condition_number,
                'regressor_corr': regressor_corr,
                'r2': r2,
                'identification_status': identification_status,
                'fit_quality': fit_quality,
                'status': status,
            })
        print('✓')
    out = pd.DataFrame(rows, columns=[
        'asof_date', 'code', 'name', 'index_code', 'index_tag', 'stock_tag',
        'k_hat', 'c_hat', 'f_self_loss', 'f_residual_loss', 'n_valid_days', 'status',
        'rank', 'condition_number', 'regressor_corr', 'r2',
        'identification_status', 'fit_quality',
    ])
    out_path = os.path.join(CSV_OUT_DIR, 'kc_estimates_time.csv')
    out.to_csv(out_path, index=False, encoding='utf-8')
    print(f'✓ {out_path} ({len(out)} 行)')


if __name__ == '__main__':
    main()