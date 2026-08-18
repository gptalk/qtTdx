# -*- coding: utf-8 -*-
# dynamics_oos_viz.py — OOS 1 步预测 + 可视化数据准备(v5.9 Task 1 scaffold)
#
# 本模块把 projection + dynamics 描述层喂给 predict_next_state,产出
# 「预测 vs 实际」对齐序列,供后续 plotly HTML 渲染(由 Task 2+ 实现)。
#
# 设计要点:
#   1. 不重写任何 projection / dynamics 数学 — 全部 import
#   2. F_self 用滚动均值代理(dynamics_1step_oos 同款策略,防恒等式陷阱)
#   3. 1 步预测 t → t+1,产出与 common_idx 对齐的 T_oos = days - 2 个点
#
# 已知坑(写代码前必看):
#   - brief 中的字段名 `mv['a_S_mag'] / mv['a_M_mag'] / mv['beta'] / mv['q_t']`
#     在实际 projection_core 里不存在 — 分别对应:
#         a_S_mag / a_M_mag → dyn['a_S_mag'] / dyn['a_M_mag']
#         beta              → mv['proj_coeff']
#         q_t               → dyn['q_t']
#     这里照 brief 描述意图映射,不重命名
#   - `compute_movement_projection` 不接 prefer_industry(基线由 load_pair 决定)
#   - `compute_dynamics` 必传 lambda_q;None = 自适应
#   - `classify_states(R, theta, E_self, thresholds) -> list[str]`
#     不是 `classify_states(dyn, ...)`(brief 描述层有出入,这里按实际签名走)
import warnings
warnings.filterwarnings('ignore')
import sys, os
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === 公共 pipeline(给 load_pair 喂数据) ===
from common import tsfresh_pipeline as P

# === 数学源头:projection / dynamics 全部 import,0 重写 ===
from backtrace.projection._projection_core import (
    load_pair,
    compute_movement_projection,
    compute_dynamics,
    classify_states,
    STATE_LABELS,
)
from backtrace.dynamics._dynamics_core import predict_next_state

# === 模块常量(供 Task 2+ 复用) ==============================================
DEFAULTS = dict(
    days=250,
    prefer_industry=True,
    k=None,
    c=None,
    lambda_q=None,
    f_self_window=10,
)
# (R_low, R_high, theta_following_rad, theta_against_rad)
# 与 classify_states 签名一致
THRESHOLDS = (0.10, 0.50, np.deg2rad(30), np.deg2rad(90))
DEFAULT_OUTPUT = 'backtrace/outputs/dynsys_oos_viz_{code}.html'


# === 1 步预测:公共入口 ======================================================
def load_oos_predictions(
    stock_code: str,
    days: int,
    *,
    prefer_industry: bool = True,
    k: float | None = None,
    c: float | None = None,
    lambda_q: float | None = None,
    f_self_window: int = 10,
) -> dict:
    """跑 OOS 1 步预测并返回「预测 vs 实际」对齐数据。

    数据流:
        load_pair → motion projection → compute_dynamics
        → F_self 滚动均值 → predict_next_state(t → t+1)

    Returns dict with keys:
        common_idx:    pd.DatetimeIndex (T_oos,) — t+1 对齐
        a_pred:        np.ndarray (T_oos, 2) — 预测的 a_pred (= v_pred - v_S_now)
        a_actual:      np.ndarray (T_oos, 2) — 实际 v_S(t+1) - v_S(t)
        state_pred:    list[str] (T_oos,) — 预测加速度幅度 → 离散标签
        state_actual:  list[str] (T_oos,) — 实际状态(从 dyn + classify_states)
        k_used:        float
        c_used:        float
        mv:            dict — 完整 motion projection(供调试)
        dyn:           dict — 完整 dynamics(供调试)
    """
    # 1) 拉数据(本地缓存,TQ 不在线时走 data/)
    loaded = load_pair(
        stock_code, days, P,
        prefer_industry=prefer_industry,
        index_code=None, lag=0,
    )
    data_stock = loaded['stock_df']
    data_index = loaded['index_df']
    common_idx = loaded['common_idx']

    # 2) 运动投影
    mv = compute_movement_projection(data_stock, data_index)

    # 3) 动力学描述层(拿到 q_t / a_S_mag / a_M_mag / R / theta / E_self)
    dyn = compute_dynamics(mv, lambda_q=lambda_q)

    # 4) k / c 兜底(brief §6.4-6.5;dyn 不返回 k_hat/c_hat,默认 0)
    k_used = float(k) if k is not None else 0.0
    c_used = float(c) if c is not None else 0.0

    # 5) 拆字段
    delta_u = mv['stock_move']                 # (T-1, 2) — v_S
    delta_v = mv['index_move']                 # (T-1, 2) — v_M
    beta = mv['proj_coeff']                    # (T-1,)
    q_t_seq = dyn['q_t']                       # (T-1,)
    T = len(delta_u)

    # d_vec:位置偏离累积(与 dynamics_1step_oos 同款公式)
    #   d(0) = 0
    #   d(t+1) = d(t) + u(t)   for t in 0..T-2
    u_vec = delta_u - beta[:, None] * delta_v
    d_vec = np.zeros_like(delta_u)
    if T >= 2:
        d_vec[1:] = np.cumsum(u_vec[:-1], axis=0)

    # 6) 实际状态标签 — 全 T-1 个
    state_actual_full = classify_states(
        dyn['R'], dyn['theta'], dyn['E_self'], THRESHOLDS,
    )

    # 7) F_self 滚动均值代理(brief §7 简化版)
    #   F_self_pred(t) = mean( delta_u[t-W : t] ),若 t < W → 0
    #   注:delta_u 严格说是"速度"(motion vector),brief 称"acceleration"是命名不严谨
    F_self_seq = []
    for t in range(T):
        if t < f_self_window:
            F_self_seq.append(np.zeros(2))
        else:
            F_self_seq.append(np.mean(delta_u[t - f_self_window:t], axis=0))

    # 8) 1 步预测主循环:t → t+1,产出 T-1 个点
    a_pred_list = []
    a_actual_list = []
    state_pred_list = []
    state_actual_list = []
    for t in range(T - 1):
        F_self_pred = F_self_seq[t]
        # v3 签名:v_S_now, v_M_now, v_M_next, beta_now, beta_next, d_now, F_self_now, k, c, q_now
        a_pred, _v_pred, _d_pred, _u_pred = predict_next_state(
            v_S_now=delta_u[t],
            v_M_now=delta_v[t],
            v_M_next=delta_v[t + 1],
            beta_now=float(beta[t]),
            beta_next=float(beta[t + 1]),
            d_now=d_vec[t],
            F_self_now=F_self_pred,
            k=k_used,
            c=c_used,
            q_now=float(q_t_seq[t]),
        )
        a_pred_list.append(a_pred)
        # 实际:下一日个股速度 v_S(t+1) = delta_u[t+1]
        a_actual_list.append(delta_u[t + 1])
        state_pred_list.append(_label_from_a(a_pred))
        state_actual_list.append(state_actual_full[t + 1])

    # 9) 索引对齐 — common_idx[1:] 是 motion projection 起点,这里再 [1:] → common_idx[2:]
    common_idx_oos = common_idx[2:2 + len(a_pred_list)]

    return {
        'common_idx': common_idx_oos,
        'a_pred': np.asarray(a_pred_list, dtype=float),
        'a_actual': np.asarray(a_actual_list, dtype=float),
        'state_pred': state_pred_list,
        'state_actual': state_actual_list,
        'k_used': k_used,
        'c_used': c_used,
        'mv': mv,
        'dyn': dyn,
    }


# === 标签助手 ==============================================================
def _label_from_a(a: np.ndarray) -> str:
    """Map predicted acceleration magnitude → discrete state label.

    Heuristic:
        |a| < 0.005 → 'none'
        0.005 ≤ |a| < 0.05 → 'follow'
        else → 'accelerating'

    注:这是 brief 给的 3 段式标签;与 description 层的 7 类 state(STATE_LABELS)
    不同,服务于「预测加速度幅度 → 弱/中/强」粗分类。HTML 渲染时颜色按
    STATE_COLORS 对应映射即可。
    """
    mag = float(np.linalg.norm(a))
    if mag < 0.005:
        return 'none'
    elif mag < 0.05:
        return 'follow'
    else:
        return 'accelerating'


# === 公共 re-export =========================================================
__all__ = [
    'load_oos_predictions',
    '_label_from_a',
    'build_oos_prediction_html',
    'DEFAULTS',
    'THRESHOLDS',
    'DEFAULT_OUTPUT',
]


# === plotly 4-row 可视化(spec §3.5) ========================================
def build_oos_prediction_html(
    common_idx: pd.DatetimeIndex,
    a_pred: np.ndarray,
    a_actual: np.ndarray,
    state_pred: list[str],
    state_actual: list[str],
    k_used: float,
    c_used: float,
    output_path: str,
    title: str = 'OOS 1-Step Prediction vs Actual',
) -> None:
    """Render OOS 1-step prediction diagnostics as a 4-row plotly HTML.

    Layout(spec §3.5):
        Row 1 — predicted (blue) vs actual (orange) |a_S| magnitude
        Row 2 — per-day error, color-coded by σ-band:
                    green  (< 0.5σ)
                    yellow (0.5σ – 1σ)
                    red    (> 1σ)
        Row 3 — 20-day rolling RMSE of error(purple)
        Row 4 — state hit rate(1 = pred matches actual, 0 = miss)

    Title contains UTF-8 `k̂` (k̂ = U+006B + U+0302) and `ĉ` (ĉ = U+0109);
    using literal escape sequences to avoid U+FFFD replacement in
    some terminal codecs.
    """
    # 1) Convert to 1-D magnitude
    a_pred_mag = np.linalg.norm(a_pred, axis=1)     # (T,)
    a_actual_mag = np.linalg.norm(a_actual, axis=1) # (T,)
    error = a_pred_mag - a_actual_mag
    T = len(common_idx)

    # 2) Error σ-band coloring
    abs_err = np.abs(error)
    sigma = float(np.nanstd(abs_err)) if T > 1 else 0.0
    if sigma == 0.0:
        sigma = 1e-9
    band_low = 0.5 * sigma
    band_high = 1.0 * sigma
    err_colors = np.where(
        abs_err < band_low, '#2ecc71',                    # green
        np.where(abs_err < band_high, '#f39c12', '#e74c3c')  # yellow / red
    )

    # 3) Rolling RMSE
    win = 20
    rolling_rmse = pd.Series(error).pow(2).rolling(win).mean().pow(0.5).to_numpy()

    # 4) State hit rate
    hit = np.array([1.0 if p == a else 0.0 for p, a in zip(state_pred, state_actual)])

    # 5) Figure layout (4 rows, shared_xaxes)
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.35, 0.25, 0.2, 0.2],
        vertical_spacing=0.05,
        subplot_titles=(
            'Row 1 — |a_S| predicted (blue) vs actual (orange)',
            'Row 2 — error (color: <0.5σ green / 0.5–1σ yellow / >1σ red)',
            f'Row 3 — {win}-day rolling RMSE of error',
            'Row 4 — state hit rate (1 = pred==actual)',
        ),
    )

    # 6) Row 1 — predicted vs actual magnitude
    fig.add_trace(
        go.Scatter(
            x=common_idx, y=a_pred_mag, name='predicted |a_S|',
            mode='lines', line=dict(color='#3498db', width=1.5),
            legendgroup='series',
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=common_idx, y=a_actual_mag, name='actual |a_S|',
            mode='lines+markers', line=dict(color='#e67e22', width=1.5),
            marker=dict(size=4),
            legendgroup='series',
        ),
        row=1, col=1,
    )

    # 7) Row 2 — error bars (colored by σ band)
    fig.add_trace(
        go.Bar(
            x=common_idx, y=error, name='error',
            marker=dict(color=err_colors.tolist()),
            showlegend=False,
            legendgroup='series',
        ),
        row=2, col=1,
    )

    # 8) Row 3 — rolling RMSE
    fig.add_trace(
        go.Scatter(
            x=common_idx, y=rolling_rmse, name=f'{win}-d rolling RMSE',
            mode='lines', line=dict(color='#9b59b6', width=2),
            legendgroup='series',
        ),
        row=3, col=1,
    )

    # 9) Row 4 — state hit rate
    fig.add_trace(
        go.Bar(
            x=common_idx, y=hit, name='state hit (1=yes, 0=no)',
            marker=dict(color=hit, colorscale=[[0, '#e74c3c'], [1, '#2ecc71']]),
            showlegend=False,
            legendgroup='series',
        ),
        row=4, col=1,
    )

    # 10) Layout
    k_str = f"{k_used:.4f}" if k_used is not None else 'auto'
    c_str = f"{c_used:.4f}" if c_used is not None else 'auto'
    # k̂ = k̂ (k + combining circumflex U+0302)
    # ĉ = č (c with caron U+0109)
    # Use literal Unicode escapes per brief to avoid U+FFFD replacement.
    KC_KHAT = 'k\u0302'
    KC_CHAT = '\u0109'
    fig.update_layout(
        title=f"{title} ({KC_KHAT}={k_str}, {KC_CHAT}={c_str})",
        height=900,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white',
    )
    fig.update_yaxes(title_text='|a_S| magnitude', row=1, col=1)
    fig.update_yaxes(title_text='error', row=2, col=1)
    fig.update_yaxes(title_text='RMSE', row=3, col=1)
    fig.update_yaxes(title_text='hit rate', row=4, col=1, range=[-0.05, 1.05])
    fig.update_xaxes(title_text='date', row=4, col=1)

    # 11) Output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn', full_html=True)
    print(f"[v5.9] wrote {output_path} ({T} OOS days, {KC_KHAT}={k_str}, {KC_CHAT}={c_str})")
