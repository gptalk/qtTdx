# -*- coding: utf-8 -*-
# dynamics_system.py — 单股离散动力系统端到端(load → describe → predict → simulate → state → HTML/CSV)
#
# 与 projection_2d.py --dynamics 的关系:
#   - 那个只跑描述层(q/θ/R/E/state/forces)→ 1 个 HTML + 1 个 CSV
#   - 这个在描述层之上 N 步前向模拟个股轨迹 → 额外 1 个 HTML + 1 个 CSV
#     (描述层 CSV/HTML 仍按 projection_2d 风格产出,本脚本同时落)
#
# 复用 backtrace/projection/_projection_core 的全部数学(单一来源真理)。
#
# 参数:
#   --code         str   个股代码(带 .SH / .SZ)。默认 002475.SZ
#   --name         str   个股中文名(仅图例)。默认从 stocks_info 反查
#   --days         int   回看天数(描述层 + 模拟起点)。默认 240
#   --horizon      int   模拟步数(N)。默认 5
#   --index        str   基线指数代码(显式覆盖);默认按个股交易所选大盘
#   --k-restore    float 恢复系数 k。F_restore = -k·d。默认 0.0
#   --c-damp       float 阻尼系数 c。F_damp = -c·u。默认 0.0
#   --lambda-q     float 锚定强度系数 λ_q。-1 走 median(‖ΔM‖) 自适应
#   --classify-thresholds str 4 浮点:R_low,R_high,theta_following_deg,theta_against_deg。默认 0.10,0.50,30,90
#   --k-from-fit / --c-from-fit flag 从 data/projection/kc_estimates.csv 加载
#
# 用法:
#   PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_system.py
#   PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_system.py --code 600519.SH --name 贵州茅台
#   PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_system.py --code 002475.SZ --horizon 10 --k-restore 0.1 --c-damp 0.05
#   PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_system.py --k-from-fit --c-from-fit
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 走本地 data/ 缓存 — 不依赖 TQ 客户端
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
from common import tsfresh_pipeline as P
from projection._projection_core import (
    load_pair, compute_movement_projection, compute_dynamics,
    classify_states, build_dynamics_df,
    compute_forces, build_forces_df,
    STATE_COLORS, STATE_LABELS_CN,
)
from dynamics import (
    build_simulation_df, simulate_trajectory,
    make_rolling_mean_f_self_predictor, make_constant_f_self_predictor,
    make_ar1_f_self_predictor,
)


CSV_OUT_DIR = 'data/dynamics'             # 描述/力/模拟 CSV 落点
OUT_DIR = 'backtrace/outputs'              # HTML 落点(CLAUDE.md 约定)
HTML_NAME = 'dynsys_simulation.html'       # 单 HTML


def parse_args():
    p = argparse.ArgumentParser(description='单股离散动力系统(load → describe → simulate → HTML/CSV)')
    p.add_argument('--code', default='002475.SZ', help='个股代码(带 .SH/.SZ)。默认 002475.SZ')
    p.add_argument('--name', default=None, help='个股中文名(仅图例);不传时按 --code 反查')
    p.add_argument('--days', type=int, default=240, help='回看天数。默认 240')
    p.add_argument('--horizon', type=int, default=5, help='N 步模拟步数。默认 5')
    p.add_argument(
        '--index', default=None,
        help='基线指数代码(显式覆盖);默认按个股交易所选大盘',
    )
    p.add_argument('--k-restore', type=float, default=0.0, help='恢复系数 k。默认 0.0')
    p.add_argument('--c-damp', type=float, default=0.0, help='阻尼系数 c。默认 0.0')
    p.add_argument('--lambda-q', type=float, default=-1.0, help='锚定强度 λ_q;-1 走 median 自适应')
    p.add_argument(
        '--classify-thresholds', default='0.10,0.50,30,90',
        help='状态分类阈值,R_low,R_high,theta_following_deg,theta_against_deg。默认 0.10,0.50,30,90',
    )
    p.add_argument('--k-from-fit', action='store_true', help='从 kc_estimates.csv 加载 k̂')
    p.add_argument('--c-from-fit', action='store_true', help='从 kc_estimates.csv 加载 ĉ')
    p.add_argument('--f-self-mode', default='rolling',
                   choices=['rolling', 'constant', 'oracle', 'ar1'],
                   help='F_self 预测:rolling=末日滚动均值(default)/constant=末日瞬时值/'
                        'oracle=末日观测外推/ar1=AR(1) 自回归(per-dim 估 ρ/μ)')
    p.add_argument('--f-self-window', type=int, default=10,
                   help='F_self 窗口(rolling=滚动均值天数;ar1=AR(1) 最少有效样本数)。默认 10')
    p.add_argument('--period', choices=['daily', '15m', '5m', '1m'], default='daily',
                   help='缓存粒度(daily = 默认)')
    return p.parse_args()


def _load_kc_for(index_tag, stock_tag):
    """从 data/projection/kc_estimates.csv 按 (index_tag, stock_tag) 查 (k̂, ĉ)。"""
    import csv as _csv
    path = 'data/projection/kc_estimates.csv'
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if row.get('index_tag') == index_tag and row.get('stock_tag') == stock_tag:
                try:
                    return {
                        'k_hat': float(row['k_hat']),
                        'c_hat': float(row['c_hat']),
                        'status': row.get('status', ''),
                    }
                except (KeyError, ValueError):
                    return None
    return None


def main():
    args = parse_args()
    os.makedirs(CSV_OUT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- 名字反查(同 projection_2d) ----------
    if args.name:
        STOCK_NAME = args.name
    else:
        from common import stocks_info
        STOCK_NAME = stocks_info.lookup_name(args.code) or {'002475.SZ': '立讯精密'}.get(args.code, '')

    # ---------- 阈值解析 ----------
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
        raise SystemExit('theta_following / theta_against 必须 0 < following < against < 180')

    LAMBDA_Q = None if args.lambda_q < 0 else args.lambda_q

    # ---------- 1. 加载数据 ----------
    print(f"加载 {args.code} {STOCK_NAME} 最近 {args.days} 日...", flush=True)
    loaded = load_pair(args.code, args.days, P, index_code=args.index, lag=0, period=args.period)
    data_stock = loaded['stock_df']
    data_index = loaded['index_df']
    common_idx = loaded['common_idx']
    INDEX_CODE = loaded['index_code']
    INDEX_NAME = loaded['index_name']
    INDEX_TAG = loaded['index_tag']
    STOCK_TAG = loaded['stock_tag']
    print(f"基线 {INDEX_CODE} ({INDEX_NAME}) × {args.code} ({STOCK_NAME}),共同交易日 {len(common_idx)} 日")

    # ---------- 2. 描述层 + 状态分类 ----------
    mv = compute_movement_projection(data_stock, data_index)
    dyn = compute_dynamics(mv, lambda_q=LAMBDA_Q)

    theta_following_rad = np.deg2rad(THETA_FOLLOWING_DEG)
    theta_against_rad = np.deg2rad(THETA_AGAINST_DEG)
    states_desc = classify_states(
        dyn['R'], dyn['theta'], dyn['E_self'],
        (R_LOW, R_HIGH, theta_following_rad, theta_against_rad),
    )
    desc_df = build_dynamics_df(common_idx[1:], dyn, states_desc, INDEX_TAG, STOCK_TAG)
    desc_csv = os.path.join(CSV_OUT_DIR, f'dynamics_{INDEX_TAG}_{STOCK_TAG}.csv')
    desc_df.to_csv(desc_csv, index=False, encoding='utf-8')
    lq_note = (
        f'{dyn["lambda_q_used"]:.4e} (median 自适应)' if LAMBDA_Q is None
        else f'{LAMBDA_Q:.4e} (用户指定)'
    )
    print(f"  描述层: λ_q={lq_note} → {desc_csv}")

    # ---------- 3. k / c 加载 + 力分解 ----------
    if args.k_from_fit or args.c_from_fit:
        kc = _load_kc_for(INDEX_TAG, STOCK_TAG)
        if kc is None:
            print(f'  [--k-from-fit/--c-from-fit] ⚠ kc_estimates.csv 中没有 ({INDEX_TAG}, {STOCK_TAG})')
        else:
            if args.k_from_fit:
                args.k_restore = kc['k_hat']
                print(f'  [--k-from-fit] k_restore ← k̂={kc["k_hat"]:+.4f}')
            if args.c_from_fit:
                args.c_damp = kc['c_hat']
                print(f'  [--c-from-fit] c_damp ← ĉ={kc["c_hat"]:+.4f}')

    frc = compute_forces(dyn, mv, k_restore=args.k_restore, c_damp=args.c_damp)
    frc_df = build_forces_df(common_idx[1:], frc, INDEX_TAG, STOCK_TAG)
    frc_csv = os.path.join(CSV_OUT_DIR, f'forces_{INDEX_TAG}_{STOCK_TAG}.csv')
    frc_df.to_csv(frc_csv, index=False, encoding='utf-8')
    print(f"  力分解: k={args.k_restore:.4f}, c={args.c_damp:.4f} → {frc_csv}")

    # ---------- 4. 准备模拟输入(Oracle 模式) ----------
    # 起点:最后一日的观测状态
    v_S_init = mv['stock_move'][-1]                          # (2,)
    # NEW(2026-08-17 时间轴重构 v2):v_M_seq / β_seq 改用 N+1 个状态量,
    # 让所有 N 个 a_M(t) = v_M(t+1) - v_M(t) 都有效(无 NaN 跳步)。
    beta_seq = mv['proj_coeff'][-(args.horizon + 1):]        # (N+1,)
    v_M_seq = mv['index_move'][-(args.horizon + 1):]         # (N+1, 2)
    # 末态 d / u 重建(同 parameter_fit.py)
    u_full = mv['stock_move'] - mv['proj_coeff'][:, None] * mv['index_move']
    d_full = np.zeros_like(mv['stock_move'])
    if len(u_full) >= 2:
        d_full[1:] = np.cumsum(u_full[:-1], axis=0)
    # NEW(2026-08-17 v2):d[t+1]=d[t]+u[t] 递推。d(0)=d_full[-1] 为自然初始条件,
    # d(1) = d_full[-1] + u_full[-1](最直观的物理定义,无需 u 补偿)
    d_init = d_full[-1]
    u_init = u_full[-1]
    # F_self 残差:末日 F_self = a_S - β·a_M + k·d + c·u;长度 N 时复制末值
    a_u_vec = np.full_like(mv['stock_move'], np.nan)
    a_v_vec = np.full_like(mv['index_move'], np.nan)
    if len(mv['stock_move']) >= 2:
        a_u_vec[:-1] = np.diff(mv['stock_move'], axis=0)
        a_v_vec[:-1] = np.diff(mv['index_move'], axis=0)
    # a_u_vec / a_v_vec 末行是 NaN(np.diff 丢一行);取倒数第二行
    # (T-2) 作为"最近一个有效加速度"用于 F_self 分解
    a_S_recent = a_u_vec[-2] if np.isfinite(a_u_vec[-2]).all() else np.array([0.0, 0.0])
    a_M_recent = a_v_vec[-2] if np.isfinite(a_v_vec[-2]).all() else np.array([0.0, 0.0])
    F_self_last = (
        a_S_recent
        - beta_seq[-1] * a_M_recent
        + args.k_restore * d_init
        + args.c_damp * u_init
    )
    # F_self 历史(用于 rolling 预测器)— F_self(τ) = a_S(τ) - β(τ)·a_M(τ) + k·d(τ) + c·u(τ)
    F_self_full = np.full_like(mv['stock_move'], np.nan)
    valid = np.isfinite(a_u_vec).all(axis=1) & np.isfinite(a_v_vec).all(axis=1)
    F_self_full[valid] = (
        a_u_vec[valid] - mv['proj_coeff'][valid, None] * a_v_vec[valid]
        + args.k_restore * d_full[valid] + args.c_damp * u_full[valid]
    )
    # F_self 预测器选择
    if args.f_self_mode == 'rolling':
        F_self_predictor = make_rolling_mean_f_self_predictor(
            F_self_full, window=args.f_self_window,
        )
        F_self_seq = None
    elif args.f_self_mode == 'constant':
        F_self_predictor = make_constant_f_self_predictor(F_self_last)
        F_self_seq = None
    elif args.f_self_mode == 'ar1':
        # AR(1) 自回归:per-dim 估 ρ/μ;数据不足自动退化
        F_self_predictor = make_ar1_f_self_predictor(
            F_self_full, min_history=args.f_self_window,
        )
        F_self_seq = None
    else:  # oracle — 末日观测残差恒定外推(旧默认)
        F_self_predictor = None
        F_self_seq = np.tile(F_self_last, (args.horizon, 1))
    print(f'[sim] F_self 模式: {args.f_self_mode}'
          + (f' W={args.f_self_window}' if args.f_self_mode in ('rolling', 'ar1') else ''))
    # q_t 序列:末日往回数
    q_t_seq = dyn['q_t'][-args.horizon:]

    # 模拟日期:末日 + horizon 个未来日(共同交易日索引往后延伸)
    # 简化:用 0..N 整数占位(避免末日之后没有共同交易日)
    sim_dates = list(range(args.horizon + 1))

    # ---------- 5. 模拟 ----------
    sim = simulate_trajectory(
        v_S_init=v_S_init,
        v_M_seq=v_M_seq,
        beta_seq=beta_seq,
        F_self_seq=F_self_seq,
        F_self_predictor=F_self_predictor,
        d_init=d_init,
        # v3:u_init 删除(派生量,simulate_trajectory 在 t=0 自动派生 u[0] = v_S[0] - β[0]·v_M[0])
        k=args.k_restore, c=args.c_damp,
        q_t_seq=q_t_seq,
        classify_thresholds=(R_LOW, R_HIGH, theta_following_rad, theta_against_rad),
    )
    sim_df = build_simulation_df(sim, sim_dates, INDEX_TAG, STOCK_TAG)
    sim_csv = os.path.join(CSV_OUT_DIR, f'simulation_{INDEX_TAG}_{STOCK_TAG}.csv')
    sim_df.to_csv(sim_csv, index=False, encoding='utf-8')
    print(f"  模拟: N={args.horizon} 步, k={args.k_restore}, c={args.c_damp} → {sim_csv}")
    from collections import Counter
    state_dist = Counter(sim['state'][:args.horizon])
    dist_str = ', '.join(f'{STATE_LABELS_CN.get(s, s)}={c}' for s, c in state_dist.most_common())
    print(f"    状态分布: {dist_str}")

    # ---------- 6. HTML 5 子图 ----------
    sim_v_S_mag = sim['v_S_mag']
    sim_E_market = sim['E_market']
    sim_E_self = sim['E_self']
    sim_E_total = sim['E_total']
    sim_R = sim['R']
    sim_theta_deg = np.degrees(sim['theta'])
    sim_states = sim['state']
    sim_F_M = sim['F_market']
    sim_F_R = sim['F_restore']
    sim_F_D = sim['F_damp']
    sim_F_S = sim['F_self']

    # 实际最后 K 日(对比用) = 末日往回数 horizon
    actual_v_S_mag = np.linalg.norm(mv['stock_move'][-args.horizon:], axis=1)
    actual_R = dyn['R'][-args.horizon:]

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        subplot_titles=(
            f'速度模长 ‖v_S‖:实际末日 → 模拟 (k={args.k_restore}, c={args.c_damp})',
            f'能量拆分 E_market / E_self / E_total',
            f'耦合度 R / 偏离角 θ (°, 右轴)',
            f'状态分类 (λ_q={dyn["lambda_q_used"]:.2e})',
            f'力分解 ‖F_M‖/‖F_R‖/‖F_D‖/‖F_S‖',
        ),
        vertical_spacing=0.04,
        row_heights=[0.20, 0.20, 0.18, 0.20, 0.22],
    )

    # Row 1: 实际(末日往回数) + 模拟
    actual_x = list(range(-args.horizon, 0))
    sim_x = list(range(0, args.horizon + 1))
    fig.add_trace(go.Scatter(
        x=actual_x, y=actual_v_S_mag, mode='lines', name='实际 ‖v_S‖ (末日往回数)',
        line=dict(color='cyan', width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sim_x, y=sim_v_S_mag, mode='lines+markers', name='模拟 ‖v_S‖',
        line=dict(color='orange', width=2), marker=dict(size=6),
    ), row=1, col=1)
    fig.add_vline(x=0, line=dict(color='gray', dash='dash'), row=1, col=1,
                  annotation_text='模拟起点', annotation_position='top')
    fig.update_yaxes(title_text='‖v_S‖', row=1, col=1)

    # Row 2: 能量堆叠
    fig.add_trace(go.Scatter(
        x=sim_x, y=sim_E_market, mode='lines', name='E_market',
        line=dict(color='cyan'), stackgroup='sim_energy',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=sim_x, y=sim_E_self, mode='lines', name='E_self',
        line=dict(color='magenta'), stackgroup='sim_energy',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=sim_x, y=sim_E_total, mode='lines', name='E_total',
        line=dict(color='lime', dash='dot'),
    ), row=2, col=1)
    fig.update_yaxes(title_text='½·‖v‖²', row=2, col=1)

    # Row 3: R + θ 双 Y 轴
    fig.add_trace(go.Scatter(
        x=sim_x, y=sim_R, mode='lines+markers', name='R 耦合度(模拟)',
        line=dict(color='green'), marker=dict(size=4),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=actual_x, y=actual_R, mode='lines', name='R 实际(末日往回数)',
        line=dict(color='cyan', dash='dot'),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=sim_x, y=sim_theta_deg, mode='lines', name='θ 偏离角(度)',
        line=dict(color='orange'), yaxis='y4',
    ), row=3, col=1)
    fig.update_yaxes(title_text='R [0,1]', range=[0, 1], row=3, col=1)

    # Row 4: 状态分类带
    # legend-only invisible traces
    for s_label, color in STATE_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=color, symbol='square'),
            name=STATE_LABELS_CN[s_label],
            showlegend=True,
        ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=sim_x, y=[0] * len(sim_x),
        mode='markers',
        marker=dict(
            size=22,
            color=[STATE_COLORS.get(s, '#7f8c8d') for s in sim_states],
            symbol='square',
            line=dict(width=0),
        ),
        text=[STATE_LABELS_CN.get(s, s) for s in sim_states],
        hovertemplate='Step %{x}<br>状态: %{text}<extra></extra>',
        showlegend=False,
    ), row=4, col=1)
    fig.update_yaxes(title_text='状态', showticklabels=False, range=[-1, 1], row=4, col=1)

    # Row 5: 力分解
    fig.add_trace(go.Scatter(
        x=sim_x, y=sim_F_M, mode='lines', name='‖F_market‖',
        line=dict(color='cyan'),
    ), row=5, col=1)
    fig.add_trace(go.Scatter(
        x=sim_x, y=sim_F_R, mode='lines', name='‖F_restore‖',
        line=dict(color='lime', dash='dot'),
    ), row=5, col=1)
    fig.add_trace(go.Scatter(
        x=sim_x, y=sim_F_D, mode='lines', name='‖F_damp‖',
        line=dict(color='magenta', dash='dot'),
    ), row=5, col=1)
    fig.add_trace(go.Scatter(
        x=sim_x, y=sim_F_S, mode='lines', name='‖F_self‖',
        line=dict(color='orange'),
    ), row=5, col=1)
    fig.update_yaxes(title_text='力 (‖·‖,原始量纲)', row=5, col=1)
    fig.update_xaxes(title_text='步数(0=模拟起点,负=末日往回数)', row=5, col=1)

    fig.update_layout(
        template='plotly_dark',
        height=1500,
        title_text=(
            f'离散动力系统轨迹: {STOCK_NAME} ({args.code}) → {INDEX_NAME} ({INDEX_CODE}) '
            f'| N={args.horizon} | k={args.k_restore} c={args.c_damp}'
        ),
        yaxis4=dict(
            title='θ (度)', overlaying='y3', side='right',
            range=[0, 180], showgrid=False,
        ),
        legend=dict(orientation='h', yanchor='bottom', y=-0.12, xanchor='right', x=1),
    )

    html_path = os.path.join(OUT_DIR, HTML_NAME).replace('\\', '/')
    fig.write_html(html_path)

    print(f"\n=== 输出 ===")
    print(f"  HTML:        {html_path}")
    print(f"  描述 CSV:    {desc_csv}  (14 列 Dyn_)")
    print(f"  力分解 CSV:  {frc_csv}  (8 列 Frc_)")
    print(f"  模拟 CSV:    {sim_csv}  (18 列 Sim_, 含 4 个力)")


if __name__ == '__main__':
    main()
