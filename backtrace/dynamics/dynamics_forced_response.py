#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v5 — 受迫系统 + G(ω) 频率响应 CLI。

从 _dynamics_core.predict_next_state 出发,纯解析推导离散时间 2D 动力系统
对正弦 β-forcing 的复频响应:

    H(jω) = V_M0·[k + c·(z-1)] / [(z-1)² + c·(z-1) + k]   其中 z = e^(jω)

(本文件实现忽略 V_M0 标量 — 不影响 |H| 形状。)

特性:
- DC gain: H(j0) = 1
- 共振:Schur 楔形外(k > c,复数极点)→ |H| 在 ω_n = arctan(√(4k-c²)/(2-c)) 处爆炸
- 滚降:Schur 楔形内(k < c)→ |H| 单调滚降,无峰值
- Schur 楔形:本线性化系统的稳定性边界为 k = c(不是 v4.7 的 c² = 4k)

输出(全 gitignored):
  - data/dynamics/transfer_function_grid.csv
  - data/dynamics/transfer_function_stability.csv
  - backtrace/outputs/dynsys_forced_response.html
  - backtrace/outputs/dynsys_forced_response_stability.html
  - backtrace/outputs/dynsys_forced_response_summary.txt
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)

import argparse
import numpy as np
import pandas as pd

CSV_OUT_DIR = "data/dynamics"
HTML_OUT_DIR = "backtrace/outputs"
DEFAULT_GRID_CSV = os.path.join(CSV_OUT_DIR, "transfer_function_grid.csv")
DEFAULT_STABILITY_CSV = os.path.join(CSV_OUT_DIR, "transfer_function_stability.csv")
DEFAULT_BODE_HTML = os.path.join(HTML_OUT_DIR, "dynsys_forced_response.html")
DEFAULT_HEATMAP_HTML = os.path.join(HTML_OUT_DIR, "dynsys_forced_response_stability.html")
DEFAULT_SUMMARY_TXT = os.path.join(HTML_OUT_DIR, "dynsys_forced_response_summary.txt")

DEFAULT_K = 2.0
DEFAULT_C = 1.5
DEFAULT_K_GRID = np.linspace(0.1, 6.0, 60)
DEFAULT_C_GRID = np.linspace(0.1, 6.0, 60)
DEFAULT_OMEGA_GRID = np.linspace(0.001, np.pi, 200)


def parse_args():
    p = argparse.ArgumentParser(description="v5 受迫系统 + G(ω) 频率响应")
    p.add_argument("--k", type=float, default=DEFAULT_K,
                   help=f"恢复系数 k (默认 {DEFAULT_K})")
    p.add_argument("--c", type=float, default=DEFAULT_C,
                   help=f"阻尼系数 c (默认 {DEFAULT_C})")
    p.add_argument("--grid-csv", default=DEFAULT_GRID_CSV,
                   help=f"频率扫描 CSV 输出路径")
    p.add_argument("--stability-csv", default=DEFAULT_STABILITY_CSV,
                   help=f"(k, c) 稳定性扫描 CSV 输出路径")
    p.add_argument("--bode-html", default=DEFAULT_BODE_HTML,
                   help=f"Bode 图 HTML 输出路径")
    p.add_argument("--heatmap-html", default=DEFAULT_HEATMAP_HTML,
                   help=f"稳定性热图 HTML 输出路径")
    p.add_argument("--summary-txt", default=DEFAULT_SUMMARY_TXT,
                   help=f"UTF-8 中文汇总输出路径")
    p.add_argument("--overlay", default="",
                   help="多对 (k, c, label) overlay,格式 'k1,c1,label1; k2,c2,label2; ...'")
    p.add_argument("--overlay-html", default=os.path.join(HTML_OUT_DIR, "dynsys_bode_overlay.html"),
                   help=f"overlay Bode HTML 输出路径")
    p.add_argument("--overlay-summary-txt", default=os.path.join(HTML_OUT_DIR, "dynsys_bode_overlay_summary.txt"),
                   help="overlay UTF-8 中文汇总输出路径")
    # v5.2 数据驱动模式 flags
    p.add_argument("--from-kc-estimates", default="",
                   help="v5.2 数据驱动:parameter_fit kc_estimates.csv 路径(与 --overlay 互斥)")
    p.add_argument("--top-n", type=int, default=5,
                   help="v5.2 选 top-N 行业(默认 5)")
    p.add_argument("--industry-agg", choices=['median', 'mean'], default='median',
                   help="v5.2 行业聚合方法(默认 median)")
    p.add_argument("--select-criterion", choices=['by_n_stocks', 'by_c_over_k', 'by_k_over_c'],
                   default='by_n_stocks',
                   help="v5.2 排序标准(默认 by_n_stocks)")
    p.add_argument("--industry-pairs-csv",
                   default=os.path.join(HTML_OUT_DIR, "dynsys_industry_overlay_pairs.csv"),
                   help="v5.2 选中行业 CSV 输出路径")
    return p.parse_args()


def transfer_function(omega, k, c):
    """复频响应 H(jω) = [k + c·(z-1)] / [(z-1)² + c·(z-1) + k],z = e^(jω)。

    (本实现忽略 V_M0 标量 — 它只放大常数倍,不影响 |H| 形状。)

    Args:
        omega: 角频率数组(ndarray 或标量)
        k: 恢复系数
        c: 阻尼系数

    Returns:
        复数 ndarray,shape 与 omega 相同
    """
    omega = np.asarray(omega)
    z = np.exp(1j * omega)
    z_minus_1 = z - 1
    numerator = k + c * z_minus_1
    denominator = z_minus_1 ** 2 + c * z_minus_1 + k
    return numerator / denominator


def natural_frequency(k, c):
    """离散自然频率 ω_n = arg(z_pole),其中 z_pole 是复数极点(Schur 楔形外)。

    极点 z = 1 - c/2 + j·√(4k-c²)/2(c² < 4k 时为复数)。
    arg(z) = arctan(√(4k-c²) / (2-c))

    当 c² >= 4k(实根)或 k <= 0 时返回 NaN。

    Returns:
        float,ω_n ∈ (0, π/2)
    """
    if c ** 2 >= 4 * k or k <= 0:
        return float('nan')
    real_part = 1 - c / 2
    imag_part = np.sqrt(4 * k - c ** 2) / 2
    return float(np.arctan2(imag_part, real_part))


def classify_response_type(k, c):
    """判别阻尼类型(基于 Schur 楔形 k = c 边界)。

    Returns:
        str ∈ {'overdamped', 'critical', 'underdamped', 'anti_damped'}
        - overdamped: k < c(Schur 内,稳定)
        - critical:   k ≈ c(Schur 边界)
        - underdamped: k > c(Schur 外,不稳定 / 共振)
        - anti_damped: k < 0(负恢复系数,反向弹簧)
    """
    if k < 0:
        return 'anti_damped'
    if abs(k - c) < 1e-9:
        return 'critical'
    if k < c:
        return 'overdamped'
    return 'underdamped'


def magnitude_phase(omega_array, k, c):
    """对给定 (k, c),算每个 ω 的 |H(jω)| 和 arg H(jω)。

    Args:
        omega_array: 一维 ndarray,角频率
        k, c: 系统参数

    Returns:
        (magnitude, phase_rad) — 两个 ndarray,shape 与 omega_array 相同
    """
    H = transfer_function(omega_array, k, c)
    magnitude = np.abs(H)
    phase_rad = np.angle(H)
    return magnitude, phase_rad


def is_in_schur_wedge(k, c, tol=1e-9):
    """Schur 楔形判定(本线性化系统):k > 0 且 k < c + tol。

    注:v4.7 的 Schur 楔形 c² >= 4k 对应不同 2×2 系统(特征多项式 det(zI-A)=z²-cz+k);
    本系统 A=[[1,1],[-k,1-c]] 的特征多项式是 (z-1)² + c(z-1) + k,稳定性边界是 k = c。
    """
    return k > 0 and k < c + tol


def bode_plot(omega_grid, k, c, output_path):
    """画 Bode 图:幅值 log-scale + 相位度数。

    (1,1) |H(jω)| 半对数 + 红虚线 ω_n + 灰虚线 |H|=1
    (1,2) arg H(jω) 度数 + 红虚线 -180°
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    magnitude, phase_rad = magnitude_phase(omega_grid, k, c)
    phase_deg = np.degrees(phase_rad)
    log_mag = np.log10(np.clip(magnitude, 1e-12, None))
    omega_n = natural_frequency(k, c)
    response_type = classify_response_type(k, c)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"|H(jω)| 半对数图 (k={k}, c={c}, {response_type})",
                        f"arg H(jω) 度数"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(go.Scatter(
        x=omega_grid, y=log_mag, mode='lines', name='|H|',
        line=dict(color='steelblue', width=2),
    ), row=1, col=1)
    if np.isfinite(omega_n):
        fig.add_vline(x=omega_n, line_dash='dash', line_color='red',
                      annotation_text=f'ω_n={omega_n:.3f}', row=1, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='gray',
                  annotation_text='|H|=1', row=1, col=1)
    fig.update_xaxes(title_text='ω (rad)', row=1, col=1)
    fig.update_yaxes(title_text='log10|H|', row=1, col=1)
    fig.add_trace(go.Scatter(
        x=omega_grid, y=phase_deg, mode='lines', name='arg H',
        line=dict(color='indianred', width=2),
    ), row=1, col=2)
    fig.add_hline(y=-180, line_dash='dash', line_color='red',
                  annotation_text='-180°', row=1, col=2)
    fig.update_xaxes(title_text='ω (rad)', row=1, col=2)
    fig.update_yaxes(title_text='arg H (degrees)', row=1, col=2,
                     range=[-200, 200])
    omega_n_str = f'{omega_n:.4f}' if np.isfinite(omega_n) else 'N/A'
    fig.update_layout(
        height=500, width=1400,
        title_text=f"v5 Bode 图 (k={k}, c={c}, ω_n={omega_n_str})",
        showlegend=False,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')


def stability_heatmap(k_grid, c_grid, output_path):
    """2D (k, c) 热图 |H(jω_n)| + Schur 楔形边界(c = k)。

    颜色:|H(jω_n)| 对数尺度
    黑色虚线:Schur 边界 c = k(本线性化系统的稳定性边界)
    楔形内(k < c):稳定(色冷);楔形外(k > c):共振爆炸(色热)
    仅对复数极点(c² < 4k 且 k > 0)画 |H(jω_n)|;其余点 NaN。
    """
    import plotly.graph_objects as go
    K, C = np.meshgrid(k_grid, c_grid)  # shape (len(c_grid), len(k_grid))
    H_at_resonance = np.full_like(K, np.nan)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            k_val = K[i, j]
            c_val = C[i, j]
            if k_val <= 0:
                continue
            omega_n = natural_frequency(k_val, c_val)
            if not np.isfinite(omega_n):
                continue
            H_val = transfer_function(omega_n, k_val, c_val)
            H_at_resonance[i, j] = np.abs(H_val)
    log_H = np.log10(np.clip(H_at_resonance, 1e-12, None))
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=k_grid, y=c_grid, z=log_H,
        colorscale='RdBu_r',
        zmid=0, zmin=-2, zmax=4,
        colorbar=dict(title='log10|H(jω_n)|'),
        hovertemplate='k=%{x:.3f}<br>c=%{y:.3f}<br>log10|H|=%{z:.3f}<extra></extra>',
    ))
    # Schur boundary c = k
    k_boundary = np.linspace(k_grid[0], k_grid[-1], 100)
    c_boundary = k_boundary  # c = k
    fig.add_trace(go.Scatter(
        x=k_boundary, y=c_boundary,
        mode='lines', name='Schur boundary c=k',
        line=dict(color='black', dash='dash', width=2),
    ))
    fig.update_layout(
        title='v5 频率响应稳定性热图 (|H(jω_n)|,Schur 楔形边界 c=k)',
        xaxis_title='k (恢复系数)',
        yaxis_title='c (阻尼系数)',
        height=700, width=900,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')
    return log_H


def write_summary(omega_grid, k_grid, c_grid, k, c, output_path):
    """写 UTF-8 中文汇总:典型 (k, c) 的频率响应特征 + (k, c) 网格统计。"""
    omega_n = natural_frequency(k, c)
    response_type = classify_response_type(k, c)
    magnitude_at_dc, _ = magnitude_phase(np.array([0.001]), k, c)
    magnitude_at_n, phase_at_n = (None, None), None
    if np.isfinite(omega_n):
        magnitude_at_n, phase_at_n = magnitude_phase(np.array([omega_n]), k, c)
    magnitude_at_pi, _ = magnitude_phase(np.array([np.pi]), k, c)
    in_wedge = is_in_schur_wedge(k, c)
    omega_n_str = f'{omega_n:.4f}' if np.isfinite(omega_n) else 'N/A'
    lines = [
        '=' * 70,
        'v5 受迫系统 + G(ω) 频率响应 汇总',
        '=' * 70,
        f'给定 (k, c) = ({k}, {c})',
        f'  阻尼类型: {response_type}',
        f'  Schur 楔形内: {in_wedge}',
        f'  自然频率 ω_n = arctan(√(4k-c²)/(2-c)) = {omega_n_str}',
        f'  |H(j0)|  ≈ {float(magnitude_at_dc[0]):.4f} (DC 增益,应 ≈ 1)',
    ]
    if np.isfinite(omega_n):
        lines.append(f'  |H(jω_n)| = {float(magnitude_at_n[0]):.4f} (共振峰)')
        lines.append(f'  arg H(jω_n) = {float(np.degrees(phase_at_n[0])):.2f}°')
    lines.append(f'  |H(jπ)|   = {float(magnitude_at_pi[0]):.4f} (Nyquist)')
    lines.extend([
        '',
        f'频率扫描: ω ∈ [{omega_grid[0]:.4f}, {omega_grid[-1]:.4f}], 共 {len(omega_grid)} 点',
        f'(k, c) 网格扫描: k ∈ [{k_grid[0]:.2f}, {k_grid[-1]:.2f}], '
        f'c ∈ [{c_grid[0]:.2f}, {c_grid[-1]:.2f}], 共 {len(k_grid)}×{len(c_grid)} = {len(k_grid)*len(c_grid)} 点',
        '',
    ])
    # 物理含义注释
    if response_type == 'overdamped':
        lines.append('物理含义:系统强阻尼,β 强迫不会引发共振,|H| 单调滚降。')
    elif response_type == 'critical':
        lines.append('物理含义:临界阻尼,边界 case,接近不振荡。')
    elif response_type == 'underdamped':
        lines.append('物理含义:欠阻尼,β 强迫在 ω_n 处被放大,|H| > 1。')
    elif response_type == 'anti_damped':
        lines.append('物理含义:反向弹簧(k<0),低频处 |H| 爆炸,系统无界。')
    lines.append('')
    lines.append('=' * 70)
    text = '\n'.join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return text


def bode_overlay(omega_grid, k_c_pairs, output_path, title="Industry G(ω) Frequency Response Comparison"):
    """多对 (k, c) Bode plot 叠加对比。

    Args:
        omega_grid: 角频率数组,shape (N,),共享
        k_c_pairs: [(k, c, label), ...] 列表
        output_path: HTML 输出路径
        title: 图表标题

    行为:
        - 2 子图:上幅频 |H(jω)| dB vs ω,下相频 arg H(jω) vs ω
        - 每对一条曲线,共享 omega_grid,不同颜色 + 实线
        - 每对额外标出 ω_n(如果有)+ 该点的 |H(jω_n)| (dB)
        - legend 显示 label(带 (k, c) 数值)
        - HTML 通过 plotly CDN 渲染(include_plotlyjs='cdn')

    Raises:
        ValueError: 空列表 / k <= 0 / c <= 0 / label 重复
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not k_c_pairs:
        raise ValueError("k_c_pairs 不能为空")
    labels = [p[2] for p in k_c_pairs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"label 重复: {labels}")
    for k, c, _ in k_c_pairs:
        if k <= 0 or c <= 0:
            raise ValueError(f"k 和 c 必须 > 0,得 (k={k}, c={c})")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("|H(jω)| (dB)", "arg H(jω) (degrees)"),
                        vertical_spacing=0.12)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for idx, (k, c, label) in enumerate(k_c_pairs):
        H = transfer_function(omega_grid, k, c)
        mag, phase = magnitude_phase(omega_grid, k, c)
        color = colors[idx % len(colors)]
        legend_name = f'{label} (k={k}, c={c})'
        mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
        fig.add_trace(go.Scatter(x=omega_grid, y=mag_db, mode='lines',
                                 name=legend_name, line=dict(color=color, width=2),
                                 legendgroup=label), row=1, col=1)
        fig.add_trace(go.Scatter(x=omega_grid, y=np.degrees(phase), mode='lines',
                                 name=legend_name, line=dict(color=color, width=2),
                                 legendgroup=label, showlegend=False), row=2, col=1)
        # 标出 ω_n(如果有)+ |H(jω_n)|
        omega_n = natural_frequency(k, c)
        if np.isfinite(omega_n):
            mag_n_scalar, _ = magnitude_phase(np.array([omega_n]), k, c)
            fig.add_trace(go.Scatter(
                x=[omega_n], y=[20 * np.log10(max(float(mag_n_scalar[0]), 1e-12))],
                mode='markers', name=f'{label} ω_n',
                marker=dict(color=color, size=10, symbol='x'),
                legendgroup=label, showlegend=False,
                hovertemplate=f'ω_n={omega_n:.4f}<br>|H|={float(mag_n_scalar[0]):.2f}<extra></extra>',
            ), row=1, col=1)

    fig.update_xaxes(title_text='ω (角频率,rad/sample)', row=2, col=1)
    fig.update_yaxes(title_text='|H(jω)| (dB)', row=1, col=1)
    fig.update_yaxes(title_text='相位 (degrees)', row=2, col=1)
    fig.update_layout(title=title, height=800, width=1000,
                      hovermode='x unified', legend=dict(orientation='v',
                                                          xanchor='left', yanchor='top',
                                                          x=1.02, y=1.0))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')


def write_overlay_summary(omega_grid, k_c_pairs, output_path):
    """多对 (k, c) 的 UTF-8 中文汇总表。

    每对一行,展示:
            - 行业/时间窗 label
            - (k, c) + 响应类型
            - ω_n + |H(jω_n)|(若有)
            - |H(j0)| (DC 增益)
            - |H(jπ)| (Nyquist)
            - Schur 楔形内/外
            - 一句业务解读
    """
    lines = [
        '=' * 80,
        f'v5.1 Industry G(ω) Frequency Response Comparison — {len(k_c_pairs)} 对 (k, c)',
        '=' * 80,
        '',
    ]
    for k, c, label in k_c_pairs:
        omega_n = natural_frequency(k, c)
        response_type = classify_response_type(k, c)
        in_wedge = is_in_schur_wedge(k, c)
        mag_dc, _ = magnitude_phase(np.array([0.001]), k, c)
        mag_pi, _ = magnitude_phase(np.array([np.pi]), k, c)
        lines.append(f'[{label}]  (k={k}, c={c})')
        lines.append(f'  响应类型: {response_type}    Schur 楔形内: {in_wedge}')
        if np.isfinite(omega_n):
            mag_n, _ = magnitude_phase(np.array([omega_n]), k, c)
            lines.append(f'  ω_n = {omega_n:.4f}    |H(jω_n)| = {float(mag_n[0]):.4f}')
        else:
            lines.append(f'  ω_n = N/A (实极点)')
        lines.append(f'  |H(j0)|  = {float(mag_dc[0]):.4f} (DC 增益)')
        lines.append(f'  |H(jπ)| = {float(mag_pi[0]):.4f} (Nyquist)')
        # 业务解读
        if not in_wedge:
            lines.append(f'  业务解读: 共振风险高,β 强迫会在 ω_n 处放大 {float(mag_n[0]):.1f} 倍')
        elif response_type == 'overdamped':
            lines.append(f'  业务解读: 低通过滤器,β 强迫不会引发共振,稳定')
        elif response_type == 'critical':
            lines.append(f'  业务解读: 临界阻尼,边界 case')
        else:
            lines.append(f'  业务解读: 标准响应')
        lines.append('')
    lines.append('=' * 80)
    text = '\n'.join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)


def parse_overlay_pairs(s):
    """解析 --overlay CLI 字符串为 [(k, c, label), ...]。

    格式:"k1,c1,label1; k2,c2,label2; ..."
    - 分号 `;` 分隔不同对
    - 逗号 `,` 分隔 k / c / label
    - label 可含逗号 / 空格(只取前两个逗号之前的为 k, c;之后全是 label)
    """
    if not s or not s.strip():
        raise ValueError("overlay 字符串为空")
    pairs = []
    for chunk in s.split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(',', 2)  # 只 split 前 2 个逗号,label 可含逗号
        if len(parts) < 3:
            raise ValueError(f"格式错误: '{chunk}' 期望 k,c,label")
        try:
            k = float(parts[0].strip())
        except ValueError:
            raise ValueError(f"k 必须为数字,得 '{parts[0]}'")
        try:
            c = float(parts[1].strip())
        except ValueError:
            raise ValueError(f"c 必须为数字,得 '{parts[1]}'")
        label = parts[2].strip()
        pairs.append((k, c, label))
    if not pairs:
        raise ValueError("未解析出任何 (k, c, label) 对")
    return pairs


def load_kc_estimates(csv_path):
    """读 parameter_fit kc_estimates.csv,验证必需列,过滤失败行。

    必需列:code, index_code, k_hat, c_hat, status(其他列可选)

    Returns:
        DataFrame,只保留 status='ok' 的行(过滤拟合失败的)

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 必需列缺失
    """
    import os
    REQUIRED_COLS = ['code', 'index_code', 'k_hat', 'c_hat', 'status']
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"kc_estimates CSV 不存在: {csv_path}\n"
            f"提示:python backtrace/projection/parameter_fit.py 先跑出 (k̂, ĉ)"
        )
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"kc_estimates CSV 缺必需列: {missing}")
    df = df[df['status'] == 'ok'].reset_index(drop=True)
    return df


def aggregate_by_industry(df, group_col='index_code', agg='median'):
    """按行业聚合 (k̂, ĉ)。

    Args:
        df: load_kc_estimates 输出
        group_col: 分组列(默认 index_code)
        agg: 聚合方法("median" / "mean"),默认 median(抗极端值)

    Returns:
        DataFrame 列:[group_col, n_stocks, k_hat, c_hat]
        按 group_col 排序
    """
    if agg not in ('median', 'mean'):
        raise ValueError(f"agg 必须 'median' 或 'mean',得 '{agg}'")
    agg_fn = np.median if agg == 'median' else np.mean
    rows = []
    for grp, sub in df.groupby(group_col):
        rows.append({
            group_col: grp,
            'n_stocks': len(sub),
            'k_hat': float(agg_fn(sub['k_hat'].values)),
            'c_hat': float(agg_fn(sub['c_hat'].values)),
        })
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def select_top_n_industries(df, criterion='by_n_stocks', n=5, group_col='index_code'):
    """从聚合 DataFrame 选 top-N 行业,转 v5.1 overlay 格式。

    Args:
        df: aggregate_by_industry 输出
        criterion: 排序标准
            - "by_n_stocks": 按股票数降序(最多成分股的行业)
            - "by_c_over_k": 按 c/k 比降序(最过阻尼,稳定)
            - "by_k_over_c": 按 k/c 比降序(最欠阻尼,危险)
        n: top N
        group_col: label 用 group_col 值,前缀 "Industry "

    Returns:
        [(k̂, ĉ, label), ...] — 直接喂给 bode_overlay
    """
    df_sorted = df.copy()
    if criterion == 'by_n_stocks':
        df_sorted = df_sorted.sort_values('n_stocks', ascending=False)
    elif criterion == 'by_c_over_k':
        df_sorted['_ratio'] = df_sorted['c_hat'] / df_sorted['k_hat'].replace(0, np.nan)
        df_sorted = df_sorted.sort_values('_ratio', ascending=False)
    elif criterion == 'by_k_over_c':
        df_sorted['_ratio'] = df_sorted['k_hat'] / df_sorted['c_hat'].replace(0, np.nan)
        df_sorted = df_sorted.sort_values('_ratio', ascending=False)
    else:
        raise ValueError(f"criterion 必须 by_n_stocks / by_c_over_k / by_k_over_c,得 '{criterion}'")
    df_sorted = df_sorted.head(n)
    pairs = []
    for _, row in df_sorted.iterrows():
        label = f"Industry {row[group_col]}"
        pairs.append((float(row['k_hat']), float(row['c_hat']), label))
    return pairs


def write_industry_pairs_csv(pairs, agg_df, output_path):
    """写选中行业的 (k̂, ĉ) + label + 行业股票数到 UTF-8 CSV(审计用)。

    Args:
        pairs: select_top_n_industries 输出 [(k, c, label), ...]
        agg_df: aggregate_by_industry 输出 DataFrame
        output_path: 输出 CSV 路径
    """
    import re
    rows = []
    for k, c, label in pairs:
        # 从 label "Industry XXX" 提取行业 code
        m = re.match(r'Industry\s+(.*)', label)
        industry_code = m.group(1) if m else label
        # 从 agg_df 查 n_stocks
        match = agg_df[agg_df['index_code'] == industry_code]
        n_stocks = int(match['n_stocks'].iloc[0]) if len(match) > 0 else 0
        rows.append({
            'industry_code': industry_code,
            'k_hat': k,
            'c_hat': c,
            'n_stocks': n_stocks,
        })
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding='utf-8-sig')


def main():
    args = parse_args()
    # v5.1 overlay 分支:有 --overlay 则跳过单对逻辑,只写 overlay 文件
    if args.overlay:
        pairs = parse_overlay_pairs(args.overlay)
        omega_grid_overlay = np.linspace(0.001, np.pi, 200)
        bode_overlay(omega_grid_overlay, pairs, args.overlay_html,
                     title=f'v5.1 Industry G(ω) Comparison — {len(pairs)} 对')
        write_overlay_summary(omega_grid_overlay, pairs, args.overlay_summary_txt)
        print(f'[v5.1 overlay] {len(pairs)} 对 (k, c) 已写入 {args.overlay_html}')
        return  # overlay-only 模式,跳过单对 main 后续
    # v5.2 数据驱动分支
    if args.from_kc_estimates:
        kc_df = load_kc_estimates(args.from_kc_estimates)
        agg_df = aggregate_by_industry(kc_df, group_col='index_code', agg=args.industry_agg)
        if len(agg_df) == 0:
            raise RuntimeError(
                f"kc_estimates.csv 没有 status='ok' 的行。请检查 {args.from_kc_estimates}"
            )
        pairs = select_top_n_industries(agg_df, criterion=args.select_criterion,
                                         n=args.top_n, group_col='index_code')
        if len(pairs) < args.top_n:
            print(f'[v5.2] 警告:实际只 {len(pairs)} 个行业(请求 {args.top_n})')
        omega_grid_overlay = np.linspace(0.001, np.pi, 200)
        bode_overlay(omega_grid_overlay, pairs, args.overlay_html,
                     title=f'v5.2 Industry G(ω) — {args.select_criterion} top-{len(pairs)}')
        write_overlay_summary(omega_grid_overlay, pairs, args.overlay_summary_txt)
        # 写行业 pairs CSV(审计用)
        write_industry_pairs_csv(pairs, agg_df, args.industry_pairs_csv)
        print(f'[v5.2] {len(pairs)} 个行业已写入 {args.overlay_html} + {args.industry_pairs_csv}')
        return
    # else: 单对模式(v5 既有逻辑,完全不变)
    omega_grid = DEFAULT_OMEGA_GRID
    k_grid = DEFAULT_K_GRID
    c_grid = DEFAULT_C_GRID
    # 1. 频率扫描(给定 k, c)
    magnitude, phase_rad = magnitude_phase(omega_grid, args.k, args.c)
    omega_n = natural_frequency(args.k, args.c)
    grid_df = pd.DataFrame({
        'omega': omega_grid,
        'magnitude': magnitude,
        'phase_rad': phase_rad,
        'phase_deg': np.degrees(phase_rad),
        'log10_magnitude': np.log10(np.clip(magnitude, 1e-12, None)),
    })
    grid_df.to_csv(args.grid_csv, index=False, encoding='utf-8-sig')
    print(f'✓ {args.grid_csv} ({len(grid_df)} 频率点)')
    # 2. (k, c) 网格扫描
    K, C = np.meshgrid(k_grid, c_grid)
    stability_rows = []
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            k_val = K[i, j]
            c_val = C[i, j]
            omega_n_val = natural_frequency(k_val, c_val) if k_val > 0 else np.nan
            H_val = (transfer_function(omega_n_val, k_val, c_val)
                     if k_val > 0 and np.isfinite(omega_n_val) else np.nan)
            stability_rows.append({
                'k': k_val, 'c': c_val,
                'omega_n': omega_n_val,
                'H_magnitude': abs(H_val) if not np.isnan(H_val) else np.nan,
                'H_log10': np.log10(abs(H_val)) if not np.isnan(H_val) else np.nan,
                'response_type': classify_response_type(k_val, c_val),
                'in_schur_wedge': is_in_schur_wedge(k_val, c_val),
            })
    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(args.stability_csv, index=False, encoding='utf-8-sig')
    print(f'✓ {args.stability_csv} ({len(stability_df)} 网格点)')
    # 3. Bode 图 HTML
    bode_plot(omega_grid, args.k, args.c, args.bode_html)
    print(f'✓ {args.bode_html}')
    # 4. 稳定性热图 HTML
    stability_heatmap(k_grid, c_grid, args.heatmap_html)
    print(f'✓ {args.heatmap_html}')
    # 5. 文本汇总
    write_summary(omega_grid, k_grid, c_grid, args.k, args.c, args.summary_txt)
    print(f'✓ {args.summary_txt}')


if __name__ == '__main__':
    main()