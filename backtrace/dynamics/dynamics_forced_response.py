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


def main():
    args = parse_args()
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