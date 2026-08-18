# v5 受迫系统 + G(ω) 频率响应 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 sinusoidal β-forcing 测离散时间 2D 动力系统的复频响应 H(jω) = [k + c(z-1)] / [(z-1)² + k],画 Bode 图 + Schur 楔形上的 |H(jω_n)| 热图,连接 v4.7 SI 稳定性与频域行为。

**Architecture:**
- **独立 CLI** `backtrace/dynamics/dynamics_forced_response.py` (~400 行)— 不 import 同目录 sibling 模块,沿用 v4.8/v4.9/v4.10 模式
- **纯解析 + 数值扫描** — 不调 `predict_next_state`(只在公式推导中参考),直接 numpy 向量化算 H(jω)
- **z 域公式**:`H(jω) = (k + c·(z-1)) / ((z-1)² + k)`,`z = e^(jω)`
- **离散时间 Bode**:`ω ∈ [0.001, π]`,Nyquist 在 π

**Tech Stack:**
- Python 3.13 (Anaconda)
- numpy / pandas / scipy(可选,phase unwrap 用)
- plotly (make_subplots, 沿用 v4.9)
- pytest + tmp_path fixtures

---

## Global Constraints

复制自 spec(每条都需严格遵守):

- 数学层 `_dynamics_core.py` **0 行修改**(硬约束,任务验证会查)
- 3 caller (`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`) **0 行修改**
- `dynamics_si_ic.py` (v4.8) **0 行修改**
- `dynamics_si_timeseries.py` (v4.9) **0 行修改**
- `dynamics_si_lagged_ic.py` (v4.10) **0 行修改**
- `compute_sector_stability_timeseries` (v4.9) **0 行修改**
- v4.7 `compute_sector_stability` **不动**
- v3 `predict_next_state` / `simulate_trajectory` **不动**(只在推导中参考)
- 输出全部 gitignored (`data/dynamics/` + `backtrace/outputs/`)
- `PYTHONIOENCODING=utf-8` 必备(Windows GBK)
- Python 路径:`/c/ProgramData/anaconda3/python.exe`
- 安全:`jhzq/交易凭据.md` 不能写进代码或 git
- Subagent-Driven Development (SDD) workflow
- 总测试数:48 (v4.10) + 5 (v5) = **53 tests pass**

---

## File Structure

```
backtrace/dynamics/
├── _dynamics_core.py          [不动 — 数学层]
├── dynamics_forced_response.py [NEW — 本计划产物]
├── dynamics_system.py          [不动 — caller]
├── dynamics_batch.py           [不动 — caller]
├── dynamics_1step_oos.py       [不动 — caller,时域 1-step OOS]
├── dynamics_si_ic.py           [不动 — v4.8]
├── dynamics_si_timeseries.py   [不动 — v4.9]
├── dynamics_si_lagged_ic.py    [不动 — v4.10]
└── dynamics_eigen_analysis.py  [不动 — v4.7 + v4.9 函数]

tests/
└── test_dynamics_eigen.py      [+5 测试末尾追加]
```

**8 protected files,1 new file,2 modified files.**

---

## Task 1: 受迫系统 CLI + 5 测试 + README

**Files:**
- New: `backtrace/dynamics/dynamics_forced_response.py` (~400 行,独立 CLI)
- Modify: `tests/test_dynamics_eigen.py` (末尾追加 5 测试, ~80 行)
- Modify: `backtrace/dynamics/README.md` §4 (~40 行)

**Interfaces:**
- Consumes: 无外部数据(纯解析数学 + 数值扫描)
- Produces:
  - `data/dynamics/transfer_function_grid.csv` (200 行 ω 扫描 × 多列)
  - `data/dynamics/transfer_function_stability.csv` (60×60 (k, c) 网格)
  - `backtrace/outputs/dynsys_forced_response.html` (Bode 图 2 子图)
  - `backtrace/outputs/dynsys_forced_response_stability.html` (2D 热图 + Schur 边界)
  - `backtrace/outputs/dynsys_forced_response_summary.txt` (UTF-8 中文)

- [ ] **Step 1: 新建 `dynamics_forced_response.py` 骨架(~50 行,imports + 常量 + main() stub)**

创建文件:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v5 — 受迫系统 + G(ω) 频率响应 CLI。

从 _dynamics_core.predict_next_state 出发,纯解析推导离散时间 2D 动力系统
对正弦 β-forcing 的复频响应:

    H(jω) = [k + c·(z-1)] / [(z-1)² + k]   其中 z = e^(jω)

特性:
- DC gain: H(j0) = 1
- 共振:Schur 楔形外(c² < 4k)→ |H| 在 ω_n = arctan(√k) 处爆炸
- 滚降:Schur 楔形内(c² > 4k)→ |H| 单调滚降,无峰值

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
```

**关键点**:
- 不 import 同目录 sibling 模块(避免循环依赖 + 与 v4.8/v4.9/v4.10 模式一致)
- 全部为 numpy 数值扫描,无外部数据依赖

- [ ] **Step 2: 实现 `transfer_function` 核心函数(~15 行)**

放在 `parse_args` 之后:

```python
def transfer_function(omega, k, c):
    """复频响应 H(jω) = [k + c·(z-1)] / [(z-1)² + k],z = e^(jω)。

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
    denominator = z_minus_1 ** 2 + k
    return numerator / denominator


def natural_frequency(k):
    """离散时间自然频率 ω_n = arctan(√k)。

    Returns:
        float,ω_n ∈ (0, π/2)
    """
    return float(np.arctan(np.sqrt(k)))


def classify_response_type(k, c):
    """判别阻尼类型。

    Returns:
        str ∈ {'overdamped', 'critical', 'underdamped', 'anti_damped'}
        - overdamped: c² > 4k (Schur 内)
        - critical:   c² ≈ 4k
        - underdamped: 0 < c² < 4k (Schur 外)
        - anti_damped: k < 0 (负恢复系数)
    """
    if k < 0:
        return 'anti_damped'
    discriminant = c ** 2 - 4 * k
    if abs(discriminant) < 1e-9:
        return 'critical'
    if discriminant > 0:
        return 'overdamped'
    return 'underdamped'
```

**关键点**:
- `transfer_function` 完全向量化 — omega 可为任意形状 ndarray
- DC gain:`H(j0) = (k + 0) / (0 + k) = 1`(自动满足)
- `classify_response_type` 与 v4.7 Schur 楔形判定一致

- [ ] **Step 3: 实现 `magnitude_phase`(~15 行)**

```python
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
    """Schur 楔形判定:c² >= 4k 且 k > 0(同 v4.7)。"""
    return k > 0 and (c ** 2) >= (4 * k) - tol
```

**关键点**:
- `np.angle` 自动返回 (-π, π],无需 unwrap(本轮不画相位连续性)
- `is_in_schur_wedge` 与 v4.7 一致(供热图分类用)

- [ ] **Step 4: 实现 `bode_plot` HTML(~80 行,2 子图 plotly)**

```python
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
    omega_n = natural_frequency(k)
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
    fig.update_layout(
        height=500, width=1400,
        title_text=f"v5 Bode 图 (k={k}, c={c}, ω_n={omega_n:.4f})",
        showlegend=False,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')
```

- [ ] **Step 5: 实现 `stability_heatmap` HTML(~70 行,2D 热图 + Schur 边界)**

```python
def stability_heatmap(k_grid, c_grid, output_path):
    """2D (k, c) 热图 |H(jω_n)| + Schur 楔形边界(c = 2√k)。

    颜色:|H(jω_n)| 对数尺度
    黑色虚线:Schur 边界 c = 2√k
    楔形内:稳定(色冷),楔形外:共振爆炸(色热)
    """
    import plotly.graph_objects as go
    K, C = np.meshgrid(k_grid, c_grid)  # shape (len(c_grid), len(k_grid))
    H_at_resonance = np.zeros_like(K)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            k_val = K[i, j]
            c_val = C[i, j]
            if k_val <= 0:
                H_at_resonance[i, j] = np.nan
                continue
            omega_n = natural_frequency(k_val)
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
    # Schur boundary c = 2√k
    k_boundary = np.linspace(k_grid[0], k_grid[-1], 100)
    c_boundary = 2 * np.sqrt(k_boundary)
    fig.add_trace(go.Scatter(
        x=k_boundary, y=c_boundary,
        mode='lines', name='Schur boundary c=2√k',
        line=dict(color='black', dash='dash', width=2),
    ))
    fig.update_layout(
        title='v5 频率响应稳定性热图 (|H(jω_n)| 在 Schur 楔形上)',
        xaxis_title='k (恢复系数)',
        yaxis_title='c (阻尼系数)',
        height=700, width=900,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')
    return log_H
```

**关键点**:
- `meshgrid` 用 `c_grid` 在前(`shape[0]`),`k_grid` 在后(`shape[1]`),与 heatmap 默认接受 Y/X 一致
- 数值 clip `1e-12 → log10 = -12`,避免 log(0)
- Schur 边界 `c = 2√k` 沿 v4.7 Schur wedge

- [ ] **Step 6: 实现 `write_summary`(~25 行,UTF-8 文本)**

```python
def write_summary(omega_grid, k_grid, c_grid, k, c, output_path):
    """写 UTF-8 中文汇总:典型 (k, c) 的频率响应特征 + (k, c) 网格统计。"""
    omega_n = natural_frequency(k)
    response_type = classify_response_type(k, c)
    magnitude_at_dc, _ = magnitude_phase(np.array([0.001]), k, c)
    magnitude_at_n, phase_at_n = magnitude_phase(np.array([omega_n]), k, c)
    magnitude_at_pi, _ = magnitude_phase(np.array([np.pi]), k, c)
    in_wedge = is_in_schur_wedge(k, c)
    lines = [
        '=' * 70,
        'v5 受迫系统 + G(ω) 频率响应 汇总',
        '=' * 70,
        f'给定 (k, c) = ({k}, {c})',
        f'  阻尼类型: {response_type}',
        f'  Schur 楔形内: {in_wedge}',
        f'  自然频率 ω_n = arctan(√k) = {omega_n:.4f}',
        f'  |H(j0)|  ≈ {float(magnitude_at_dc[0]):.4f} (DC 增益,应 ≈ 1)',
        f'  |H(jω_n)| = {float(magnitude_at_n[0]):.4f} (共振峰)',
        f'  arg H(jω_n) = {float(np.degrees(phase_at_n[0])):.2f}°',
        f'  |H(jπ)|   = {float(magnitude_at_pi[0]):.4f} (Nyquist)',
        '',
        f'频率扫描: ω ∈ [{omega_grid[0]:.4f}, {omega_grid[-1]:.4f}], 共 {len(omega_grid)} 点',
        f'(k, c) 网格扫描: k ∈ [{k_grid[0]:.2f}, {k_grid[-1]:.2f}], '
        f'c ∈ [{c_grid[0]:.2f}, {c_grid[-1]:.2f}], 共 {len(k_grid)}×{len(c_grid)} = {len(k_grid)*len(c_grid)} 点',
        '',
    ]
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
```

- [ ] **Step 7: 单元测试 1 + 2:`test_transfer_function_dc_gain` + `test_transfer_function_stable_rolloff`**

放在 `tests/test_dynamics_eigen.py` 末尾:

```python
def test_transfer_function_dc_gain():
    """DC gain 验证:H(jω→0) ≈ 1.0(任意 k, c,k>0)。"""
    pytest.importorskip("backtrace.dynamics.dynamics_forced_response")
    from backtrace.dynamics.dynamics_forced_response import transfer_function
    for k, c in [(0.5, 0.5), (2.0, 1.5), (4.0, 4.0), (5.0, 0.1)]:
        H_dc = transfer_function(np.array([0.001]), k, c)
        assert abs(abs(H_dc[0]) - 1.0) < 1e-3, (
            f'DC gain failed for (k={k}, c={c}): |H(j0)|={abs(H_dc[0]):.4f}'
        )


def test_transfer_function_stable_rolloff():
    """Schur 内 (k=2, c=4):高频滚降 |H(jπ)| < |H(j0.1)|。"""
    pytest.importorskip("backtrace.dynamics.dynamics_forced_response")
    from backtrace.dynamics.dynamics_forced_response import magnitude_phase
    omega = np.array([0.1, np.pi])
    mag, _ = magnitude_phase(omega, k=2.0, c=4.0)
    assert mag[1] < mag[0], (
        f'expected high-freq rolloff, but |H(jπ)|={mag[1]:.4f} >= |H(j0.1)|={mag[0]:.4f}'
    )
    assert mag[1] < 0.5, f'|H(jπ)|={mag[1]:.4f} 应 < 0.5(强阻尼)'


def test_transfer_function_resonance_peak():
    """欠阻尼 (k=4, c=0.5):|H(jω)| 在 ω ≈ arctan(2) ≈ 1.107 处有局部峰值。"""
    pytest.importorskip("backtrace.dynamics.dynamics_forced_response")
    from backtrace.dynamics.dynamics_forced_response import (
        magnitude_phase, natural_frequency, classify_response_type,
    )
    assert classify_response_type(4.0, 0.5) == 'underdamped'
    omega_n = natural_frequency(4.0)
    assert abs(omega_n - np.arctan(2.0)) < 1e-6
    omega_grid = np.linspace(0.01, np.pi, 1000)
    mag, _ = magnitude_phase(omega_grid, k=4.0, c=0.5)
    # 找峰值
    peak_idx = np.argmax(mag)
    peak_omega = omega_grid[peak_idx]
    peak_mag = mag[peak_idx]
    # 峰值应在 ω_n 附近(±0.3 弧度)
    assert abs(peak_omega - omega_n) < 0.3, (
        f'peak at ω={peak_omega:.4f}, expected near ω_n={omega_n:.4f}'
    )
    assert peak_mag > 1.0, f'欠阻尼应有 peak > 1, got {peak_mag:.4f}'


def test_transfer_function_unstable_blowup():
    """严重欠阻尼 (k=4, c=0.05):|H(jω_n)| > 5(共振爆炸)。"""
    pytest.importorskip("backtrace.dynamics.dynamics_forced_response")
    from backtrace.dynamics.dynamics_forced_response import (
        magnitude_phase, natural_frequency,
    )
    omega_n = natural_frequency(4.0)
    mag, _ = magnitude_phase(np.array([omega_n]), k=4.0, c=0.05)
    assert mag[0] > 5.0, f'严重欠阻尼应 |H(jω_n)| > 5, got {mag[0]:.4f}'


def test_classify_response_type():
    """4 种阻尼类型判定。"""
    pytest.importorskip("backtrace.dynamics.dynamics_forced_response")
    from backtrace.dynamics.dynamics_forced_response import classify_response_type
    assert classify_response_type(2.0, 4.0) == 'overdamped'   # c²=16 > 4k=8
    assert classify_response_type(4.0, 4.0) == 'critical'     # c²=16 = 4k=16
    assert classify_response_type(4.0, 0.5) == 'underdamped'  # c²=0.25 < 4k=16
    assert classify_response_type(-1.0, 0.5) == 'anti_damped'  # k<0
```

- [ ] **Step 8: 实现 `main()`(~50 行,端到端)**

```python
def main():
    args = parse_args()
    omega_grid = DEFAULT_OMEGA_GRID
    k_grid = DEFAULT_K_GRID
    c_grid = DEFAULT_C_GRID
    # 1. 频率扫描(给定 k, c)
    magnitude, phase_rad = magnitude_phase(omega_grid, args.k, args.c)
    omega_n = natural_frequency(args.k)
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
            omega_n_val = natural_frequency(k_val) if k_val > 0 else np.nan
            H_val = transfer_function(omega_n_val, k_val, c_val) if k_val > 0 else np.nan
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
```

**关键点**:
- `encoding='utf-8-sig'` 沿用 v4.8/v4.9/v4.10 CSV 惯例(Excel 友好)
- 网格扫描 naive loop(60×60=3600 次 `transfer_function` 调用,< 1 秒)

- [ ] **Step 9: 跑全部 53 个测试**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest \
    tests/test_dynamics_eigen.py -v
```

Expected:
- `test_transfer_function_dc_gain` PASS
- `test_transfer_function_stable_rolloff` PASS
- `test_transfer_function_resonance_peak` PASS
- `test_transfer_function_unstable_blowup` PASS
- `test_classify_response_type` PASS
- 总共 53/53 PASS (48 v4.10 + 5 v5)

- [ ] **Step 10: 端到端 CLI 测试**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_forced_response.py
# 默认: k=2.0, c=1.5 (Schur 外)
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_forced_response.py --k 2.0 --c 4.0
# Schur 内:无共振峰
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_forced_response.py --k 4.0 --c 0.5
# 欠阻尼:共振峰
```

Expected:
- 5 个新文件存在
- `transfer_function_grid.csv` 200 行
- `transfer_function_stability.csv` 3600 行(60×60)
- Bode 图与热图正常渲染

- [ ] **Step 11: 更新 `backtrace/dynamics/README.md` §4(~40 行)**

在 §3.10 (v4.10)之后追加:

```markdown
### §4 v5 — 受迫系统 + G(ω) 频率响应

v4.7-v4.10 把 SI 当成被动的"稳定性指标"。v5 扩展到受迫:用 sinusoidal β(t) 主动驱动,测系统的复频响应 H(jω),把 SI 与频域行为耦合。

**核心公式**(离散 z 域):
\`\`\`
H(jω) = [k + c·(z-1)] / [(z-1)² + k]    其中 z = e^(jω)
\`\`\`

**关键性质**:
- **DC gain**:H(j0) = 1(任意 k, c > 0)
- **共振**:Schur 楔形外(c² < 4k)→ |H| 在 ω_n = arctan(√k) 处爆炸
- **滚降**:Schur 楔形内(c² > 4k)→ |H| 单调滚降,无峰值
- **抗阻尼**:k < 0 → 低频 |H| 爆炸,系统无界

**输出**:
- `data/dynamics/transfer_function_grid.csv` — ω × (|H|, arg H) 200 点
- `data/dynamics/transfer_function_stability.csv` — 60×60 (k, c) 网格
- `backtrace/outputs/dynsys_forced_response.html` — Bode 图 2 子图
  - (1,1) |H(jω)| 半对数 + ω_n 红虚线 + |H|=1 灰虚线
  - (1,2) arg H(jω) 度数 + -180° 红虚线
- `backtrace/outputs/dynsys_forced_response_stability.html` — 2D 热图
  - (1,1) |H(jω_n)| 颜色 + Schur 边界 c = 2√k 黑虚线
- `backtrace/outputs/dynsys_forced_response_summary.txt` — UTF-8 中文汇总

**CLI**:
\`\`\`bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_forced_response.py
# 默认: k=2.0, c=1.5 (Schur 外,有共振)
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_forced_response.py --k 2.0 --c 4.0
# Schur 内:稳定,无共振
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_forced_response.py --k 4.0 --c 0.5
# 欠阻尼:共振爆炸
\`\`\`

**与现有代码的关系**:
- 推导基于 `_dynamics_core.predict_next_state` (v3) 但**不调用**
- 与 `dynamics_1step_oos.py` 互补:时域 OOS vs 频域解析
- 与 v4.7 SI 计算互补:SI 给出 (k̂, ĉ) → 本 spec 给出该点的频率响应

**已知陷阱**:
- ω=0 时 `e^(jω)-1 = 0`,但分母 = k ≠ 0(无 DC 奇异);omega_grid 从 0.001 起避免数值噪声
- |H| 爆炸(超 1e10)时 log-scale 显示,数值上 `np.clip(..., 1e-12, None)`
- 离散 Bode ≠ 连续 Bode — Nyquist 在 ω=π(不是 ∞)
- 5 测试用合成 (k, c),非真实数据 — 真实 SI 频率响应是 v5.3 候选

### 与 v4.7-v4.10 的关系

| 版 | commit | 主题 | 数学层 |
|---|---|---|---|
| v4.7 | `c63e783` | SI 单一指标(描述性) | Schur 楔形 |
| v4.8 | `dbd367d` | SI × forward return IC ≈ 0(SI 不是预测性) | Spearman IC |
| v4.9 | `f2178a3` | SI 时序 + 漂移检测 | 时序 + z-score |
| v4.10 | `d002a0e` | 时序 SI 的 lagged IC 评估 | Lagged Spearman IC |
| **v5** | **(本版)** | **受迫系统 + G(ω) 频率响应** | **z 域 H(jω)** |
```

- [ ] **Step 12: 提交(2 commits,先代码后文档)**

Commit 1 (代码 + 测试):
```bash
git add backtrace/dynamics/dynamics_forced_response.py tests/test_dynamics_eigen.py
git commit -m "feat(dynamics): v5 — 受迫系统 + G(ω) 频率响应

新 CLI dynamics_forced_response.py(~400 行,独立):
- transfer_function 解析公式 H(jω) = [k+c(z-1)] / [(z-1)²+k], z=e^(jω)
- magnitude_phase + natural_frequency + classify_response_type(4 种阻尼)
- bode_plot 2 子图(幅值 log + 相位度数)
- stability_heatmap 2D (k, c) 热图 + Schur 边界 c=2√k
- 5 个新单元测试(53 total): DC gain / 滚降 / 共振峰 / 爆炸 / 阻尼分类

约束兑现:_dynamics_core.py / 3 caller / dynamics_si_ic.py(v4.8) /
dynamics_si_timeseries.py(v4.9) / dynamics_si_lagged_ic.py(v4.10) /
compute_sector_stability_timeseries(v4.9) / v4.7 compute_sector_stability 0 行修改。"
```

Commit 2 (文档):
```bash
git add backtrace/dynamics/README.md
git commit -m "docs(dynamics): README §4 v5 受迫系统 + G(ω)"
```

---

## Self-Review(Plan v1)

1. **Spec 覆盖**:
   - [x] spec §3.1-3.4 数学推导 → Step 2 `transfer_function` / `natural_frequency` / `classify_response_type`
   - [x] spec §4.1 频率扫描 200 点 → Step 8 `DEFAULT_OMEGA_GRID = np.linspace(0.001, π, 200)`
   - [x] spec §4.2 (k, c) 60×60 网格 → Step 8 `DEFAULT_K_GRID` / `DEFAULT_C_GRID` (60 点)
   - [x] spec §5.2 5 输出文件 → Step 8 main() 5 个写出
   - [x] spec §6.1-6.2 HTML 布局 → Step 4 Bode + Step 5 热图
   - [x] spec §7 5 测试 → Steps 7(2 测试) + 隐含 Step 7 剩余 3 测试
   - [x] spec §8 8 protected files 0 行修改 → Step 11 + Step 12 commit message 注明
   - [x] spec §11 验证清单 7 项 → Step 9 pytest + Step 10 E2E

2. **Placeholder scan**: 无 TBD / TODO。

3. **类型一致性**:
   - `transfer_function(omega, k, c)` → 复数 ndarray(Steps 2 + 3 都接受此签名)✓
   - `magnitude_phase(omega_array, k, c)` → (magnitude, phase_rad) ✓
   - `natural_frequency(k)` → float ✓
   - `classify_response_type(k, c)` → str ✓
   - `is_in_schur_wedge(k, c, tol)` → bool(与 v4.7 兼容)✓

4. **潜在风险**:
   - **DC gain 测试**:Step 7 用 `np.array([0.001])` 避免 ω=0 数值奇异,所有 (k, c) 都应 ≈ 1;测试已验证多组值
   - **共振峰测试**:Step 7 找 `argmax(mag)`,断言 `peak_omega` 在 `ω_n ± 0.3` 内(留 0.3 弧度容差)
   - **网格扫描性能**:60×60 = 3600 次 `transfer_function`,每次数组操作 → < 1 秒;无需优化
   - **HTML 渲染**:Bode 1×2 + 热图 1×1 plotly,沿用 v4.8/v4.9/v4.10 模式

5. **修复记录**(inline):
   - Step 4 `log_mag = np.log10(np.clip(magnitude, 1e-12, None))` 防止 log(0) → -∞
   - Step 5 `H_at_resonance[i, j] = np.nan` 在 k≤0 时跳过
   - Step 6 `magnitude_at_dc` 用 0.001 而非 0.0(数值稳定)
   - Step 8 CSV `encoding='utf-8-sig'` 沿用 v4.8/v4.9/v4.10
   - Step 12 commit message 明确列出 8 protected files(防止回归)

---

## 执行提示

**SDD 推荐**(沿用 v4.7/v4.8/v4.9/v4.10 模式):
- 1 implementer subagent 跑 Task 1 → task review → fix rounds → push
- 1 final code reviewer 整 branch 扫一遍 → push

**关键文件**:
- `backtrace/dynamics/dynamics_forced_response.py` (Task 1 new)
- `tests/test_dynamics_eigen.py` (Task 1 末尾追加 5 测试)
- `backtrace/dynamics/README.md` (Task 1 Step 11)

**预期产出**:
- 2 commits total
- 53 tests pass
- 0 行修改:8 protected files(`_dynamics_core.py` / 3 caller / 4 v4.7-v4.10)

**模型选择**:
- Implementer:haiku(机械转录,plan 含完整代码)
- Task reviewer:sonnet(常规代码审查)
- Final reviewer:opus(数学正确性 + 边界 case,v4.10 教训)