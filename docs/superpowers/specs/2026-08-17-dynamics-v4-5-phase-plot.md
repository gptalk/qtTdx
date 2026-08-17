# v4.5 — (k, c) phase plot + 11 classification colors

**Date:** 2026-08-17
**Parent plan:** v4.3 全市场经验分布 + v4.4 (1,4) bar chart label 增强
**Status:** Draft

## Background

v4.3 在 `dynamics_eigen_analysis.py` 2×4 HTML 里把"全市场 (k̂, ĉ) 经验分布"画了出来,但 (k, c) 散点只在 (2,1) 子图以 **连续 wedge distance 着色** 呈现。**离散的 11 类稳定性分类**(CLASS_COLORS) 没有自己的散点图。

数据上,每只票都属于 11 类中的 1 类:`stable_oscillatory` / `stable_overdamped` / `stable_critical_damping` / `oscillatory_divergent` / `monotonic_divergent` / `anti_restoring` / `critical_periodic` / `critical_period2` / `critical_real_unit` / `marginal_const` / `jordan_drift`。看 11 类在 (k, c) 平面的分布 = 看"股票市场动力学的拓扑结构"。

## Goal

**新增独立 HTML 文件 `dynsys_eigen_phase.html`,画 (k̂, ĉ) 散点 + 11 类离散着色 + 楔形稳定区边界 overlay。**

## Scope

**In scope:**
- `dynamics_eigen_analysis.py` 加 `build_phase_plot_html(summary_df, output_path)` 函数
- 加 `--phase-plot` CLI flag(默认 off,保持现有 2x4 行为不变)
- 楔形边界 3 段线段 + 11 类颜色 + 1 类 1 个图例条目
- 2 个新单元测试
- README + spec 注脚

**Out of scope:**
- 密度等高线(YAGNI)
- 交互式筛选器(下钻某类)
- 行业 / 交易所专用 phase plot(后续 v4.6+ 再说)
- 替换 (2,1) 子图(不动 v4.3 已推的 2x4)

## Design

### 1. 新函数 `wedge_boundary_polygon(k_max=4.0, n=100)`

```python
def wedge_boundary_polygon(k_max: float = 4.0, n: int = 100) -> dict:
    """楔形稳定区边界 3 段折线。
    
    Returns:
        dict with keys: 'k_axis', 'c_axis', 'upper_curve', 'k_max'
        - k_axis: list[(k, c)]   — c = 0, k ∈ [0, k_max]
        - c_axis: list[(k, c)]   — k = 0, c ∈ [0, 2]
        - upper_curve: list[(k, c)] — c = 2*sqrt(k+1), k ∈ [0, k_max]
    """
    k_axis = [(k, 0.0) for k in np.linspace(0, k_max, n)]
    c_axis = [(0.0, c) for c in np.linspace(0, 2.0, n)]
    upper_curve = [(k, 2.0 * np.sqrt(k + 1.0)) for k in np.linspace(0, k_max, n)]
    return {'k_axis': k_axis, 'c_axis': c_axis, 'upper_curve': upper_curve, 'k_max': k_max}
```

**注意:** 楔形是 A 矩阵 Schur 稳定区的边界(c² = 4(k+1) 抛物线 + c ≥ 0, k ≥ 0)。`analyze_eigenvalues` 已实现 `distance_to_wedge`,这个 helper 只画边界曲线。

### 2. 新函数 `build_phase_plot_html(summary_df, output_path)`

```python
def build_phase_plot_html(summary_df: pd.DataFrame, output_path: str) -> None:
    """画 (k̂, ĉ) 散点 + 11 类颜色 + 楔形边界 overlay。"""
    fig = go.Figure()
    
    # 楔形稳定区填充(浅绿)
    boundary = wedge_boundary_polygon(k_max=summary_df['k_hat'].quantile(0.99))
    fill_k = [k for k, c in boundary['upper_curve']] + [k for k, c in boundary['k_axis']][::-1]
    fill_c = [c for k, c in boundary['upper_curve']] + [c for k, c in boundary['k_axis']][::-1]
    fig.add_trace(go.Scatter(
        x=fill_k, y=fill_c, fill='toself', fillcolor='rgba(44, 160, 44, 0.08)',
        line=dict(color='rgba(0,0,0,0)'), name='楔形稳定区', showlegend=True, hoverinfo='skip',
    ))
    
    # 楔形边界 3 段线
    for label, pts in [('c=0', boundary['k_axis']), 
                       ('k=0', boundary['c_axis']),
                       ('c=2√(k+1)', boundary['upper_curve'])]:
        fig.add_trace(go.Scatter(
            x=[k for k, c in pts], y=[c for k, c in pts],
            mode='lines', line=dict(color='black', width=1.5, dash='dash'),
            name=label, showlegend=False, hoverinfo='skip',
        ))
    
    # 11 类散点(每类 1 trace,图例 1 entry)
    for cls in CLASS_COLORS:
        sub = summary_df[summary_df['classification'] == cls]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub['k_hat'], y=sub['c_hat'],
            mode='markers',
            marker=dict(color=CLASS_COLORS[cls], size=6, opacity=0.7, line=dict(width=0)),
            name=f'{CLASS_LABEL_CN[cls]} ({len(sub)})',
            hovertemplate=f'<b>{cls}</b><br>k̂=%{{x:.4f}}<br>ĉ=%{{y:.4f}}<extra></extra>',
            showlegend=True,
        ))
    
    fig.update_layout(
        title='全市场 (k̂, ĉ) 11 类稳定性分类 phase plot',
        xaxis_title='k̂ (回复力强度)',
        yaxis_title='ĉ (阻尼系数)',
        width=1100, height=750,
        legend=dict(title='11 类分类', x=1.02, y=1, bgcolor='rgba(255,255,255,0.9)'),
        template='plotly_white',
    )
    fig.write_html(output_path, include_plotlyjs='cdn')
```

**关键点:**
- 楔形稳定区填充透明度 0.08(背景,不抢眼)
- 楔形边界 3 段虚线(`dash='dash'`)
- 11 类每类 1 trace,图例显示中文 + 计数
- 楔形 k_max 用 99 分位数(覆盖 99% 票,避免超长 k_max 把图压扁)
- 如果某类 0 只票,跳过

### 3. CLI flag `--phase-plot`

```python
# parse_args() 加:
p.add_argument('--phase-plot', action='store_true', help='画 (k,c) 11 类 phase plot 到独立 HTML')

# main() 末尾:
if args.phase_plot:
    phase_path = DEFAULT_OUTPUT_HTML.replace('dynsys_eigen.html', 'dynsys_eigen_phase.html')
    build_phase_plot_html(summary_df, phase_path)
    print(f'[eigen] ✓ phase plot: {phase_path}')
```

**默认 off** — `phase plot` 是可选附加,不破坏现有 2x4 行为。

### 4. 测试

`tests/test_dynamics_eigen.py` 加 2 个测试:

```python
def test_wedge_boundary_polygon():
    """楔形边界 3 段:左 c=0 / 底 c=0 / 上 c=2√(k+1)"""
    boundary = wedge_boundary_polygon(k_max=4.0, n=50)
    # 上边界:起点 (k=0, c=2),终点 (k=4, c=2√5 ≈ 4.47)
    assert boundary['upper_curve'][0] == (0.0, 2.0)
    assert abs(boundary['upper_curve'][-1][0] - 4.0) < 1e-9
    assert abs(boundary['upper_curve'][-1][1] - 2.0 * np.sqrt(5.0)) < 1e-9
    # k 轴:起点 (0, 0),终点 (4, 0)
    assert boundary['k_axis'][0] == (0.0, 0.0)
    assert boundary['k_axis'][-1] == (4.0, 0.0)
    # c 轴:起点 (0, 0),终点 (0, 2)
    assert boundary['c_axis'][0] == (0.0, 0.0)
    assert boundary['c_axis'][-1] == (0.0, 2.0)


def test_phase_plot_html_smoke(tmp_path):
    """build_phase_plot_html 写文件成功 + HTML 包含 11 类 marker。"""
    # mock 11 类样本,每类 5 只
    rng = np.random.default_rng(42)
    rows = []
    for cls in CLASS_COLORS:
        for _ in range(5):
            rows.append({
                'code': f'{rng.integers(0, 999999):06d}.SH',
                'k_hat': rng.uniform(0, 4),
                'c_hat': rng.uniform(0, 4),
                'classification': cls,
            })
    df = pd.DataFrame(rows)
    out = tmp_path / 'phase.html'
    build_phase_plot_html(df, str(out))
    assert out.exists()
    content = out.read_text(encoding='utf-8')
    assert 'scatter' in content.lower()  # plotly HTML
    for cls in CLASS_COLORS:
        assert cls in content  # 11 类都在 HTML 里
```

**已有 26 + v4.4 + 1 = 27 旧测试,加 2 = 29 total。**

### 5. 文档

**README §3.5 末尾追加 `### 3.6 v4.5 phase plot (2026-08-17)`:**
```markdown
### 3.6 v4.5 phase plot (2026-08-17)

新增独立 HTML `dynsys_eigen_phase.html`,画 (k̂, ĉ) 散点 + 11 类离散着色 + 楔形稳定区边界 overlay。
启用:`--phase-plot` 标志(默认 off)。

11 类: stable_oscillatory / stable_overdamped / stable_critical_damping / 
oscillatory_divergent / monotonic_divergent / anti_restoring / 
critical_periodic / critical_period2 / critical_real_unit / 
marginal_const / jordan_drift(CLASS_COLORS 字典定义)
```

**spec §3.7 末尾追加 `### 3.8 v4.5 phase plot (2026-08-17)`:**
```markdown
### 3.8 v4.5 phase plot (2026-08-17)— (k,c) 11 类颜色叠加

新增独立 HTML `dynsys_eigen_phase.html`,画 (k̂, ĉ) 散点 + 11 类离散着色 + 楔形稳定区边界 overlay。
启用: `--phase-plot` 标志(默认 off)。
2 新测试 + 1 文件改动,数学层 / 3 caller 零修改。
```

## Constraints (carryover)

- **数学层 `_dynamics_core.py` 0 行修改**
- **3 caller (`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`) 0 行修改**
- `analyze_eigenvalues` / `simulate_trajectory` 函数签名不变
- 输出全部 gitignored(`data/` + `backtrace/outputs/`)

## Files

| 文件 | 改动 |
|---|---|
| `backtrace/dynamics/dynamics_eigen_analysis.py` | +105 / -2 行(`wedge_boundary_polygon` + `build_phase_plot_html` + `--phase-plot` flag) |
| `tests/test_dynamics_eigen.py` | +34 / 0 行(2 tests) |
| `backtrace/dynamics/README.md` | +10 / 0 行(§3.6 v4.5 节) |
| `docs/superpowers/specs/2026-08-16-dynamics-system-design.md` | +6 / 0 行(§3.8 v4.5 注脚) |

## Verification

```bash
# 1. 单元测试
cd c:/Users/yellow/mcp/qtTdx
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v
# → 29 passed (27 旧 + 2 新)

# 2. 端到端冒烟(enable phase plot)
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_eigen_analysis.py --limit 50 --phase-plot
# → 5 旧 outputs + 1 新 dynsys_eigen_phase.html,exit 0

# 3. 默认 off 验证(不传 --phase-plot 不产出 phase HTML)
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_eigen_analysis.py --limit 50
# → 5 旧 outputs,无 dynsys_eigen_phase.html,exit 0

# 4. 浏览器打开检查(肉眼可见)
# backtrace/outputs/dynsys_eigen_phase.html
# 期待: (k̂, ĉ) 散点 + 楔形绿色填充 + 3 段虚线边界 + 11 类颜色图例
```

## Risk

| 风险 | 缓解 |
|---|---|
| 楔形 k_max 选 99 分位数,有 1% 票落在外面 | 视觉上不重要(plot 区外自动裁剪);可加 hover 文本说明 |
| 11 类中 `jordan_drift` / `marginal_const` 在 N=4972 里 0 只票 | 跳过该 trace(`if len(sub) == 0: continue`),不报错 |
| HTML 大小增加(~2-3MB,但 outputs gitignored) | 接受,跟 v4.3 5.8MB 一致 |
| `--phase-plot` 必须与现有 2x4 兼容(默认 off) | `action='store_true'`,默认 False |

## 与 v4.3 / v4.4 的关系

| 版 | 焦点 | 状态 |
|---|---|---|
| v4.3 | 2x4 HTML + 文本汇总 + 行业 / 交易所聚合 | ✅ done |
| v4.4 | (1,4) bar chart label 增强(共享 lookup) | ✅ done |
| **v4.5** | **(k,c) phase plot + 11 类颜色 overlay** | **本文** |
| v4.6+ | 行业 / 交易所专用 phase plot,密度等高线... | future |
| v6 | 受迫系统 + G(ω) 频率响应 | future |
