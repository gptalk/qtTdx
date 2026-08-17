# v4.5 — (k, c) phase plot + 11 classification colors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增独立 HTML `dynsys_eigen_phase.html`,画 (k̂, ĉ) 散点 + 11 类离散着色 + 楔形稳定区边界 overlay。

**Architecture:** `dynamics_eigen_analysis.py` 加 2 函数(`wedge_boundary_polygon` + `build_phase_plot_html`)+ 1 CLI flag(`--phase-plot`,默认 off,向后兼容)。复用 `CLASS_COLORS` / `CLASS_LABEL_CN` 字典。`build_phase_plot_html` 写独立 HTML,不修改 v4.3 2x4 输出。

**Tech Stack:** Python 3.13 / pandas / numpy / plotly / pytest / tsfresh 全栈环境(`/c/ProgramData/anaconda3/python.exe`)

## Global Constraints

- 数学层 `_dynamics_core.py` 0 行修改
- 3 caller(`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`)0 行修改
- `analyze_eigenvalues` / `simulate_trajectory` 函数签名不变
- 输出全部 gitignored(`data/` + `backtrace/outputs/`)
- `--phase-plot` 默认 off,保持 v4.3 2x4 行为完全不变
- `PYTHONIOENCODING=utf-8` 必备;Windows 用 `/c/ProgramData/anaconda3/python.exe`(python 不在 PATH)
- 29 tests pass 目标(27 旧 + 2 新)

## 现状(实施前必读)

- `backtrace/dynamics/dynamics_eigen_analysis.py:46-72` — `CLASS_COLORS` 11 类配色 + `CLASS_LABEL_CN` 中文标签(已存在,直接复用)
- `backtrace/dynamics/dynamics_eigen_analysis.py:75-87` — `parse_args()`(加 `--phase-plot`)
- `backtrace/dynamics/dynamics_eigen_analysis.py:206-219` — `_industry_name_lookup` helper(v4.4 新增)
- `backtrace/dynamics/dynamics_eigen_analysis.py:221-302` — `write_text_summary`(v4.4 扩展过)
- `backtrace/dynamics/dynamics_eigen_analysis.py:304-657` — `main()`(在末尾 L656 后加 phase plot 调用)
- `backtrace/dynamics/dynamics_eigen_analysis.py:37-44` — constants(`DEFAULT_OUTPUT_HTML`)
- `backtrace/dynamics/dynamics_eigen_analysis.py:57-58` — `marginal_const` / `jordan_drift` 配色(实际 N=4972 这两类 0 只)
- `tests/test_dynamics_eigen.py:428` 行,目前 27 个测试函数
- `backtrace/dynamics/README.md` §3.5 是 v4.3 节(commit fbaff0f)
- `docs/superpowers/specs/2026-08-16-dynamics-system-design.md` §3.7 是 v4.4 子节(commit 180f01c)

---

## Task 1: 楔形边界 + phase plot + 2 测试 + 2 文档注脚

**Files:**
- Modify: `backtrace/dynamics/dynamics_eigen_analysis.py` (3 处:`+wedge_boundary_polygon`, `+build_phase_plot_html`, `+--phase-plot` flag in parse_args, `+main()` call)
- Modify: `tests/test_dynamics_eigen.py` (新增 `test_wedge_boundary_polygon` + `test_phase_plot_html_smoke`)
- Modify: `backtrace/dynamics/README.md` (§3.5 末尾追加 §3.6 v4.5 节)
- Modify: `docs/superpowers/specs/2026-08-16-dynamics-system-design.md` (§3.7 末尾追加 §3.8 v4.5 注脚)

**Interfaces:**
- 新增: `wedge_boundary_polygon(k_max: float = 4.0, n: int = 100) -> dict`
  - 返回 dict with keys: `k_axis`, `c_axis`, `upper_curve`, `k_max`
  - k_axis: list[(k, c)] — c = 0, k ∈ [0, k_max]
  - c_axis: list[(k, c)] — k = 0, c ∈ [0, 2]
  - upper_curve: list[(k, c)] — c = 2√(k+1), k ∈ [0, k_max]
- 新增: `build_phase_plot_html(summary_df: pd.DataFrame, output_path: str) -> None`
  - 写 1 个 HTML 文件: (k̂, ĉ) 散点 + 楔形稳定区填充 + 3 段边界虚线 + 11 类颜色 + 11 类图例
- CLI: `--phase-plot` action='store_true', 默认 False

### Step 1.1: 给 `wedge_boundary_polygon` 写失败测试

打开 `tests/test_dynamics_eigen.py`,在文件末尾(第 27 个测试函数之后,3 个空行)追加:

```python
def test_wedge_boundary_polygon():
    """楔形边界 3 段:左 c=0 / 底 c=0 / 上 c=2√(k+1)"""
    from dynamics.dynamics_eigen_analysis import wedge_boundary_polygon

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

    # 长度: n 个点(由 n=50 参数)
    assert len(boundary['k_axis']) == 50
    assert len(boundary['c_axis']) == 50
    assert len(boundary['upper_curve']) == 50

    # k_max 字段
    assert boundary['k_max'] == 4.0
```

### Step 1.2: 跑测试,确认失败

```bash
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_wedge_boundary_polygon -v
```

**Expected:** `ImportError: cannot import name 'wedge_boundary_polygon' from 'dynamics.dynamics_eigen_analysis'`

### Step 1.3: 实现 `wedge_boundary_polygon`

打开 `backtrace/dynamics/dynamics_eigen_analysis.py`,在 `write_text_summary` 函数后(约 L302,`def main()` 之前,留 2 空行)插入:

```python
def wedge_boundary_polygon(k_max: float = 4.0, n: int = 100) -> dict:
    """楔形稳定区边界 3 段折线。

    Schur 稳定区定义: c² ≤ 4(k+1) AND c ≥ 0 AND k ≥ 0
    边界曲线:
      - k 轴: c = 0, k ∈ [0, k_max]
      - c 轴: k = 0, c ∈ [0, 2]
      - 上抛物线: c = 2√(k+1), k ∈ [0, k_max]

    Returns:
        dict with keys: 'k_axis', 'c_axis', 'upper_curve', 'k_max'
        每段都是 list[(k, c)] 长度 n。
    """
    k_axis = [(k, 0.0) for k in np.linspace(0, k_max, n)]
    c_axis = [(0.0, c) for c in np.linspace(0, 2.0, n)]
    upper_curve = [(k, 2.0 * np.sqrt(k + 1.0)) for k in np.linspace(0, k_max, n)]
    return {'k_axis': k_axis, 'c_axis': c_axis, 'upper_curve': upper_curve, 'k_max': k_max}
```

### Step 1.4: 跑测试,确认通过

```bash
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_wedge_boundary_polygon -v
```

**Expected:** PASS

### Step 1.5: 给 `build_phase_plot_html` 写失败测试

打开 `tests/test_dynamics_eigen.py`,在 `test_wedge_boundary_polygon` 之后追加:

```python
def test_phase_plot_html_smoke(tmp_path):
    """build_phase_plot_html 写文件成功 + HTML 包含 11 类 marker。"""
    from dynamics.dynamics_eigen_analysis import build_phase_plot_html

    # mock 11 类样本,每类 5 只票
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
    # 11 类名称都出现在 HTML 里(CLASS_LABEL_CN 中文标签)
    for cn in CLASS_LABEL_CN.values():
        assert cn in content
```

**注意:** `CLASS_COLORS` / `CLASS_LABEL_CN` 已在文件顶部 import 范围(via `from dynamics import dynamics_eigen_analysis as EA`),需要在测试里直接 import 或者用 EA 引出:

```python
# 在测试内部:
from dynamics.dynamics_eigen_analysis import CLASS_COLORS, CLASS_LABEL_CN, build_phase_plot_html
```

### Step 1.6: 跑测试,确认失败

```bash
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_phase_plot_html_smoke -v
```

**Expected:** `ImportError: cannot import name 'build_phase_plot_html' from 'dynamics.dynamics_eigen_analysis'`

### Step 1.7: 实现 `build_phase_plot_html`

打开 `backtrace/dynamics/dynamics_eigen_analysis.py`,在 `wedge_boundary_polygon` 之后(留 2 空行)插入:

```python
def build_phase_plot_html(summary_df: pd.DataFrame, output_path: str) -> None:
    """画 (k̂, ĉ) 散点 + 11 类颜色 + 楔形稳定区边界 overlay。

    独立 HTML,不动 v4.3 2x4 输出。被 main() 通过 --phase-plot flag 调用。
    """
    fig = go.Figure()

    # 楔形稳定区填充(浅绿背景)
    k_max = summary_df['k_hat'].quantile(0.99)
    boundary = wedge_boundary_polygon(k_max=k_max)
    fill_k = [k for k, c in boundary['upper_curve']] + [k for k, c in boundary['k_axis']][::-1]
    fill_c = [c for k, c in boundary['upper_curve']] + [c for k, c in boundary['k_axis']][::-1]
    fig.add_trace(go.Scatter(
        x=fill_k, y=fill_c, fill='toself', fillcolor='rgba(44, 160, 44, 0.08)',
        line=dict(color='rgba(0,0,0,0)'), name='楔形稳定区', showlegend=True, hoverinfo='skip',
    ))

    # 楔形边界 3 段虚线
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

### Step 1.8: 跑测试,确认通过

```bash
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_phase_plot_html_smoke -v
```

**Expected:** PASS

### Step 1.9: 加 `--phase-plot` CLI flag + main() 调用

打开 `backtrace/dynamics/dynamics_eigen_analysis.py:75-87`(`parse_args()`),在 `p.add_argument('--sw2-members', ...)` 之后追加:

```python
    p.add_argument('--phase-plot', action='store_true', help='画 (k,c) 11 类 phase plot 到独立 HTML(默认 off)')
```

打开 `main()` 末尾(实际位置:在 `write_text_summary(...)` 调用结束于 L658,`if __name__ == '__main__':` 在 L661 之间),插入:

```python
    # ---------- 5. (可选) (k,c) phase plot ----------
    if args.phase_plot:
        phase_path = args.output.replace('dynsys_eigen.html', 'dynsys_eigen_phase.html')
        build_phase_plot_html(summary_df, phase_path)
        print(f'[eigen] ✓ phase plot: {phase_path}')
```

**位置确认:** L654-658 是 `write_text_summary` 调用,L659 空白,L660 空白,L661-662 是 `if __name__ == '__main__': main()`。插入点在 L659 之前。

### Step 1.10: 跑全部 29 测试,确认通过

```bash
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v
```

**Expected:** 29 passed (27 旧 + 2 新)

### Step 1.11: 端到端冒烟

```bash
# 1. 默认 off 验证
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_eigen_analysis.py --limit 50
# → 5 旧 outputs,无 dynsys_eigen_phase.html,exit 0

# 2. enable phase plot 验证
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_eigen_analysis.py --limit 50 --phase-plot
# → 5 旧 outputs + 1 新 dynsys_eigen_phase.html,exit 0
```

**Expected:** exit 0,phase HTML 大小 1-3MB,在浏览器打开可见 11 类散点 + 楔形填充。

### Step 1.12: 更新 README §3.6 v4.5 节

打开 `backtrace/dynamics/README.md`,在第 3.5 节末尾、下一节之前,追加:

```markdown
### 3.6 v4.5 phase plot (2026-08-17)

新增独立 HTML `dynsys_eigen_phase.html`,画 (k̂, ĉ) 散点 + 11 类离散着色 + 楔形稳定区边界 overlay(浅绿背景 + 3 段虚线)。
启用:`--phase-plot` 标志(默认 off,不影响 v4.3 2x4 行为)。

11 类分类:`CLASS_COLORS` 字典定义(11 种颜色 + 11 中文标签)。
楔形 boundary 由 `wedge_boundary_polygon(k_max, n)` helper 提供。

用法:
```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_eigen_analysis.py --limit 50 --phase-plot
# 产出 backtrace/outputs/dynsys_eigen_phase.html
```

输出示例(实证发现,N=4972):
- 大多数票落在 `monotonic_divergent` (橙) + `anti_restoring` (棕) + `stable_overdamped` (蓝) 三大区域
- `jordan_drift` / `marginal_const` 在 N=4972 样本中 0 只 → 该 trace 跳过(代码已处理)
```

### Step 1.13: 更新 spec §3.8 v4.5 注脚

打开 `docs/superpowers/specs/2026-08-16-dynamics-system-design.md`,在 §3.7 末尾(由 v4.4 commit 180f01c 加的)追加:

```markdown
### 3.8 v4.5 phase plot (2026-08-17)— (k,c) 11 类颜色叠加

新增独立 HTML `dynsys_eigen_phase.html`,画 (k̂, ĉ) 散点 + 11 类离散着色 + 楔形稳定区边界 overlay。
启用: `--phase-plot` 标志(默认 off)。
2 新测试 (`test_wedge_boundary_polygon` + `test_phase_plot_html_smoke`) + 1 文件改动,数学层 / 3 caller 零修改。
```

### Step 1.14: Commit

```bash
cd "C:\Users\yellow\mcp\qtTdx"
git add backtrace/dynamics/dynamics_eigen_analysis.py tests/test_dynamics_eigen.py backtrace/dynamics/README.md docs/superpowers/specs/2026-08-16-dynamics-system-design.md
git commit -m "feat(dynamics): v4.5 — (k,c) phase plot + 11 类颜色 overlay

- 新增 wedge_boundary_polygon(k_max, n) helper(3 段折线)
- 新增 build_phase_plot_html(summary_df, output_path)(独立 HTML)
- CLI flag --phase-plot (默认 off,完全向后兼容)
- 楔形稳定区:浅绿背景 + 3 段虚线边界
- 11 类散点 + 11 类中文图例(每类 1 trace)
- 2 新测试: test_wedge_boundary_polygon + test_phase_plot_html_smoke
- README §3.6 + spec §3.8 注脚
- 数学层 / 3 caller 零修改,29 tests pass"
```

---

## 显式不做

- ❌ 密度等高线(YAGNI)
- ❌ 交互式筛选器(下钻某类)
- ❌ 行业 / 交易所专用 phase plot(v4.6+ 再说)
- ❌ 替换 (2,1) 子图(不动 v4.3 已推的 2x4)
- ❌ v6 受迫系统 + G(ω) 频率响应

## 验证清单

- [ ] Step 1.1: 测试 wedge_boundary_polygon 写入
- [ ] Step 1.2: 测试失败 (ImportError)
- [ ] Step 1.3: helper 实现
- [ ] Step 1.4: 单测试通过
- [ ] Step 1.5: 测试 build_phase_plot_html 写入
- [ ] Step 1.6: 测试失败 (ImportError)
- [ ] Step 1.7: build_phase_plot_html 实现
- [ ] Step 1.8: 单测试通过
- [ ] Step 1.9: --phase-plot flag + main() 调用
- [ ] Step 1.10: 29 tests pass
- [ ] Step 1.11: 端到端冒烟 (--limit 50 --phase-plot 和默认两种)
- [ ] Step 1.12: README §3.6 v4.5 节
- [ ] Step 1.13: spec §3.8 v4.5 注脚
- [ ] Step 1.14: commit

## 风险

| 风险 | 缓解 |
|---|---|
| 楔形 k_max 选 99 分位数,有 1% 票落在外面 | 视觉上不重要(plot 区外自动裁剪);hover 文本显示具体值 |
| 11 类中 `jordan_drift` / `marginal_const` 在 N=4972 里 0 只票 | `if len(sub) == 0: continue` 跳过,代码已处理 |
| HTML 大小增加(~2-3MB) | outputs gitignored,接受 |
| `--phase-plot` 必须与现有 2x4 兼容 | `action='store_true'` 默认 False,验证过(Step 1.11.1) |
