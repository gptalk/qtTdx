# v5.4 Dual-Pane Bode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `build_animated_overlay_html` from single subplot (|H(jω)| dB) to dual subplot (|H(jω)| dB + ∠H(jω) deg), satisfying v5.3 spec narrative that initially promised two subplots.

**Architecture:** Single-function modification. `build_animated_overlay_html` replaces `go.Figure` with `go.Figure(make_subplots(2, 1, shared_xaxes=True))`, doubles per-frame trace count (magnitude + phase), keeps everything else (slider, Play/Pause, animation_frame) intact. All other 5 functions + parse_args + main unchanged.

**Tech Stack:** Python 3.10+, plotly>=3.0 (`plotly.subplots.make_subplots`), pandas, numpy, pytest.

## Global Constraints

These constraints bind every task. Any conflict with task text — task text is wrong.

### File protection (only 1 file has bounded modifications)

- ❌ `_dynamics_core.py` — 0 修改
- ❌ `dynamics_forced_response.py` — 0 修改
- ❌ `dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py` — 0 修改
- ❌ `dynamics_si_ic.py` / `dynamics_si_timeseries.py` / `dynamics_si_lagged_ic.py` / `dynamics_eigen_analysis.py` — 0 修改
- ❌ `parameter_fit.py` — 0 修改
- ✓ v5.4 是**单函数内部扩展**:`backtrace/dynamics/dynamics_si_freq_response.py::build_animated_overlay_html` 内部从单子图 → 双子图
- ✓ `tests/test_dynamics_eigen.py` 1 个 test 扩充(`test_cli_si_freq_response_mode`)
- ✓ `backtrace/dynamics/README.md` §4.1.2 加 1 行 v5.4 提示

### Signature protection

- v5.4 已有 5 函数签名 0 修改:`load_kc_time_series` / `aggregate_by_industry_per_date` / `select_top_n_per_date` / `write_animated_summary_txt` / `write_animated_pairs_csv`
- `build_animated_overlay_html` 签名 0 修改(参数 `pairs_per_date, omega_grid, output_path, title` 不变)
- `parse_args()` 0 修改
- `main()` 0 修改

### Test count

- **72 → 72 tests pass**(1 test 更新,总数不变)
- 旧测试**全部不动**(任何失败 → fix 而非删测试)

### Runtime

- `PYTHONIOENCODING=utf-8` 必备
- Python: `/c/ProgramData/anaconda3/python.exe`
- 测试命令: `cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v`

---

## Task 1: Modify `build_animated_overlay_html` (single → dual subplot) + extend test

**Files:**
- Modify: `backtrace/dynamics/dynamics_si_freq_response.py:build_animated_overlay_html` (replace function body)
- Modify: `tests/test_dynamics_eigen.py::test_cli_si_freq_response_mode` (extend 1 assertion)

**Interfaces:**
- Consumes: `build_animated_overlay_html` v5.3 signature (unchanged)
- Produces: HTML with **2 subplots** (magnitude + phase), shared x-axis, animated via `animation_frame`

### Step 1: Extend the CLI subprocess test assertions

Modify `tests/test_dynamics_eigen.py::test_cli_si_freq_response_mode`:

Replace the existing single-subplot assertion block:

```python
    # HTML 含 plotly animation_frame + frames + 3 个日期
    html_text = html_path.read_text(encoding='utf-8')
    assert 'plotly' in html_text.lower()
    assert 'animation_frame' in html_text or 'frames' in html_text
    assert '2024-09-30' in html_text and '2024-10-31' in html_text and '2024-11-30' in html_text
```

With:

```python
    # HTML 含 plotly animation_frame + frames + 3 个日期 + 双子图元素
    html_text = html_path.read_text(encoding='utf-8')
    assert 'plotly' in html_text.lower()
    assert 'animation_frame' in html_text or 'frames' in html_text
    assert '2024-09-30' in html_text and '2024-10-31' in html_text and '2024-11-30' in html_text
    # v5.4 dual-pane: at least 2 xaxis/yaxis pairs (subplot 1 + subplot 2)
    assert html_text.count('xaxis') >= 2, 'v5.4 dual-pane: 至少 2 个 xaxis'
    assert html_text.count('yaxis') >= 2, 'v5.4 dual-pane: 至少 2 个 yaxis'
    # phase subplot 关键词(中文或英文都行)
    assert any(kw in html_text for kw in ('∠H', 'phase', 'arg H', '相角')), 'v5.4: phase 子图存在'
```

Also update the size assertion:

```python
    assert html_path.exists() and html_path.stat().st_size > 1000
```

To:

```python
    assert html_path.exists() and html_path.stat().st_size > 2000  # v5.4 dual-pane 加倍
```

### Step 2: Run test to verify it fails

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_si_freq_response_mode -v
```

Expected: FAIL — `xaxis` count < 2 OR `yaxis` count < 2 OR phase keyword missing.

### Step 3: Replace `build_animated_overlay_html` function body

In `backtrace/dynamics/dynamics_si_freq_response.py`:

1. Add import at top of file (if not already present):
```python
from plotly.subplots import make_subplots
```

2. Replace the entire `build_animated_overlay_html` function body with:

```python
def build_animated_overlay_html(
    pairs_per_date: list,
    omega_grid: np.ndarray,
    output_path: str,
    title: str = 'Industry G(ω) Frequency Response — Time Series',
) -> None:
    """构建 plotly 动画 slider:每帧一个 asof_date,每帧 N × 2 条 industry Bode 曲线。

    v5.4 双子图:
        - 上子图 |H(jω)| dB vs ω
        - 下子图 ∠H(jω) degrees vs ω(共享 x 轴)

    Args:
        pairs_per_date: [(asof_date, k̂, ĉ, label), ...] from select_top_n_per_date
        omega_grid: 共享 ω 网格(np.ndarray,默认 linspace(0.001, π, 200))
        output_path: HTML 输出路径
        title: 图表标题

    Raises:
        ValueError: pairs_per_date 为空
    """
    if not pairs_per_date:
        raise ValueError('pairs_per_date 为空,无法构建动画')

    dates = sorted(set(p[0] for p in pairs_per_date))
    initial_date = dates[0]

    # Phase 1: build initial-figure traces using go.Figure (one row = one subplot)
    # We use make_subplots(2, 1, shared_xaxes=True) for the dual-pane layout.
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=('|H(jω)| dB', '∠H(jω) deg'),
        vertical_spacing=0.10,
    )

    def _magnitude_db(p):
        return magnitude_phase(omega_grid * 1j, p[1], p[2])[0].tolist()

    def _phase_deg(p):
        return magnitude_phase(omega_grid * 1j, p[1], p[2])[1].tolist()

    # Initial-state traces (first date)
    for p in (p_ for p_ in pairs_per_date if p_[0] == initial_date):
        fig.add_trace(go.Scatter(x=omega_grid.tolist(), y=_magnitude_db(p),
                                 mode='lines', name=p[3], legendgroup=p[3]),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=omega_grid.tolist(), y=_phase_deg(p),
                                 mode='lines', name=p[3], legendgroup=p[3],
                                 showlegend=False),
                      row=2, col=1)

    # Phase 2: build frames — one frame per date, each frame has 2 × N traces
    frames = []
    for date in dates:
        frame_traces = []
        for p in (p_ for p_ in pairs_per_date if p_[0] == date):
            frame_traces.append(go.Scatter(x=omega_grid.tolist(), y=_magnitude_db(p),
                                           mode='lines', name=p[3], legendgroup=p[3]))
            frame_traces.append(go.Scatter(x=omega_grid.tolist(), y=_phase_deg(p),
                                           mode='lines', name=p[3], legendgroup=p[3],
                                           showlegend=False))
        frames.append(go.Frame(data=frame_traces, name=date))

    fig.frames = frames

    # Slider
    slider_steps = [
        dict(
            method='animate',
            args=[[date], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': True}}],
            label=date,
        )
        for date in dates
    ]

    # Play/Pause buttons
    play_button = dict(
        label='Play',
        method='animate',
        args=[None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
    )
    pause_button = dict(
        label='Pause',
        method='animate',
        args=[[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
    )

    fig.update_layout(
        title=title,
        # Only the bottom subplot shows the x-axis title (shared axes)
        xaxis2_title='ω (rad/day)',
        yaxis_title='|H(jω)| dB',
        yaxis2_title='∠H(jω) deg',
        updatemenus=[dict(
            type='buttons', showactive=False, y=1.15, x=0.5, xanchor='center',
            buttons=[play_button, pause_button],
        )],
        sliders=[dict(active=0, steps=slider_steps, x=0.1, len=0.9, xanchor='left',
                      y=0, yanchor='top', currentvalue=dict(prefix='asof_date: ', visible=True))],
        height=700,  # taller to accommodate 2 subplots
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')
```

### Step 4: Run test to verify it passes

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_si_freq_response_mode -v
```

Expected: PASS.

### Step 5: Run full test suite to verify count + zero-regression

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v
```

Expected: **72 passed** (no change in count, 1 test updated).

### Step 6: Verify zero-modification of protected files

```bash
cd c:/Users/yellow/mcp/qtTdx && git diff --stat f1d17f7 -- backtrace/dynamics/_dynamics_core.py backtrace/dynamics/dynamics_forced_response.py backtrace/dynamics/dynamics_system.py backtrace/dynamics/dynamics_batch.py backtrace/dynamics/dynamics_1step_oos.py backtrace/dynamics/dynamics_si_ic.py backtrace/dynamics/dynamics_si_timeseries.py backtrace/dynamics/dynamics_si_lagged_ic.py backtrace/dynamics/dynamics_eigen_analysis.py backtrace/projection/parameter_fit.py
```

Expected: empty (no changes to 10 protected files).

### Step 7: Commit

```bash
cd c:/Users/yellow/mcp/qtTdx && git add backtrace/dynamics/dynamics_si_freq_response.py tests/test_dynamics_eigen.py && git commit -m "feat(dynamics): v5.4 — dual-pane Bode (|H(jω)| + ∠H(jω))

build_animated_overlay_html 内部从单子图 → 双子图:
- 上子图 |H(jω)| dB(保留 v5.3)
- 下子图 ∠H(jω) deg(新增,共享 x 轴)
- plotly make_subplots(2, 1, shared_xaxes=True)
- 每帧 2 × N traces(magnitude + phase per industry)
- 动画 slider + Play/Pause 联动 2 子图

签名 0 修改。其他 5 函数 + parse_args + main 0 修改。
1 test 扩充(加 xaxis/yaxis 计数 + phase 关键词断言)。
72 → 72 tests pass。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: README §4.1.2 v5.4 footnote

**Files:**
- Modify: `backtrace/dynamics/README.md` (append 1 line to §4.1.2)

### Step 1: Find §4.1.2 anchor

```bash
cd c:/Users/yellow/mcp/qtTdx && grep -n "^### §4.1.2" backtrace/dynamics/README.md
```

Expected: shows line number of `### §4.1.2 v5.3 — Real SI Frequency Response (时序动画)`.

### Step 2: Append 1 line after the §4.1.2 v5.3 reserved heading `### §4.1.3 (Reserved — v5.4+)`

Change:
```markdown
### §4.1.3 (Reserved — v5.4+)
```

To:
```markdown
### §4.1.3 v5.4 — Dual-Pane Bode (|H(jω)| + ∠H(jω))

v5.4 把 v5.3 的单子图 (|H(jω)| dB) 扩成**双子图 Bode**:
- 上子图 |H(jω)| dB vs ω
- 下子图 ∠H(jω) deg vs ω(共享 x 轴)

实现: `build_animated_overlay_html` 内部用 `plotly.subplots.make_subplots(2, 1, shared_xaxes=True)`,每帧 2 × N traces。

CLI/输出/签名 0 变化,只影响 HTML 渲染(从单图 → 双图,size ~400KB)。
```

### Step 3: Verify only README.md modified

```bash
cd c:/Users/yellow/mcp/qtTdx && git diff --stat HEAD -- backtrace/dynamics/README.md
```

Expected: only `backtrace/dynamics/README.md` modified.

### Step 4: Commit

```bash
cd c:/Users/yellow/mcp/qtTdx && git add backtrace/dynamics/README.md && git commit -m "docs(dynamics): README §4.1.3 v5.4 — dual-pane Bode footnote

记录 v5.4 把 v5.3 单子图扩成双子图 (|H| + ∠H),共享 x 轴。
其他 0 变化。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Spec Coverage Checklist

After both tasks, verify:

- [ ] §1 问题 — Task 1 解决(单子图 → 双子图)
- [ ] §2 目标 — 1 函数 modify + 1 test update,其他 0 修改
- [ ] §3.1 架构 — `make_subplots(2, 1, shared_xaxes=True)` 实现
- [ ] §3.2 修改范围 — 仅 `build_animated_overlay_html` 内部
- [ ] §3.3 CLI — 0 改动 ✓
- [ ] §3.4 输出 — 与 v5.3 相同
- [ ] §3.5 测试 — 1 个 test 扩充,72 tests pass ✓
- [ ] §4 约束兑现 — 10 保护文件 0 修改
- [ ] §5 关键文件 — 1 函数修改 + 1 test 更新 + 1 README 1-line

## Self-Review Notes

1. **Spec coverage:** All 9 spec sections have a task. ✓
2. **Placeholder scan:** No "TBD" / "TODO" — all code values are exact (function bodies, test assertions, magic numbers).
3. **Type consistency:** `build_animated_overlay_html` signature unchanged (`pairs_per_date, omega_grid, output_path, title`). Other 5 functions untouched.
4. **Single-task vs multi-task split:** Could combine Tasks 1+2 into one, but separating them gives each its own review gate. The README change is docs-only and trivial.
