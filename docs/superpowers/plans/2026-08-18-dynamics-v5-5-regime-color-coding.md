# v5.5 Regime Color Coding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Color each industry's Bode curves (magnitude + phase) by `classify_response_type` regime (4 colors), so business can see at a glance which industries are stable vs resonating.

**Architecture:** Single-function modification. `build_animated_overlay_html` adds `_regime_color` closure that maps `(k, c) → hex` using existing `classify_response_type` from `dynamics_forced_response.py:130`. All 2 × N traces (per frame) get `line=dict(color=...)` bound. Plus a small annotation explaining the color mapping.

**Tech Stack:** Python 3.10+, plotly>=3.0 (already used), pytest.

## Global Constraints

These constraints bind every task. Any conflict with task text — task text is wrong.

### File protection (only 1 file has bounded modifications)

- ❌ `_dynamics_core.py` — 0 修改
- ❌ `dynamics_forced_response.py` — 0 修改(`classify_response_type` 签名 0 变化,只被 v5.5 调)
- ❌ `dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py` — 0 修改
- ❌ `dynamics_si_ic.py` / `dynamics_si_timeseries.py` / `dynamics_si_lagged_ic.py` / `dynamics_eigen_analysis.py` — 0 修改
- ❌ `parameter_fit.py` — 0 修改
- ✓ v5.5 是**单函数内部扩展**: `backtrace/dynamics/dynamics_si_freq_response.py::build_animated_overlay_html` 内部加 `_regime_color` 闭包 + 颜色绑定 + 注释
- ✓ `tests/test_dynamics_eigen.py` 1 个 test 扩充(`test_cli_si_freq_response_mode`)
- ✓ `backtrace/dynamics/README.md` §4.1.4 加 v5.5 提示

### Signature protection

- v5.5 已有 6 函数签名 0 修改: `load_kc_time_series` / `aggregate_by_industry_per_date` / `select_top_n_per_date` / `write_animated_summary_txt` / `write_animated_pairs_csv` / `build_animated_overlay_html`
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

## Task 1: Add `_regime_color` closure + color binding + extend test

**Files:**
- Modify: `backtrace/dynamics/dynamics_si_freq_response.py:build_animated_overlay_html` (add closure + binding)
- Modify: `tests/test_dynamics_eigen.py::test_cli_si_freq_response_mode` (extend 1 assertion block)

**Interfaces:**
- Consumes: `build_animated_overlay_html` v5.4 signature (unchanged)
- Produces: HTML with regime-colored traces + legend annotation

### Step 1: Extend the CLI subprocess test assertions

Modify `tests/test_dynamics_eigen.py::test_cli_si_freq_response_mode`. After the existing v5.4 dual-pane assertions, add:

```python
    # v5.5 regime color: HTML 含至少 2 种 regime 颜色 hex
    # fixture 里有 2 industries × 3 dates,Industry A 始终 overdamped,Industry B 始终 underdamped
    html_text = html_path.read_text(encoding='utf-8')
    assert '#2ca02c' in html_text or '#d62728' in html_text, 'v5.5: 至少一种 regime 颜色 hex'
    # v5.5 颜色注释(过阻尼绿 / 欠阻尼红 / 临界橙 / anti-damped 紫 任一关键词)
    assert any(kw in html_text for kw in ('过阻尼', '欠阻尼', '临界阻尼', 'regime')), 'v5.5: 颜色注释'
```

### Step 2: Run test to verify it fails

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_si_freq_response_mode -v
```

Expected: FAIL — `#2ca02c` not in HTML OR `#d62728` not in HTML OR regime comment keyword missing.

### Step 3: Modify `build_animated_overlay_html` to add color binding

In `backtrace/dynamics/dynamics_si_freq_response.py`:

1. **Add import at top of file** (if not already present, after existing imports):
```python
from backtrace.dynamics.dynamics_forced_response import classify_response_type
```

**重要**: 这是**同级 module import**(`backtrace/dynamics/dynamics_si_freq_response.py` 已经 import `_dynamics_core`,后者 import `dynamics_forced_response`)。需要避免循环 import。如果直接 import 失败,fallback:
```python
# fallback (only if direct import causes cycle):
import importlib
_mod = importlib.import_module('backtrace.dynamics.dynamics_forced_response')
classify_response_type = _mod.classify_response_type
```

2. **Add `_regime_color` closure** inside `build_animated_overlay_html` (after `_phase_deg` definition):

```python
    def _regime_color(k, c):
        """Map (k, c) to regime color hex per classify_response_type."""
        regime = classify_response_type(k, c)
        return {
            'overdamped': '#2ca02c',    # 绿, Schur 内稳定
            'critical':   '#ff7f0e',    # 橙, Schur 边界
            'underdamped': '#d62728',   # 红, Schur 外共振
            'anti_damped': '#9467bd',   # 紫, 病态
        }.get(regime, '#7f7f7f')        # 灰 fallback (不应触发)
```

3. **Bind color in initial-state traces** (Phase 1 — after `_magnitude_db` and `_phase_deg`):

```python
    # Initial-state traces (first date)
    for p in (p_ for p_ in pairs_per_date if p_[0] == initial_date):
        color = _regime_color(p[1], p[2])
        fig.add_trace(go.Scatter(x=omega_grid.tolist(), y=_magnitude_db(p),
                                 mode='lines', name=p[3], legendgroup=p[3],
                                 line=dict(color=color)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=omega_grid.tolist(), y=_phase_deg(p),
                                 mode='lines', name=p[3], legendgroup=p[3],
                                 showlegend=False,
                                 line=dict(color=color)),
                      row=2, col=1)
```

4. **Bind color in frame traces** (Phase 2):

```python
    # Phase 2: build frames — one frame per date, each frame has 2 × N traces
    frames = []
    for date in dates:
        frame_traces = []
        for p in (p_ for p_ in pairs_per_date if p_[0] == date):
            color = _regime_color(p[1], p[2])
            frame_traces.append(go.Scatter(x=omega_grid.tolist(), y=_magnitude_db(p),
                                           mode='lines', name=p[3], legendgroup=p[3],
                                           line=dict(color=color)))
            frame_traces.append(go.Scatter(x=omega_grid.tolist(), y=_phase_deg(p),
                                           mode='lines', name=p[3], legendgroup=p[3],
                                           showlegend=False,
                                           line=dict(color=color)))
        frames.append(go.Frame(data=frame_traces, name=date))
```

5. **Add color legend annotation** (after `fig.update_layout` in the existing layout block, OR as a separate `add_annotation` call). Use `fig.add_annotation` before `fig.write_html`:

```python
    # v5.5 color legend annotation (top-right corner)
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.99, y=1.08, xanchor='right', yanchor='top',
        showarrow=False,
        text=('颜色 = 阻尼 regime: '
              '<span style="color:#2ca02c">●</span> 过阻尼(stable)  '
              '<span style="color:#ff7f0e">●</span> 临界  '
              '<span style="color:#d62728">●</span> 欠阻尼(resonance)  '
              '<span style="color:#9467bd">●</span> anti-damped'),
        align='left',
        font=dict(size=11),
    )
```

Note: plotly annotation text supports basic HTML tags including `<span style="color:...">`. This produces a small inline legend in the top-right of the figure.

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
cd c:/Users/yellow/mcp/qtTdx && git diff --stat 7e02782 -- backtrace/dynamics/_dynamics_core.py backtrace/dynamics/dynamics_forced_response.py backtrace/dynamics/dynamics_system.py backtrace/dynamics/dynamics_batch.py backtrace/dynamics/dynamics_1step_oos.py backtrace/dynamics/dynamics_si_ic.py backtrace/dynamics/dynamics_si_timeseries.py backtrace/dynamics/dynamics_si_lagged_ic.py backtrace/dynamics/dynamics_eigen_analysis.py backtrace/projection/parameter_fit.py
```

Expected: empty (no changes to 10 protected files).

### Step 7: Commit

```bash
cd c:/Users/yellow/mcp/qtTdx && git add backtrace/dynamics/dynamics_si_freq_response.py tests/test_dynamics_eigen.py && git commit -m "feat(dynamics): v5.5 — regime color coding on dual-pane Bode

build_animated_overlay_html 内部:
- 加 _regime_color(k, c) 闭包,复用 dynamics_forced_response.classify_response_type
  (v5 已有函数, 0 修改)
- 4 种 regime → 4 种 hex (overdamped 绿 / critical 橙 / underdamped 红 / anti-damped 紫)
- 每个 industry 的 magnitude + phase trace 绑定 regime 颜色
- 跨 asof_date 同 industry 颜色随 (k̂, ĉ) 变化(时序漂移可见)
- 右上角加颜色 ↔ regime 注释(HTML annotation)

签名 0 修改。其他 6 函数 + parse_args + main 0 修改。
1 test 扩充(加 regime hex + 注释关键词断言)。
72 → 72 tests pass。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: README §4.1.4 v5.5 footnote

**Files:**
- Modify: `backtrace/dynamics/README.md` (add §4.1.4 v5.5 section)

### Step 1: Find §4.1.3 anchor

```bash
cd c:/Users/yellow/mcp/qtTdx && grep -n "^### §4.1.3" backtrace/dynamics/README.md
```

Expected: shows line number of `### §4.1.3 v5.4 — Dual-Pane Bode (|H(jω)| + ∠H(jω))`.

### Step 2: Append §4.1.4 v5.5 section after §4.1.3

After the §4.1.3 v5.4 section (find a natural break — likely the next `###` heading or end of subsection), add:

```markdown
### §4.1.4 v5.5 — Regime Color Coding (regime 颜色编码)

v5.5 在 v5.4 双子图基础上**按阻尼 regime 给曲线着色**,业务一眼区分稳定/共振:

| Regime | 颜色 | 业务语义 |
|---|---|---|
| overdamped (k<c) | 🟢 绿 `#2ca02c` | Schur 内,稳定 |
| critical (k≈c) | 🟠 橙 `#ff7f0e` | Schur 边界,临界 |
| underdamped (k>c) | 🔴 红 `#d62728` | Schur 外,共振风险 |
| anti_damped (k<0) | 🟣 紫 `#9467bd` | 病态(负恢复系数) |

实现: `build_animated_overlay_html` 内 `_regime_color(k, c)` 闭包 → 复用 v5 `classify_response_type` (`dynamics_forced_response.py:130`)。

**业务可读性升级**: 拖 slider 时同 industry 颜色随 (k̂, ĉ) 漂移变化 — 业务可直观看到"哪些 industry 从绿(稳定)漂到红(共振)"。

CLI/输出/签名 0 变化,只影响 HTML 颜色 + 注释(右上角 inline 颜色 ↔ regime 注释)。
```

### Step 3: Verify only README.md modified

```bash
cd c:/Users/yellow/mcp/qtTdx && git diff --stat HEAD -- backtrace/dynamics/README.md
```

Expected: only `backtrace/dynamics/README.md` modified.

### Step 4: Commit

```bash
cd c:/Users/yellow/mcp/qtTdx && git add backtrace/dynamics/README.md && git commit -m "docs(dynamics): README §4.1.4 v5.5 — regime color coding footnote

记录 v5.5 在 v5.4 双子图基础上按 4 种 regime (over/critical/under/anti-damped) 着色。
复用 v5 classify_response_type, 时序漂移业务可读。
其他 0 变化。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Spec Coverage Checklist

After both tasks, verify:

- [ ] §1 问题 — Task 1 解决(颜色编码 regime 信息)
- [ ] §2 目标 — 1 函数 modify + 1 test update,其他 0 修改
- [ ] §3.1 架构 — `_regime_color` 闭包实现
- [ ] §3.2 颜色映射 — 4 种 hex 颜色
- [ ] §3.3 修改范围 — 仅 `build_animated_overlay_html` 内部
- [ ] §3.4 CLI — 0 改动 ✓
- [ ] §3.5 输出 — 与 v5.4 相同(HTML 增加颜色 + 注释)
- [ ] §3.6 测试 — 1 个 test 扩充,72 tests pass ✓
- [ ] §4 约束兑现 — 10 保护文件 0 修改
- [ ] §5 关键文件 — 1 函数修改 + 1 test 更新 + 1 README 1-line

## Self-Review Notes

1. **Spec coverage:** All 9 spec sections have a task. ✓
2. **Placeholder scan:** No "TBD" / "TODO" — all code values are exact (color hex, function names, magic numbers).
3. **Type consistency:** `build_animated_overlay_html` signature unchanged. `_regime_color` is a closure (not exported).
4. **Single-task vs multi-task split:** Could combine Tasks 1+2 into one, but separating them gives each its own review gate. The README change is docs-only and trivial.
5. **Import cycle risk:** v5.5 imports `classify_response_type` from `dynamics_forced_response`. Both are in `backtrace/dynamics/`. The plan provides a fallback via `importlib` if direct import causes cycle. Implementer should try direct import first.
