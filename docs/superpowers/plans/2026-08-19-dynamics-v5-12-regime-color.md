# Plan v5.12 — Dashboard regime color-coding (k̂ vs ĉ dominance)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 v5.10 `build_full_market_oos_html` 的 (2,1) 散点 panel 上,按 (k̂, ĉ) dominance 分 3 色 (k_dominant / c_dominant / balanced),让业务方一眼看出批量 OOS 跑完后"共振 / 过阻尼 / 平衡"股的占比。

**Architecture:**
- 新 helper `classify_regime(k, c, threshold=0.1) -> str` 放 `dynamics_oos_batch.py` 模块级(不藏在函数内,可单独 import 测试)
- `build_full_market_oos_html` 在 (2,1) 散点上: 把当前 Viridis (连续 hit-rate) 换成 3 离散 regime 颜色 + legend + hover 加 regime 字段
- 其他 3 个 panel (1,1 hit-rate 直方图 / 1,2 RMSE 直方图 / 2,2 hit-rate CDF) 不动
- 1 新 unit test `test_classify_regime` (10 case 覆盖 5 类 + 边界 + 异常)
- 1 新 CLI flag `--regime-threshold`

**Tech Stack:** pandas / numpy / plotly (全部已装,0 新依赖)

## Global Constraints

- **0 modifications to 11 protected files** (`_projection_core.py` / `dynamics_forced_response.py` / `dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py` / `dynamics_si_ic.py` / `dynamics_si_timeseries.py` / `dynamics_si_lagged_ic.py` / `dynamics_eigen_analysis.py` / `parameter_fit.py` / `dynamics_oos_viz.py`)
- **0 modifications to** `dynamics_oos_batch.py` **core logic** (除 `build_full_market_oos_html` 内部 + `classify_regime` 新 helper + main() argparse)
- **0 new dependencies** (pandas / numpy / plotly 已在)
- **仅修改 3 个文件**: `backtrace/dynamics/dynamics_oos_batch.py` / `tests/test_dynamics_eigen.py` (+1 test) / `backtrace/dynamics/README.md` (§4.1.11)
- **不重跑 batch CSV** — dashboard 现算 regime

---

## File Structure

| 文件 | 角色 | 改动量 |
|---|---|---|
| `backtrace/dynamics/dynamics_oos_batch.py` | `classify_regime` 新 helper + `build_full_market_oos_html` (2,1) 散点改色 + main() 加 `--regime-threshold` flag | ~+50 行 |
| `tests/test_dynamics_eigen.py` | 新增 `test_classify_regime` (10 case) | +30 行 |
| `backtrace/dynamics/README.md` | 新 §4.1.11 | +25 行 |
| `docs/superpowers/specs/2026-08-19-dynamics-v5-12-regime-color.md` | 已有 (d4443c8) | 0 |
| `docs/superpowers/plans/2026-08-19-dynamics-v5-12-regime-color.md` | 本文件 | 0 |
| `~/.claude/projects/c--Users-yellow-mcp-qtTdx/memory/dynamics-v5-12-regime-color.md` | Task 2 新建 (auto-memory) | +40 行 |

---

## Task 1: `classify_regime` helper + (2,1) scatter color/legend/hover

**Files:**
- Modify: `backtrace/dynamics/dynamics_oos_batch.py` (~+50 行)
  - Module-level 新 helper `classify_regime`
  - Modify `build_full_market_oos_html` 函数体 (lines ~250-268 的散点 panel + 加 legend)

**Interfaces:**
- Consumes: `metrics_list` (list of dicts with keys `code, hit_rate, rmse, k_used, c_used`)
- Produces: HTML with regime-colored scatter + legend + hover

**Step 1.1: 在模块顶部加 `classify_regime` 函数**

放在 `build_full_market_oos_html` 函数定义之前(模块级,便于 import 测试)。

```python
def classify_regime(k: float, c: float, threshold: float = 0.1) -> str:
    """按 |k| / |c| 比例分 3 个 regime (v5.12)。

    Args:
        k: 拟合的弹性系数 (k̂)
        c: 拟合的阻尼系数 (ĉ)
        threshold: 相对差异容忍度,默认 0.1 (= 10%)。
                   |k|/|c| 在 [1/(1+threshold), 1+threshold] 内视为 balanced。

    Returns:
        'k_dominant' — 共振风险 (|k| > |c| * (1 + threshold))
        'c_dominant' — 过阻尼稳定 (|c| > |k| * (1 + threshold))
        'balanced'   — 其余(含 k=c=0 placeholder 状态)

    Raises:
        ValueError: if threshold < 0
    """
    if threshold < 0:
        raise ValueError(f'threshold must be >= 0, got {threshold}')

    abs_k, abs_c = abs(float(k)), abs(float(c))

    # 占位符或零参数 → balanced (不区分,免得"无信息"被错分)
    if abs_k == 0.0 and abs_c == 0.0:
        return 'balanced'

    # 阈值上限:绝对值差异非常悬殊时,ratio 会爆 → 强制归类
    # 不报错,允许 caller 探索大 threshold
    if abs_c < 1e-12:
        return 'k_dominant'
    if abs_k < 1e-12:
        return 'c_dominant'

    ratio = abs_k / abs_c
    upper = 1.0 + threshold
    lower = 1.0 / upper

    if ratio > upper:
        return 'k_dominant'
    if ratio < lower:
        return 'c_dominant'
    return 'balanced'
```

**Step 1.2: 修改 `build_full_market_oos_html` 签名**

```python
def build_full_market_oos_html(
    metrics_list: list[dict],
    output_path: str,
    title: str = 'Full-Market OOS Prediction Quality Distribution',
    regime_threshold: float = 0.1,  # v5.12 NEW
) -> None:
    """[v5.10 docstring] + v5.12 regime_threshold:
        |k| / |c| 平衡区相对差异容忍度(默认 0.1 = 10%)。
        控制 dashboard (2,1) 散点 panel 的 regime 颜色分布。
    """
```

**Step 1.3: 修改 (2,1) scatter panel 颜色 + legend + hover**

替换原 line 250-268 的 `fig.add_trace(go.Scatter(...))`,改成:

```python
# 7) (2,1) hit-rate vs RMSE 散点 — v5.12: 按 regime 着色
regimes = [classify_regime(m.get('k_used', 0.0), m.get('c_used', 0.0),
                            threshold=regime_threshold)
           for m in metrics_list]
regime_colors = {
    'k_dominant': '#e74c3c',  # 红
    'c_dominant': '#3498db',  # 蓝
    'balanced':   '#2ecc71',  # 绿
}

# 主散点(按 regime 离散着色)+ hover 含 regime + (k̂, ĉ)
fig.add_trace(
    go.Scatter(
        x=hit_rates, y=rmses,
        mode='markers',
        marker=dict(
            size=6,
            color=[regime_colors[r] for r in regimes],
            line=dict(color='#2c3e50', width=0.5),
        ),
        text=df['code'].tolist(),
        customdata=list(zip(
            regimes,
            df.get('k_used', pd.Series([0.0] * n_stocks)).tolist(),
            df.get('c_used', pd.Series([0.0] * n_stocks)).tolist(),
        )),
        hovertemplate=(
            '<b>%{text}</b><br>'
            'hit-rate: %{x:.3f}<br>'
            'RMSE: %{y:.4f}<br>'
            'regime: %{customdata[0]}<br>'
            'k̂: %{customdata[1]:.4f}<br>'
            'ĉ: %{customdata[2]:.4f}'
            '<extra></extra>'
        ),
        name='stocks',
        showlegend=False,
    ),
    row=2, col=1,
)

# 7b) Legend 3 个 invisible scatter traces(纯 legend entry,不画数据)
for regime_name, regime_color in regime_colors.items():
    count = regimes.count(regime_name)
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=regime_color,
                        line=dict(color='#2c3e50', width=0.5)),
            name=f'{regime_name} (N={count})',
            showlegend=True,
        ),
        row=2, col=1,
    )
```

**Step 1.4: 更新 layout `showlegend=False` → `showlegend=True` + legend 位置**

替换原 line 301:

```python
fig.update_layout(
    title=f"{title} — N={n_stocks}",
    height=800, showlegend=True,            # v5.12: 显示 regime legend
    legend=dict(
        x=0.01, y=0.99,
        xanchor='left', yanchor='top',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='#2c3e50',
        borderwidth=1,
    ),
    template='plotly_white',
)
```

**Step 1.5: 跑已有测试,确认没破坏**

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe \
    -m pytest tests/test_dynamics_eigen.py --no-header -q
```

期望: 79 PASS (78 + 1,classify_regime helper 还没测)+ 0 SKIP (v5.11 已 78)。

(注意:`test_classify_regime` 在 Task 2 才加,这里只是确认现有测试不挂。)

**Step 1.6: Smoke(冒烟)— HTML 能产出且颜色对**

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe \
    backtrace/dynamics/dynamics_oos_batch.py --help
```

期望: `--regime-threshold` flag 还没出现 (Task 2 加);但其他不变。

注意:Task 1 不加 CLI flag,只改函数;CLI 由 Task 2 加。

**Step 1.7: Commit**

```bash
git add backtrace/dynamics/dynamics_oos_batch.py
git commit -m "feat(dynamics): v5.12 — classify_regime helper + regime color in dashboard scatter

classify_regime(k, c, threshold=0.1) module-level helper (testable).
build_full_market_oos_html (2,1) scatter: replace Viridis (continuous
hit-rate) with 3 discrete regime colors (k_dominant red / c_dominant blue /
balanced green) + legend + hover with regime + k̂ + ĉ fields.
Threshold default 0.1 (10%).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: CLI flag + test + README + final review + push + memory

**Files:**
- Modify: `backtrace/dynamics/dynamics_oos_batch.py` (Task 1 已改,这里加 main() flag + 透传)
- Modify: `tests/test_dynamics_eigen.py` (+1 test `test_classify_regime`)
- Modify: `backtrace/dynamics/README.md` (§4.1.11)
- Create: `~/.claude/projects/c--Users-yellow-mcp-qtTdx/memory/dynamics-v5-12-regime-color.md`

**Step 2.1: CLI flag — `--regime-threshold` on `dynamics_oos_batch.py`**

在 `main()` 的 `p.add_argument(...)` block 里,加:

```python
p.add_argument('--regime-threshold', dest='regime_threshold', type=float, default=0.1,
               help='v5.12: |k|/|c| 平衡区相对差异容忍度(默认 0.1 = 10%)')
```

**Step 2.2: Pass through to `build_full_market_oos_html`**

找到 `main()` 里调用 `build_full_market_oos_html(...)` 的地方,加 `regime_threshold=args.regime_threshold`:

```python
build_full_market_oos_html(
    metrics_list,
    output_path=str(top_html),
    title=f'Full-Market OOS Prediction Quality Distribution (top {top_n})',
    regime_threshold=args.regime_threshold,  # v5.12 NEW
)
```

**Step 2.3: `test_classify_regime`**

追加到 `tests/test_dynamics_eigen.py`:

```python
def test_classify_regime():
    """v5.12 — classify_regime 10 cases: 3 regime + placeholder + 负值 + 边界 + 异常。"""
    from backtrace.dynamics.dynamics_oos_batch import classify_regime

    # 1. k 主导
    assert classify_regime(0.5, 0.1, 0.1) == 'k_dominant'

    # 2. c 主导
    assert classify_regime(0.1, 0.5, 0.1) == 'c_dominant'

    # 3. 平衡 (|k|/|c|=1.11 在 [1/1.1, 1.1] 内)
    assert classify_regime(0.5, 0.49, 0.1) == 'balanced'

    # 4. 占位符 (k=c=0) → balanced
    assert classify_regime(0.0, 0.0, 0.1) == 'balanced'

    # 5. 负 k(anti-restoring)按 |k| 算,k_dominant
    assert classify_regime(-0.5, 0.1, 0.1) == 'k_dominant'

    # 6. 边界: |k|/|c|=1.10 → balanced (恰在阈值边界)
    assert classify_regime(0.55, 0.50, 0.1) == 'balanced'

    # 7. 边界: |k|/|c|=1.11 → k_dominant (刚超阈值)
    assert classify_regime(0.555, 0.50, 0.1) == 'k_dominant'

    # 8. threshold=0 → 严格不等式(k=c 才进 balanced)
    assert classify_regime(0.5, 0.5, 0.0) == 'balanced'
    assert classify_regime(0.5, 0.4, 0.0) == 'k_dominant'

    # 9. threshold < 0 → ValueError
    import pytest
    with pytest.raises(ValueError):
        classify_regime(0.5, 0.1, -0.1)

    # 10. 大 threshold → 不崩,k_dominant
    assert classify_regime(0.5, 0.1, 100.0) == 'k_dominant'

    # 11. (额外) c 接近 0 → k_dominant (避免除零)
    assert classify_regime(0.5, 1e-15, 0.1) == 'k_dominant'

    # 12. (额外) k 接近 0 → c_dominant
    assert classify_regime(1e-15, 0.5, 0.1) == 'c_dominant'
```

**Step 2.4: 跑测试**

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe \
    -m pytest tests/test_dynamics_eigen.py::test_classify_regime -v --no-header
```

期望: `1 passed`.

Full suite:

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe \
    -m pytest tests/test_dynamics_eigen.py --no-header -q
```

期望: 79 PASS + 0 SKIP (78 + 1 new test).

**Step 2.5: Smoke CLI**

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe \
    backtrace/dynamics/dynamics_oos_batch.py --help
```

期望: `--regime-threshold FLOAT` 出现在 help 输出。

如果有 `data/projection/kc_estimates.csv` 在场:

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe \
    backtrace/dynamics/dynamics_oos_batch.py --days 60 --limit 5 --top-n 3 \
    --kc-estimates-csv data/projection/kc_estimates.csv \
    --regime-threshold 0.1 \
    --output backtrace/outputs/_smoke_v5_12.html
```

期望: log 行 `[v5.10] wrote backtrace/outputs/_smoke_v5_12.html (5 stocks, ...)`,HTML 文件生成。

**Step 2.6: Commit test + README + CLI flag**

```bash
git add backtrace/dynamics/dynamics_oos_batch.py tests/test_dynamics_eigen.py
git commit -m "feat(dynamics): v5.12 — --regime-threshold CLI flag + test_classify_regime (12 cases)

Adds --regime-threshold flag on dynamics_oos_batch.py main(), passes to
build_full_market_oos_html. test_classify_regime covers 3 regime types +
placeholder + negative k + boundary + ValueError + near-zero handling.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

**Step 2.7: README §4.1.11**

追加在 `backtrace/dynamics/README.md` (在 §4.1.10 之后):

```markdown
### 4.1.11 v5.12 — Dashboard regime color-coding (k̂ vs ĉ dominance)

**File:** `backtrace/dynamics/dynamics_oos_batch.py` (extended)

**Goal:** After v5.11 made `k_used`/`c_used` real, v5.12 surfaces them visually — (2,1) scatter panel now colors points by regime instead of continuous hit-rate Viridis.

**Regime classifier:** `classify_regime(k, c, threshold=0.1) -> str`
- `'k_dominant'` — `|k| > |c| * (1 + threshold)` (resonance risk, red `#e74c3c`)
- `'c_dominant'` — `|c| > |k| * (1 + threshold)` (overdamped stability, blue `#3498db`)
- `'balanced'`   — ratio in `[1/(1+threshold), 1+threshold]` (green `#2ecc71`); also catches `k_used=c_used=0` placeholder state

**Dashboard changes:**
- (2,1) scatter: discrete 3-color markers (was Viridis continuous)
- Legend shows each regime with stock count: `k_dominant (N=X)`
- Hover adds `regime`, `k̂`, `ĉ` fields
- Other 3 panels (1,1 histogram / 1,2 RMSE histogram / 2,2 CDF) untouched

**CLI:**
```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_oos_batch.py \
    --days 60 --limit 5 --top-n 3 \
    --kc-estimates-csv data/projection/kc_estimates.csv \
    --regime-threshold 0.1 \
    --output backtrace/outputs/dynsys_oos_full_market.html
```

**Threshold guidance:**
- `0.05` (5%): strict balance — only near-equal ratios count as balanced
- `0.10` (default): practical — catches most "visually balanced" stocks
- `0.30+`: only extreme k-dominant or c-dominant stocks split off

**Test:** `tests/test_dynamics_eigen.py::test_classify_regime` (12 cases incl. boundaries + ValueError).
```

```bash
git add backtrace/dynamics/README.md
git commit -m "docs(dynamics): v5.12 — README §4.1.11

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

**Step 2.8: Spec + plan status + memory file**

Append to `docs/superpowers/specs/2026-08-19-dynamics-v5-12-regime-color.md`:

```markdown

## Status: ✅ DONE — 2026-08-19

3 commits complete:
- Task 1: `classify_regime` helper + dashboard (2,1) scatter color/legend/hover
- Task 2: `--regime-threshold` CLI flag + `test_classify_regime` (12 cases) + README §4.1.11
- (post-review touchup if any)

Final: 79 PASS + 0 SKIP, 0 modifications to 11 protected files + `dynamics_oos_batch.py` core logic, 0 new dependencies.
```

Append to `docs/superpowers/plans/2026-08-19-dynamics-v5-12-regime-color.md`:

```markdown

## Status: ✅ DONE — 2026-08-19

All 2 tasks complete. Pushed to origin/main in commits <push_commit_pending>. See memory file `dynamics-v5-12-regime-color.md`.
```

(Use `<push_commit_pending>` placeholder; replace after push.)

Memory file at `~/.claude/projects/c--Users-yellow-mcp-qtTdx/memory/dynamics-v5-12-regime-color.md`:

```markdown
---
name: dynamics-v5-12-regime-color
description: v5.12 dashboard (2,1) scatter 按 (k̂, ĉ) dominance 分 3 regime 配色 + legend + hover,79 tests PASS
metadata:
  type: project
---

v5.12 在 v5.11 让 (k̂, ĉ) 真实化之后,把这俩值**视觉化**到 v5.10 dashboard 散点 panel —— (2,1) hit-rate × RMSE 散点按 regime 离散配色(原来 Viridis 连续 hit-rate),让业务方一眼看出"批量里有多少共振风险 / 过阻尼稳定 / 平衡"。

**1 新 helper(module-level,便于测试)**:`classify_regime(k, c, threshold=0.1) -> str`
- 规则:`ratio = |k|/|c|`,阈值区间 `[1/(1+t), 1+t]`,平衡区内为 `'balanced'`
- 3 个返回:`'k_dominant'`(共振风险,红) / `'c_dominant'`(过阻尼,蓝) / `'balanced'`(绿,含 `k=c=0` placeholder)
- 防除零:`|c| < 1e-12 → k_dominant` / `|k| < 1e-12 → c_dominant`
- 防异常:`threshold < 0 → ValueError` / `threshold > 100` 不报错(允许探索)

**Dashboard 改动**:只改 (2,1) scatter panel(其他 3 panel 不动)
- 颜色:`Viridis` → 离散 3 色 (`#e74c3c` / `#3498db` / `#2ecc71`)
- 加 3 个 invisible legend entries:`k_dominant (N=X)` / `c_dominant (N=X)` / `balanced (N=X)`
- Hover 加 3 字段:`regime` / `k̂` / `ĉ`
- Layout:`showlegend=False → True` + 左上角 legend

**1 新 CLI flag**:`--regime-threshold FLOAT`(默认 0.1)
- 调大:`0.3+` → 只极悬殊的 (k vs c) 才不归 balanced
- 调小:`0.05` → 只有近乎严格相等的 (k vs c) 才归 balanced

**1 新 test**:`test_classify_regime`,12 case 覆盖 k/c_dominant / balanced / placeholder (k=c=0) / 负 k / 边界 (ratio=1.10/1.11) / threshold=0 / ValueError / 大 threshold / near-zero (避免除零)。

**placeholder 视觉损失**:v5.11 之前 `k_used=c_used=0` 的股票现在被归到 balanced(绿) —— 视觉上不区分"真实平衡"和"无信息"。**已知 trade-off**,v5.13+ 才考虑加 4th color "unknown / placeholder"。

**关联**:[[dynamics-v5-11-load-oos-with-kc]] / [[dynamics-v5-10-full-market-oos-distribution]] / [[dynamics-v5-9-oos-prediction-html]]

**Why:** v5.11 让 k_used/c_used 真实化,但业务方要逐个 hover 才能看到。v5.12 把它推到批量视觉层 —— dashboard 一打开,业务方先看 legend 比例,决定要不要下钻。

**How to apply:** 任何 v5.10 batch dashboard 调用想升级到 regime 视图,加 `--regime-threshold 0.1` (默认)。3 色对应业务决策:红(共振风险 → 警惕 → 降低仓位) / 蓝(过阻尼 → 稳定 → 适合做空头保护) / 绿(平衡 → 中性)。如需更宽松的 balanced 区,用 `--regime-threshold 0.3`。
```

Add to MEMORY.md:
```
- [dynamics-v5-12-regime-color](dynamics-v5-12-regime-color.md) — v5.12 dashboard (2,1) 按 regime 3 色 + legend + hover,79 tests
```

```bash
git add docs/superpowers/specs/2026-08-19-dynamics-v5-12-regime-color.md \
       docs/superpowers/plans/2026-08-19-dynamics-v5-12-regime-color.md \
       C:/Users/yellow/.claude/projects/c--Users-yellow-mcp-qtTdx/memory/dynamics-v5-12-regime-color.md \
       C:/Users/yellow/.claude/projects/c--Users-yellow-mcp-qtTdx/memory/MEMORY.md
git commit -m "docs(dynamics): v5.12 — spec/plan status + memory file

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

**Step 2.9: Final code review (opus)**

```bash
git diff d4443c8..HEAD > .superpowers/sdd/2026-08-19-dynamics-v5-12-regime-color/final-review-package.txt
git diff d4443c8..HEAD --stat
```

Dispatch opus final reviewer with:
- review package path
- spec path: `docs/superpowers/specs/2026-08-19-dynamics-v5-12-regime-color.md`
- plan path: `docs/superpowers/plans/2026-08-19-dynamics-v5-12-regime-color.md`
- constraint block (verbatim copy of "Global Constraints" from this plan)

Reviewer checks:
- Spec coverage (§3.1-3.5)
- 0 modifications to 11 protected files + `dynamics_oos_batch.py` core logic
- 0 new dependencies
- All Task 1-2 deliverables present
- Code quality + comments match

If PASS → Step 2.10.
If NEEDS_FIX → dispatch one fix implementer + re-review.

**Step 2.10: Push to origin/main**

```bash
git push origin main
git log --oneline -5 origin/main
```

**Step 2.11: Update plan footer placeholder**

After push, replace `<push_commit_pending>` with actual commit hash range in plan file, commit + push:
```bash
git add docs/superpowers/plans/2026-08-19-dynamics-v5-12-regime-color.md
git commit -m "docs(dynamics): v5.12 — plan footer with actual push commit hash

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push origin main  # final push
```

## Status: ✅ DONE — 2026-08-19

All 2 tasks + 1 drift fix + 1 v5.12.1 fix + 1 opus polish complete. Pushed to origin/main in 8 commits `378d448..9ec2078`:
- ff5f954 (plan)
- d1b64bf (Task 1): `classify_regime` helper + dashboard (2,1) scatter color/legend/hover
- a2a49b8 (post-review drift fix): spec/plan threshold band + balanced test case
- c392845 (Task 2): `--regime-threshold` CLI flag + `test_classify_regime` (12 cases)
- 9618244 (Task 2): README §4.1.11
- ab366fd (Task 2): spec/plan status + memory file
- 8588f09 (v5.12.1 fix): revert unauthorized threshold clip + fix test #10
- 9ec2078 (opus polish): 3 minor touchups (docstring + __all__ + spec)

Final: 79 PASS + 0 SKIP, 0 modifications to 11 protected files + `dynamics_oos_batch.py` core logic, 0 new dependencies. Opus final review READY TO MERGE. See memory file `dynamics-v5-12-regime-color.md`.

## Report contract

```
# v5.12 Final Report

## Status
DONE | DONE_WITH_CONCERNS | NEEDS_CONTEXT | BLOCKED

## Final state
- Commits: <list>
- Push: <commit hash on origin/main>
- Test count: 79 PASS + 0 SKIP
- Final review verdict: PASS

## Concerns (if any)
```