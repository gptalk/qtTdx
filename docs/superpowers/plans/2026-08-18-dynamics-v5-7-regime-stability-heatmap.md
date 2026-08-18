# Plan — v5.7 Regime Stability Heatmap

**Date:** 2026-08-18
**Goal:** 加 2D cell-based PNG heatmap (rows=dates, cols=industries, color=regime) — dashboard view, 业务周报 / 复盘用。

**Spec:** [`docs/superpowers/specs/2026-08-18-dynamics-v5-7-regime-stability-heatmap.md`](../specs/2026-08-18-dynamics-v5-7-regime-stability-heatmap.md)
**Tech:** matplotlib 3.10.6 (already in), 0 新依赖
**Architecture:** Feature flags 极简 ─ 1 新 function + 1 CLI flag + 1 main() 行 + 1 test + 1 README 节

## Global Constraints

- 0 protected file modifications (`_dynamics_core.py` / `dynamics_forced_response.py` / `dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py` / `dynamics_si_ic.py` / `dynamics_si_timeseries.py` / `dynamics_si_lagged_ic.py` / `dynamics_eigen_analysis.py` / `projection/parameter_fit.py`)
- 0 新依赖
- 73 → 74 tests pass (1 new test)
- 4-color hex 与 v5.5 / v5.6 完全一致 (REJIME_COLORS 模块级 dict)
- 顶部 4 色 mpatches legend (v5.6 I-1 模式)
- `matplotlib.use('Agg')` 已在 v5.6 设置,不需要重新设置

---

## Task 1: `build_regime_heatmap` + CLI flag + main() + test

**Files:**
- Modify: `backtrace/dynamics/dynamics_si_freq_response.py` (+50 lines)
- Modify: `tests/test_dynamics_eigen.py` (+44 lines)

**Interfaces:**
- Consumes: `pairs_per_date` from `select_top_n_per_date` (existing v5.6 function)
- Produces: `build_regime_heatmap(pairs_per_date, output_path, title='...', dpi=100)` — 1 new function
- CLI flag: `--heatmap-output PATH` (default `backtrace/outputs/dynsys_regime_heatmap.png`)
- main() 末尾: `build_regime_heatmap(pairs, args.heatmap_output)` + 1 print

### Step 1: Add new test

Append AFTER `test_cli_static_grid_mode` in `tests/test_dynamics_eigen.py`:

```python
def test_cli_regime_heatmap_mode(tmp_path):
    """v5.7: CLI regime heatmap mode — 验证 build_regime_heatmap 输出 PNG."""
    pytest.importorskip("matplotlib")

    import subprocess
    import sys
    import os

    # 合成 3 dates × 2 industries CSV (复用 v5.6 fixture 模式)
    csv_path = tmp_path / 'kc_time.csv'
    rows = []
    for date_str in ['2024-09-30', '2024-10-31', '2024-11-30']:
        for code, k, c in [('AAA', 0.5, 2.0), ('BBB', 3.5, 0.5)]:
            rows.append({
                'code': code, 'index_code': f'Industry_{code}',
                'asof_date': date_str, 'k_hat': k, 'c_hat': c,
                'status': 'ok', 'n_valid_days': 200,
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')

    heatmap_png = tmp_path / 'heatmap.png'
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cli_script = os.path.join(repo_root, 'backtrace', 'dynamics', 'dynamics_si_freq_response.py')
    cmd = [
        sys.executable, cli_script,
        '--kc-time-csv', str(csv_path),
        '--top-n-industries', '2',
        '--heatmap-output', str(heatmap_png),
    ]
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(cmd, capture_output=True, env=env, timeout=60)
    assert result.returncode == 0, f'CLI failed: {result.stderr.decode("utf-8", errors="ignore")}'

    # 验证 PNG 存在 + 字节头 + size
    assert heatmap_png.exists(), f'PNG not created: {heatmap_png}'
    assert heatmap_png.stat().st_size > 5000, f'PNG too small: {heatmap_png.stat().st_size}'
    with open(heatmap_png, 'rb') as fh:
        header = fh.read(8)
    assert header.startswith(b'\x89PNG'), f'Not a valid PNG: header={header!r}'
```

### Step 2: Run test to verify FAIL

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_regime_heatmap_mode -v
```

Expected: FAIL — `--heatmap-output` flag not recognized OR `build_regime_heatmap` not defined.

### Step 3: Add `build_regime_heatmap` function to `dynamics_si_freq_response.py`

Append AFTER `build_static_bode_grid` (v5.6 function):

```python
REGIME_ABBREV = {
    'overdamped':  'over',
    'critical':    'crit',
    'underdamped': 'under',
    'anti_damped': 'anti',
}


def build_regime_heatmap(
    pairs_per_date: list,
    output_path: str,
    title: str = 'Industry Regime Stability — Heatmap',
    dpi: int = 100,
) -> None:
    """Render regime for each (date, industry) as a 2D heatmap.

    Args:
        pairs_per_date: [(asof_date, k̂, ĉ, label), ...] from select_top_n_per_date
        output_path: PNG 输出路径
        title: figure 标题
        dpi: PNG 分辨率(默认 100)

    Raises:
        ValueError: pairs_per_date 为空
    """
    if not pairs_per_date:
        raise ValueError('pairs_per_date 为空,无法构建 heatmap')

    dates = sorted(set(p[0] for p in pairs_per_date))
    industries = sorted(set(p[3] for p in pairs_per_date))
    n_rows = len(dates)
    n_cols = len(industries)

    # 索引 (date, industry) → (k̂, ĉ)
    pair_lookup = {}
    for p in pairs_per_date:
        pair_lookup[(p[0], p[3])] = (p[1], p[2])

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * n_cols), max(4, 0.6 * n_rows)))

    # 画 cell
    for i, date in enumerate(dates):
        for j, industry in enumerate(industries):
            k, c = pair_lookup.get((date, industry), (None, None))
            if k is None:
                color = '#7f7f7f'  # 灰 (无数据)
                text = '?'
            else:
                regime = classify_response_type(k, c)
                color = REGIME_COLORS.get(regime, '#7f7f7f')
                text = REGIME_ABBREV.get(regime, '?')

            # cell 边界 + 填色
            rect = mpatches.Rectangle(
                (j, i), 1, 1,
                facecolor=color, edgecolor='white', linewidth=1.5,
            )
            ax.add_patch(rect)
            # cell 中心文字
            ax.text(j + 0.5, i + 0.5, text,
                    ha='center', va='center',
                    fontsize=10, color='black')

    # 行/列 label
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(industries, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(dates, fontsize=9)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()  # 日期早的在顶部
    ax.set_aspect('equal')

    # 顶部 4 色 legend (v5.6 I-1 模式)
    regime_patches = [
        mpatches.Patch(color='#2ca02c', label='overdamped (stable)'),
        mpatches.Patch(color='#ff7f0e', label='critical'),
        mpatches.Patch(color='#d62728', label='underdamped (resonance)'),
        mpatches.Patch(color='#9467bd', label='anti_damped'),
    ]
    fig.legend(handles=regime_patches, loc='upper center',
               bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False, fontsize=9)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
```

### Step 4: Add `--heatmap-output` CLI flag

In `parse_args()` (existing function), add ONE new argument AFTER `--static-output`:

```python
    parser.add_argument(
        '--heatmap-output',
        type=str,
        default='backtrace/outputs/dynsys_regime_heatmap.png',
        help='Regime heatmap PNG 输出路径',
    )
```

### Step 5: Add main() call to new function

In `main()` (existing function), AFTER the existing `build_static_bode_grid(...)` call and print, add:

```python
    build_regime_heatmap(pairs, args.heatmap_output)
    print(f'[v5.7] regime heatmap 已写入 {args.heatmap_output}')
```

### Step 6: Run test to verify PASS

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_regime_heatmap_mode -v
```

Expected: PASS.

### Step 7: Run full test suite to verify count + zero-regression

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v
```

Expected: **74 passed** (1 new test, 73 unchanged).

### Step 8: Verify zero-modification of protected files

```bash
cd c:/Users/yellow/mcp/qtTdx && git diff --stat 74199d8 -- backtrace/dynamics/_dynamics_core.py backtrace/dynamics/dynamics_forced_response.py backtrace/dynamics/dynamics_system.py backtrace/dynamics/dynamics_batch.py backtrace/dynamics/dynamics_1step_oos.py backtrace/dynamics/dynamics_si_ic.py backtrace/dynamics/dynamics_si_timeseries.py backtrace/dynamics/dynamics_si_lagged_ic.py backtrace/dynamics/dynamics_eigen_analysis.py backtrace/projection/parameter_fit.py
```

Expected: empty (no changes to 10 protected files).

### Step 9: Verify only 2 files modified in this task

```bash
cd c:/Users/yellow/mcp/qtTdx && git diff --stat 74199d8
```

Expected: ONLY `backtrace/dynamics/dynamics_si_freq_response.py` + `tests/test_dynamics_eigen.py` modified.

### Step 10: Commit

```bash
cd c:/Users/yellow/mcp/qtTdx && git add backtrace/dynamics/dynamics_si_freq_response.py tests/test_dynamics_eigen.py && git commit -m "feat(dynamics): v5.7 — regime stability heatmap (2D cells, dashboard view)

build_regime_heatmap 新函数:
- 2D 网格 (rows = unique asof_date, cols = unique industry)
- 每个 cell = REGIME_COLORS.get(classify_response_type(k, c)) + 4 字 abbreviation
  - over (overdamped 绿) / crit (critical 橙) / under (underdamped 红) / anti (anti_damped 紫)
- 顶部 4 色 mpatches legend (v5.6 I-1 模式)
- 0 新依赖 (matplotlib 3.10.6 已在 v5.6 引入)

新增:
- REGIME_ABBREV 模块级 dict (4 字映射)
- 1 CLI flag --heatmap-output PATH
- main() 末尾 1 行调用 build_regime_heatmap
- 1 新 test (PNG 字节头 + 文件大小校验)

与其他 7 函数 + parse_args 签名 0 变化(只加 1 flag)。
73 → 74 tests pass。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: README §4.1.6 v5.7 footnote

**Files:**
- Modify: `backtrace/dynamics/README.md` (+15 lines)

### Step 1: Find §4.1.5 anchor

```bash
cd c:/Users/yellow/mcp/qtTdx && grep -n "^### §4.1.5" backtrace/dynamics/README.md
```

Expected: shows line number of `### §4.1.5 v5.6 — Static 2D Grid PNG (matplotlib 导出)`.

### Step 2: Add §4.1.6 v5.7 section

After the §4.1.5 section ends, append:

```markdown
### §4.1.6 v5.7 — Regime Stability Heatmap (matplotlib cells)

v5.7 给同一份数据加 **dashboard 视图** — 业务写周报 / 复盘用:

- 函数: `build_regime_heatmap(pairs_per_date, output_path, dpi=100)`
- 布局: 2D 网格 (rows = unique asof_date, cols = industry)
- 每个 cell: 背景色 = regime color, 中心文字 = 4 字 abbreviation (over/crit/under/anti)
- 0 新依赖 (matplotlib 3.10.6 已在 v5.6 引入)
- CLI: `--heatmap-output PATH` (默认 `backtrace/outputs/dynsys_regime_heatmap.png`)

**v5.5 / v5.6 / v5.7 关系**:
- v5.5 HTML: 交互曲线, 浏览器拖 slider
- v5.6 PNG: 静态曲线, 嵌 PDF / PPT
- v5.7 PNG: **静态 cells, dashboard 执行摘要**
- 三者共用 `classify_response_type` + REGIME_COLORS → 0 重复, 0 不一致
```

### Step 3: Verify only README.md modified

```bash
cd c:/Users/yellow/mcp/qtTdx && git diff --stat <T1-commit> -- backtrace/dynamics/README.md
```

Expected: only `backtrace/dynamics/README.md` modified.

### Step 4: Commit

```bash
cd c:/Users/yellow/mcp/qtTdx && git add backtrace/dynamics/README.md && git commit -m "docs(dynamics): README §4.1.6 v5.7 — regime stability heatmap footnote

记录 v5.7 给同一份数据加 2D cell-based heatmap (dashboard 视图),
业务周报 / 复盘用, 与 v5.5 HTML / v5.6 PNG 互补, 0 新依赖, 颜色逻辑 0 重复。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
