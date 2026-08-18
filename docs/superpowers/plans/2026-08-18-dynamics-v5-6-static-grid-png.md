# v5.6 Static 2D Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `build_static_bode_grid` function that uses matplotlib to render all dates' Bode curves as a 2D grid PNG (rows = dates, cols = |H| + ∠H), usable in reports/PDFs.

**Architecture:** New function `build_static_bode_grid` (NOT modification of `build_animated_overlay_html`). Uses matplotlib 3.10.6 (already in env). Reuses `magnitude_phase` (v5.1) + `classify_response_type` (v5.1) + REGIME_COLORS dict (mirrors v5.5 closure).

**Tech Stack:** Python 3.10+, matplotlib 3.10.6 (already installed), pytest.

## Global Constraints

These constraints bind every task. Any conflict with task text — task text is wrong.

### File protection (only 1 file has bounded modifications)

- ❌ `_dynamics_core.py` — 0 修改
- ❌ `dynamics_forced_response.py` — 0 修改(`magnitude_phase` / `classify_response_type` 只被 v5.6 调用, 不变)
- ❌ `dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py` — 0 修改
- ❌ `dynamics_si_ic.py` / `dynamics_si_timeseries.py` / `dynamics_si_lagged_ic.py` / `dynamics_eigen_analysis.py` — 0 修改
- ❌ `parameter_fit.py` — 0 修改
- ✓ v5.6 是 `backtrace/dynamics/dynamics_si_freq_response.py` 修改 + 1 新函数 + 1 import + 1 module-level dict + 1 CLI flag + main() 1 行
- ✓ `tests/test_dynamics_eigen.py` 1 个新 test
- ✓ `backtrace/dynamics/README.md` §4.1.5 加 v5.6 提示

### Signature protection

- v5+v5.1+v5.2+v5.3+v5.4+v5.5 已有 6 函数签名 0 修改:`load_kc_time_series` / `aggregate_by_industry_per_date` / `select_top_n_per_date` / `write_animated_summary_txt` / `write_animated_pairs_csv` / `build_animated_overlay_html`
- `parse_args()` 签名 0 变化(只加 1 个 optional flag,默认 None)

### Test count

- **72 → 73 tests pass**(1 新增,总数 +1)
- 旧测试**全部不动**

### Runtime

- `PYTHONIOENCODING=utf-8` 必备
- Python: `/c/ProgramData/anaconda3/python.exe`
- matplotlib 3.10.6 已装(pre-flight 验证)
- 测试命令: `cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v`

---

## Task 1: Add `build_static_bode_grid` + CLI flag + main + test

**Files:**
- Modify: `backtrace/dynamics/dynamics_si_freq_response.py` (add 1 import + REGIME_COLORS dict + 1 new function + 1 CLI flag + main() 1 line)
- Modify: `tests/test_dynamics_eigen.py` (add 1 new test)

### Step 1: Add new test

Modify `tests/test_dynamics_eigen.py` — append AFTER `test_cli_si_freq_response_mode`:

```python
def test_cli_static_grid_mode(tmp_path):
    """v5.6: CLI static PNG export mode — 验证 build_static_bode_grid 输出 PNG."""
    pytest.importorskip("matplotlib")
    import subprocess
    import sys
    import os

    # 合成 3 dates × 2 industries CSV (复用 v5.3 fixture 模式)
    csv_path = tmp_path / 'kc_time.csv'
    rows = []
    for date_str, codes in [('2024-09-30', ['AAA', 'BBB']), ('2024-10-31', ['AAA', 'BBB']), ('2024-11-30', ['AAA', 'BBB'])]:
        for code in codes:
            if code == 'AAA':
                k, c = 0.5, 2.0  # overdamped
            else:
                k, c = 3.5, 0.5  # underdamped
            rows.append({
                'code': code, 'index_code': f'Industry_{code}',
                'asof_date': date_str, 'k_hat': k, 'c_hat': c,
                'status': 'ok', 'n_valid_days': 200,
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')

    static_png = tmp_path / 'static.png'
    cmd = [
        sys.executable, '-c',
        f'import sys, os; sys.path.insert(0, {repr(str(tmp_path / "backtrace_parent"))}); '
        f'os.chdir({repr(str(tmp_path))}); '
        f'exec(open({repr(str(__file__))}).read())'  # 简化,见下面
    ]
    # 简化: 直接用 CLI 子进程
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cli_script = os.path.join(repo_root, 'backtrace', 'dynamics', 'dynamics_si_freq_response.py')
    cmd = [
        sys.executable, cli_script,
        '--kc-time-csv', str(csv_path),
        '--top-n-industries', '2',
        '--static-output', str(static_png),
    ]
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(cmd, capture_output=True, env=env, timeout=60)
    assert result.returncode == 0, f'CLI failed: {result.stderr.decode("utf-8", errors="ignore")}'

    # 验证 PNG 存在 + 字节头 + size
    assert static_png.exists(), f'PNG not created: {static_png}'
    assert static_png.stat().st_size > 5000, f'PNG too small: {static_png.stat().st_size}'
    with open(static_png, 'rb') as f:
        header = f.read(8)
    assert header.startswith(b'\\x89PNG'), f'Not a valid PNG: header={header!r}'
```

**Note**: 测试需要 `pd` + `pytest` + `matplotlib` + `subprocess` 在文件顶部已经 import (file-level imports — v5.3 fixture 已用)。如果 `pd` 或 `matplotlib` 顶层 import 没在文件里,加 inline:
```python
import pandas as pd
import matplotlib  # noqa: F401  # 触发 pytest.importorskip if missing
```

### Step 2: Run test to verify it fails

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_static_grid_mode -v
```

Expected: FAIL — `build_static_bode_grid` not defined OR `--static-output` flag not recognized.

### Step 3: Add matplotlib import + REGIME_COLORS dict to `dynamics_si_freq_response.py`

At TOP of `backtrace/dynamics/dynamics_si_freq_response.py` (after existing imports), add:

```python
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (no display required)
import matplotlib.pyplot as plt
```

**Important**: `matplotlib.use('Agg')` MUST be called before `import matplotlib.pyplot` — sets non-interactive backend so subprocess doesn't fail on machines without GUI.

After existing imports, add module-level constant:

```python
REGIME_COLORS = {
    'overdamped':  '#2ca02c',   # 绿, Schur 内稳定
    'critical':    '#ff7f0e',   # 橙, Schur 边界
    'underdamped': '#d62728',   # 红, Schur 外共振
    'anti_damped': '#9467bd',   # 紫, 病态
}
```

### Step 4: Add `build_static_bode_grid` function

Append to `backtrace/dynamics/dynamics_si_freq_response.py` (after `build_animated_overlay_html`):

```python
def build_static_bode_grid(
    pairs_per_date: list,
    omega_grid: np.ndarray,
    output_path: str,
    title: str = 'Industry G(ω) Frequency Response — Static Grid',
    dpi: int = 100,
) -> None:
    """Render all dates' Bode curves as a 2D matplotlib grid (rows = dates, cols = |H| + ∠H).

    Args:
        pairs_per_date: [(asof_date, k̂, ĉ, label), ...] from select_top_n_per_date
        omega_grid: 共享 ω 网格
        output_path: PNG 输出路径
        title: figure 标题
        dpi: PNG 分辨率(默认 100)

    Raises:
        ValueError: pairs_per_date 为空
    """
    if not pairs_per_date:
        raise ValueError('pairs_per_date 为空,无法构建 static grid')

    dates = sorted(set(p[0] for p in pairs_per_date))
    n_rows = len(dates)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows), sharex=True, sharey='col')
    if n_rows == 1:
        axes = np.array([axes])  # 2D array for indexing

    def _mag_db(p):
        return magnitude_phase(omega_grid * 1j, p[1], p[2])[0].tolist()

    def _phase_deg(p):
        return np.degrees(magnitude_phase(omega_grid * 1j, p[1], p[2])[1]).tolist()

    def _color(k, c):
        return REGIME_COLORS.get(classify_response_type(k, c), '#7f7f7f')

    for i, date in enumerate(dates):
        ax_mag = axes[i, 0]
        ax_phase = axes[i, 1]
        for p in (p_ for p_ in pairs_per_date if p_[0] == date):
            color = _color(p[1], p[2])
            ax_mag.plot(omega_grid, _mag_db(p), color=color, label=p[3], linewidth=1.5)
            ax_phase.plot(omega_grid, _phase_deg(p), color=color, label=p[3], linewidth=1.5)
        ax_mag.set_ylabel('|H(jω)| dB' if i == 0 else '')
        ax_phase.set_ylabel('∠H(jω) deg' if i == 0 else '')
        ax_mag.set_title(f'{date}')
        ax_mag.grid(True, alpha=0.3)
        ax_phase.grid(True, alpha=0.3)
        if i == 0:
            ax_mag.legend(loc='upper right', fontsize=8)

    # Bottom row xlabel
    axes[-1, 0].set_xlabel('ω (rad/day)')
    axes[-1, 1].set_xlabel('ω (rad/day)')

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
```

### Step 5: Add `--static-output` CLI flag

In `parse_args()` (existing function), add ONE new argument:

```python
    parser.add_argument(
        '--static-output',
        type=str,
        default='backtrace/outputs/dynsys_si_freq_response_static.png',
        help='PNG 静态网格输出路径',
    )
```

Place it after the existing `--pairs-csv-output` argument (logical grouping).

### Step 6: Add main() call to new function

In `main()` (existing function), AFTER the existing `write_animated_pairs_csv` call, add:

```python
    build_static_bode_grid(pairs, omega_grid, args.static_output)
    print(f'[v5.6] 静态 PNG 已写入 {args.static_output}')
```

### Step 7: Run test to verify it passes

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_static_grid_mode -v
```

Expected: PASS.

### Step 8: Run full test suite to verify count + zero-regression

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v
```

Expected: **73 passed** (1 new test, 72 unchanged).

### Step 9: Verify zero-modification of protected files

```bash
cd c:/Users/yellow/mcp/qtTdx && git diff --stat 179a579 -- backtrace/dynamics/_dynamics_core.py backtrace/dynamics/dynamics_forced_response.py backtrace/dynamics/dynamics_system.py backtrace/dynamics/dynamics_batch.py backtrace/dynamics/dynamics_1step_oos.py backtrace/dynamics/dynamics_si_ic.py backtrace/dynamics/dynamics_si_timeseries.py backtrace/dynamics/dynamics_si_lagged_ic.py backtrace/dynamics/dynamics_eigen_analysis.py backtrace/projection/parameter_fit.py
```

Expected: empty (no changes to 10 protected files).

### Step 10: Commit

```bash
cd c:/Users/yellow/mcp/qtTdx && git add backtrace/dynamics/dynamics_si_freq_response.py tests/test_dynamics_eigen.py && git commit -m "feat(dynamics): v5.6 — static 2D grid PNG export (matplotlib)

build_static_bode_grid 新函数:
- matplotlib 2D 网格 (rows = unique asof_date, cols = |H| + ∠H)
- 4 种 regime 颜色与 v5.5 一致 (overdamped 绿 / critical 橙 / underdamped 红 / anti_damped 紫)
- 0 新依赖 (matplotlib 3.10.6 已装)
- 业务报告 / PDF 嵌入用

新增:
- 1 import (matplotlib.use('Agg') backend)
- REGIME_COLORS 模块级 dict (与 v5.5 闭包 dict 镜像)
- 1 CLI flag --static-output PATH
- main() 末尾 1 行调用 build_static_bode_grid
- 1 新 test (PNG 字节头 + 文件大小校验)

其他 6 函数 + parse_args 签名 0 变化(只加 1 flag)。
72 → 73 tests pass。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: README §4.1.5 v5.6 footnote

**Files:**
- Modify: `backtrace/dynamics/README.md` (add §4.1.5 v5.6 section after §4.1.4 v5.5)

### Step 1: Find §4.1.4 anchor

```bash
cd c:/Users/yellow/mcp/qtTdx && grep -n "^### §4.1.4" backtrace/dynamics/README.md
```

Expected: shows line number of `### §4.1.4 v5.5 — Regime Color Coding`.

### Step 2: Add §4.1.5 v5.6 section

After the §4.1.4 section ends, append:

```markdown
### §4.1.5 v5.6 — Static 2D Grid PNG (matplotlib 导出)

v5.6 给同一份数据加**静态 PNG 导出** — 业务写报告 / 嵌 PDF 用:

- 函数: `build_static_bode_grid(pairs_per_date, omega_grid, output_path, dpi=100)`
- 布局: 2D 网格 (rows = unique asof_date, cols = |H|/∠H)
- 颜色: 与 v5.5 一致 (4 种 regime hex)
- 0 新依赖 (matplotlib 3.10.6 已装)
- CLI: `--static-output PATH` (默认 `backtrace/outputs/dynsys_si_freq_response_static.png`)

**v5.5 vs v5.6 关系**:
- v5.5 HTML: 交互, 浏览器拖 slider
- v5.6 PNG: 静态, 嵌 PDF / PPT
- 两者共用 `magnitude_phase` + `classify_response_type` + REGIME_COLORS → 0 重复, 0 不一致
```

### Step 3: Verify only README.md modified

```bash
cd c:/Users/yellow/mcp/qtTdx && git diff --stat <T1-commit> -- backtrace/dynamics/README.md
```

Expected: only `backtrace/dynamics/README.md` modified.

### Step 4: Commit

```bash
cd c:/Users/yellow/mcp/qtTdx && git add backtrace/dynamics/README.md && git commit -m "docs(dynamics): README §4.1.5 v5.6 — static 2D grid PNG footnote

记录 v5.6 给同一份数据加 matplotlib 静态 PNG 导出 (业务报告 / PDF 嵌入),
与 v5.5 HTML 互补, 0 新依赖, 颜色逻辑 0 重复。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Spec Coverage Checklist

After both tasks, verify:

- [ ] §1 问题 — Task 1 解决(静态 PNG 导出)
- [ ] §2 目标 — 1 新函数 + 1 CLI flag + 1 新 test, 其他 0 修改
- [ ] §3.1 架构 — `build_static_bode_grid` + matplotlib 2D 网格
- [ ] §3.2 2D 网格布局 — rows = dates, cols = |H| + ∠H
- [ ] §3.3 v5.6 修改范围 — 1 import + REGIME_COLORS dict + 1 新函数 + 1 CLI flag + main() 1 行
- [ ] §3.4 CLI 扩展 — `--static-output` 1 flag
- [ ] §3.5 输出 — 新增 PNG(~50-200KB)
- [ ] §3.6 测试 — 1 新 test, 72 → 73 tests pass
- [ ] §4 约束兑现 — 10 保护文件 0 修改
- [ ] §5 关键文件 — 1 函数 + 1 test + 1 README

## Self-Review Notes

1. **Spec coverage:** All 9 spec sections have a task. ✓
2. **Placeholder scan:** No "TBD" / "TODO" — all code values are exact (function signature, axis labels, dpi).
3. **Type consistency:** `build_static_bode_grid` signature matches spec §3.3. `REGIME_COLORS` dict keys match v5.5 closure dict keys.
4. **matplotlib.use('Agg')** MUST be called before `import matplotlib.pyplot` — non-interactive backend.
5. **Test scaffolding:** Test imports `pd` inline if not at top-level. Uses subprocess with `sys.executable` (not hardcoded path).
6. **Single-task vs multi-task split:** Task 1 = code + test, Task 2 = docs. Trivial docs split.
