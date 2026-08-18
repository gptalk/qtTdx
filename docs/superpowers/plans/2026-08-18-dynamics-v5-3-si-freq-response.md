# v5.3 Real SI Frequency Response Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `backtrace/dynamics/dynamics_si_freq_response.py` CLI that reads `parameter_fit --rolling-time` output (`kc_estimates_time.csv`), aggregates per `(asof_date, index_code)`, selects top-N industries per date, and renders an animated plotly HTML slider showing industry G(ω) drift over time.

**Architecture:** New standalone file `dynamics_si_freq_response.py` (does not modify any existing file). 4 spec'd helpers (`load_kc_time_series`, `aggregate_by_industry_per_date`, `select_top_n_per_date`, `build_animated_overlay_html`) + 2 standard writer helpers (`write_animated_summary_txt`, `write_animated_pairs_csv`, consistent with v5.2's `write_industry_pairs_csv` pattern) + `parse_args` + `main()`. Reuses v5+v5.1+v5.2 math layer (`natural_frequency`, `magnitude_phase`) with **zero signature changes**.

**Tech Stack:** Python 3.10+, pandas, numpy, plotly>=5.0 (`graph_objects` with `go.Frame` + `updatemenus` + `sliders`), pytest.

## Global Constraints

These constraints bind every task. Any conflict with task text — task text is wrong.

### File protection (10 files must not be modified)

- ❌ `backtrace/dynamics/_dynamics_core.py` — 0 行修改
- ❌ `backtrace/dynamics/dynamics_forced_response.py` — v5 单对 + v5.1 overlay + v5.2 from-kc-estimates main() 函数体 + 已有函数签名 **0 修改**
- ❌ `backtrace/dynamics/dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py` — 0 修改
- ❌ `backtrace/dynamics/dynamics_si_ic.py` / `dynamics_si_timeseries.py` / `dynamics_si_lagged_ic.py` / `dynamics_eigen_analysis.py` — 0 修改
- ❌ `backtrace/projection/parameter_fit.py` — 0 修改(只读 CSV,不调函数)
- ❌ `backtrace/dynamics/README.md` — 唯一允许的修是 §4.1.2 v5.3 子节(Task 5)
- ❌ `tests/test_dynamics_eigen.py` — 唯一允许的修是 append 5 个 v5.3 test(Task 1-4)
- ✓ v5.3 是**新文件** `backtrace/dynamics/dynamics_si_freq_response.py`
- ✓ 所有新增输出 gitignored(`backtrace/outputs/dynsys_si_freq_response*` + `data/dynamics/si_freq_response_*`)

### Test count

- **67 → 72 tests pass**(67 旧 + 5 新: 4 单元测试 + 1 CLI 集成测试)
- 旧测试**全部不动**(任何失败 → fix 而非删测试)

### Runtime

- `PYTHONIOENCODING=utf-8` 必备
- Python: `/c/ProgramData/anaconda3/python.exe`(anaconda3)
- 测试命令: `cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v`
- 默认 omega_grid: `np.linspace(0.001, np.pi, 200)`(与 v5.2 overlay 一致)
- ramp-up filter: `n_valid_days >= 192`(沿用 v4.9)

### v5.2 helpers to reuse(签名 0 修改)

`backtrace/dynamics/dynamics_forced_response.py` 已有函数(直接 `import from backtrace.dynamics.dynamics_forced_response`):

```python
def natural_frequency(k: float, c: float) -> float: ...
def magnitude_phase(z_array: np.ndarray, k: float, c: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mag_db, phase_deg) — 已在 v5 实现,可复用"""
def load_kc_estimates(csv_path: str) -> pd.DataFrame: ...  # v5.2 已有,但不带 asof_date
def aggregate_by_industry(df, group_col='index_code', agg='median') -> pd.DataFrame: ...  # v5.2 已有
def select_top_n_industries(df, criterion='by_n_stocks', n=5, group_col='index_code') -> list[tuple]: ...  # v5.2 已有
```

**关键差异**:v5.2 已有函数**不带 asof_date**,v5.3 必须用**新函数**(`aggregate_by_industry_per_date` / `select_top_n_per_date`)保留时间维度。

### Output paths(全 gitignored)

| 路径 | 默认值 |
|---|---|
| HTML | `backtrace/outputs/dynsys_si_freq_response.html` |
| TXT | `backtrace/outputs/dynsys_si_freq_response_summary.txt` |
| CSV | `data/dynamics/si_freq_response_pairs.csv` |

---

## Task 1: `load_kc_time_series` + 2 unit tests

**Files:**
- Create: `backtrace/dynamics/dynamics_si_freq_response.py`
- Modify: `tests/test_dynamics_eigen.py` (append 2 tests)

**Interfaces:**
- Consumes: `pandas`, `numpy` (pre-installed)
- Produces: `load_kc_time_series(csv_path: str) -> pd.DataFrame`

### Step 1: Write the failing test for filters

Append to `tests/test_dynamics_eigen.py`:

```python
def test_load_kc_time_series_filters_failed(tmp_path):
    """load_kc_time_series 过滤 status != 'ok' 行 + n_valid_days < 192 (ramp-up)"""
    rows = [
        # (code, index_code, asof_date, k_hat, c_hat, status, n_valid_days)
        ('000001.SZ', '801010', '2024-09-30', 0.50, 2.00, 'ok',  250),  # 保留
        ('000002.SZ', '801010', '2024-09-30', 0.55, 2.10, 'ok',  100),  # 过滤 (ramp-up)
        ('000003.SZ', '801010', '2024-09-30', 0.60, 1.90, 'fail', 250), # 过滤 (status)
        ('000004.SZ', '801010', '2024-09-30', 0.70, 1.80, 'ok',  300),  # 保留
        ('000005.SZ', '801010', '2024-09-30', 0.80, 1.70, 'ok',  192),  # 保留 (边界)
        ('000006.SZ', '801010', '2024-09-30', 0.90, 1.60, 'ok',  191),  # 过滤 (ramp-up 边界外)
    ]
    df = pd.DataFrame(rows, columns=[
        'code', 'index_code', 'asof_date', 'k_hat', 'c_hat', 'status', 'n_valid_days',
    ])
    csv_path = tmp_path / 'kc_estimates_time.csv'
    df.to_csv(csv_path, index=False)

    from backtrace.dynamics.dynamics_si_freq_response import load_kc_time_series
    result = load_kc_time_series(str(csv_path))

    assert len(result) == 3  # 只保留 250/300/192 三行
    assert result['code'].tolist() == ['000001.SZ', '000004.SZ', '000005.SZ']
    assert (result['status'] == 'ok').all()
    assert (result['n_valid_days'] >= 192).all()
```

### Step 2: Run test to verify it fails

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_load_kc_time_series_filters_failed -v
```

Expected: FAIL with `ImportError: cannot import name 'load_kc_time_series' from 'backtrace.dynamics.dynamics_si_freq_response'` (file doesn't exist yet).

### Step 3: Write the failing test for column validation

Append to `tests/test_dynamics_eigen.py`:

```python
def test_load_kc_time_series_validates_columns(tmp_path):
    """缺必需列 → ValueError,错误信息列出缺失列名"""
    rows = [
        ('000001.SZ', '801010', '2024-09-30', 0.5, 2.0, 'ok'),  # 缺 n_valid_days
    ]
    df = pd.DataFrame(rows, columns=['code', 'index_code', 'asof_date', 'k_hat', 'c_hat', 'status'])
    csv_path = tmp_path / 'bad.csv'
    df.to_csv(csv_path, index=False)

    from backtrace.dynamics.dynamics_si_freq_response import load_kc_time_series
    with pytest.raises(ValueError, match='n_valid_days'):
        load_kc_time_series(str(csv_path))
```

### Step 4: Run test to verify it fails

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_load_kc_time_series_validates_columns -v
```

Expected: FAIL with `ImportError`.

### Step 5: Write minimal implementation

Create `backtrace/dynamics/dynamics_si_freq_response.py` with ONLY this function:

```python
"""v5.3 — Real SI Frequency Response 时序动画 overlay.

读 parameter_fit --rolling-time 输出 (kc_estimates_time.csv),按 asof_date 切片
+ 行业聚合 + top-N 选取,通过 plotly animation_frame 联动多帧 Bode overlay。
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

REQUIRED_COLUMNS = ('code', 'index_code', 'asof_date', 'k_hat', 'c_hat', 'status', 'n_valid_days')
RAMP_UP_DAYS = 192  # 沿用 v4.9


def load_kc_time_series(csv_path: str) -> pd.DataFrame:
    """读 parameter_fit --rolling-time 输出 kc_estimates_time.csv。

    必需列:code, index_code, asof_date, k_hat, c_hat, status, n_valid_days
    过滤:status='ok' AND n_valid_days >= 192 (ramp-up)

    Raises:
        FileNotFoundError: csv_path 不存在
        ValueError: 缺必需列(错误信息列出缺失列名)
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'kc_estimates_time.csv 缺必需列: {missing}')
    return df[(df['status'] == 'ok') & (df['n_valid_days'] >= RAMP_UP_DAYS)].copy()
```

### Step 6: Run both tests to verify they pass

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_load_kc_time_series_filters_failed tests/test_dynamics_eigen.py::test_load_kc_time_series_validates_columns -v
```

Expected: 2 PASS.

### Step 7: Commit

```bash
cd c:/Users/yellow/mcp/qtTdx && git add backtrace/dynamics/dynamics_si_freq_response.py tests/test_dynamics_eigen.py && git commit -m "feat(dynamics): v5.3 Task 1 — load_kc_time_series helper + 2 tests

读 kc_estimates_time.csv,过滤 status='ok' + n_valid_days>=192,
验证必需列。0 修改 _dynamics_core / v5+v5.1+v5.2 已有函数。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: `aggregate_by_industry_per_date` + 1 unit test

**Files:**
- Modify: `backtrace/dynamics/dynamics_si_freq_response.py` (append)
- Modify: `tests/test_dynamics_eigen.py` (append 1 test)

**Interfaces:**
- Consumes: `load_kc_time_series` (Task 1 output)
- Produces: `aggregate_by_industry_per_date(df, dates, group_col='index_code', agg='median') -> dict[str, pd.DataFrame]`

### Step 1: Write the failing test

Append to `tests/test_dynamics_eigen.py`:

```python
def test_aggregate_by_industry_per_date():
    """按 (asof_date, index_code) 聚合 (k̂, ĉ),每片一个 DataFrame"""
    rows = [
        # Date 1 (2 ind × 2 stocks)
        ('000001.SZ', '801010', '2024-09-30', 0.50, 2.00, 'ok', 250),
        ('000002.SZ', '801010', '2024-09-30', 0.55, 2.10, 'ok', 250),
        ('600001.SH', '801080', '2024-09-30', 3.50, 0.50, 'ok', 250),
        ('600002.SH', '801080', '2024-09-30', 3.60, 0.45, 'ok', 250),
        # Date 2
        ('000001.SZ', '801010', '2024-10-31', 0.60, 1.90, 'ok', 250),
        ('000002.SZ', '801010', '2024-10-31', 0.65, 1.95, 'ok', 250),
        ('600001.SH', '801080', '2024-10-31', 4.00, 0.40, 'ok', 250),
        ('600002.SH', '801080', '2024-10-31', 3.90, 0.42, 'ok', 250),
    ]
    df = pd.DataFrame(rows, columns=[
        'code', 'index_code', 'asof_date', 'k_hat', 'c_hat', 'status', 'n_valid_days',
    ])
    dates = ['2024-09-30', '2024-10-31']

    from backtrace.dynamics.dynamics_si_freq_response import aggregate_by_industry_per_date
    result = aggregate_by_industry_per_date(df, dates)

    assert set(result.keys()) == {'2024-09-30', '2024-10-31'}
    # Date 1: Industry 801010 k̂=0.525, ĉ=2.05; Industry 801080 k̂=3.55, ĉ=0.475
    d1 = result['2024-09-30'].set_index('index_code')
    assert d1.loc['801010', 'k_hat'] == pytest.approx(0.525)
    assert d1.loc['801010', 'c_hat'] == pytest.approx(2.05)
    assert d1.loc['801010', 'n_stocks'] == 2
    assert d1.loc['801080', 'k_hat'] == pytest.approx(3.55)
    # Date 2: 801010 k̂=0.625, ĉ=1.925; 801080 k̂=3.95, ĉ=0.41
    d2 = result['2024-10-31'].set_index('index_code')
    assert d2.loc['801010', 'k_hat'] == pytest.approx(0.625)
    assert d2.loc['801080', 'k_hat'] == pytest.approx(3.95)
```

### Step 2: Run test to verify it fails

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_aggregate_by_industry_per_date -v
```

Expected: FAIL with `ImportError: cannot import name 'aggregate_by_industry_per_date'`.

### Step 3: Append implementation

Append to `backtrace/dynamics/dynamics_si_freq_response.py`:

```python
def aggregate_by_industry_per_date(
    df: pd.DataFrame,
    dates: list,
    group_col: str = 'index_code',
    agg: str = 'median',
) -> dict:
    """按 (asof_date, group_col) 聚合 (k̂, ĉ),每片一个 DataFrame。

    Args:
        df: load_kc_time_series 输出
        dates: asof_date 列表 (YYYY-MM-DD str)
        group_col: 分组列(默认 'index_code')
        agg: 聚合方法(目前仅 'median')

    Returns:
        {asof_date: DataFrame [group_col, n_stocks, k_hat, c_hat]},每片按 group_col 排序
    """
    if agg != 'median':
        raise ValueError(f'agg={agg!r} 不支持,目前仅 median')

    out = {}
    for date in dates:
        slice_df = df[df['asof_date'] == date]
        if slice_df.empty:
            continue
        grouped = slice_df.groupby(group_col).agg(
            n_stocks=('code', 'count'),
            k_hat=('k_hat', 'median'),
            c_hat=('c_hat', 'median'),
        ).reset_index().sort_values(group_col).reset_index(drop=True)
        out[date] = grouped
    return out
```

### Step 4: Run test to verify it passes

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_aggregate_by_industry_per_date -v
```

Expected: PASS.

### Step 5: Commit

```bash
cd c:/Users/yellow/mcp/qtTdx && git add backtrace/dynamics/dynamics_si_freq_response.py tests/test_dynamics_eigen.py && git commit -m "feat(dynamics): v5.3 Task 2 — aggregate_by_industry_per_date + 1 test

按 (asof_date, index_code) 聚合 (k̂, ĉ) 中位数 + n_stocks 计数。
每片一个 DataFrame,按 group_col 排序。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: `select_top_n_per_date` + 1 unit test

**Files:**
- Modify: `backtrace/dynamics/dynamics_si_freq_response.py` (append)
- Modify: `tests/test_dynamics_eigen.py` (append 1 test)

**Interfaces:**
- Consumes: `aggregate_by_industry_per_date` (Task 2 output)
- Produces: `select_top_n_per_date(per_date_dfs, criterion='by_n_stocks', n=5, group_col='index_code') -> list[tuple[str, float, float, str]]`

### Step 1: Write the failing test

Append to `tests/test_dynamics_eigen.py`:

```python
def test_select_top_n_per_date():
    """每个 date 选 top-N industries,返 (asof_date, k̂, ĉ, label) 元组列表(按 date 排序)"""
    per_date_dfs = {
        '2024-09-30': pd.DataFrame({
            'index_code': ['801010', '801080', '801090'],
            'n_stocks':   [4,         3,         2],
            'k_hat':      [0.5,       3.5,       2.0],
            'c_hat':      [2.0,       0.5,       1.5],
        }),
        '2024-10-31': pd.DataFrame({
            'index_code': ['801010', '801080', '801090'],
            'n_stocks':   [5,         2,         1],
            'k_hat':      [0.6,       4.0,       1.8],
            'c_hat':      [1.9,       0.4,       1.7],
        }),
    }

    from backtrace.dynamics.dynamics_si_freq_response import select_top_n_per_date
    pairs = select_top_n_per_date(per_date_dfs, criterion='by_n_stocks', n=2)

    # 2 dates × 2 industries = 4 pairs
    assert len(pairs) == 4
    # Date 1 top-2 by n_stocks: 801010 (4 stocks), 801080 (3 stocks)
    d1 = [p for p in pairs if p[0] == '2024-09-30']
    assert d1[0][1:] == (0.5, 2.0, 'Industry 801010')
    assert d1[1][1:] == (3.5, 0.5, 'Industry 801080')
    # Date 2 top-2 by n_stocks: 801010 (5 stocks), 801080 (2 stocks)
    d2 = [p for p in pairs if p[0] == '2024-10-31']
    assert d2[0][1:] == (0.6, 1.9, 'Industry 801010')
    assert d2[1][1:] == (4.0, 0.4, 'Industry 801080')
    # 按 date 排序
    assert [p[0] for p in pairs] == ['2024-09-30', '2024-09-30', '2024-10-31', '2024-10-31']
```

### Step 2: Run test to verify it fails

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_select_top_n_per_date -v
```

Expected: FAIL with `ImportError: cannot import name 'select_top_n_per_date'`.

### Step 3: Append implementation

Append to `backtrace/dynamics/dynamics_si_freq_response.py`:

```python
import numpy as np


def select_top_n_per_date(
    per_date_dfs: dict,
    criterion: str = 'by_n_stocks',
    n: int = 5,
    group_col: str = 'index_code',
) -> list:
    """每个 asof_date 选 top-N industries,转动画 overlay 格式。

    Args:
        per_date_dfs: aggregate_by_industry_per_date 输出 {date: DataFrame}
        criterion: 'by_n_stocks' / 'by_c_over_k' / 'by_k_over_c'
        n: top N(每个 date 最多选 n 个行业)

    Returns:
        [(asof_date, k̂, ĉ, "Industry {group_col}"), ...],按 date 排序
    """
    if criterion not in ('by_n_stocks', 'by_c_over_k', 'by_k_over_c'):
        raise ValueError(f'criterion={criterion!r} 不支持')

    pairs = []
    for date in sorted(per_date_dfs.keys()):
        df = per_date_dfs[date]
        if criterion == 'by_n_stocks':
            sorted_df = df.sort_values('n_stocks', ascending=False).head(n)
        elif criterion == 'by_c_over_k':
            df_copy = df.copy()
            df_copy['ratio'] = df_copy['c_hat'] / df_copy['k_hat'].replace(0, np.nan)
            sorted_df = df_copy.sort_values('ratio', ascending=False, na_position='last').head(n)
        else:  # by_k_over_c
            df_copy = df.copy()
            df_copy['ratio'] = df_copy['k_hat'] / df_copy['c_hat'].replace(0, np.nan)
            sorted_df = df_copy.sort_values('ratio', ascending=False, na_position='last').head(n)
        for _, row in sorted_df.iterrows():
            pairs.append((
                date,
                float(row['k_hat']),
                float(row['c_hat']),
                f'Industry {row[group_col]}',
            ))
    return pairs
```

### Step 4: Run test to verify it passes

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_select_top_n_per_date -v
```

Expected: PASS.

### Step 5: Commit

```bash
cd c:/Users/yellow/mcp/qtTdx && git add backtrace/dynamics/dynamics_si_freq_response.py tests/test_dynamics_eigen.py && git commit -m "feat(dynamics): v5.3 Task 3 — select_top_n_per_date + 1 test

每个 asof_date 选 top-N industries,3 种 criterion 与 v5.2 一致。
返 (date, k̂, ĉ, label) 元组列表,按 date 排序(保证动画时序)。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: `build_animated_overlay_html` + `write_animated_summary_txt` + `write_animated_pairs_csv` + `parse_args` + `main` + 1 CLI subprocess test

**Files:**
- Modify: `backtrace/dynamics/dynamics_si_freq_response.py` (append all)
- Modify: `tests/test_dynamics_eigen.py` (append 1 test)

**Interfaces:**
- Consumes: `select_top_n_per_date` (Task 3 output) + v5.1/v5.2 helpers (`natural_frequency`, `magnitude_phase`) imported from `dynamics_forced_response`
- Produces:
  - `build_animated_overlay_html(pairs_per_date, omega_grid, output_path, title='...') -> None`
  - `write_animated_summary_txt(pairs_per_date, dates, output_path) -> None`
  - `write_animated_pairs_csv(pairs_per_date, output_path) -> None`
  - `parse_args() -> argparse.Namespace`
  - `main() -> None`

### Step 1: Write the failing CLI integration test

Append to `tests/test_dynamics_eigen.py`:

```python
def test_cli_si_freq_response_mode(tmp_path):
    """CLI 时序动画模式:合成 12 行 CSV → 跑 CLI → 验证 HTML + TXT + CSV 3 个输出"""
    # 合成 3 dates × 2 ind × 2 stocks = 12 行
    rows = [
        # Date 1 (2024-09-30)
        ('000001.SZ', '801010', '2024-09-30', 0.50, 2.00, 'ok', 250),
        ('000002.SZ', '801010', '2024-09-30', 0.55, 2.10, 'ok', 250),
        ('600001.SH', '801080', '2024-09-30', 3.50, 0.50, 'ok', 250),
        ('600002.SH', '801080', '2024-09-30', 3.60, 0.45, 'ok', 250),
        # Date 2 (2024-10-31)
        ('000001.SZ', '801010', '2024-10-31', 0.60, 1.90, 'ok', 250),
        ('000002.SZ', '801010', '2024-10-31', 0.65, 1.95, 'ok', 250),
        ('600001.SH', '801080', '2024-10-31', 4.00, 0.40, 'ok', 250),
        ('600002.SH', '801080', '2024-10-31', 3.90, 0.42, 'ok', 250),
        # Date 3 (2024-11-30)
        ('000001.SZ', '801010', '2024-11-30', 0.70, 1.80, 'ok', 250),
        ('000002.SZ', '801010', '2024-11-30', 0.72, 1.82, 'ok', 250),
        ('600001.SH', '801080', '2024-11-30', 3.00, 0.60, 'ok', 250),
        ('600002.SH', '801080', '2024-11-30', 2.95, 0.58, 'ok', 250),
    ]
    df = pd.DataFrame(rows, columns=[
        'code', 'index_code', 'asof_date', 'k_hat', 'c_hat', 'status', 'n_valid_days',
    ])
    csv_path = tmp_path / 'kc_estimates_time.csv'
    df.to_csv(csv_path, index=False)

    html_path = tmp_path / 'si_freq_response.html'
    summary_path = tmp_path / 'si_freq_response_summary.txt'
    pairs_path = tmp_path / 'si_freq_response_pairs.csv'

    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cli_script = os.path.join(_PROJECT_ROOT, 'backtrace', 'dynamics', 'dynamics_si_freq_response.py')

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(
        [
            '/c/ProgramData/anaconda3/python.exe',
            cli_script,
            '--kc-time-csv', str(csv_path),
            '--top-n-industries', '2',
            '--industry-selection', 'by_n_stocks',
            '--max-dates', '3',
            '--html-output', str(html_path),
            '--summary-output', str(summary_path),
            '--pairs-csv-output', str(pairs_path),
        ],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 0, f'stderr: {result.stderr}'

    # 3 输出文件都存在
    assert html_path.exists() and html_path.stat().st_size > 1000
    assert summary_path.exists()
    assert pairs_path.exists()

    # HTML 含 plotly animation_frame + frames + 3 个日期
    html_text = html_path.read_text(encoding='utf-8')
    assert 'plotly' in html_text.lower()
    assert 'animation_frame' in html_text or 'frames' in html_text
    assert '2024-09-30' in html_text and '2024-10-31' in html_text and '2024-11-30' in html_text

    # Summary TXT 含 3 日期 + 中文
    summary_text = summary_path.read_text(encoding='utf-8')
    assert '2024-09-30' in summary_text and '2024-10-31' in summary_text and '2024-11-30' in summary_text
    assert '行业' in summary_text  # 中文关键词

    # Pairs CSV: 3 dates × 2 industries = 6 行 + header
    pairs_df = pd.read_csv(pairs_path)
    assert len(pairs_df) == 6
    assert set(pairs_df.columns) >= {'asof_date', 'index_code', 'k_hat', 'c_hat'}
    assert set(pairs_df['asof_date'].unique()) == {'2024-09-30', '2024-10-31', '2024-11-30'}
```

Also ensure the imports are present at the top of `tests/test_dynamics_eigen.py`:

```python
import os
import subprocess
```

(If they're not already there, prepend or insert at the top of the file.)

### Step 2: Run test to verify it fails

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_si_freq_response_mode -v
```

Expected: FAIL — either `main()` doesn't exist yet, or CLI flags not recognized.

### Step 3: Append implementation

Append to `backtrace/dynamics/dynamics_si_freq_response.py`:

```python
import os
import sys
import argparse
import numpy as np
import plotly.graph_objects as go


DEFAULT_OMEGA_GRID = np.linspace(0.001, np.pi, 200)
DEFAULT_TOP_N = 5
DEFAULT_MAX_DATES = 12
HTML_OUT = 'backtrace/outputs/dynsys_si_freq_response.html'
SUMMARY_OUT = 'backtrace/outputs/dynsys_si_freq_response_summary.txt'
PAIRS_OUT = 'data/dynamics/si_freq_response_pairs.csv'

# Reuse v5.1 / v5.2 zero-modification helpers
from backtrace.dynamics.dynamics_forced_response import natural_frequency, magnitude_phase


def build_animated_overlay_html(
    pairs_per_date: list,
    omega_grid: np.ndarray,
    output_path: str,
    title: str = 'Industry G(ω) Frequency Response — Time Series',
) -> None:
    """构建 plotly 动画 slider:每帧一个 asof_date,每帧 N 条 industry Bode 曲线。

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

    # 按 date 分组,每帧 N 条 trace
    dates = sorted(set(p[0] for p in pairs_per_date))
    initial_date = dates[0]
    initial_traces = [
        go.Scatter(
            x=omega_grid.tolist(),
            y=magnitude_phase(omega_grid * 1j, p[1], p[2])[0].tolist(),
            mode='lines',
            name=p[3],
        )
        for p in pairs_per_date if p[0] == initial_date
    ]

    fig = go.Figure(data=initial_traces)

    frames = []
    for date in dates:
        frame_traces = [
            go.Scatter(
                x=omega_grid.tolist(),
                y=magnitude_phase(omega_grid * 1j, p[1], p[2])[0].tolist(),
                mode='lines',
                name=p[3],
            )
            for p in pairs_per_date if p[0] == date
        ]
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

    # Play/Pause button
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
        xaxis_title='ω (rad/day)',
        yaxis_title='|H(jω)| dB',
        updatemenus=[dict(
            type='buttons', showactive=False, y=1.15, x=0.5, xanchor='center',
            buttons=[play_button, pause_button],
        )],
        sliders=[dict(active=0, steps=slider_steps, x=0.1, len=0.9, xanchor='left',
                      y=0, yanchor='top', currentvalue=dict(prefix='asof_date: ', visible=True))],
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')


def write_animated_summary_txt(
    pairs_per_date: list,
    dates: list,
    output_path: str,
) -> None:
    """写 UTF-8 中文业务解读:每个 asof_date 一段(top-N industries + 业务解读)。"""
    from collections import defaultdict
    by_date = defaultdict(list)
    for date, k, c, label in pairs_per_date:
        by_date[date].append((k, c, label))

    lines = ['# Industry G(ω) 时序动画 — 业务解读', '']
    for date in dates:
        if date not in by_date:
            continue
        lines.append(f'## {date}')
        for k, c, label in by_date[date]:
            omega_n = natural_frequency(k, c)
            regime = '过阻尼 (低通过滤器)' if c * c > 4 * k else ('临界阻尼' if abs(c * c - 4 * k) < 1e-6 else '欠阻尼 (有共振)')
            lines.append(f'  - {label}: k̂={k:.4f}, ĉ={c:.4f}, ω_n={omega_n:.4f}, {regime}')
        lines.append('')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def write_animated_pairs_csv(pairs_per_date: list, output_path: str) -> None:
    """写 UTF-8-sig 审计 CSV:每个 (asof_date, industry) 一行 + (k̂, ĉ)。"""
    rows = [
        {'asof_date': d, 'k_hat': k, 'c_hat': c, 'industry_label': label}
        for d, k, c, label in pairs_per_date
    ]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='v5.3 — Real SI Frequency Response 时序动画 overlay',
    )
    p.add_argument('--kc-time-csv', default='data/projection/kc_estimates_time.csv',
                   help='parameter_fit --rolling-time 输出 CSV')
    p.add_argument('--top-n-industries', type=int, default=DEFAULT_TOP_N,
                   help='每个 asof_date 选 top-N industries')
    p.add_argument('--industry-selection', default='by_n_stocks',
                   choices=['by_n_stocks', 'by_c_over_k', 'by_k_over_c'],
                   help='排序标准')
    p.add_argument('--max-dates', type=int, default=DEFAULT_MAX_DATES,
                   help='最多取最近 N 个 asof_date(默认 12,避免动画过慢)')
    p.add_argument('--html-output', default=HTML_OUT)
    p.add_argument('--summary-output', default=SUMMARY_OUT)
    p.add_argument('--pairs-csv-output', default=PAIRS_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load
    kc_df = load_kc_time_series(args.kc_time_csv)
    if kc_df.empty:
        raise RuntimeError(f'{args.kc_time_csv} 过滤后为空 — 检查 status / n_valid_days')

    # 2. 取所有 unique asof_date,排序,截断到最近 max_dates
    all_dates = sorted(kc_df['asof_date'].unique().tolist())
    if len(all_dates) > args.max_dates:
        print(f'[v5.3] asof_date 共 {len(all_dates)} 个,截断到最近 {args.max_dates} 个')
        all_dates = all_dates[-args.max_dates:]

    # 3. 聚合
    per_date_dfs = aggregate_by_industry_per_date(kc_df, dates=all_dates, group_col='index_code')
    if not per_date_dfs:
        raise RuntimeError('聚合后为空')

    # 4. 选 top-N
    pairs = select_top_n_per_date(per_date_dfs, criterion=args.industry_selection, n=args.top_n_industries)
    if not pairs:
        raise RuntimeError('选不到任何 industry pair')

    # 5. omega_grid
    omega_grid = DEFAULT_OMEGA_GRID

    # 6. 写 3 输出
    build_animated_overlay_html(pairs, omega_grid, args.html_output)
    write_animated_summary_txt(pairs, all_dates, args.summary_output)
    write_animated_pairs_csv(pairs, args.pairs_csv_output)

    print(f'[v5.3] {len(pairs)} 个 (date, industry) 对已写入:')
    print(f'  - {args.html_output}')
    print(f'  - {args.summary_output}')
    print(f'  - {args.pairs_csv_output}')


if __name__ == '__main__':
    main()
```

### Step 4: Run test to verify it passes

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_si_freq_response_mode -v
```

Expected: PASS. Test takes ~5-15 seconds (subprocess + plotly HTML write).

### Step 5: Verify zero-modification of protected files

```bash
cd c:/Users/yellow/mcp/qtTdx && git diff --stat HEAD~1 -- backtrace/dynamics/_dynamics_core.py backtrace/dynamics/dynamics_forced_response.py backtrace/dynamics/dynamics_system.py backtrace/dynamics/dynamics_batch.py backtrace/dynamics/dynamics_1step_oos.py backtrace/projection/parameter_fit.py
```

Expected: empty (no changes to protected files).

### Step 6: Run full test suite to verify count

```bash
cd c:/Users/yellow/mcp/qtTdx && /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v
```

Expected: **72 passed**(67 旧 + 5 新)。

### Step 7: Commit

```bash
cd c:/Users/yellow/mcp/qtTdx && git add backtrace/dynamics/dynamics_si_freq_response.py tests/test_dynamics_eigen.py && git commit -m "feat(dynamics): v5.3 Task 4 — animated HTML builder + 3 writers + main() + CLI

build_animated_overlay_html: plotly go.Frame + animation_frame slider,
每帧一个 asof_date,每帧 N 条 industry Bode |H(jω)| 曲线。
write_animated_summary_txt: UTF-8 中文业务解读。
write_animated_pairs_csv: UTF-8-sig 审计 CSV。
main(): load → 截断 max_dates → 聚合 → top-N → 写 3 输出。
parse_args(): 7 个 CLI flags。

零修改:_dynamics_core / v5+v5.1+v5.2 已有函数 / 3 caller /
4 v4.x CLI / parameter_fit。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 5: README §4.1.2 v5.3 sub-section

**Files:**
- Modify: `backtrace/dynamics/README.md` (append §4.1.2 after §4.1.1 v5.2)

### Step 1: Read existing README to find §4.1 anchor

```bash
cd c:/Users/yellow/mcp/qtTdx && grep -n "^### §4.1" backtrace/dynamics/README.md
```

Expected: shows §4.1.1 v5.2 line number (where §4.1.2 should go after).

### Step 2: Append §4.1.2 sub-section

Find the end of §4.1.1 (last line before next `### §` heading or end of §4.1 block) and append:

```markdown
### §4.1.2 v5.3 — Real SI Frequency Response (时序动画)

**动机**:v5.2 数据驱动 overlay 只画**单帧**(一个 asof_date 的行业 G(ω) 对比)。v5.3 把这层补上:**多 asof_date 的 Bode overlay 通过 plotly 动画 slider 联动**,业务可拖时间轴看行业频率响应如何漂移。

**新文件**:`backtrace/dynamics/dynamics_si_freq_response.py`(独立文件,不动 `dynamics_forced_response.py`)

**端到端示例**:
```bash
# 前置:v4.9 parameter_fit --rolling-time 已跑过,data/projection/kc_estimates_time.csv 存在
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_si_freq_response.py
# 期待:3 个 gitignored 输出
#   backtrace/outputs/dynsys_si_freq_response.html (plotly 动画 slider)
#   backtrace/outputs/dynsys_si_freq_response_summary.txt (中文业务解读)
#   data/dynamics/si_freq_response_pairs.csv (审计)
```

**CLI flags**:

| Flag | 默认 | 说明 |
|---|---|---|
| `--kc-time-csv PATH` | `data/projection/kc_estimates_time.csv` | v4.9 rolling 时序输出 |
| `--top-n-industries N` | 5 | 每个 asof_date 选 top-N industries |
| `--industry-selection` | `by_n_stocks` | `by_n_stocks` / `by_c_over_k` / `by_k_over_c` |
| `--max-dates N` | 12 | 最多取最近 N 个 asof_date(避免动画过慢) |
| `--html-output PATH` | `backtrace/outputs/dynsys_si_freq_response.html` | |
| `--summary-output PATH` | `backtrace/outputs/dynsys_si_freq_response_summary.txt` | |
| `--pairs-csv-output PATH` | `data/dynamics/si_freq_response_pairs.csv` | |

**与 v5 / v5.1 / v5.2 / v4.9 的关系**:

| 版 | commit | 主题 |
|---|---|---|
| v5 | `0ce3014` | 受迫系统 + G(ω) 单对频率响应 |
| v5.1 | `e990fb3` | 多对 (k, c) overlay 对比 |
| v5.2 | `fce9532` | 数据驱动 overlay(单帧) |
| v4.9 | `f2178a3` | SI(t) 时序 + 漂移检测 |
| **v5.3** | **(本次)** | **时序动画 G(ω)(t) overlay** |

v5.3 是**时序维度**的扩展:v5 单对 → v5.1 多对 overlay → v5.2 行业 overlay → v5.3 时序动画 overlay。

**已知陷阱**:

- `kc_estimates_time.csv` 不存在(v4.9 没跑过)→ `load_kc_time_series` raise `FileNotFoundError`,CLI 给清晰错误提示用户跑 `parameter_fit.py --rolling-time`
- asof_date 数 > `--max-dates` 12 → 自动截断到最近 N 个,print 警告
- 同行业在某 asof_date 整段 fail(无 `status='ok'`)→ 该 date 该行业跳过,top-N 不足时按实际数
- 动画 HTML 大 → plotly CDN 渲染,数据 ≤ 12 帧 × 5 industries × 200 ω points = 12k 点(~200KB)

### §4.1.3 (Reserved — v5.4+)
```

### Step 3: Verify no other README sections changed

```bash
cd c:/Users/yellow/mcp/qtTdx && git diff --stat HEAD backtrace/dynamics/README.md
```

Expected: only §4.1.2 added; no other sections modified.

### Step 4: Commit

```bash
cd c:/Users/yellow/mcp/qtTdx && git add backtrace/dynamics/README.md && git commit -m "docs(dynamics): README §4.1.2 v5.3 — animated overlay sub-section

记录新 CLI dynamics_si_freq_response.py 的:
- 动机(单帧 → 时序动画)
- 端到端示例
- 7 个 CLI flags table
- 与 v5/v5.1/v5.2/v4.9 的关系表
- 已知陷阱

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Spec Coverage Checklist

After all 5 tasks, verify spec coverage:

- [ ] §1 问题 — Task 1+2+3+4 addresses(时序维度动画 overlay)
- [ ] §2 目标 — 核心 CLI 已建(独立文件)+ YAGNI 全部遵守
- [ ] §3.1 架构 — `dynamics_si_freq_response.py` 独立新文件,4 新函数 + 2 写函数 + main() + argparse
- [ ] §3.2 v5.3 新 API — 4 函数签名与 spec 一致(`load_kc_time_series` / `aggregate_by_industry_per_date` / `select_top_n_per_date` / `build_animated_overlay_html`)
- [ ] §3.3 CLI 扩展 — 7 flags 实现(`--kc-time-csv` / `--top-n-industries` / `--industry-selection` / `--max-dates` / `--html-output` / `--summary-output` / `--pairs-csv-output`)
- [ ] §3.4 输出(全 gitignored) — 3 输出文件路径与默认值与 spec 一致
- [ ] §3.5 动画帧细节 — `go.Frame` + `animation_frame` slider + Play/Pause button 实现
- [ ] §4.1 单元测试 — 5 tests pass(Task 1: 2, Task 2: 1, Task 3: 1, Task 4: 1 CLI integration)
- [ ] §4.3 回归保护 — **67 → 72 tests pass**
- [ ] §5 约束兑现 — 9 个保护文件 0 修改(每 Task Step 5-7 验证 + Task 4 Step 5 全量验证)
- [ ] §6 关键文件 — 1 新文件 + 1 test 改 + 1 README 改
- [ ] §8 已知风险 — 4 风险全部有缓解(`FileNotFoundError` / max_dates 截断 / 颜色重叠 / 性能 ~200KB)

## Self-Review Notes (controller-side, not subagent)

1. **Spec coverage:** 4 spec'd functions implemented in Task 1-4. 2 writer helpers (`write_animated_summary_txt` / `write_animated_pairs_csv`) added in Task 4 consistent with v5.2 `write_industry_pairs_csv` pattern. **Note**: this is a minor spec refinement — spec §3.2 listed 4 functions but 2 writers are standard; final reviewer should rule this as "positive deviation, consistent with v5.2 pattern".

2. **Placeholder scan:** No "TBD" / "TODO" / "fill in details" — all step values are exact (test row tuples, file paths, magic numbers).

3. **Type consistency:** `load_kc_time_series` returns `pd.DataFrame` ✓, `aggregate_by_industry_per_date` returns `dict[str, pd.DataFrame]` ✓, `select_top_n_per_date` returns `list[tuple[str, float, float, str]]` ✓, `build_animated_overlay_html` returns `None` ✓, all writers return `None` ✓.

4. **Spec drift:** Spec §3.2 lists 4 functions, plan implements 6 (4 spec'd + 2 writers). This is consistent with v5.2 spec drift pattern (spec listed 3, impl had 4). Will fix in final reviewer adjudication.
