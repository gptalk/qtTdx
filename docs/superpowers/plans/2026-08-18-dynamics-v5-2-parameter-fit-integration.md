# v5.2 parameter_fit Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 v5.1 overlay 框架基础上,加 `--from-kc-estimates` CLI flag,从 `parameter_fit` 的 `kc_estimates.csv` 自动读数据 → 按行业聚合 → 选 top-N → 喂给 v5.1 `bode_overlay` + `write_overlay_summary`。

**Architecture:**
- 复用 v5.1 已有函数(`bode_overlay` / `write_overlay_summary` / `parse_overlay_pairs`)**0 修改**
- 新增 3 个 helper 函数(`load_kc_estimates` / `aggregate_by_industry` / `select_top_n_industries`)+ 1 个 CSV writer(`write_industry_pairs_csv`)
- 在 `main()` 的 v5.1 if-return 之后插入 v5.2 if-return,3 个分支互斥:`--overlay` (v5.1) → `--from-kc-estimates` (v5.2) → else 单对模式 (v5)
- 测试加在 `tests/test_dynamics_eigen.py` 末尾
- 输出全 gitignored,新增 1 个 CSV 文件记录选中行业

**Tech Stack:** Python 3.x / numpy / pandas / plotly(沿用 v5.1)

## Global Constraints

[v5 + v5.1 沿用 + v5.2 新增]

- `_dynamics_core.py` 0 行修改
- v5 + v5.1 已有函数(`transfer_function` / `natural_frequency` / `magnitude_phase` / `classify_response_type` / `is_in_schur_wedge` / `bode_plot` / `stability_heatmap` / `write_summary` / `bode_overlay` / `write_overlay_summary` / `parse_overlay_pairs`)签名 0 修改
- 3 caller(`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`)0 行修改
- 4 v4.x CLI(`dynamics_si_ic.py` / `dynamics_si_timeseries.py` / `dynamics_si_lagged_ic.py` / `dynamics_eigen_analysis.py`)0 修改
- `parameter_fit.py` 0 修改(只读 CSV,不调函数)
- v5 单对模式 main() 函数体 0 修改
- v5.1 `--overlay` 模式 main() 分支 0 修改
- 互斥规则:`--from-kc-estimates` 与 `--overlay` 不能同时传;同时传 → argparse 报错退出
- 新增输出全 gitignored:`backtrace/outputs/dynsys_bode_overlay.html` + `dynsys_bode_overlay_summary.txt` + `dynsys_industry_overlay_pairs.csv`
- 61 → 66 tests pass(61 旧 + 5 新)
- Python: `/c/ProgramData/anaconda3/python.exe`
- `PYTHONIOENCODING=utf-8` 必备
- 函数命名沿用 v5.1 风格(snake_case / docstring 中文)

---

### Task 1: `load_kc_estimates()` 函数 + 2 单元测试

**Files:**
- Modify: `backtrace/dynamics/dynamics_forced_response.py` (在 `parse_overlay_pairs` 之后,`main()` 之前)
- Modify: `tests/test_dynamics_eigen.py` (末尾新增 2 tests)

**Interfaces:**
- Consumes: CSV file path(`kc_estimates.csv`)
- Produces: `load_kc_estimates(csv_path)` → `pd.DataFrame`(只保留 status='ok' 的行)

- [ ] **Step 1: 写失败的测试**

在 `tests/test_dynamics_eigen.py` 末尾新增:

```python
def test_load_kc_estimates_filters_failed(tmp_path):
    """load_kc_estimates 过滤 status != 'ok' 的行。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    csv_path = tmp_path / "kc.csv"
    csv_path.write_text(
        "code,index_code,k_hat,c_hat,status\n"
        "600000.SH,801010,0.5,2.0,ok\n"
        "600001.SH,801010,0.6,1.9,ok\n"
        "600002.SH,801020,2.0,1.5,ok\n"
        "600003.SH,801020,2.1,1.4,fail\n",  # ← 应被过滤
        encoding='utf-8',
    )
    df = DFR.load_kc_estimates(str(csv_path))
    assert len(df) == 3, f"应过滤 fail 行,剩 3 行,得 {len(df)}"
    assert "600003.SH" not in df['code'].values


def test_load_kc_estimates_validates_columns(tmp_path):
    """缺必需列 → ValueError。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    csv_path = tmp_path / "kc.csv"
    csv_path.write_text(
        "code,k_hat,c_hat\n"  # ← 缺 index_code + status
        "600000.SH,0.5,2.0\n",
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match="index_code"):
        DFR.load_kc_estimates(str(csv_path))
```

- [ ] **Step 2: 跑测试,验证失败**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_load_kc_estimates_filters_failed tests/test_dynamics_eigen.py::test_load_kc_estimates_validates_columns -v
```

期望:FAIL — `AttributeError: module 'backtrace.dynamics.dynamics_forced_response' has no attribute 'load_kc_estimates'`

- [ ] **Step 3: 实现 `load_kc_estimates`**

在 `backtrace/dynamics/dynamics_forced_response.py` 中,`parse_overlay_pairs` 函数之后,`main()` 之前新增:

```python
def load_kc_estimates(csv_path):
    """读 parameter_fit kc_estimates.csv,验证必需列,过滤失败行。

    必需列:code, index_code, k_hat, c_hat, status(其他列可选)

    Returns:
        DataFrame,只保留 status='ok' 的行(过滤拟合失败的)

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 必需列缺失
    """
    import os
    REQUIRED_COLS = ['code', 'index_code', 'k_hat', 'c_hat', 'status']
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"kc_estimates CSV 不存在: {csv_path}\n"
            f"提示:python backtrace/projection/parameter_fit.py 先跑出 (k̂, ĉ)"
        )
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"kc_estimates CSV 缺必需列: {missing}")
    df = df[df['status'] == 'ok'].reset_index(drop=True)
    return df
```

- [ ] **Step 4: 跑测试,验证通过**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_load_kc_estimates_filters_failed tests/test_dynamics_eigen.py::test_load_kc_estimates_validates_columns -v
```

期望:2 PASS

- [ ] **Step 5: Commit**

```bash
git add backtrace/dynamics/dynamics_forced_response.py tests/test_dynamics_eigen.py
git commit -m "feat(dynamics): v5.2 — load_kc_estimates() + 2 unit tests"
```

---

### Task 2: `aggregate_by_industry()` 函数 + 1 单元测试

**Files:**
- Modify: `backtrace/dynamics/dynamics_forced_response.py` (在 `load_kc_estimates` 之后)
- Modify: `tests/test_dynamics_eigen.py` (末尾新增 1 test)

**Interfaces:**
- Consumes: `load_kc_estimates` 输出 DataFrame
- Produces: `aggregate_by_industry(df, group_col, agg)` → DataFrame (列: `group_col`, `n_stocks`, `k_hat`, `c_hat`)

- [ ] **Step 1: 写失败的测试**

```python
def test_aggregate_by_industry_median():
    """agg='median' 对 (k̂, ĉ) 中位数聚合 + n_stocks 计数。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    import pandas as pd
    df = pd.DataFrame({
        'code': ['A1', 'A2', 'A3', 'B1', 'B2'],
        'index_code': ['801010', '801010', '801010', '801020', '801020'],
        'k_hat': [0.5, 0.6, 0.7, 2.0, 2.1],
        'c_hat': [2.0, 1.9, 2.1, 1.5, 1.4],
    })
    agg_df = DFR.aggregate_by_industry(df, group_col='index_code', agg='median')
    assert len(agg_df) == 2
    # 801010 中位数 k=0.6, c=2.0, n=3
    row_a = agg_df[agg_df['index_code'] == '801010'].iloc[0]
    assert row_a['n_stocks'] == 3
    assert abs(row_a['k_hat'] - 0.6) < 1e-9
    assert abs(row_a['c_hat'] - 2.0) < 1e-9
    # 801020 中位数 k=2.05, c=1.45, n=2
    row_b = agg_df[agg_df['index_code'] == '801020'].iloc[0]
    assert row_b['n_stocks'] == 2
    assert abs(row_b['k_hat'] - 2.05) < 1e-9
    assert abs(row_b['c_hat'] - 1.45) < 1e-9
```

- [ ] **Step 2: 跑测试,验证失败**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_aggregate_by_industry_median -v
```

期望:FAIL — `AttributeError: ... has no attribute 'aggregate_by_industry'`

- [ ] **Step 3: 实现 `aggregate_by_industry`**

```python
def aggregate_by_industry(df, group_col='index_code', agg='median'):
    """按行业聚合 (k̂, ĉ)。

    Args:
        df: load_kc_estimates 输出
        group_col: 分组列(默认 index_code)
        agg: 聚合方法("median" / "mean"),默认 median(抗极端值)

    Returns:
        DataFrame 列:[group_col, n_stocks, k_hat, c_hat]
        按 group_col 排序
    """
    if agg not in ('median', 'mean'):
        raise ValueError(f"agg 必须 'median' 或 'mean',得 '{agg}'")
    agg_fn = np.median if agg == 'median' else np.mean
    rows = []
    for grp, sub in df.groupby(group_col):
        rows.append({
            group_col: grp,
            'n_stocks': len(sub),
            'k_hat': float(agg_fn(sub['k_hat'].values)),
            'c_hat': float(agg_fn(sub['c_hat'].values)),
        })
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)
```

- [ ] **Step 4: 跑测试,验证通过**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_aggregate_by_industry_median -v
```

期望:1 PASS

- [ ] **Step 5: Commit**

```bash
git add backtrace/dynamics/dynamics_forced_response.py tests/test_dynamics_eigen.py
git commit -m "feat(dynamics): v5.2 — aggregate_by_industry() + 1 unit test"
```

---

### Task 3: `select_top_n_industries()` 函数 + 2 单元测试

**Files:**
- Modify: `backtrace/dynamics/dynamics_forced_response.py` (在 `aggregate_by_industry` 之后)
- Modify: `tests/test_dynamics_eigen.py` (末尾新增 2 tests)

**Interfaces:**
- Consumes: `aggregate_by_industry` 输出 DataFrame
- Produces: `select_top_n_industries(df, criterion, n, group_col)` → `list[tuple[float, float, str]]`

- [ ] **Step 1: 写失败的测试**

```python
def test_select_top_n_by_n_stocks():
    """criterion='by_n_stocks' 按股票数降序排,选 top N。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    import pandas as pd
    df = pd.DataFrame({
        'index_code': ['A', 'B', 'C'],
        'n_stocks': [10, 5, 2],
        'k_hat': [1.0, 2.0, 3.0],
        'c_hat': [1.5, 1.5, 1.5],
    })
    pairs = DFR.select_top_n_industries(df, criterion='by_n_stocks', n=2)
    assert len(pairs) == 2
    # A (10 stocks) 第一, B (5 stocks) 第二
    assert pairs[0] == (1.0, 1.5, 'Industry A')
    assert pairs[1] == (2.0, 1.5, 'Industry B')


def test_select_top_n_by_c_over_k():
    """criterion='by_c_over_k' 按 c/k 比降序排,选 top N。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    import pandas as pd
    df = pd.DataFrame({
        'index_code': ['A', 'B', 'C'],
        'n_stocks': [5, 5, 5],
        'k_hat': [0.5, 2.0, 1.0],
        'c_hat': [2.0, 1.5, 1.0],  # c/k: 4.0, 0.75, 1.0
    })
    pairs = DFR.select_top_n_industries(df, criterion='by_c_over_k', n=2)
    # A (c/k=4.0) 第一, C (c/k=1.0) 第二
    assert pairs[0][2] == 'Industry A'
    assert pairs[1][2] == 'Industry C'
```

- [ ] **Step 2: 跑测试,验证失败**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_select_top_n_by_n_stocks tests/test_dynamics_eigen.py::test_select_top_n_by_c_over_k -v
```

期望:2 FAIL

- [ ] **Step 3: 实现 `select_top_n_industries`**

```python
def select_top_n_industries(df, criterion='by_n_stocks', n=5, group_col='index_code'):
    """从聚合 DataFrame 选 top-N 行业,转 v5.1 overlay 格式。

    Args:
        df: aggregate_by_industry 输出
        criterion: 排序标准
            - "by_n_stocks": 按股票数降序(最多成分股的行业)
            - "by_c_over_k": 按 c/k 比降序(最过阻尼,稳定)
            - "by_k_over_c": 按 k/c 比降序(最欠阻尼,危险)
        n: top N
        group_col: label 用 group_col 值,前缀 "Industry "

    Returns:
        [(k̂, ĉ, label), ...] — 直接喂给 bode_overlay
    """
    df_sorted = df.copy()
    if criterion == 'by_n_stocks':
        df_sorted = df_sorted.sort_values('n_stocks', ascending=False)
    elif criterion == 'by_c_over_k':
        df_sorted['_ratio'] = df_sorted['c_hat'] / df_sorted['k_hat'].replace(0, np.nan)
        df_sorted = df_sorted.sort_values('_ratio', ascending=False)
    elif criterion == 'by_k_over_c':
        df_sorted['_ratio'] = df_sorted['k_hat'] / df_sorted['c_hat'].replace(0, np.nan)
        df_sorted = df_sorted.sort_values('_ratio', ascending=False)
    else:
        raise ValueError(f"criterion 必须 by_n_stocks / by_c_over_k / by_k_over_c,得 '{criterion}'")
    df_sorted = df_sorted.head(n)
    pairs = []
    for _, row in df_sorted.iterrows():
        label = f"Industry {row[group_col]}"
        pairs.append((float(row['k_hat']), float(row['c_hat']), label))
    return pairs
```

- [ ] **Step 4: 跑测试,验证通过**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_select_top_n_by_n_stocks tests/test_dynamics_eigen.py::test_select_top_n_by_c_over_k -v
```

期望:2 PASS

- [ ] **Step 5: Commit**

```bash
git add backtrace/dynamics/dynamics_forced_response.py tests/test_dynamics_eigen.py
git commit -m "feat(dynamics): v5.2 — select_top_n_industries() + 2 unit tests"
```

---

### Task 4: `write_industry_pairs_csv()` + `--from-kc-estimates` CLI flag + main() 中间分支 + 1 CLI 集成测试

**Files:**
- Modify: `backtrace/dynamics/dynamics_forced_response.py` (加 `write_industry_pairs_csv` + 4 new CLI flags + main() v5.2 分支)
- Modify: `tests/test_dynamics_eigen.py` (末尾新增 1 CLI test)

**Interfaces:**
- 新增:`write_industry_pairs_csv(pairs, agg_df, output_path)` → None(写 UTF-8 CSV)
- 新增 CLI flags:`--from-kc-estimates PATH` / `--top-n N` (default 5) / `--industry-agg {median|mean}` (default median) / `--select-criterion {...}` (default by_n_stocks) / `--industry-pairs-csv PATH`

**关键设计**:v5.2 分支在 main() 中插入位置 — 在 v5.1 `--overlay` 分支 **之后**,在单对 main() 函数体 **之前**(原 v5.1 分支已有 `return`,所以插入位置不会与 v5 单对冲突):

```python
def main():
    args = parse_args()

    # v5.1 --overlay 分支(已有,不变)
    if args.overlay:
        ...
        return

    # v5.2 --from-kc-estimates 分支(新增)
    if args.from_kc_estimates:
        ...
        return

    # else: v5 单对模式(函数体不动)
    omega_grid = DEFAULT_OMEGA_GRID
    ...
```

- [ ] **Step 1: 写失败的 CLI 集成测试**

```python
def test_cli_from_kc_estimates_mode(tmp_path):
    """CLI --from-kc-estimates 模式读合成 CSV → 选 top-N → 写 overlay + 行业 CSV。"""
    import subprocess
    # 写合成 kc_estimates.csv 到 cwd
    cwd = tmp_path
    csv_path = cwd / "kc_estimates.csv"
    csv_path.write_text(
        "code,index_code,k_hat,c_hat,status\n"
        "600000.SH,801010,0.5,2.0,ok\n"
        "600001.SH,801010,0.6,1.9,ok\n"
        "600002.SH,801010,0.7,2.1,ok\n"
        "600010.SH,801020,2.0,1.5,ok\n"
        "600011.SH,801020,2.1,1.4,ok\n"
        "600020.SH,801030,3.5,0.5,ok\n",
        encoding='utf-8',
    )
    out_html = cwd / "overlay.html"
    out_txt = cwd / "overlay_summary.txt"
    out_pairs = cwd / "industry_pairs.csv"
    result = subprocess.run([
        sys.executable,
        "backtrace/dynamics/dynamics_forced_response.py",
        "--from-kc-estimates", str(csv_path),
        "--top-n", "2",
        "--industry-agg", "median",
        "--select-criterion", "by_n_stocks",
        "--overlay-html", str(out_html),
        "--overlay-summary-txt", str(out_txt),
        "--industry-pairs-csv", str(out_pairs),
    ], capture_output=True, text=True, cwd=str(cwd))
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out_html.exists()
    assert out_txt.exists()
    assert out_pairs.exists()
    pairs_content = out_pairs.read_text(encoding='utf-8')
    # 801010 有 3 只股票(最多),801020 有 2 只,801030 有 1 只
    # by_n_stocks top-2: 801010 + 801020
    assert "801010" in pairs_content
    assert "801020" in pairs_content
```

**注意**:test 用 `cwd=tmp_path`,这样 overlay / summary / pairs 输出落到 tmp_path(用 `--overlay-html` 等覆盖默认路径)。需用绝对 script 路径,见 Task 4 v5.1 的 fix pattern。

为简化,可以用 **绝对 script 路径**:

```python
import os
_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "backtrace", "dynamics", "dynamics_forced_response.py")
```

然后 subprocess 用 `[_SCRIPT, ...]`。

- [ ] **Step 2: 跑测试,验证失败**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_from_kc_estimates_mode -v
```

期望:FAIL — `--from-kc-estimates` flag 未定义

- [ ] **Step 3: 修改 `parse_args` 加 5 个新 flags**

在 `backtrace/dynamics/dynamics_forced_response.py` 中,`parse_args()` 函数 `--overlay-summary-txt` 之后新增:

```python
    # v5.2 数据驱动模式 flags
    p.add_argument("--from-kc-estimates", default="",
                   help="v5.2 数据驱动:parameter_fit kc_estimates.csv 路径(与 --overlay 互斥)")
    p.add_argument("--top-n", type=int, default=5,
                   help="v5.2 选 top-N 行业(默认 5)")
    p.add_argument("--industry-agg", choices=['median', 'mean'], default='median',
                   help="v5.2 行业聚合方法(默认 median)")
    p.add_argument("--select-criterion", choices=['by_n_stocks', 'by_c_over_k', 'by_k_over_c'],
                   default='by_n_stocks',
                   help="v5.2 排序标准(默认 by_n_stocks)")
    p.add_argument("--industry-pairs-csv",
                   default=os.path.join(HTML_OUT_DIR, "dynsys_industry_overlay_pairs.csv"),
                   help="v5.2 选中行业 CSV 输出路径")
```

- [ ] **Step 4: 在 main() 中插入 v5.2 分支**

在 `main()` 中,v5.1 `if args.overlay:` 分支(及其 `return`)之后,单对 `omega_grid = DEFAULT_OMEGA_GRID` 之前,新增:

```python
    # v5.2 数据驱动分支
    if args.from_kc_estimates:
        kc_df = load_kc_estimates(args.from_kc_estimates)
        agg_df = aggregate_by_industry(kc_df, group_col='index_code', agg=args.industry_agg)
        if len(agg_df) == 0:
            raise RuntimeError(
                f"kc_estimates.csv 没有 status='ok' 的行。请检查 {args.from_kc_estimates}"
            )
        pairs = select_top_n_industries(agg_df, criterion=args.select_criterion,
                                         n=args.top_n, group_col='index_code')
        if len(pairs) < args.top_n:
            print(f'[v5.2] 警告:实际只 {len(pairs)} 个行业(请求 {args.top_n})')
        omega_grid_overlay = np.linspace(0.001, np.pi, 200)
        bode_overlay(omega_grid_overlay, pairs, args.overlay_html,
                     title=f'v5.2 Industry G(ω) — {args.select_criterion} top-{len(pairs)}')
        write_overlay_summary(omega_grid_overlay, pairs, args.overlay_summary_txt)
        # 写行业 pairs CSV(审计用)
        write_industry_pairs_csv(pairs, agg_df, args.industry_pairs_csv)
        print(f'[v5.2] {len(pairs)} 个行业已写入 {args.overlay_html} + {args.industry_pairs_csv}')
        return
    # else: 单对模式(v5 既有)
```

- [ ] **Step 5: 实现 `write_industry_pairs_csv`**

在 `select_top_n_industries` 之后,`main()` 之前新增:

```python
def write_industry_pairs_csv(pairs, agg_df, output_path):
    """写选中行业的 (k̂, ĉ) + label + 行业股票数到 UTF-8 CSV(审计用)。

    Args:
        pairs: select_top_n_industries 输出 [(k, c, label), ...]
        agg_df: aggregate_by_industry 输出 DataFrame
        output_path: 输出 CSV 路径
    """
    import re
    rows = []
    for k, c, label in pairs:
        # 从 label "Industry XXX" 提取行业 code
        m = re.match(r'Industry\s+(.*)', label)
        industry_code = m.group(1) if m else label
        # 从 agg_df 查 n_stocks
        match = agg_df[agg_df['index_code'] == industry_code]
        n_stocks = int(match['n_stocks'].iloc[0]) if len(match) > 0 else 0
        rows.append({
            'industry_code': industry_code,
            'k_hat': k,
            'c_hat': c,
            'n_stocks': n_stocks,
        })
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding='utf-8-sig')
```

- [ ] **Step 6: 跑测试,验证通过**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_from_kc_estimates_mode -v
```

期望:1 PASS

- [ ] **Step 7: 跑全套测试,验证 66 tests pass**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v
```

期望:66 PASS(61 旧 + 5 新)

- [ ] **Step 8: 手动验证 v5 + v5.1 模式未被破坏**

```bash
# v5 单对
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_forced_response.py --k 2.0 --c 1.5 \
    --grid-csv "$HOME/v52_v5_grid.csv" --stability-csv "$HOME/v52_v5_stab.csv" \
    --bode-html "$HOME/v52_v5_bode.html" --heatmap-html "$HOME/v52_v5_heat.html" \
    --summary-txt "$HOME/v52_v5_sum.txt"

# v5.1 --overlay
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_forced_response.py \
    --overlay "0.5,2.0,A; 2.0,1.5,B" \
    --overlay-html "$HOME/v52_v51_bode.html" \
    --overlay-summary-txt "$HOME/v52_v51_sum.txt"
```

(Windows 可用 `%TEMP%\v52_xxx.ext` 或 `C:/Users/yellow/AppData/Local/Temp/v52_xxx.ext`。)

期望:两种模式都跑通,无 stderr,输出各自对应。

- [ ] **Step 9: Commit**

```bash
git add backtrace/dynamics/dynamics_forced_response.py tests/test_dynamics_eigen.py
git commit -m "feat(dynamics): v5.2 — --from-kc-estimates CLI flag + main() 分支 + 1 集成测试"
```

---

### Task 5: README §4.1.1 + 最终 commit

**Files:**
- Modify: `backtrace/dynamics/README.md` (§4.1 v5.1 子节末尾新增 §4.1.1 v5.2 子节)

- [ ] **Step 1: 定位 §4.1 v5.1 子节末尾**

```bash
grep -n "^### \|^#### " backtrace/dynamics/README.md | head -30
```

找 `### §4.1 v5.1` 章节末尾(应该在文件最后部分)。

- [ ] **Step 2: 追加 §4.1.1 v5.2 子节**

在 README 中 §4.1 v5.1 子节末尾新增:

```markdown
### §4.1.1 v5.2 — parameter_fit Integration (数据驱动 overlay)

把 `parameter_fit.py` 的 `kc_estimates.csv` 数据接到 v5.1 overlay,从"对比框架"升级到"真实行业 G(ω) 对比"。

#### 新增 CLI flag

| flag | 类型 | 说明 |
|---|---|---|
| `--from-kc-estimates` | path | parameter_fit kc_estimates.csv 路径(与 `--overlay` 互斥) |
| `--top-n` | int | 选 top-N 行业(默认 5) |
| `--industry-agg` | str | 行业聚合方法:`median` / `mean`(默认 median) |
| `--select-criterion` | str | 排序标准:`by_n_stocks` / `by_c_over_k` / `by_k_over_c`(默认 by_n_stocks) |
| `--industry-pairs-csv` | path | 选中行业 CSV 输出(默认 `backtrace/outputs/dynsys_industry_overlay_pairs.csv`) |

#### 端到端示例

```bash
# 前提:parameter_fit.py 已跑过,data/projection/kc_estimates.csv 存在
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_forced_response.py \
    --from-kc-estimates data/projection/kc_estimates.csv \
    --top-n 5 \
    --select-criterion by_n_stocks
# 期待:3 个 gitignored 输出
#   backtrace/outputs/dynsys_bode_overlay.html
#   backtrace/outputs/dynsys_bode_overlay_summary.txt
#   backtrace/outputs/dynsys_industry_overlay_pairs.csv
```

#### 排序标准

| criterion | 含义 | 业务用途 |
|---|---|---|
| `by_n_stocks` | 按行业股票数降序 | 默认:成分股最多的行业(覆盖广) |
| `by_c_over_k` | 按 c/k 比降序 | 最过阻尼 / 最稳 / 低通过滤器 |
| `by_k_over_c` | 按 k/c 比降序 | 最欠阻尼 / 共振风险高 / 危险行业 |

#### 与 v5.1 的关系

v5.2 是 v5.1 的**数据接入层** — v5.1 提供"对比框架",v5.2 提供"真实数据 → 框架输入"转换。两者组合 = 业务可决策的行业 G(ω) 对比。

#### 与 parameter_fit 的接口契约(只读)

```python
# v5.2 期望 kc_estimates.csv 的列:
# code: str — 股票代码
# index_code: str — 申万二级代码
# k_hat: float — OLS 拟合恢复系数
# c_hat: float — OLS 拟合阻尼系数
# status: str — "ok" / "fail" (过滤 fail 行)
#
# 其他列可选。**不调任何 parameter_fit 函数** — CSV 是 stable 接口
```

#### 已知陷阱

- `kc_estimates.csv` 必须先存在(跑 `parameter_fit.py`),否则 FileNotFoundError
- 必需列缺失 → ValueError(列出缺失列名)
- `--from-kc-estimates` 与 `--overlay` 互斥,不能同时传
- `select_top_n_industries` 只取实际存在的行业数,如果少于 `--top-n` 会 print 警告
```

- [ ] **Step 3: Commit**

```bash
git add backtrace/dynamics/README.md
git commit -m "docs(dynamics): README §4.1.1 — v5.2 data-driven overlay 使用 + 接口契约"
```

---

## Final state

- **6 commits**(Task 1-5 + spec + plan)
- **66 tests pass**(61 旧 + 5 新)
- **0 修改**:`_dynamics_core.py` / v5 + v5.1 已有函数 / 3 caller / 4 v4.x CLI / `parameter_fit.py` / v5 单对 main() / v5.1 overlay 分支
- **新增**:4 函数(`load_kc_estimates` / `aggregate_by_industry` / `select_top_n_industries` / `write_industry_pairs_csv`)+ 5 个 CLI flags + 3 个 gitignored 输出

## Self-Review Checklist

- [x] Spec 覆盖:每个 spec 章节都有对应 task
- [x] 无 placeholder / TODO / TBD
- [x] 类型一致:`load_kc_estimates` / `aggregate_by_industry` / `select_top_n_industries` / `write_industry_pairs_csv` 跨 task 签名一致
- [x] 函数命名沿用 v5.1 风格(snake_case + 中文 docstring)
- [x] 测试与函数一一对应
- [x] main() 改动互斥:`--overlay` (v5.1)→ `--from-kc-estimates` (v5.2)→ else 单对 (v5),3 个分支 early-return
- [x] 所有输出路径 gitignored(`backtrace/outputs/`)