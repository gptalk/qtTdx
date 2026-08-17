# v4.4 — (1,4) bar chart label enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (1,4) bar chart x-axis 显示 `电力(881459.SH)` 格式人类可读标签,与文本汇总输出一致。

**Architecture:** 抽 `_industry_name_lookup(sw2_path)` 私有 helper(sector_code → sector_name)。`write_text_summary` 接受 `name_lookup` 可选参数;`main()` 在 bar chart 上复用同一个 lookup 构建 `industry_label` 列。tickangle=-30 加可读性。

**Tech Stack:** Python 3.13 / pandas / plotly / pytest / tsfresh 全栈环境(`/c/ProgramData/anaconda3/python.exe`)

## Global Constraints

- 数学层 `_dynamics_core.py` 0 行修改
- 3 caller(`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`)0 行修改
- `analyze_eigenvalues` / `simulate_trajectory` 函数签名不变
- 输出全部 gitignored(`data/` + `backtrace/outputs/`)
- `write_text_summary` 旧调用方式(不传 `name_lookup`)仍要能跑 —— 内部 fallback 到 helper
- 中文 / UTF-8:`PYTHONIOENCODING=utf-8` 必备

## 现状(实施前必读)

- `backtrace/dynamics/dynamics_eigen_analysis.py:206` — `write_text_summary` 函数(已含 `sw2_members_path` 参数,commit 5b9e788)
- `backtrace/dynamics/dynamics_eigen_analysis.py:230-239` — `write_text_summary` 内联 `pd.read_csv('data/sw2/members.csv')` 块 → 应抽出
- `backtrace/dynamics/dynamics_eigen_analysis.py:486-505` — (1,4) bar chart, `x=agg_l1['industry_l1']`(`industry_l1` 是 sector_code `881459.SH`)
- `backtrace/dynamics/dynamics_eigen_analysis.py:513` — `tickangle=-30` 已设
- `backtrace/dynamics/dynamics_eigen_analysis.py:638` — `write_text_summary` 调用点,已传 `sw2_members_path=args.sw2_members`
- `tests/test_dynamics_eigen.py:428` 行,目前 26 个测试函数
- `backtrace/dynamics/README.md` §3.5 是 v4.3 节(commit fbaff0f)
- `docs/superpowers/specs/2026-08-16-dynamics-system-design.md` §3.6 是 v4.3 子节(commit 05910a3)

---

## Task 1: 抽 helper + bar chart label 增强 + 1 测试 + 2 文档注脚

**Files:**
- Modify: `backtrace/dynamics/dynamics_eigen_analysis.py` (4 处:`+_industry_name_lookup` helper, `write_text_summary` 内联删, `write_text_summary` 签名加 `name_lookup`, `main()` bar chart x-axis + helper 调用)
- Modify: `tests/test_dynamics_eigen.py` (新增 `test_industry_name_lookup` 函数)
- Modify: `backtrace/dynamics/README.md` (§3.5 末尾加 v4.4 注脚)
- Modify: `docs/superpowers/specs/2026-08-16-dynamics-system-design.md` (§3.6 末尾加 v4.4 注脚)

**Interfaces:**
- 新增: `_industry_name_lookup(sw2_members_path: str) -> dict[str, str]`
  - 输入:sw2/members.csv 路径(str)
  - 输出:sector_code → sector_name 反查表(`Dict[str, str]`);文件不存在/缺列返回空 dict
- 扩展: `write_text_summary(..., name_lookup: dict[str, str] | None = None)` — 旧 7 位置参数不变,新增第 8 可选 kwarg
- `main()` 内部: `agg_l1_label['industry_label'] = agg_l1_label['industry_l1'].map(lambda c: f'{name_lookup.get(c, c)}({c})' if c else '(未知)')`

### Step 1.1: 给 `_industry_name_lookup` 写失败测试

打开 `tests/test_dynamics_eigen.py`,在文件末尾追加以下测试函数(在所有现有测试之后,3 个空行 + 这个函数):

```python
def test_industry_name_lookup(tmp_path):
    """_industry_name_lookup 3 个 case: 正常 / 缺文件 / 缺关键列。"""
    from dynamics.dynamics_eigen_analysis import _industry_name_lookup

    # 1. 正常:写 mock sw2/members.csv,验证返回 dict
    sw2 = tmp_path / 'sw2.csv'
    sw2.write_text(
        'sector_code,sector_name,member_code\n'
        '881459.SH,电力,600000.SH\n'
        '881001.SH,银行,600001.SH\n',
        encoding='utf-8',
    )
    result = _industry_name_lookup(str(sw2))
    assert result == {'881459.SH': '电力', '881001.SH': '银行'}

    # 2. 缺文件:返回空 dict
    assert _industry_name_lookup(str(tmp_path / 'nope.csv')) == {}

    # 3. 缺关键列:返回空 dict
    bad = tmp_path / 'bad.csv'
    bad.write_text('foo,bar\n1,2\n', encoding='utf-8')
    assert _industry_name_lookup(str(bad)) == {}
```

### Step 1.2: 运行测试,确认失败

```bash
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_industry_name_lookup -v
```

**Expected:** `ImportError: cannot import name '_industry_name_lookup' from 'dynamics.dynamics_eigen_analysis'`

### Step 1.3: 实现 `_industry_name_lookup` helper

打开 `backtrace/dynamics/dynamics_eigen_analysis.py`,在 `write_text_summary` 函数定义之前(约 L205,紧邻 `def write_text_summary(` 之上,留 2 空行)插入:

```python
def _industry_name_lookup(sw2_members_path: str = 'data/sw2/members.csv') -> dict:
    """sector_code → sector_name 反查表。

    文件不存在 / 缺关键列 → 返回空 dict(让 caller 走 fallback)。
    """
    if not os.path.exists(sw2_members_path):
        print(f'[eigen] ⚠ sw2_members 不存在: {sw2_members_path},行业 label 走 fallback')
        return {}
    df = pd.read_csv(sw2_members_path, dtype={'sector_code': str})
    if 'sector_code' not in df.columns or 'sector_name' not in df.columns:
        print(f'[eigen] ⚠ sw2_members 缺关键列: {sw2_members_path}')
        return {}
    return df.drop_duplicates('sector_code').set_index('sector_code')['sector_name'].to_dict()
```

### Step 1.4: 跑测试,确认通过

```bash
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_industry_name_lookup -v
```

**Expected:** PASS

### Step 1.5: 更新 `write_text_summary` 签名 + 删内联读 csv

打开 `backtrace/dynamics/dynamics_eigen_analysis.py:206-213`,把签名改成:

```python
def write_text_summary(
    summary_df: pd.DataFrame,
    cls_count: Counter,
    agg_l1: pd.DataFrame,
    l1_threshold: int,
    agg_ex: pd.DataFrame,
    path: str,
    sw2_members_path: str = 'data/sw2/members.csv',
    name_lookup: dict | None = None,
) -> None:
    """写 dynsys_eigen_summary.txt 纯文本汇总(UTF-8)。

    行业 label 增强:agg_l1.groupby key 是 industry_l1(sector_code),
    这里再读 sw2/members.csv 把 sector_name 拼过来,显示更可读。
    sw2_members_path 默认即仓库内 data/sw2/members.csv。
    name_lookup 若 caller 已构造好(测试隔离 / main() 复用)则用之,否则 fallback 到 helper。
    """
```

然后把 `backtrace/dynamics/dynamics_eigen_analysis.py:232-241` 的内联块:

```python
    # 行业 label 增强:industry_l1(sector_code) → industry_l2(sector_name)
    sb = pd.DataFrame()
    try:
        sb = pd.read_csv(sw2_members_path, dtype={'sector_code': str})
        if 'sector_code' in sb.columns and 'sector_name' in sb.columns:
            name_lookup = sb.drop_duplicates('sector_code').set_index('sector_code')['sector_name'].to_dict()
        else:
            name_lookup = {}
    except FileNotFoundError:
        name_lookup = {}
```

替换成:

```python
    # 行业 label 增强:industry_l1(sector_code) → industry_l2(sector_name)
    # 优先用 caller 传入的 lookup(避免重复读 csv),否则 fallback 到 helper
    if name_lookup is None:
        name_lookup = _industry_name_lookup(sw2_members_path)
```

### Step 1.6: 更新 `main()` bar chart x-axis 用 `industry_label`

打开 `backtrace/dynamics/dynamics_eigen_analysis.py`,在 `agg_l1, l1_threshold = aggregate_by_industry(summary_df)` 之后(约 L484,紧邻下一行,bar chart `fig.add_trace(go.Bar(...))` 之前),插入:

```python
    # 行业 label 增强(sector_code → sector_name),(1,4) bar chart + 文本汇总共用
    name_lookup = _industry_name_lookup(args.sw2_members)
    agg_l1_label = agg_l1.copy()
    agg_l1_label['industry_label'] = agg_l1_label['industry_l1'].map(
        lambda c: f'{name_lookup.get(c, c)}({c})' if c else '(未知)'
    )
```

然后把 `write_text_summary` 调用(约 L638-641)改成传 `name_lookup`:

```python
    write_text_summary(
        summary_df, cls_count, agg_l1, l1_threshold, agg_ex, DEFAULT_TXT_OUTPUT,
        sw2_members_path=args.sw2_members, name_lookup=name_lookup,
    )
```

最后改 bar chart x-axis(约 L488),把 `x=agg_l1['industry_l1']` 改成 `x=agg_l1_label['industry_label']`:

```python
            go.Bar(
                x=agg_l1_label['industry_label'],
                y=agg_l1['rho_median'],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=agg_l1['rho_p75'] - agg_l1['rho_median'],
                    arrayminus=agg_l1['rho_median'] - agg_l1['rho_p25'],
                    color='black',
                    thickness=1.5,
                    width=4,
                ),
                marker_color='steelblue',
                name=f'行业 top10 (n≥{l1_threshold})',
                text=[f"n={n}<br>ρ={r:.2f}" for n, r in zip(agg_l1['n_stocks'], agg_l1['rho_median'])],
                hovertemplate='<b>%{x}</b><br>ρ 中位数: %{y:.3f}<br>%{text}<extra></extra>',
                showlegend=False,
            ),
```

**注意:** `tickangle=-30` 已在 L513 设置,无需重复改。

### Step 1.7: 跑全部 dynamics 测试,确认 27 个通过

```bash
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v
```

**Expected:** 27 passed (26 旧 + 1 新 `test_industry_name_lookup`)

### Step 1.8: 端到端冒烟(验证 bar chart label)

```bash
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_eigen_analysis.py --limit 50
```

**Expected:**
- exit 0
- stdout 末尾打印 HTML 路径
- `backtrace/outputs/dynsys_eigen_summary.txt` 仍显示 `电力(881459.SH)` 格式(无回归)
- `backtrace/outputs/dynsys_eigen.html` (1,4) bar chart x-axis 显示 `电力(881459.SH)` 格式(肉眼可见)

### Step 1.9: 更新 README §3.5 注脚

打开 `backtrace/dynamics/README.md`,在第 3.5 节标题(由 fbaff0f 加的)末尾、下一节之前,追加:

```markdown
**v4.4 (2026-08-17)**: (1,4) bar chart x-axis 也升级到 `电力(881459.SH)` 格式(文本汇总原已支持),通过抽出 `_industry_name_lookup` helper 复用 lookup。
```

### Step 1.10: 更新 spec §3.6 注脚

打开 `docs/superpowers/specs/2026-08-16-dynamics-system-design.md`,在 §3.6 末尾(由 2944f7a 重排)追加:

```markdown
### 3.7 v4.4 bar chart label 增强(2026-08-17)— 复用 v4.3 lookup

(1,4) bar chart x-axis 由 `industry_l1` (sector_code `881459.SH`) 升级到 `industry_label` (`电力(881459.SH)`)。
文本汇总 v4.3 已支持,v4.4 通过 `_industry_name_lookup(sw2_path)` 私有 helper 复用,消除两份输出的视觉不一致。
1 新测试 + 2 文件改动,数学层 / 3 caller 零修改。
```

### Step 1.11: Commit

```bash
cd "C:\Users\yellow\mcp\qtTdx"
git add backtrace/dynamics/dynamics_eigen_analysis.py tests/test_dynamics_eigen.py backtrace/dynamics/README.md docs/superpowers/specs/2026-08-16-dynamics-system-design.md
git commit -m "feat(dynamics): v4.4 — (1,4) bar chart label 增强 (T3.2 收尾)

- 抽 _industry_name_lookup(sw2_path) 私有 helper
- write_text_summary 加 name_lookup 可选 kwarg(旧调用兼容)
- main() bar chart x-axis 改用 industry_label 列
- 文本汇总无回归,HTML bar chart 视觉一致
- 1 新测试(test_industry_name_lookup 3 cases)
- README §3.5 + spec §3.7 各加 1 段注脚
- 数学层 / 3 caller 零修改,27 tests pass"
```

---

## 显式不做

- ❌ T3.1 HTML 5.8MB > 2-4MB 预算(继续 deferred,可接受)
- ❌ T3.3 exchange asc vs industry desc 排序方向
- ❌ T3.4 text 2 decimals vs hover 3 decimals
- ❌ T3.5 aggregate_by_industry 0-row fallback 区分
- ❌ v4.5 (k,c) 相图 + 7 状态颜色
- ❌ v6 受迫系统 + G(ω) 频率响应

## 验证清单

- [ ] Step 1.1: 测试代码写入
- [ ] Step 1.2: 测试失败(ImportError)
- [ ] Step 1.3: helper 实现
- [ ] Step 1.4: 单测试通过
- [ ] Step 1.5: write_text_summary 改签名 + 删内联
- [ ] Step 1.6: main() bar chart x-axis 改
- [ ] Step 1.7: 27 tests pass
- [ ] Step 1.8: 端到端冒烟 exit 0
- [ ] Step 1.9: README 注脚
- [ ] Step 1.10: spec 注脚
- [ ] Step 1.11: commit

## 风险

| 风险 | 缓解 |
|---|---|
| `write_text_summary` 加 `name_lookup` 关键字参数可能干扰 positional 调用 | 全部旧 caller 用 keyword,未传 `name_lookup` 默认 None → 内部 fallback |
| bar chart tickangle=-30 + 10 个长标签可能截断 | `automargin=True` 是备选(本次先不加,若肉眼可见截断再加) |
| `_industry_name_lookup` 重复读 csv | 用 `name_lookup` 模式避免:helper 算一次,main() 复用给两边 |
