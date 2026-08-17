# v4.4 — (1,4) bar chart label enhancement

**Date:** 2026-08-17
**Parent plan:** v4.3 全市场经验分布
**Status:** Draft

## Background

v4.3 在 `dynamics_eigen_analysis.py` 加了 2×4 HTML 报告,把"行业 ρ 中位数 top10"放进 (1,4) 子图。但 x-axis 直接显示 `industry_l1` (sector_code,如 `881459.SH`),与文本汇总里 `电力(881459.SH)` 的可读命名格式不一致 — T3.2 deferred finding。

文本汇总已经做了行业 label 增强(读 `data/sw2/members.csv` 把 sector_code 映射到 sector_name),但 bar chart 没有复用这段逻辑。视觉上同一份数据的两种呈现方式不一致。

## Goal

**让 (1,4) bar chart x-axis 显示 `电力(881459.SH)` 格式的人类可读标签,与文本汇总输出一致。**

## Scope

**In scope:**
- 抽出 `_industry_name_lookup(sw2_members_path)` 私有 helper
- `write_text_summary` 接受 `name_lookup` 参数(或内部调用 helper),消除内联重复
- `main()` bar chart 用增强后的 `industry_label` 列,加 tick 旋转(`tickangle=-30`)
- 1 个新单元测试

**Out of scope(继续 deferred):**
- T3.1 HTML 5.8MB > 2-4MB 预算(spec 端不修,实际可接受)
- T3.3 exchange asc vs industry desc 排序方向
- T3.4 text 2 decimals vs hover 3 decimals 一致性
- T3.5 aggregate_by_industry 0-row fallback 区分
- v4.5 roadmap 项((k,c) 相图 + 7 状态颜色)
- v6 受迫系统 + G(ω) 频率响应

## Design

### 1. 新 helper `_industry_name_lookup`

```python
def _industry_name_lookup(sw2_members_path: str = 'data/sw2/members.csv') -> dict[str, str]:
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

### 2. `write_text_summary` 签名扩展

```python
def write_text_summary(
    summary_df: pd.DataFrame,
    cls_count: Counter,
    agg_l1: pd.DataFrame,
    l1_threshold: int,
    agg_ex: pd.DataFrame,
    path: str,
    sw2_members_path: str = 'data/sw2/members.csv',
    name_lookup: dict[str, str] | None = None,  # ← 新增
) -> None:
    """... 行业 label 增强:industry_l1(sector_code) → industry_l2(sector_name) ..."""
    ...
    # 优先用 caller 传入的 lookup,caller 没传则自己算
    if name_lookup is None:
        name_lookup = _industry_name_lookup(sw2_members_path)
    # (删掉之前内联的 pd.read_csv('data/sw2/members.csv') 块)
    ...
```

**向后兼容:** `name_lookup=None` → 内部调用 helper(保留"自给自足"行为);显式传 → 调用方拥有控制权(便于测试隔离)。

### 3. `main()` bar chart x-axis

```python
# 现有代码(伪):
fig.add_trace(go.Bar(x=agg_l1['industry_l1'], y=agg_l1['rho_median'], ...), row=1, col=4)

# 改为:
name_lookup = _industry_name_lookup(args.sw2_members)
agg_l1_label = agg_l1.copy()
agg_l1_label['industry_label'] = agg_l1_label['industry_l1'].map(
    lambda c: f'{name_lookup.get(c, c)}({c})' if c else '(未知)'
)
fig.add_trace(go.Bar(x=agg_l1_label['industry_label'], y=agg_l1['rho_median'], ...), row=1, col=4)
```

**xaxis4 配置(若之前没设):**
```python
fig.update_xaxes(tickangle=-30, row=1, col=4)
```

### 4. 测试

`tests/test_dynamics_eigen.py` 加 1 个新测试:

```python
def test_industry_name_lookup(tmp_path, monkeypatch):
    """_industry_name_lookup 正常 + 缺文件 + 缺列 3 个 case。"""
    # 1. 正常:写 mock sw2/members.csv,验证返回 dict
    sw2 = tmp_path / 'sw2.csv'
    sw2.write_text('sector_code,sector_name,member_code\n'
                   '881459.SH,电力,600000.SH\n'
                   '881001.SH,银行,600001.SH\n', encoding='utf-8')
    assert _industry_name_lookup(str(sw2)) == {'881459.SH': '电力', '881001.SH': '银行'}
    
    # 2. 缺文件:返回空 dict
    assert _industry_name_lookup(str(tmp_path / 'nope.csv')) == {}
    
    # 3. 缺关键列:返回空 dict
    bad = tmp_path / 'bad.csv'
    bad.write_text('foo,bar\n1,2\n', encoding='utf-8')
    assert _industry_name_lookup(str(bad)) == {}
```

**现有测试 `test_html_2x4_layout_and_text_summary` 不需要改动**:它已经 mock sw2_members 文件,改完后行为仍正确。

### 5. 与上一轮 fix round 的衔接

fix round 1(`commit 5b9e788`)已经把 `args.sw2_members` 贯穿到 `write_text_summary`。v4.4 在此基础上:
- 抽出 helper 让两份输出共用
- bar chart 也接 `args.sw2_members`

## Constraints (carryover)

- **数学层 `_dynamics_core.py` 零修改**
- **3 caller(`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`) 零修改**
- `analyze_eigenvalues` / `simulate_trajectory` 函数签名不变
- 输出全部 gitignored

## Files

| 文件 | 改动量 |
|---|---|
| `backtrace/dynamics/dynamics_eigen_analysis.py` | +18 / -11 行 |
| `tests/test_dynamics_eigen.py` | +18 / 0 行 |
| `docs/superpowers/specs/2026-08-16-dynamics-system-design.md` | §3.6 末尾加 1 段 v4.4 注脚 |
| `backtrace/dynamics/README.md` | §3.5 末尾加 1 段 v4.4 注脚 |

## Verification

```bash
# 1. 单元测试
cd c:/Users/yellow/mcp/qtTdx
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_eigen.py -v
# → 27 passed (26 + 1 new)

# 2. 端到端冒烟(--limit 50,看 bar chart x-axis)
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_eigen_analysis.py --limit 50
# → 在浏览器打开 backtrace/outputs/dynsys_eigen.html
#   (1,4) bar chart x-axis 显示 "电力(881459.SH)" 格式,不是 "881459.SH"

# 3. 文本汇总回归
cat backtrace/outputs/dynsys_eigen_summary.txt | grep "ρ_med"
# → 仍然显示 "电力(881459.SH)" 格式,无回归
```

## Risk

| 风险 | 缓解 |
|---|---|
| `write_text_summary` 改签名(加 `name_lookup` 关键字) | 旧 callers 不传也兼容(默认 None → 内部算) |
| 26 旧测试中若有依赖 `write_text_summary` 位置参数顺序 | 全部用 keyword 传,不受影响 |
| x-axis tick 旋转后被截断 | tickangle=-30,10 个标签应足够;若被截断再加 `automargin=True` |
