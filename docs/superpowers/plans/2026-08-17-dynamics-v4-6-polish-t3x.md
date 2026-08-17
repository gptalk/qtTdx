# v4.6 — Polish T3.3/T3.4/T3.5 deferred findings

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 v4.3 final review 留下的 3 个 minor findings — exchange sort 方向不一致(T3.3),bar chart text decimal precision(T3.4),0-row fallback 区分(T3.5)。

**Architecture:** `dynamics_eigen_analysis.py` 3 处小改 + 1 个新测试。T3.5 涉及 `aggregate_by_industry` 返回值语义(threshold=0 表示无数据),其他 2 处是直接 fix。

**Tech Stack:** Python 3.13 / pandas / numpy / plotly / pytest / tsfresh 全栈环境(`/c/ProgramData/anaconda3/python.exe`)

## Global Constraints

- 数学层 `_dynamics_core.py` 0 行修改
- 3 caller(`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`)0 行修改
- `analyze_eigenvalues` / `simulate_trajectory` 函数签名不变
- 输出全部 gitignored(`data/` + `backtrace/outputs/`)
- `PYTHONIOENCODING=utf-8` 必备;Windows 用 `/c/ProgramData/anaconda3/python.exe`
- 30 tests pass 目标(29 旧 + 1 新)

## 现状(实施前必读)

- `backtrace/dynamics/dynamics_eigen_analysis.py:163-189` — `aggregate_by_industry(df, min_stocks=50, fallback_min=30)` 返回 `(agg, threshold)`。T3.5: 两阈值都 < 5 行时返回 fallback_min 但 agg 可能是 0 行,无 "no data" 区分
- `backtrace/dynamics/dynamics_eigen_analysis.py:191-204` — `aggregate_by_exchange(df)` 排序 `sort_values('rho_median')` (asc)。T3.3: 行业是 desc,交易所 asc,视觉不一致
- `backtrace/dynamics/dynamics_eigen_analysis.py:611` — bar chart text `f"n={n}<br>ρ={r:.2f}"` (2 decimals)。T3.4: 与 hovertemplate `.3f` 不一致
- `backtrace/dynamics/dynamics_eigen_analysis.py:718` — exchange bar chart 同样 `.2f` 问题
- `tests/test_dynamics_eigen.py` 现有 29 测试(`test_industry_aggregation_rho_median` L265 已经覆盖 3-industry 触发 fallback_min 路径)
- 现有 `test_exchange_split_correctness` 用 `dict(zip(...,))` 断言,排序改变不影响

---

## Task 1: T3.3 / T3.4 / T3.5 三处 polish + 1 测试

**Files:**
- Modify: `backtrace/dynamics/dynamics_eigen_analysis.py` (3 处:`aggregate_by_exchange` sort desc, `aggregate_by_industry` 0-row 分支, 2 个 `.2f` → `.3f`)
- Modify: `tests/test_dynamics_eigen.py` (新增 `test_aggregate_by_industry_no_data`)

**Interfaces:**
- `aggregate_by_industry` 返回阈值新增语义: `threshold=0` 表示"无数据"(empty agg),`threshold>0` 表示"data 存在但都 < 阈值"
- `aggregate_by_exchange` 排序方向: `rho_median` 降序(与 industry 一致)

### Step 1.1: 给 T3.5 写失败测试

打开 `tests/test_dynamics_eigen.py`,在文件末尾追加:

```python
def test_aggregate_by_industry_no_data():
    """T3.5: 0-row fallback 区分 — 无任何行业时返回 threshold=0(无数据)

    构造 0 行 df 应返回 (empty_df, 0)。这是与"有数据但 < 阈值"区分的标志。
    """
    # 空 df
    df_empty = pd.DataFrame(columns=['code', 'industry_l1', 'spectral_radius',
                                      'k_hat', 'c_hat', 'schur_stable',
                                      'in_wedge', 'distance_to_wedge'])
    agg, threshold = EA.aggregate_by_industry(df_empty)
    assert len(agg) == 0
    assert threshold == 0  # T3.5 修复: 0 表示无数据

    # 单行业 50 只(>= 默认阈值 50,但 < 5 industries): threshold > 0
    rng = np.random.default_rng(13)
    rows = []
    for i in range(50):
        rows.append({'code': f'X{i:03d}', 'industry_l1': '881999.SH',
                     'spectral_radius': rng.uniform(0.5, 1.0),
                     'k_hat': 0.0, 'c_hat': 1.0,
                     'schur_stable': True, 'in_wedge': True,
                     'distance_to_wedge': 0.1})
    df_one = pd.DataFrame(rows)
    agg, threshold = EA.aggregate_by_industry(df_one)
    assert len(agg) == 1  # 1 个行业
    assert threshold > 0  # 有数据(虽然只 1 个行业),threshold >= 30 (fallback_min)
```

**关键:** 测试 2 个 case — (a) 0 数据 → `(empty, 0)`, (b) 有数据但 < 5 → `(agg, >0)`。

### Step 1.2: 跑测试,确认失败

```bash
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_aggregate_by_industry_no_data -v
```

**Expected:** FAIL — `assert threshold == 0` fails(当前空 df 返回 `threshold=fallback_min=30`,不是 0)

### Step 1.3: 修复 T3.5 — `aggregate_by_industry` 0-row 分支

打开 `backtrace/dynamics/dynamics_eigen_analysis.py:163-189`,把 `aggregate_by_industry` 末尾的 fallback 逻辑:

```python
    for thr in (min_stocks, fallback_min):
        agg = df.groupby('industry_l1').agg(
            n_stocks=('code', 'count'),
            rho_median=('spectral_radius', 'median'),
            rho_p25=('spectral_radius', lambda s: s.quantile(0.25)),
            rho_p75=('spectral_radius', lambda s: s.quantile(0.75)),
            k_hat_median=('k_hat', 'median'),
            c_hat_median=('c_hat', 'median'),
            schur_stable_pct=('schur_stable', 'mean'),
            in_wedge_pct=('in_wedge', 'mean'),
            dist_wedge_median=('distance_to_wedge', 'median'),
        ).reset_index()
        agg = agg[agg['n_stocks'] >= thr].sort_values('rho_median', ascending=False).head(10)
        if len(agg) >= 5:
            return agg, thr
    return agg, fallback_min
```

替换为:

```python
    for thr in (min_stocks, fallback_min):
        agg = df.groupby('industry_l1').agg(
            n_stocks=('code', 'count'),
            rho_median=('spectral_radius', 'median'),
            rho_p25=('spectral_radius', lambda s: s.quantile(0.25)),
            rho_p75=('spectral_radius', lambda s: s.quantile(0.75)),
            k_hat_median=('k_hat', 'median'),
            c_hat_median=('c_hat', 'median'),
            schur_stable_pct=('schur_stable', 'mean'),
            in_wedge_pct=('in_wedge', 'mean'),
            dist_wedge_median=('distance_to_wedge', 'median'),
        ).reset_index()
        agg = agg[agg['n_stocks'] >= thr].sort_values('rho_median', ascending=False).head(10)
        if len(agg) >= 5:
            return agg, thr
    # T3.5: 区分"无数据" vs "有数据但 < 阈值"
    if len(agg) == 0:
        return agg, 0  # 0 = 无数据
    return agg, fallback_min  # 有数据但都 < 5
```

### Step 1.4: 跑测试,确认通过

```bash
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_aggregate_by_industry_no_data -v
```

**Expected:** PASS

### Step 1.5: 修复 T3.3 — `aggregate_by_exchange` 排序方向

打开 `backtrace/dynamics/dynamics_eigen_analysis.py:203`,把:

```python
    ).reset_index().sort_values('rho_median')
```

改成:

```python
    ).reset_index().sort_values('rho_median', ascending=False)
```

### Step 1.6: 修复 T3.4 — bar chart text decimal precision

打开 `backtrace/dynamics/dynamics_eigen_analysis.py:611`,把:

```python
                text=[f"n={n}<br>ρ={r:.2f}" for n, r in zip(agg_l1['n_stocks'], agg_l1['rho_median'])],
```

改成:

```python
                text=[f"n={n}<br>ρ={r:.3f}" for n, r in zip(agg_l1['n_stocks'], agg_l1['rho_median'])],
```

打开 `backtrace/dynamics/dynamics_eigen_analysis.py:718`,把:

```python
            text=[f"n={n}<br>ρ={r:.2f}" for n, r in zip(agg_ex['n_stocks'], agg_ex['rho_median'])],
```

改成:

```python
            text=[f"n={n}<br>ρ={r:.3f}" for n, r in zip(agg_ex['n_stocks'], agg_ex['rho_median'])],
```

### Step 1.7: 跑全部 30 测试,确认通过

```bash
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v
```

**Expected:** 30 passed (29 旧 + 1 新)

### Step 1.8: 端到端冒烟(2 路径)

```bash
# 1. 默认 off — 行业聚合应正常排序 desc,ex 也 desc
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_eigen_analysis.py --limit 50
# → 5 旧 outputs,exit 0
# 验证:文本汇总里 行业 top10 (SH / SZ 顺序都按 ρ_med 降序)

# 2. 验证 text summary 输出顺序
cat backtrace/outputs/dynsys_eigen_summary.txt | grep "ρ_med" | head -15
# 期待: 行业第 1 行 ρ_med 最大, 后面递减;交易所类似
```

### Step 1.9: Commit

```bash
cd "C:\Users\yellow\mcp\qtTdx"
git add backtrace/dynamics/dynamics_eigen_analysis.py tests/test_dynamics_eigen.py
git commit -m "polish(dynamics): v4.6 — T3.3 / T3.4 / T3.5 三处 deferred polish

- T3.3: aggregate_by_exchange 排序改 descending(与 industry 一致)
- T3.4: 2 处 bar chart text 改 .3f(.2f -> .3f, 与 hovertemplate 一致)
- T3.5: aggregate_by_industry 0-row fallback 区分 — threshold=0 表示无数据
- 1 新测试 test_aggregate_by_industry_no_data(2 cases: empty df + 1 industry < 5)
- 30 tests pass,数学层 / 3 caller 零修改"
```

---

## 显式不做

- ❌ T3.1 HTML 5.8MB 减重(gitignored,浏览器加载不是问题)
- ❌ v4.5 parked 4 findings (plotly CDN pinning, test design, json import, HTML template duplication)
- ❌ v4.6 roadmap 新功能(行业 / 交易所专用 phase plot,密度等高线)
- ❌ v6 受迫系统 + G(ω) 频率响应

## 验证清单

- [ ] Step 1.1: 测试写入
- [ ] Step 1.2: 测试失败(threshold != 0)
- [ ] Step 1.3: T3.5 修复
- [ ] Step 1.4: 测试通过
- [ ] Step 1.5: T3.3 sort 改 descending
- [ ] Step 1.6: T3.4 2 处 .2f → .3f
- [ ] Step 1.7: 30 tests pass
- [ ] Step 1.8: 端到端冒烟 + 文本汇总顺序验证
- [ ] Step 1.9: commit

## 风险

| 风险 | 缓解 |
|---|---|
| T3.5 改 `aggregate_by_industry` 返回值语义,下游 caller 可能不识别 threshold=0 | `main()` 唯一 caller 已用 `l1_threshold` 在 print 输出,加 0 检查:`if l1_threshold == 0: print('no data')`;本 task 不动 main()(输出看似 'n>=0' 无害,grep 友好) |
| T3.3 改 sort 方向,`test_exchange_split_correctness` 失效 | 测试用 `dict(zip(...))` 断言,顺序无关,验证过 |
| T3.4 改 `.2f` → `.3f`,文本汇总没有 bar chart 文本,不受影响 | 文本汇总用 `.3f` 已统一,验证过 |
| 现有 `test_industry_aggregation_rho_median` 触发 fallback_min 不触发 0-row | 3 个行业 50+30+20 → 触发 fallback_min=10,新逻辑下仍返回 `(agg, 10)`,验证过 |
