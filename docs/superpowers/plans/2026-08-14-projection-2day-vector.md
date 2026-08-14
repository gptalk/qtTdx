# Projection 2 日向量 (4-D) 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `backtrace/projection/` 的 2-D 量价向量投影扩展为可选 4-D(当日 + 前一日 `(Vol, Amt)`),由 CLI flag `--two-day-vec` 显式开启,默认行为完全不变。

**Architecture:** 三个 core 函数 (`_projection_core.load_pair` / `compute_vectors` / `build_result_df`) 各自新增 `lag: int = 0` 参数。`lag=0` 时函数体不动(回归零风险);`lag>=1` 时计算并加入 `Volume_prev` / `Amount_prev` 列,丢首行,扩展向量维度。两个调用脚本 (`projection_batch.py` / `projection_2d.py`) 加 `--two-day-vec` flag 把 lag 透传下去;`compute_projections` 维度无关,**完全不改**。

**Tech Stack:** Python 3 + pandas + numpy + plotly(只在 projection_2d.py 用,本次不改 plot 代码)+ pytest(测试沿用 `tests/conftest.py` 的 sys.path 注入)。不引入新依赖。

---

## Global Constraints

1. **不修改** `compute_projections`(投影算子维度无关,`proj[1]/proj[0]` 语义在 4-D 下保持"今日 Amount/Volume 比")
2. **不修改** `find_resi_positive.py`(依赖 `Resi_Price` 列存在,本次保留其语义)
3. **`lag=0` 时所有函数行为与改动前逐字节一致**(包括 `Norm_Params` 字符串格式)— 这是回归不变量
4. **`--two-day-vec` 默认 off**(沿用 2-D 行为);开启时输出 CSV 文件名**不变**(`projection_<IX>_<ST>.csv`)— 这是显式取舍,见 spec §5/§8
5. **`projection_2d.py` HTML 输出前缀在 4-D 模式切换为 `proj2d_4d_`**,避免覆盖 2-D 输出
6. **TQ 客户端不需启动**:本次改动全部走 `use_tq=False` 本地缓存路径(`CLAUDE.md` 已说明)
7. **`PYTHONIOENCODING=utf-8`** 仍然必须在命令行加(Windows GBK 终端)— 见 `CLAUDE.md`
8. **每个 commit 后跑 pytest**: `cd c:\Users\yellow\mcp\qtTdx && python -m pytest tests/ -x -q`

---

## File Structure

```
backtrace/
├── projection/
│   ├── _projection_core.py      ← 改: load_pair / compute_vectors / build_result_df 各加 lag=0 参数
│   ├── projection_batch.py      ← 改: 加 --two-day-vec flag
│   └── projection_2d.py         ← 改: 加 --two-day-vec flag + FILE_PREFIX 切换
tests/
├── test_projection_core.py      ← 新增: 单元测试 compute_vectors / build_result_df / load_pair 三函数 lag=0/1 行为
└── test_projection_cli.py       ← 新增: 子进程跑 projection_2d.py / projection_batch.py,断言 CSV 列数 / status
```

每个文件一个职责:
- `_projection_core.py` — 共享数学;参数化后同时支持 2-D 和 4-D
- `projection_batch.py` — 批量入口;thin wrapper,只负责 flag 透传 + CSV 落盘
- `projection_2d.py` — 单股入口;thin wrapper,只负责 flag 透传 + HTML 落盘
- `tests/test_projection_core.py` — 核心函数单元测试(`compute_vectors` / `build_result_df` / `load_pair`)
- `tests/test_projection_cli.py` — CLI / process_one 单测(用 `_FakePipeline` 替身避免 subprocess)

---

### Task 1: 扩展 `compute_vectors` 与 `build_result_df` 支持 `lag` 参数

**Files:**
- Modify: `backtrace/projection/_projection_core.py:201-289`(两个函数体)
- Create: `tests/test_projection_core.py`

**Interfaces:**
- Consumes: 标准 pandas `DataFrame`(列: `Volume`, `Amount`,可选 `Volume_prev`, `Amount_prev`, `Close`)
- Produces:
  - `compute_vectors(stock_df, index_df, index_tag, stock_tag, lag: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]`
  - `build_result_df(common_idx, vec_index, vec_stock, vec_index_norm, vec_stock_norm, projections, residuals, dot_after, proj_coeffs, proj_mags, proj_prices, resi_prices, norm_params, index_tag, stock_tag, lag: int = 0) -> pd.DataFrame`

- [ ] **Step 1: 写失败测试 — `compute_vectors` lag=1 输出 4 列**

`tests/test_projection_core.py`:

```python
# -*- coding: utf-8 -*-
"""backtrace/projection/_projection_core.py 单元测试 — 覆盖 lag=0 / lag=1 双路径。"""
import sys, os
import numpy as np
import pandas as pd
import pytest

# 与脚本同一套导入约定:backtrace/ 进 sys.path
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTRACE = os.path.join(REPO, 'backtrace')
PROJECTION = os.path.join(BACKTRACE, 'projection')
if BACKTRACE not in sys.path:
    sys.path.insert(0, BACKTRACE)

from projection._projection_core import compute_vectors, build_result_df


def _make_pair(n=10):
    """造一对 (stock_df, index_df),index=stock 索引,数据有差异。"""
    idx = pd.date_range('2026-07-01', periods=n, freq='D')
    index_df = pd.DataFrame({
        'Volume': np.linspace(1e7, 1.5e7, n),
        'Amount': np.linspace(1e11, 1.5e11, n),
        'Close':  np.linspace(3000, 3200, n),
    }, index=idx)
    stock_df = pd.DataFrame({
        'Volume': np.linspace(2e6, 3e6, n),
        'Amount': np.linspace(2e10, 3e10, n),
        'Close':  np.linspace(20, 25, n),
    }, index=idx)
    return stock_df, index_df


def _add_prev(df, vol_prev, amt_prev):
    out = df.copy()
    out['Volume_prev'] = vol_prev
    out['Amount_prev'] = amt_prev
    return out


def test_compute_vectors_lag0_returns_2_columns():
    """lag=0 默认: 输出向量 shape=(T, 2)。"""
    stock_df, index_df = _make_pair(10)
    v_ix, v_st, v_ix_n, v_st_n, _ = compute_vectors(stock_df, index_df, '000001', '002475')
    assert v_ix.shape == (10, 2)
    assert v_st.shape == (10, 2)
    assert v_ix_n.shape == (10, 2)
    assert v_st_n.shape == (10, 2)


def test_compute_vectors_lag1_returns_4_columns():
    """lag=1: 输出向量 shape=(T, 4),顺序 Vol_t, Amt_t, Vol_prev, Amt_prev。"""
    stock_df, index_df = _make_pair(10)
    stock_df = _add_prev(stock_df, np.linspace(1.9e6, 2.8e6, 10), np.linspace(1.9e10, 2.8e10, 10))
    index_df = _add_prev(index_df, np.linspace(0.95e7, 1.45e7, 10), np.linspace(0.95e11, 1.45e11, 10))
    v_ix, v_st, v_ix_n, v_st_n, _ = compute_vectors(stock_df, index_df, '000001', '002475', lag=1)
    assert v_ix.shape == (10, 4)
    assert v_st.shape == (10, 4)
    # 前 2 列 = 今日 (与 lag=0 一致)
    np.testing.assert_array_equal(v_ix[:, :2], index_df[['Volume', 'Amount']].values)
    # 后 2 列 = 昨日
    np.testing.assert_array_equal(v_ix[:, 2:], index_df[['Volume_prev', 'Amount_prev']].values)


def test_compute_vectors_norm_range_in_unit_interval():
    """归一化后每列在 [0, 1]。"""
    stock_df, index_df = _make_pair(10)
    stock_df = _add_prev(stock_df, np.linspace(1.9e6, 2.8e6, 10), np.linspace(1.9e10, 2.8e10, 10))
    index_df = _add_prev(index_df, np.linspace(0.95e7, 1.45e7, 10), np.linspace(0.95e11, 1.45e11, 10))
    _, _, v_ix_n, v_st_n, _ = compute_vectors(stock_df, index_df, '000001', '002475', lag=1)
    assert (0.0 <= v_ix_n).all() and (v_ix_n <= 1.0).all()
    assert (0.0 <= v_st_n).all() and (v_st_n <= 1.0).all()


def test_compute_vectors_norm_params_lists_four_ranges_at_lag1():
    """lag=1 时 norm_params 字符串包含 4 个范围(每个 tag × 2 个列 + 2 个 prev 列)。"""
    stock_df, index_df = _make_pair(10)
    stock_df = _add_prev(stock_df, np.linspace(1.9e6, 2.8e6, 10), np.linspace(1.9e10, 2.8e10, 10))
    index_df = _add_prev(index_df, np.linspace(0.95e7, 1.45e7, 10), np.linspace(0.95e11, 1.45e11, 10))
    _, _, _, _, params = compute_vectors(stock_df, index_df, '000001', '002475', lag=1)
    # 期望含 4 个 vol_/amt_ 范围 + 4 个 vol_prev/amt_prev 范围 → 8 个 "["
    assert params.count('[') == 8
```

- [ ] **Step 2: 跑测试,确认失败**

Run:
```bash
cd c:\Users\yellow\mcp\qtTdx && python -m pytest tests/test_projection_core.py -x -q
```
Expected: FAIL — `compute_vectors` 收到未预期的 `lag` 参数 (`TypeError: compute_vectors() got an unexpected keyword argument 'lag'`)。

- [ ] **Step 3: 写失败测试 — `build_result_df` lag=1 输出 27 列**

接上,同一文件追加:

```python
def test_build_result_df_lag0_returns_19_columns():
    """回归: lag=0 保持 19 列。"""
    n = 10
    common_idx = pd.date_range('2026-07-01', periods=n, freq='D')
    v_ix = np.random.rand(n, 2)
    v_st = np.random.rand(n, 2)
    proj = {'projections': np.zeros((n, 2)), 'residuals': np.zeros((n, 2)),
            'dot_after': np.zeros(n), 'proj_coeffs': np.zeros(n),
            'proj_mags': np.zeros(n), 'proj_prices': np.zeros(n), 'resi_prices': np.zeros(n)}
    df = build_result_df(
        common_idx, v_ix, v_st, v_ix, v_st,
        proj['projections'], proj['residuals'], proj['dot_after'],
        proj['proj_coeffs'], proj['proj_mags'], proj['proj_prices'], proj['resi_prices'],
        "vol_000001:[1,2] amt_000001:[1,2] vol_002475:[1,2] amt_002475:[1,2]",
        '000001', '002475',
    )
    assert df.shape == (n, 19)


def test_build_result_df_lag1_returns_27_columns():
    """lag=1: 增加 8 个 prev 列 = 27 列。"""
    n = 10
    common_idx = pd.date_range('2026-07-01', periods=n, freq='D')
    v_ix = np.random.rand(n, 4)
    v_st = np.random.rand(n, 4)
    proj = {'projections': np.zeros((n, 4)), 'residuals': np.zeros((n, 4)),
            'dot_after': np.zeros(n), 'proj_coeffs': np.zeros(n),
            'proj_mags': np.zeros(n), 'proj_prices': np.zeros(n), 'resi_prices': np.zeros(n)}
    df = build_result_df(
        common_idx, v_ix, v_st, v_ix, v_st,
        proj['projections'], proj['residuals'], proj['dot_after'],
        proj['proj_coeffs'], proj['proj_mags'], proj['proj_prices'], proj['resi_prices'],
        "vol_000001:[1,2] amt_000001:[1,2] vol_002475:[1,2] amt_002475:[1,2] "
        "vol_prev_000001:[1,2] amt_prev_000001:[1,2] vol_prev_002475:[1,2] amt_prev_002475:[1,2]",
        '000001', '002475', lag=1,
    )
    assert df.shape == (n, 27)
    # 检查 prev_raw + prev_norm 4 对列都在
    for tag in ('000001', '002475'):
        for kind in ('Vol', 'Amt'):
            assert f'{kind}_{tag}_prev_raw' in df.columns
            assert f'{kind}_{tag}_prev_norm' in df.columns


def test_build_result_df_lag1_preserves_projection_columns_after_prev_block():
    """lag=1 时,Proj_Vol 仍是第 18 列(prev 块在 10-17)。"""
    n = 5
    common_idx = pd.date_range('2026-07-01', periods=n, freq='D')
    v_ix = np.random.rand(n, 4)
    v_st = np.random.rand(n, 4)
    proj = {'projections': np.zeros((n, 4)), 'residuals': np.zeros((n, 4)),
            'dot_after': np.zeros(n), 'proj_coeffs': np.zeros(n),
            'proj_mags': np.zeros(n), 'proj_prices': np.zeros(n), 'resi_prices': np.zeros(n)}
    df = build_result_df(
        common_idx, v_ix, v_st, v_ix, v_st,
        proj['projections'], proj['residuals'], proj['dot_after'],
        proj['proj_coeffs'], proj['proj_mags'], proj['proj_prices'], proj['resi_prices'],
        "x", '000001', '002475', lag=1,
    )
    cols = list(df.columns)
    assert cols[0] == 'Date'
    assert cols[17] == 'Proj_Vol'
    assert cols[18] == 'Proj_Amt'


def test_build_result_df_lag1_resi_price_present():
    """find_resi_positive.py 依赖 Resi_Price 列,lag=1 必须保留。"""
    n = 5
    common_idx = pd.date_range('2026-07-01', periods=n, freq='D')
    v_ix = np.random.rand(n, 4)
    v_st = np.random.rand(n, 4)
    proj = {'projections': np.zeros((n, 4)), 'residuals': np.zeros((n, 4)),
            'dot_after': np.zeros(n), 'proj_coeffs': np.zeros(n),
            'proj_mags': np.zeros(n), 'proj_prices': np.zeros(n), 'resi_prices': np.zeros(n)}
    df = build_result_df(
        common_idx, v_ix, v_st, v_ix, v_st,
        proj['projections'], proj['residuals'], proj['dot_after'],
        proj['proj_coeffs'], proj['proj_mags'], proj['proj_prices'], proj['resi_prices'],
        "x", '000001', '002475', lag=1,
    )
    assert 'Resi_Price' in df.columns
```

Run:
```bash
cd c:\Users\yellow\mcp\qtTdx && python -m pytest tests/test_projection_core.py -x -q
```
Expected: 所有 `lag=1` 测试 FAIL,`lag=0` 测试(`test_build_result_df_lag0_returns_19_columns`)PASS。

- [ ] **Step 4: 改 `compute_vectors` — 加 lag 参数**

`backtrace/projection/_projection_core.py` 中 `compute_vectors` (line 201-226),把签名改为:

```python
def compute_vectors(stock_df, index_df, index_tag, stock_tag, lag: int = 0):
    """Min-Max 归一化 Vol/Amt(及可选的 Vol_prev/Amt_prev)。

    Args:
        lag: 0 = 当前 (Volume, Amount) 2-D(默认,与改动前一致);
             >=1 时还取 Volume.shift(1) / Amount.shift(1), 输出向量维度 = 2 * (lag + 1)。
             本次仅实现 lag=0 / lag=1。
    """
    cols = ['Volume', 'Amount']
    if lag >= 1:
        cols += ['Volume_prev', 'Amount_prev']
    # 防呆: DataFrame 缺列时直接报错(比 KeyError 友好)
    for c in cols:
        if c not in stock_df.columns:
            raise KeyError(f"compute_vectors(lag={lag}) 需要 stock_df 含列 {c!r}")
        if c not in index_df.columns:
            raise KeyError(f"compute_vectors(lag={lag}) 需要 index_df 含列 {c!r}")

    vec_index = index_df[cols].values
    vec_stock = stock_df[cols].values

    # 每个维度独立 Min-Max(向量化)
    norms_ix = np.zeros_like(vec_index)
    norms_st = np.zeros_like(vec_stock)
    params_parts = []
    for j, c in enumerate(cols):
        v_min_ix, v_max_ix = vec_index[:, j].min(), vec_index[:, j].max()
        v_min_st, v_max_st = vec_stock[:, j].min(), vec_stock[:, j].max()
        norms_ix[:, j] = _safe_minmax(vec_index[:, j], v_min_ix, v_max_ix - v_min_ix)
        norms_st[:, j] = _safe_minmax(vec_stock[:, j], v_min_st, v_max_st - v_min_st)
        params_parts.append(f"{c}_{index_tag}:[{v_min_ix:.2e},{v_max_ix:.2e}]")
        params_parts.append(f"{c}_{stock_tag}:[{v_min_st:.2e},{v_max_st:.2e}]")
    norm_params = " ".join(params_parts)

    return vec_index, vec_stock, norms_ix, norms_st, norm_params
```

- [ ] **Step 5: 改 `build_result_df` — 加 lag 参数**

同文件 line 265,签名尾部加 `, lag: int = 0`。函数体在 `f'Amt_{stock_tag}_norm': vec_stock_norm[:, 1]` 这一行后插入:

```python
prev_cols_raw = {}
prev_cols_norm = {}
if lag >= 1:
    # 假设 vec_* 已是 4 列 (Vol_t, Amt_t, Vol_prev, Amt_prev),取 [2:4]
    assert vec_index.shape[1] >= 4 and vec_stock.shape[1] >= 4, (
        f"build_result_df(lag={lag}) 需要 vec_index/vec_stock 有 4 列,"
        f" 实际 shape={vec_index.shape}, {vec_stock.shape}"
    )
    prev_cols_raw = {
        f'Vol_{index_tag}_prev_raw': vec_index[:, 2],
        f'Amt_{index_tag}_prev_raw': vec_index[:, 3],
        f'Vol_{stock_tag}_prev_raw': vec_stock[:, 2],
        f'Amt_{stock_tag}_prev_raw': vec_stock[:, 3],
    }
    prev_cols_norm = {
        f'Vol_{index_tag}_prev_norm': vec_index_norm[:, 2],
        f'Amt_{index_tag}_prev_norm': vec_index_norm[:, 3],
        f'Vol_{stock_tag}_prev_norm': vec_stock_norm[:, 2],
        f'Amt_{stock_tag}_prev_norm': vec_stock_norm[:, 3],
    }

return pd.DataFrame({
    'Date': common_idx,
    f'Vol_{index_tag}_raw': vec_index[:, 0],
    f'Amt_{index_tag}_raw': vec_index[:, 1],
    f'Vol_{stock_tag}_raw': vec_stock[:, 0],
    f'Amt_{stock_tag}_raw': vec_stock[:, 1],
    f'Vol_{index_tag}_norm': vec_index_norm[:, 0],
    f'Amt_{index_tag}_norm': vec_index_norm[:, 1],
    f'Vol_{stock_tag}_norm': vec_stock_norm[:, 0],
    f'Amt_{stock_tag}_norm': vec_stock_norm[:, 1],
    **prev_cols_raw,    # lag=0 时为空 dict,不引入新列
    **prev_cols_norm,
    'Proj_Vol': projections[:, 0],
    'Proj_Amt': projections[:, 1],
    'Residual_Vol': residuals[:, 0],
    'Residual_Amt': residuals[:, 1],
    'Proj_Coeff': proj_coeffs,
    'Proj_Magnitude': proj_mags,
    'Proj_Price': proj_prices,
    'Resi_Price': resi_prices,
    'Dot_After_Proj': dot_after,
    'Norm_Params': [norm_params] * len(common_idx),
})
```

- [ ] **Step 6: 跑测试,确认全绿**

Run:
```bash
cd c:\Users\yellow\mcp\qtTdx && python -m pytest tests/test_projection_core.py -v
```
Expected: 7 PASS。

- [ ] **Step 7: 跑全套测试,确认未破坏 2-D 路径**

Run:
```bash
cd c:\Users\yellow\mcp\qtTdx && python -m pytest tests/ -x -q
```
Expected: 全绿(包括已有的 `test_fallback.py` / `test_data_store.py` 等)— 因为 `compute_vectors` / `build_result_df` 默认参数 `lag=0` 行为完全等同旧版。

- [ ] **Step 8: 提交**

```bash
cd c:\Users\yellow\mcp\qtTdx && git add backtrace/projection/_projection_core.py tests/test_projection_core.py
git commit -m "feat(projection): compute_vectors / build_result_df 新增 lag=0 参数 (4-D 支持)

- lag=0 默认行为与改动前逐字节一致(回归不变量)
- lag=1 时向量扩为 (Vol_t, Amt_t, Vol_prev, Amt_prev) 4-D,
  build_result_df 输出 27 列(原 19 列 + 8 个 prev 列,顺序如 spec §3.1)
- Resi_Price 语义保持,find_resi_positive.py 无需改动
- 7 个单元测试覆盖 lag=0 回归 + lag=1 向量维度 / 归一化 / norm_params

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: 扩展 `load_pair` 支持 `lag` 参数

**Files:**
- Modify: `backtrace/projection/_projection_core.py:127-189`(`load_pair` 函数体)
- Modify: `tests/test_projection_core.py`(追加测试)

**Interfaces:**
- Consumes: `tsfresh_pipeline` 实例(同前);`lag: int = 0`
- Produces: 同前(dict 含 `stock_df` / `index_df` / `common_idx` 等),但 `lag=1` 时 `stock_df` / `index_df` 含 `Volume_prev` / `Amount_prev` 列,且 `common_idx` 已丢弃首行

- [ ] **Step 1: 写失败测试 — `load_pair` lag=0 与 lag=1 共用 input/output shape 差异**

`tests/test_projection_core.py` 顶部 import 改成:

```python
from projection._projection_core import compute_vectors, build_result_df, load_pair
```

追加:

```python
class _FakePipeline:
    """最小化的 tsfresh_pipeline 替身:返回内存中的 DataFrame,不读 data/。"""

    def __init__(self, df_by_code):
        self._df = df_by_code

    def load_ohlcva(self, code, use_tq=False, verbose=False):
        return self._df.get(code)


def _make_ohlcv(n, base_vol, base_amt):
    idx = pd.date_range('2026-07-01', periods=n, freq='D')
    return pd.DataFrame({
        'Volume': np.linspace(base_vol, base_vol * 1.5, n),
        'Amount': np.linspace(base_amt, base_amt * 1.5, n),
        'Close':  np.linspace(100, 110, n),
    }, index=idx)


def test_load_pair_lag0_does_not_add_prev_columns():
    """lag=0: 返回的 stock_df / index_df 不含 Volume_prev。"""
    df = _make_ohlcv(10, 1e6, 1e10)
    pipe = _FakePipeline({'000001.SH': df, '002475.SZ': df})
    out = load_pair('002475.SZ', days=10, pipeline=pipe, index_code='000001.SH')
    assert 'Volume_prev' not in out['stock_df'].columns
    assert 'Volume_prev' not in out['index_df'].columns
    assert len(out['common_idx']) == 10


def test_load_pair_lag1_adds_prev_columns_and_drops_first_row():
    """lag=1: 返回的 df 含 prev 列,common_idx 比原始少 1 行(首行 prev=NaN 被 dropna)。"""
    df = _make_ohlcv(10, 1e6, 1e10)
    pipe = _FakePipeline({'000001.SH': df, '002475.SZ': df})
    out = load_pair('002475.SZ', days=10, pipeline=pipe, index_code='000001.SH', lag=1)
    assert 'Volume_prev' in out['stock_df'].columns
    assert 'Amount_prev' in out['stock_df'].columns
    assert 'Volume_prev' in out['index_df'].columns
    assert 'Amount_prev' in out['index_df'].columns
    assert len(out['common_idx']) == 9, "首行 prev=NaN 应被 dropna 丢弃"
    # 第 0 行(index_df.iloc[0])的 prev 列应等于原第 1 行(df.iloc[0]→df.iloc[1])
    assert out['index_df']['Volume_prev'].iloc[0] == df['Volume'].iloc[1]


def test_load_pair_lag1_raises_when_data_too_short():
    """数据 < 2 行时 lag=1 必须 ValueError。"""
    df = _make_ohlcv(1, 1e6, 1e10)
    pipe = _FakePipeline({'000001.SH': df, '002475.SZ': df})
    with pytest.raises(ValueError, match="≥2"):
        load_pair('002475.SZ', days=10, pipeline=pipe, index_code='000001.SH', lag=1)
```

- [ ] **Step 2: 跑测试,确认失败**

Run:
```bash
cd c:\Users\yellow\mcp\qtTdx && python -m pytest tests/test_projection_core.py -x -q
```
Expected: 3 个新测试 FAIL(`load_pair() got an unexpected keyword argument 'lag'`)。

- [ ] **Step 3: 改 `load_pair` — 加 lag 参数**

`backtrace/projection/_projection_core.py:127-189` 改写为:

```python
def load_pair(stock_code, days, pipeline, prefer_industry=False, index_code=None, lag: int = 0):
    """从本地 data/ 缓存加载 (stock_df, index_df) 共同交易日的最近 `days` 行。

    Args:
        ... (同前)
        lag: 0 = 当前(Volume, Amount)(默认,行为与改动前一致);
             >=1 时还附带 Volume.shift(1) / Amount.shift(1),首行 prev=NaN 被 dropna。
             本次仅实现 lag=0 / lag=1。

    基线选择优先级:同前。

    Returns:
        dict: stock_df, index_df, common_idx, index_code, index_name,
              index_tag, stock_tag。

    Raises:
        RuntimeError: 本地缓存缺失(同前)。
        ValueError: lag >= 1 但 stock/index 数据 < 2 行。
    """
    if index_code:
        index_name = resolve_index_name(index_code)
    elif prefer_industry:
        try:
            index_code, index_name = resolve_industry(stock_code)
        except ValueError:
            index_code, index_name = resolve_index(stock_code)
    else:
        index_code, index_name = resolve_index(stock_code)
    index_tag = index_code.split('.')[0]
    stock_tag = stock_code.split('.')[0]

    data_index_full = pipeline.load_ohlcva(index_code, use_tq=False, verbose=True)
    data_stock_full = pipeline.load_ohlcva(stock_code, use_tq=False, verbose=True)
    if data_index_full is None:
        raise RuntimeError(
            f"本地缓存缺失 {index_code}。请先跑:\n"
            f"  PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py"
        )
    if data_stock_full is None:
        raise RuntimeError(
            f"本地缓存缺失 {stock_code}。请先跑:\n"
            f"  PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py"
        )

    # lag >= 1 时附加 prev 列;前置 < 2 行检查(避免 shift 全 NaN 后静默丢所有行)
    if lag >= 1:
        if len(data_index_full) < 2 or len(data_stock_full) < 2:
            raise ValueError(
                f"--two-day-vec 需要 ≥2 行数据,"
                f"实际 {index_code}={len(data_index_full)} 行, {stock_code}={len(data_stock_full)} 行"
            )
        data_index_full = data_index_full.assign(
            Volume_prev=data_index_full['Volume'].shift(1),
            Amount_prev=data_index_full['Amount'].shift(1),
        )
        data_stock_full = data_stock_full.assign(
            Volume_prev=data_stock_full['Volume'].shift(1),
            Amount_prev=data_stock_full['Amount'].shift(1),
        )

    cols = ['Volume', 'Amount', 'Close']
    if lag >= 1:
        cols = ['Volume', 'Amount', 'Volume_prev', 'Amount_prev', 'Close']
    data_index = data_index_full[cols].tail(days).dropna()
    data_stock = data_stock_full[cols].tail(days).dropna()
    common_idx = data_index.index.intersection(data_stock.index)
    return {
        'stock_df': data_stock.loc[common_idx],
        'index_df': data_index.loc[common_idx],
        'common_idx': common_idx,
        'index_code': index_code,
        'index_name': index_name,
        'index_tag': index_tag,
        'stock_tag': stock_tag,
    }
```

- [ ] **Step 4: 跑测试,确认全绿**

Run:
```bash
cd c:\Users\yellow\mcp\qtTdx && python -m pytest tests/test_projection_core.py -v
```
Expected: 10 PASS。

- [ ] **Step 5: 跑全套测试**

Run:
```bash
cd c:\Users\yellow\mcp\qtTdx && python -m pytest tests/ -x -q
```
Expected: 全绿。

- [ ] **Step 6: 提交**

```bash
cd c:\Users\yellow\mcp\qtTdx && git add backtrace/projection/_projection_core.py tests/test_projection_core.py
git commit -m "feat(projection): load_pair 新增 lag=0 参数 (4-D 支持)

- lag=0 行为完全等同旧版(回归不变量)
- lag=1 时附 Volume_prev / Amount_prev 列,首行被 dropna 丢弃
- 数据 < 2 行时抛 ValueError(防呆)
- 3 个新单元测试覆盖 lag=0 回归 + lag=1 prev 列 + 数据不足

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: `projection_batch.py` 加 `--two-day-vec` flag

**Files:**
- Modify: `backtrace/projection/projection_batch.py`(imports, `parse_args`, `process_one`, `main`)
- Create: `tests/test_projection_cli.py`

**Interfaces:**
- Produces: 新 CLI 参数 `--two-day-vec`(action='store_true', 默认 off)
- `process_one` 签名追加 `lag: int` 形参

- [ ] **Step 1: 写失败测试 — `parse_args` 接受 `--two-day-vec`**

`tests/test_projection_cli.py`:

```python
# -*- coding: utf-8 -*-
"""projection_batch.py / projection_2d.py 的 CLI / process_one 单测。

策略: 不调 subprocess(DATA_DIR 在子进程里没法 monkeypatch,脆弱)。
改为:
- 用 argparse.Namespace 模拟 parse_args 输出,直接调 process_one
- monkeypatch tsfresh_pipeline.load_ohlcva 返回内存 DataFrame
- 临时切到 tmp cwd 避免污染 data/projection/
"""
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTRACE = os.path.join(REPO, 'backtrace')
PROJECTION = os.path.join(BACKTRACE, 'projection')
if BACKTRACE not in sys.path:
    sys.path.insert(0, BACKTRACE)
if PROJECTION not in sys.path:
    sys.path.insert(0, PROJECTION)


class _FakePipeline:
    """最小化的 tsfresh_pipeline 替身:返回内存中的 DataFrame,不读 data/。"""

    def __init__(self, df_by_code):
        self._df = df_by_code

    def load_ohlcva(self, code, use_tq=False, verbose=False):
        return self._df.get(code)


def _fake_pair(n=5):
    idx = pd.date_range('2026-07-01', periods=n, freq='D')
    base = pd.DataFrame({
        'Volume': np.linspace(1e6, 1.5e6, n),
        'Amount': np.linspace(1e7, 1.5e7, n),
        'Close':  np.linspace(100, 110, n),
    }, index=idx)
    return base


def test_batch_process_one_lag0_writes_19_col_csv(tmp_path, monkeypatch):
    """process_one 默认 lag=0 写出 19 列 CSV。"""
    # 切 cwd 到 tmp,避免污染 data/projection/
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('projection_batch.CSV_OUT_DIR', str(tmp_path))

    df = _fake_pair(5)
    fake_pipe = _FakePipeline({'000001.SH': df.copy(), '002475.SZ': df.copy()})
    # patch P 引用本身(在 projection_batch 模块里是 tsfresh_pipeline 别名)
    import projection_batch as pb_mod
    monkeypatch.setattr(pb_mod, 'P', fake_pipe)

    row = pb_mod.process_one(
        stock_code='002475.SZ', stock_name='立讯精密',
        days=5, prefer_industry=False, index_code='000001.SH', lag=0,
    )
    assert row['status'] == 'ok', row
    assert row['rows'] == 5
    assert os.path.exists(row['csv_path'])
    csv_df = pd.read_csv(row['csv_path'])
    assert csv_df.shape[1] == 19
    assert 'Resi_Price' in csv_df.columns


def test_batch_process_one_lag1_writes_27_col_csv(tmp_path, monkeypatch):
    """process_one lag=1 写出 27 列 CSV,首行降。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('projection_batch.CSV_OUT_DIR', str(tmp_path))

    df = _fake_pair(5)
    fake_pipe = _FakePipeline({'000001.SH': df.copy(), '002475.SZ': df.copy()})
    import projection_batch as pb_mod
    monkeypatch.setattr(pb_mod, 'P', fake_pipe)

    row = pb_mod.process_one(
        stock_code='002475.SZ', stock_name='立讯精密',
        days=5, prefer_industry=False, index_code='000001.SH', lag=1,
    )
    assert row['status'] == 'ok', row
    assert row['rows'] == 4, f"5 行 - 首行 = 4 行,实际 {row['rows']}"
    csv_df = pd.read_csv(row['csv_path'])
    assert csv_df.shape[1] == 27
    assert 'Resi_Price' in csv_df.columns
    assert 'Vol_000001_prev_raw' in csv_df.columns
    assert 'Vol_000001_prev_norm' in csv_df.columns
```

- [ ] **Step 2: 跑测试,确认失败**

Run:
```bash
cd c:\Users\yellow\mcp\qtTdx && python -m pytest tests/test_projection_cli.py -x -q
```
Expected: FAIL — `TypeError: process_one() got an unexpected keyword argument 'lag'`。

- [ ] **Step 3: 改 `projection_batch.py` — 加 `--two-day-vec` + 透传 `lag`**

`backtrace/projection/projection_batch.py`:

1. `parse_args` (line 66-85) 在 `--index` 后追加:
```python
    parser.add_argument(
        '--two-day-vec', action='store_true',
        help=(
            '将向量扩展为 (Vol_today, Amt_today, Vol_yesterday, Amt_yesterday) 4-D;'
            '首日丢弃。默认 2-D。'
        ),
    )
```

2. `process_one` 签名追加 `lag`,函数体三处调用加 `lag=lag` 透传(行 111-130):

```python
def process_one(stock_code, stock_name, days, prefer_industry, index_code, lag: int = 0):
    """处理一只股票。返回 manifest 行 dict(失败也返回,status 字段说明原因)。"""
    try:
        loaded = load_pair(stock_code, days, P, prefer_industry=prefer_industry,
                           index_code=index_code, lag=lag)
        data_stock = loaded['stock_df']
        data_index = loaded['index_df']
        common_idx = loaded['common_idx']
        index_code = loaded['index_code']
        index_name = loaded['index_name']
        index_tag = loaded['index_tag']
        stock_tag = loaded['stock_tag']

        vec_index, vec_stock, vec_index_norm, vec_stock_norm, norm_params = compute_vectors(
            data_stock, data_index, index_tag, stock_tag, lag=lag,
        )
        proj = compute_projections(vec_stock_norm, vec_index_norm)

        result_df = build_result_df(
            common_idx, vec_index, vec_stock, vec_index_norm, vec_stock_norm,
            proj['projections'], proj['residuals'], proj['dot_after'],
            proj['proj_coeffs'], proj['proj_mags'], proj['proj_prices'], proj['resi_prices'],
            norm_params, index_tag, stock_tag, lag=lag,
        )
        # ... 后续不变(csv_name / csv_path / to_csv / manifest dict)...
```

3. `main` (line 162-207):
- 在 `print(f"投影基线: {baseline}")` 后新增一行 `向量维度` 信息
- 计算 `lag = 1 if args.two_day_vec else 0`
- `process_one` 调用追加 `lag=lag`

```python
    print(f"输入: {args.input} ({len(stock_list)} 只)")
    print(f"回看天数: {args.days}")
    print(f"投影基线: {baseline}")
    print(f"向量维度: {'4-D (今日+前一日 Vol/Amt, --two-day-vec)' if args.two_day_vec else '2-D (今日 Vol/Amt)'}")
    print(f"输出目录: {CSV_OUT_DIR}\n")

    lag = 1 if args.two_day_vec else 0
    manifest = []
    for i, (code, name) in enumerate(stock_list, 1):
        label = f"{code} ({name})" if name else code
        print(f"[{i}/{len(stock_list)}] {label}...", end=' ', flush=True)
        row = process_one(code, name, args.days, prefer_industry, args.index, lag)
        manifest.append(row)
        if row['status'] == 'ok':
            print(f"✓ {row['rows']} 行 → {row['csv_path']}")
        else:
            print(f"✗ {row['status']}")
```

- [ ] **Step 4: 跑测试,确认全绿**

Run:
```bash
cd c:\Users\yellow\mcp\qtTdx && python -m pytest tests/test_projection_cli.py -v
```
Expected: 2 PASS。

- [ ] **Step 5: 跑全套测试**

Run:
```bash
cd c:\Users\yellow\mcp\qtTdx && python -m pytest tests/ -x -q
```
Expected: 全绿。

- [ ] **Step 6: 提交**

```bash
cd c:\Users\yellow\mcp\qtTdx && git add backtrace/projection/projection_batch.py tests/test_projection_cli.py
git commit -m "feat(projection): projection_batch 新增 --two-day-vec flag

- 默认 2-D 行为完全不变
- --two-day-vec 透传到 load_pair / compute_vectors / build_result_df (lag=1)
- main 多打印一行 '向量维度' 信息
- 2 个单测覆盖 process_one lag=0 (19 列) + lag=1 (27 列)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: `projection_2d.py` 加 `--two-day-vec` flag + HTML 前缀切换

**Files:**
- Modify: `backtrace/projection/projection_2d.py`(`parse_args`, `FILE_PREFIX` 切换, 三处调用透传)

- [ ] **Step 1: 写失败测试 — `parse_args` 接受 `--two-day-vec`**

`tests/test_projection_cli.py` 追加:

```python
def test_single_2d_does_not_set_two_day_vec_by_default():
    """parse_args 默认解析后 two_day_vec=False。"""
    from projection_2d import parse_args
    args = parse_args.__wrapped__(['--code', '002475.SZ']) if hasattr(parse_args, '__wrapped__') else None
    # projection_2d.parse_args 直接吃 sys.argv,这里跳过 — 直接验证 FILE_PREFIX 默认值
    import importlib
    import projection_2d as p2d_mod
    importlib.reload(p2d_mod)  # 不传 --two-day-vec
    assert not p2d_mod.TWO_DAY_VEC
    assert p2d_mod.FILE_PREFIX == 'proj2d_'
    assert p2d_mod.LAG == 0


def test_single_two_day_vec_sets_4d_prefix(tmp_path, monkeypatch):
    """parse_args 收到 --two-day-vec 后 FILE_PREFIX='proj2d_4d_' 且 LAG=1。"""
    # 用 monkeypatch sys.argv 模拟 CLI
    monkeypatch.setattr(sys, 'argv', [
        'projection_2d.py', '--code', '002475.SZ',
        '--name', '立讯精密', '--days', '5', '--index', '000001.SH',
        '--two-day-vec',
    ])
    import importlib
    import projection_2d as p2d_mod
    importlib.reload(p2d_mod)
    assert p2d_mod.TWO_DAY_VEC is True
    assert p2d_mod.FILE_PREFIX == 'proj2d_4d_'
    assert p2d_mod.LAG == 1
```

- [ ] **Step 2: 跑测试,确认失败**

Run:
```bash
cd c:\Users\yellow\mcp\qtTdx && python -m pytest tests/test_projection_cli.py -x -q
```
Expected: FAIL — `TWO_DAY_VEC` 不存在 (AttributeError)。

- [ ] **Step 3: 改 `projection_2d.py`**

`backtrace/projection/projection_2d.py`:

1. `parse_args` (line 49-64) 在 `--index` 后追加:
```python
    p.add_argument(
        '--two-day-vec', action='store_true',
        help='将向量扩展为 4-D (今日 + 前一日 Vol/Amt);首日丢弃。默认 2-D。',
    )
```

2. 全局变量 `args = parse_args()` 后追加(在 `args` 之后):
```python
TWO_DAY_VEC = args.two_day_vec
LAG = 1 if TWO_DAY_VEC else 0
```

3. `OUT_DIR` 块(line 73-76)改 FILE_PREFIX:
```python
OUT_DIR = 'backtrace/outputs'
FILE_PREFIX = 'proj2d_4d_' if TWO_DAY_VEC else 'proj2d_'
CSV_OUT = 'data/projection'
```

4. `load_pair` / `compute_vectors` / `build_result_df` 三处调用加 `lag=LAG` 透传(位置: line 79, 106, 314):

```python
loaded = load_pair(STOCK_CODE, days, P, index_code=INDEX_OVERRIDE, lag=LAG)
# ...
vec_index, vec_stock, vec_index_norm, vec_stock_norm, norm_params = compute_vectors(
    data_stock, data_index, INDEX_TAG, STOCK_TAG, lag=LAG,
)
# ...
result_df = build_result_df(
    common_idx, vec_index, vec_stock, vec_index_norm, vec_stock_norm,
    projections, residuals, dot_products_after,
    proj_coefficients, proj_magnitudes, proj_prices, resi_prices,
    norm_params, INDEX_TAG, STOCK_TAG, lag=LAG,
)
```

5. `baseline_kind` 那行(line 90-93)后面追加 `print` 提示向量维度:
```python
print(f"基线选择: {baseline_kind}")
print(f"向量维度: {'4-D (今日+前一日 Vol/Amt)' if TWO_DAY_VEC else '2-D (今日 Vol/Amt)'}")
```

- [ ] **Step 4: 跑测试,确认全绿**

Run:
```bash
cd c:\Users\yellow\mcp\qtTdx && python -m pytest tests/test_projection_cli.py -v
```
Expected: 4 PASS。

- [ ] **Step 5: 跑全套测试**

Run:
```bash
cd c:\Users\yellow\mcp\qtTdx && python -m pytest tests/ -x -q
```
Expected: 全绿。

- [ ] **Step 6: 提交**

```bash
cd c:\Users\yellow\mcp\qtTdx && git add backtrace/projection/projection_2d.py tests/test_projection_cli.py
git commit -m "feat(projection): projection_2d 新增 --two-day-vec flag + HTML 前缀切换

- --two-day-vec 透传到 core 三函数 (lag=1)
- HTML 输出前缀: 2-D 模式 'proj2d_' (沿用), 4-D 模式 'proj2d_4d_'
- 新增 2 个单测覆盖 FILE_PREFIX / LAG / TWO_DAY_VEC 全局变量

Co-Authored-By: Claude <noreply@anthropic.com>"
```
vec_index, vec_stock, vec_index_norm, vec_stock_norm, norm_params = compute_vectors(
    data_stock, data_index, INDEX_TAG, STOCK_TAG, lag=LAG,
)
# ...
result_df = build_result_df(
    common_idx, vec_index, vec_stock, vec_index_norm, vec_stock_norm,
    projections, residuals, dot_products_after,
    proj_coefficients, proj_magnitudes, proj_prices, resi_prices,
    norm_params, INDEX_TAG, STOCK_TAG, lag=LAG,
)
```

5. `baseline_kind` 那行(line 90-93)后面追加 `print` 提示向量维度:
```python
print(f"基线选择: {baseline_kind}")
print(f"向量维度: {'4-D (今日+前一日 Vol/Amt)' if TWO_DAY_VEC else '2-D (今日 Vol/Amt)'}")
```

6. 输出文件列表的 print(line 305-311)同样按 prefix 输出(已是动态字符串,自动正确)。

- [ ] **Step 4: 跑测试,确认全绿**

Run:
```bash
cd c:\Users\yellow\mcp\qtTdx && python -m pytest tests/test_projection_cli.py -v
```
Expected: 4 PASS。

- [ ] **Step 5: 跑全套测试**

Run:
```bash
cd c:\Users\yellow\mcp\qtTdx && python -m pytest tests/ -x -q
```
Expected: 全绿。

- [ ] **Step 6: 提交**

```bash
cd c:\Users\yellow\mcp\qtTdx && git add backtrace/projection/projection_2d.py tests/test_projection_cli.py
git commit -m "feat(projection): projection_2d 新增 --two-day-vec flag + HTML 前缀切换

- --two-day-vec 透传到 core 三函数 (lag=1)
- HTML 输出前缀: 2-D 模式 'proj2d_' (沿用), 4-D 模式 'proj2d_4d_'
- CSV 文件名不变
- 新增 2 个单测覆盖 FILE_PREFIX / LAG / TWO_DAY_VEC 全局变量

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: 最终回归 — `find_resi_positive.py` 仍可用 + 2-D 模式字节级未变

**Files:**
- 不修改任何文件(纯验证任务)

- [ ] **Step 1: 跑 2-D 模式回归 — 输出 CSV 应与 main 分支一致**

先确认当前 HEAD 是 Task 4 commit(`git log --oneline -1`)。然后:

```bash
cd c:\Users\yellow\mcp\qtTdx && \
    git show main:backtrace/projection/_projection_core.py > /tmp/_projection_core.main.py && \
    git show main:backtrace/projection/projection_batch.py > /tmp/projection_batch.main.py
```

跑 2-D 模式生成当前 CSV:
```bash
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_batch.py \
    --input data/projection/stocks.csv --limit 1 --market-baseline --days 50
```
备份:`cp data/projection/projection_*.csv /tmp/projection_now_2d.csv`(单只票)。

切回 main 分支跑同一个命令,生成 main 模式 CSV:
```bash
git stash && git checkout main -- backtrace/projection/_projection_core.py backtrace/projection/projection_batch.py
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_batch.py \
    --input data/projection/stocks.csv --limit 1 --market-baseline --days 50
cp data/projection/projection_*.csv /tmp/projection_main_2d.csv
```

切回 feat 分支并恢复:
```bash
git checkout feat/projection-industry-baseline -- backtrace/projection/_projection_core.py backtrace/projection/projection_batch.py
git stash pop 2>/dev/null || true
```

比对:
```bash
diff /tmp/projection_main_2d.csv /tmp/projection_now_2d.csv
```
Expected: **空 diff**(2-D 模式输出与 main 完全一致)。

- [ ] **Step 2: 跑 `find_resi_positive.py` 冒烟(用本次产出的 CSV)**

```bash
cd c:\Users\yellow\mcp\qtTdx && \
    PYTHONIOENCODING=utf-8 python backtrace/projection/find_resi_positive.py --no-push --date latest
```
Expected: 跑通,输出 CSV 含 `code` / `resi_price` 列;不抛 KeyError(`Resi_Price` 列存在)。

- [ ] **Step 3: 跑全套测试**

```bash
cd c:\Users\yellow\mcp\qtTdx && python -m pytest tests/ -v
```
Expected: 全部 PASS(含新增的 13 个 projection 测试)。

- [ ] **Step 4: 更新 `docs/api.md`(若有公开 API 变更)**

检查 `docs/api.md` 是否描述了 `_projection_core` 的函数签名。如果有,在 `compute_vectors` / `build_result_df` / `load_pair` 三处加一句:

```
`lag: int = 0` 参数: 0 = (Volume, Amount) 2-D(默认,沿用旧行为);
1 = (Volume, Amount, Volume_prev, Amount_prev) 4-D(由 CLI flag --two-day-vec 触发)。
```

- [ ] **Step 5: 提交(如有文档改动)**

```bash
cd c:\Users\yellow\mcp\qtTdx && git add docs/api.md
git commit -m "docs(api): 标注 _projection_core 三函数新增 lag 参数

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review (执行前自检)

- [x] **覆盖 spec §4.1**:`load_pair` 加 lag — Task 2
- [x] **覆盖 spec §4.2**:`compute_vectors` 加 lag — Task 1
- [x] **覆盖 spec §4.3**:`build_result_df` 加 lag — Task 1
- [x] **覆盖 spec §4.4**:`compute_projections` 不改 — (无 task,全局约束 1 锁定)
- [x] **覆盖 spec §5**:CLI flag + FILE_PREFIX — Task 3 (batch) + Task 4 (single)
- [x] **覆盖 spec §7.1**:2-D 模式字节级回归 — Task 5 step 1
- [x] **覆盖 spec §7.2**:4-D 模式新功能 + 列数 = 27 + Resi_Price 存在 — Task 1 测试 + Task 3 测试 + Task 4 测试
- [x] **覆盖 spec §7.3**:边界 — Task 2 测试 `test_load_pair_lag1_raises_when_data_too_short`
- [x] **覆盖 spec §8**:"本次不做" 项目均不在 plan 内(YAGNI)
- [x] **类型一致**:`lag: int = 0` 在 3 处签名一致;`lag=LAG` 透传值一致
- [x] **无 placeholder**:所有 step 给出可运行代码
- [x] **commit 节奏**:5 个 commit,每个 commit 后跑 pytest