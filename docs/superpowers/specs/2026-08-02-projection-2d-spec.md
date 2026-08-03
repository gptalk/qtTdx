# `projection_2d.py` — 迁移 / 复用 spec

> 写于 2026-08-02,为 [backtrace/legacy/projection_2d.py](../../../../backtrace/legacy/projection_2d.py) 333 行 legacy 脚本补一份"可迁移 / 可复用"契约。
> 读者:未来要把这个 2-D 投影实验搬到别的股票 / 别的数据源 / 别的可视化栈的人。
> **本文档不动原脚本**(legacy 性质,仅留档)。

---

## 1. Why this exists(动机)

早期正交性可视化实验:**把个股 (002475.SZ 立讯精密) 的二维"成交量-成交额"向量,投影到大盘 (000001.SH 上证指数) 同样二维向量张成的方向上**,看个股的"大盘解释不了"残差有多大。

- **假设**:个股的量价行为 ≈ α × 大盘的量价行为 + 残差(α 是标量系数)。残差垂直分量越大,说明个股当天有独立于大盘的"独特故事"。
- **输出**:7 个 plotly HTML 用于肉眼检验投影是否合理,1 个 CSV 留作后续指标计算。
- **结论**:这个视角后来没用上,正式研究改用 [backtrace/vbt/](../../../../backtrace/vbt/) 和 [backtrace/tsfresh/](../../../../backtrace/tsfresh/) 系列。本脚本在 CLAUDE.md 和 [docs/api.md](../../api.md) 都标为"已不推荐"。

---

## 2. Inputs contract(输入)

| 项 | 现状 | 备注 |
|---|---|---|
| 数据源 | `tq.get_market_data(...)` | 走 [C:/new_tdx_mock/PYPlugins/user/tqcenter.py](C:/new_tdx_mock/PYPlugins/user/tqcenter.py);TQ 客户端必须先启动 |
| 字段 | `['Volume', 'Amount', 'Close']` | front 除权;`fill_data=True` |
| 股票 | `['000001.SH', '002475.SZ']` (硬编码 L19) | 第 1 只 = "大盘基准 v";第 2 只 = "标的 u" |
| 回看 | `days=240` (硬编码 L20) | 实际拉 `days+30` 天;Min-Max 归一化只用最近 `days` 天的数据 |
| 归一化 | Min-Max 到 `[0, 1]` | **每只股票 / 每个字段独立归一化**(4 个独立的 min/max),**不是**全局归一化 |
| 输出目录 | `backtrace/` 根目录(相对 cwd) | 脚本假设从仓库根运行(`cwd == qtTdx/`);从别处跑会写到错误位置 |

---

## 3. Outputs contract(输出)

### 3.1 7 个 plotly HTML

| 文件 | 内容 | 主要看什么 |
|---|---|---|
| `backtrace/vector_scatter.html` | 全期 u / v 向量散点对比 | 两组向量是否真的"接近",还是基本正交 |
| `backtrace/projection_verify.html` | 4 个采样日(0 / 50 / 100 / 174)的 u / v / proj / residual 四向量 + 标注点积 | 直观验证"投影公式"是否被正确实现 |
| `backtrace/orthogonality_check.html` | `(u - proj) · v` 的时序,理想恒为 0 | 这是**算法不变量验证**(数值误差应该 ~1e-16,不应该有非零趋势) |
| `backtrace/proj_coefficient.html` | 投影系数 `u·v / v·v` 时序 | "大盘方向对个股的标量贡献"如何随时间变 |
| `backtrace/proj_magnitude.html` | 投影向量模长 + 002475 收盘价(归一化) | 投影大小是否跟股价联动 |
| `backtrace/proj_function.html` | u 在单位圆上变化、v 固定 `(1,0)` 的投影函数曲线 | 纯函数层面验证投影算子的几何含义 |
| `backtrace/residual_magnitude.html` | 残差模长 + 收盘价 | 个股"独立故事"的强度时序 |

### 3.2 CSV

`backtrace/projection_result.csv`,UTF-8(默认),列:

| 列 | 类型 | 说明 |
|---|---|---|
| `Date` | `datetime` | 共同交易日索引 |
| `Vol_000001_raw` | `float` | 000001.SH 原始 Volume |
| `Amt_000001_raw` | `float` | 000001.SH 原始 Amount |
| `Vol_002475_raw` | `float` | 002475.SZ 原始 Volume |
| `Amt_002475_raw` | `float` | 002475.SZ 原始 Amount |
| `Vol_000001_norm` | `float ∈ [0,1]` | Min-Max 归一化 |
| `Amt_000001_norm` | `float ∈ [0,1]` | 同上 |
| `Vol_002475_norm` | `float ∈ [0,1]` | 同上 |
| `Amt_002475_norm` | `float ∈ [0,1]` | 同上 |
| `Proj_Vol` | `float` | 投影向量在 Volume 轴的分量 |
| `Proj_Amt` | `float` | 投影向量在 Amount 轴的分量 |
| `Residual_Vol` | `float` | 残差向量在 Volume 轴的分量 |
| `Residual_Amt` | `float` | 残差向量在 Amount 轴的分量 |
| `Dot_After_Proj` | `float` | `(u - proj) · v`(理论 0,数值 ~1e-16) |
| `Norm_Params` | `str` | 4 个 Min-Max 范围,每行复制(冗余字段,用于离线复算) |

---

## 4. Algorithm(算法,6 步)

1. **Load**: `tq.get_market_data` 拉 2 只票 × 3 字段 × (240+30) 天 → `tq.price_df` 转 wide DataFrame,取共同交易日(`df.index.intersection`)。
2. **Normalize**: 每只票 / 每个字段独立做 Min-Max 到 [0, 1];得到 `vec_000001_norm` / `vec_002475_norm` 两个 `(N, 2)` ndarray。
3. **Project loop**: 对每个共同交易日:
   - `u = vec_002475_norm[i]`, `v = vec_000001_norm[i]`
   - `proj = project_u_onto_v(u, v)` = `(u·v / v·v) × v`
   - `residual = u - proj`
   - 累计 `proj_coefficients[i] = u·v / v·v`、`proj_magnitudes[i] = ||proj||`
4. **Plot**: 7 个 plotly HTML(见 §3.1)。
5. **Save CSV**: 15 列(见 §3.2)。
6. **Close**: `tq.close()`。

**关键不变量**:
- `(u - proj) · v` 恒为 0(浮点误差 ~1e-16)— 这是数学保证
- `proj_magnitudes ≤ 1`(因为 `||proj|| = |u·v| / ||v||` 且 u 已被归一化)
- `residual_magnitudes ≤ sqrt(2)`(2-D 单位立方体对角线)

---

## 5. Parametrization points(参数化点 / 迁移时改哪里)

| 位置 | 写死值 | 改它做什么 |
|---|---|---|
| L19 `stock_list` | `['000001.SH', '002475.SZ']` | 换标的 / 换大盘基准(注意顺序:第 1 个 = 大盘 v,第 2 个 = 个股 u) |
| L20 `days` | `240` | 改回看窗口 |
| L137 `sample_indices = [0, 50, 100, 174]` | 4 个采样日索引 | 在 `projection_verify.html` 里展示哪 4 天;改成 mid-range / recent 都可以 |
| L271 `theta = np.linspace(0, 2*np.pi, 100)` | 单位圆采样数 | 控制 `proj_function.html` 曲线的光滑度 |
| L228-229 模长 × Close 重缩放 | hardcoded | 想换其他量价指标(成交笔数 / 主动买量)时,这一段要换 |
| 输出路径(L132 / L189 / L208 / L223 / L248 / L268 / L301 / L330) | `'backtrace/*.html'` / `'backtrace/*.csv'` | 想输出到 `outputs/` 或别处要批量替换 |

**不需要改**:
- `project_u_onto_v`(纯数学算子,标量函数,永远正确)
- 归一化公式(Min-Max,标准)
- 投影公式(线性代数标准)

---

## 6. Skeleton code(可运行骨架,迁移者 fork 起点)

下面是一段**可直接 `python xxx.py` 跑通**的 ~80 行骨架,把原 333 行拆成 6 个纯函数。**不依赖 TQ 客户端**(用合成数据演示);真实场景下把 `load_vectors` 换成 TQ 调用即可。

```python
# -*- coding: utf-8 -*-
"""
projection_2d skeleton — 2-D 量价向量投影实验。
用法:python this.py(从仓库根运行,会在 ./out/ 下生成 7 HTML + 1 CSV)
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT_DIR = './out'                      # 迁移时改成你的目标目录
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- 1. 加载(原 TQ → 这里用合成数据)----------
def load_vectors(stock_u: str, stock_v: str, days: int = 240) -> pd.DataFrame:
    """
    返回 {Date, Vol_u, Amt_u, Close_u, Vol_v, Amt_v, Close_v}
    原版: tq.get_market_data + tq.price_df + index.intersection
    """
    dates = pd.bdate_range(end='2025-01-01', periods=days)
    rng = np.random.RandomState(42)
    rows = []
    for d in dates:
        rows.append({
            'Date': d,
            f'Vol_{stock_u}':  abs(rng.normal(1e6, 2e5)),
            f'Amt_{stock_u}':  abs(rng.normal(1e10, 2e9)),
            f'Close_{stock_u}': abs(rng.normal(20, 3)),
            f'Vol_{stock_v}':  abs(rng.normal(2e6, 3e5)),
            f'Amt_{stock_v}':  abs(rng.normal(3e10, 4e9)),
            f'Close_{stock_v}': abs(rng.normal(3000, 100)),
        })
    return pd.DataFrame(rows).set_index('Date')


# ---------- 2. Min-Max 归一化(每列独立)----------
def minmax_normalize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """返回每列 Min-Max 到 [0,1] 后的 ndarray;同时记录 min/max"""
    mins, maxs = {}, {}
    norm = np.zeros((len(df), len(cols)))
    for j, c in enumerate(cols):
        v = df[c].values
        mins[c], maxs[c] = v.min(), v.max()
        norm[:, j] = (v - mins[c]) / (maxs[c] - mins[c])
    return norm, mins, maxs


# ---------- 3. 投影算子(标量函数,永远不改)----------
def project_u_onto_v(u, v):
    """u → v 方向投影;v_norm=0 时返回 0"""
    v_norm_sq = np.dot(v, v)
    if v_norm_sq == 0:
        return np.zeros_like(u)
    return (np.dot(u, v) / v_norm_sq) * v


# ---------- 4. 主循环:每天一个 (proj, residual)----------
def project_loop(u_arr, v_arr):
    """返回 dict 包含 projections, residuals, dot_after, coeffs, mag_proj"""
    n = len(u_arr)
    out = dict(projections=np.zeros_like(u_arr),
               residuals=np.zeros_like(u_arr),
               dot_after=np.zeros(n),
               coeffs=np.zeros(n),
               mag_proj=np.zeros(n))
    for i in range(n):
        u, v = u_arr[i], v_arr[i]
        proj = project_u_onto_v(u, v)
        out['projections'][i] = proj
        out['residuals'][i]   = u - proj
        out['dot_after'][i]   = np.dot(u - proj, v)          # 应 ~0
        out['coeffs'][i]      = np.dot(u, v) / np.dot(v, v)
        out['mag_proj'][i]    = np.linalg.norm(proj)
    return out


# ---------- 5. 7 个 HTML(为了 spec 简洁只画 3 个代表性图)----------
def plot_scatter(u_norm, v_norm, stock_u, stock_v):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v_norm[:, 0], y=v_norm[:, 1], mode='markers', name=stock_v))
    fig.add_trace(go.Scatter(x=u_norm[:, 0], y=u_norm[:, 1], mode='markers', name=stock_u))
    fig.update_layout(title=f'{stock_u} vs {stock_v} 2-D normalized',
                      xaxis_title='Vol', yaxis_title='Amt', template='plotly_dark')
    fig.write_html(f'{OUT_DIR}/vector_scatter.html')


def plot_orthogonality(dot_after, dates):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=dot_after, mode='lines', name='(u - proj) · v'))
    fig.add_trace(go.Scatter(x=dates, y=[0]*len(dates), mode='lines', name='ideal = 0'))
    fig.update_layout(title='Orthogonality check (should be 0)',
                      template='plotly_dark', height=400)
    fig.write_html(f'{OUT_DIR}/orthogonality_check.html')


def plot_projection_curve():
    """u 走单位圆、v 固定 (1,0),画投影曲线"""
    theta = np.linspace(0, 2*np.pi, 100)
    u_circ = np.column_stack([np.cos(theta), np.sin(theta)])
    v_fix  = np.array([1.0, 0.0])
    projs  = np.array([project_u_onto_v(u, v_fix) for u in u_circ])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines', name='u on unit circle'))
    fig.add_trace(go.Scatter(x=projs[:, 0],     y=projs[:, 1],     mode='lines', name='proj(u -> v)'))
    fig.update_layout(title='Projection curve (u varies, v fixed)',
                      xaxis=dict(range=[-1.5, 1.5]), yaxis=dict(range=[-1.5, 1.5]),
                      template='plotly_dark')
    fig.write_html(f'{OUT_DIR}/proj_function.html')


# ---------- 6. CSV 输出 ----------
def save_csv(df_raw, u_norm, v_norm, out, stock_u, stock_v, mins, maxs):
    # 4 个 Min-Max 范围拼成 1 个字符串,每行复制(冗余但方便离线复算)
    norm_str = (
        f"vol_{stock_v}:[{mins[f'Vol_{stock_v}']:.2e},{maxs[f'Vol_{stock_v}']:.2e}] "
        f"amt_{stock_v}:[{mins[f'Amt_{stock_v}']:.2e},{maxs[f'Amt_{stock_v}']:.2e}] "
        f"vol_{stock_u}:[{mins[f'Vol_{stock_u}']:.2e},{maxs[f'Vol_{stock_u}']:.2e}] "
        f"amt_{stock_u}:[{mins[f'Amt_{stock_u}']:.2e},{maxs[f'Amt_{stock_u}']:.2e}]"
    )
    result = pd.DataFrame({
        'Date': df_raw.index,
        f'Vol_{stock_v}_raw': df_raw[f'Vol_{stock_v}'],
        f'Amt_{stock_v}_raw': df_raw[f'Amt_{stock_v}'],
        f'Vol_{stock_u}_raw': df_raw[f'Vol_{stock_u}'],
        f'Amt_{stock_u}_raw': df_raw[f'Amt_{stock_u}'],
        f'Vol_{stock_v}_norm': v_norm[:, 0],
        f'Amt_{stock_v}_norm': v_norm[:, 1],
        f'Vol_{stock_u}_norm': u_norm[:, 0],
        f'Amt_{stock_u}_norm': u_norm[:, 1],
        'Proj_Vol': out['projections'][:, 0],
        'Proj_Amt': out['projections'][:, 1],
        'Residual_Vol': out['residuals'][:, 0],
        'Residual_Amt': out['residuals'][:, 1],
        'Dot_After_Proj': out['dot_after'],
        'Norm_Params': [norm_str] * len(df_raw),
    })
    result.to_csv(f'{OUT_DIR}/projection_result.csv', index=False, encoding='utf-8')


# ---------- main ----------
def main(stock_u='002475.SZ', stock_v='000001.SH', days=240):
    df = load_vectors(stock_u, stock_v, days)
    u_cols = [f'Vol_{stock_u}', f'Amt_{stock_u}']
    v_cols = [f'Vol_{stock_v}', f'Amt_{stock_v}']
    u_norm, mins_u, maxs_u = minmax_normalize(df, u_cols)
    v_norm, mins_v, maxs_v = minmax_normalize(df, v_cols)
    out = project_loop(u_norm, v_norm)

    plot_scatter(u_norm, v_norm, stock_u, stock_v)
    plot_orthogonality(out['dot_after'], df.index)
    plot_projection_curve()
    save_csv(df, u_norm, v_norm, out, stock_u, stock_v,
             {**mins_u, **mins_v}, {**maxs_u, **maxs_v})

    # 不变量 sanity check
    assert np.abs(out['dot_after']).max() < 1e-12, '正交性破坏'
    assert (0 <= u_norm).all() and (u_norm <= 1).all(), 'Min-Max 越界'
    print(f'OK: {len(df)} days, max |dot_after| = {np.abs(out["dot_after"]).max():.2e}')


if __name__ == '__main__':
    main()
```

骨架刻意只画 3 个 HTML(原版 7 个);读者按 §5 的清单把缺的 4 个补上即可。

---

## 7. Migration recipes(迁移示例)

### 7.1 换股票(标的 + 大盘)

```python
# 原: 002475.SZ ← 000001.SH
# 改成: 600519.SH 贵州茅台 ← 000300.SH 沪深300
main(stock_u='600519.SH', stock_v='000300.SH')
```

只需换参数,其余不动。如果 `stock_v` 不是指数而是另一只票(比如同行业),语义从"大盘投影"变成"行业投影"。

### 7.2 换数据源(TQ → 本地 CSV)

> 前置条件:`backtrace/{stock}_daily.csv` 必须存在,且至少含 `Volume / Close` 列(`Amount` 列可选;缺时按 `Volume × Close` 估算)。
> 没有可用的本地 CSV 时,**先**用 [backtrace/common/tsfresh_pipeline.py](../../../../backtrace/common/tsfresh_pipeline.py) 的 `P.load_ohlcva(stock, verbose=True)` 拉一份。

把 §6 的 `load_vectors` 换成:

```python
def load_vectors(stock_u, stock_v, days=240):
    base = P.BACKTRACE_DIR   # backtrace/common/tsfresh_pipeline as P
    df_u = pd.read_csv(f'{base}/{stock_u.replace(".", "_")}_daily.csv', index_col=0, parse_dates=True)
    df_v = pd.read_csv(f'{base}/{stock_v.replace(".", "_")}_daily.csv', index_col=0, parse_dates=True)
    common = df_u.index.intersection(df_v.index)[-days:]
    df_u, df_v = df_u.loc[common], df_v.loc[common]
    return pd.DataFrame({
        'Date': common,
        f'Vol_{stock_u}':  df_u['Volume'],
        f'Amt_{stock_u}':  df_u['Amount'] if 'Amount' in df_u else df_u['Volume'] * df_u['Close'],
        f'Close_{stock_u}': df_u['Close'],
        f'Vol_{stock_v}':  df_v['Volume'],
        f'Amt_{stock_v}':  df_v['Amount'] if 'Amount' in df_v else df_v['Volume'] * df_v['Close'],
        f'Close_{stock_v}': df_v['Close'],
    }).set_index('Date')
```

### 7.3 换可视化栈(plotly → matplotlib)

`plot_*` 函数每个只用了 `go.Figure / add_trace / update_layout / write_html` 4 个 API。matplotlib 等价:
- `go.Figure()` → `fig, ax = plt.subplots()`
- `add_trace(go.Scatter(...))` → `ax.plot(...)` 或 `ax.scatter(...)`
- `update_layout(...)` → `ax.set_title / set_xlabel / set_ylabel`
- `write_html(...)` → `fig.savefig(...)`(PNG)

子图(`projection_verify.html`)需要 `make_subplots` → `plt.subplots(2, 2)` + `for ax in axes.flat`。

---

## 8. Verification(契约不变性 — 跑通后必须过)

| 检查项 | 期望值 | 怎么查 |
|---|---|---|
| 正交性 | `\|dot_after\|.max() < 1e-12` | `assert np.abs(out['dot_after']).max() < 1e-12` |
| Min-Max 范围 | `0 ≤ norm ≤ 1`(每列独立) | `assert (0 <= u_norm).all() and (u_norm <= 1).all()` |
| CSV 列数 | 15 列(`Date` + 13 数据列 + `Norm_Params`) | `pd.read_csv(...).shape[1] == 15` |
| 共同交易日数 | 至少 100 天(避免 Min-Max 噪声) | `len(common_idx) >= 100` |

骨架末尾的 `assert` 已经覆盖前两项;后两项由调用方按需加。

---

## 9. Out of scope(本文档不覆盖)

- 把 `projection_2d.py` 重构成 importable module(legacy,不动)
- 把脚本里的 7 HTML 重做成 plotly 模板 / Dash app
- 把投影公式换成更复杂的(主成分分析 / 独立成分分析)— 是另一个 spec
- 用结果 CSV 做下游策略回测 — 走 [backtrace/vbt/](../../../../backtrace/vbt/) 或 [backtrace/tsfresh/](../../../../backtrace/tsfresh/) 体系

---

## 10. 相关引用

- 源码:[backtrace/legacy/projection_2d.py](../../../../backtrace/legacy/projection_2d.py)
- API 文档:[docs/api.md §legacy](../../api.md#backtracelegacy--过时模板5-个文件已不推荐)
- 父项目说明:[CLAUDE.md](../../../../CLAUDE.md)
- TQ 数据接入:[docs/api.md §tqcenter 跨引用](../../api.md#tqcenter-跨引用外部)