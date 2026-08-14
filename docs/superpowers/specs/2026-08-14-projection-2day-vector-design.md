# `projection` — 2 日向量投影(4-D)spec

> 写于 2026-08-14,为 `backtrace/projection/` 增加"2 日向量"维度扩展。
> 读者:想把个股 / 指数向量从 `(Vol_today, Amt_today)` 扩展到 `(Vol_today, Amt_today, Vol_yesterday, Amt_yesterday)` 的人。
> **现有 2-D 行为保持不变**(默认关闭,CLI flag `--two-day-vec` 显式开启)。

---

## 1. Why this exists(动机)

当前 projection 2-D 模型只用单日的 `(Volume, Amount)` 作向量。这把"量价"当天行为压成 1 个二维点,丢失了**隔夜持续性**信息:

- 个股昨日大单放量 + 今日缩量,在 2-D 空间是两个独立点,看不出"从放量 → 缩量"的方向
- "个股相对大盘的昨日偏差"在 2-D 里被合并到当日残差里,无法区分"今日新故事"和"昨日延续"

直觉上,引入前一日 `(Vol_prev, Amt_prev)` 形成 4-D 向量后,**残差**能分解为「今日新偏离」和「昨日延续偏离」两个分量(由投影矩阵的列空间结构决定),便于回测二者对次日收益的解释力。

---

## 2. Inputs contract(新增项)

| 项 | 现状 | 2-D 模式 (lag=0,默认) | 4-D 模式 (lag=1, --two-day-vec) |
|---|---|---|---|
| 数据源 | 本地 `data/` 缓存(沿用 `P.load_ohlcva(..., use_tq=False)`) | 同 | 同 |
| 字段(个股) | `['Volume', 'Amount', 'Close']` | 同 | + `Volume.shift(1)`, `Amount.shift(1)` 后丢弃首行 |
| 字段(指数) | 同上 | 同 | 同上 |
| 回看 | `--days 240` | 取最近 240 行 | 取最近 240 行后丢弃首行 → 最多 239 行 |
| 向量维度 | 2 | 2 | 4 |
| Min-Max 归一化 | 每列独立到 [0, 1] | 2 个范围 | 4 个范围 |

**关键约束**:启用 4-D 时,数据至少需要 **2 行**(否则 `shift(1)` 全为 NaN)。

---

## 3. Outputs contract

### 3.1 CSV

`data/projection/projection_<INDEX_TAG>_<STOCK_TAG>.csv`(文件名不变,沿用 `<INDEX_TAG>_<STOCK_TAG>` 命名)。列差异:

| 列 | 2-D (lag=0) | 4-D (lag=1) |
|---|---|---|
| `Date` | ✓ | ✓ |
| `Vol_<ix>_raw` | ✓ | ✓ |
| `Amt_<ix>_raw` | ✓ | ✓ |
| `Vol_<st>_raw` | ✓ | ✓ |
| `Amt_<st>_raw` | ✓ | ✓ |
| `Vol_<ix>_norm` | ✓ | ✓ |
| `Amt_<ix>_norm` | ✓ | ✓ |
| `Vol_<st>_norm` | ✓ | ✓ |
| `Amt_<st>_norm` | ✓ | ✓ |
| `Vol_<ix>_prev_raw` | — | ✓ (新增) |
| `Amt_<ix>_prev_raw` | — | ✓ (新增) |
| `Vol_<st>_prev_raw` | — | ✓ (新增) |
| `Amt_<st>_prev_raw` | — | ✓ (新增) |
| `Vol_<ix>_prev_norm` | — | ✓ (新增) |
| `Amt_<ix>_prev_norm` | — | ✓ (新增) |
| `Vol_<st>_prev_norm` | — | ✓ (新增) |
| `Amt_<st>_prev_norm` | — | ✓ (新增) |
| `Proj_Vol` | ✓ | ✓ (今日 Vol 分量) |
| `Proj_Amt` | ✓ | ✓ (今日 Amt 分量) |
| `Residual_Vol` | ✓ | ✓ |
| `Residual_Amt` | ✓ | ✓ |
| `Proj_Coeff` | ✓ | ✓ |
| `Proj_Magnitude` | ✓ | ✓ |
| `Proj_Price` | ✓ | ✓ (`proj[1]/proj[0]`,今日 Amount/Volume 比) |
| `Resi_Price` | ✓ | ✓ (`residual[1]/residual[0]`,今日 Amount/Volume 比) |
| `Dot_After_Proj` | ✓ | ✓ |
| `Norm_Params` | ✓(2 个范围) | ✓(4 个范围,含 prev 维度) |
| **列数** | **19** | **27** |

**`Resi_Price` 语义保持不变**(今日 Amount/Volume 残差比)→ `find_resi_positive.py` 无需任何改动。

### 3.2 HTML (projection_2d.py 单股模式)

输出前缀:
- 2-D: `backtrace/outputs/proj2d_<name>.html`(沿用)
- 4-D: `backtrace/outputs/proj2d_4d_<name>.html`(新前缀,避免覆盖)

6 个图的内容变化:
- `vector_scatter.html` / `4d`: x 轴仍是 `Vol_today_norm`, y 轴仍是 `Amt_today_norm`(只看前 2 个轴)。新增 `vector_scatter_prev.html` 画 `(Vol_prev, Amt_prev)` 散点(可选,本次不做 — YAGNI)
- 其余 5 个图(投影验证、正交性、系数、价格比)逻辑不变,只是数据来自 4-D 投影后的结果

**HTML 改动范围**:本次实现**只改前缀**,散点图仍只画前 2 轴(YAGNI — 多视图留给后续)。

---

## 4. Algorithm(算法变更,3 处)

### 4.1 `load_pair(..., lag: int = 0)`

新增 `lag` 参数。`lag >= 1` 时:

```python
data_index_full = pipeline.load_ohlcva(index_code, use_tq=False, verbose=True)
data_stock_full = pipeline.load_ohlcva(stock_code, use_tq=False, verbose=True)
# ... 缺失检查沿用 ...

if lag >= 1:
    if len(data_index_full) < 2 or len(data_stock_full) < 2:
        raise ValueError(f"--two-day-vec 需要 ≥2 行数据,实际 {index_code}={len(data_index_full)}, {stock_code}={len(data_stock_full)}")
    # 计算 prev 列
    data_index_full = data_index_full.assign(
        Volume_prev=data_index_full['Volume'].shift(1),
        Amount_prev=data_index_full['Amount'].shift(1),
    )
    data_stock_full = data_stock_full.assign(
        Volume_prev=data_stock_full['Volume'].shift(1),
        Amount_prev=data_stock_full['Amount'].shift(1),
    )

data_index = data_index_full[['Volume', 'Amount', 'Volume_prev', 'Amount_prev', 'Close']].tail(days).dropna()
data_stock = data_stock_full[['Volume', 'Amount', 'Volume_prev', 'Amount_prev', 'Close']].tail(days).dropna()
common_idx = data_index.index.intersection(data_stock.index)
```

**`dropna()` 自动丢首行**(因为 prev 是 NaN)。`lag == 0` 时不加 prev 列,行为完全等同旧版本。

### 4.2 `compute_vectors(..., lag: int = 0)`

新增 `lag` 参数。当 `lag >= 1`:

```python
cols = ['Volume', 'Amount']
if lag >= 1:
    cols += ['Volume_prev', 'Amount_prev']

vec_index = index_df[cols].values   # shape: (T, 2*(lag+1))
vec_stock = stock_df[cols].values
# 之后每个维度独立 Min-Max(向量化循环 / list comprehension)
# norm_params 字符串列出全部维度
```

**lag == 0 时函数体不动**(行为完全等同旧版本)。

### 4.3 `build_result_df(...)`

新增 `lag` 参数。当 `lag >= 1`,在 raw/norm 块(列 2-9)之后插入 8 个 prev 列(列 10-17,顺序:`Vol_ix_prev_raw, Amt_ix_prev_raw, Vol_st_prev_raw, Amt_st_prev_raw, Vol_ix_prev_norm, Amt_ix_prev_norm, Vol_st_prev_norm, Amt_st_prev_norm`)。其余 10 列(`Proj_Vol` 起)顺序不变。

**lag == 0 时函数体不动**。

### 4.4 `compute_projections(...)`

**完全不改**。`proj[1]/proj[0]` 和 `residual[1]/residual[0]` 在 4-D 下语义保持(用今日 Vol/Amt 轴)。

---

## 5. CLI changes

### `projection_batch.py`

新增:
```python
parser.add_argument(
    '--two-day-vec', action='store_true',
    help=(
        '将向量扩展为 (Vol_today, Amt_today, Vol_yesterday, Amt_yesterday) 4-D;'
        '首日丢弃。默认 2-D。'
    ),
)
```

`process_one` 把 `args.two_day_vec` 透传给 `load_pair` / `compute_vectors` / `build_result_df`。

`CSV_OUT_DIR` 文件名:**不变**(`projection_<INDEX_TAG>_<STOCK_TAG>.csv`)— 同一只股票跑 2-D 和 4-D 会互相覆盖。如需共存,后续可加 `--output-suffix` (本次不做)。

### `projection_2d.py`

新增:
```python
p.add_argument(
    '--two-day-vec', action='store_true',
    help='将向量扩展为 4-D (今日 + 前一日 Vol/Amt);首日丢弃。',
)
```

`FILE_PREFIX` 根据 flag 切换:
- 2-D: `'proj2d_'`
- 4-D: `'proj2d_4d_'`

其余脚本逻辑不变(只是把 flag 透传到 core 函数)。

---

## 6. Parametrization points

| 位置 | 改它做什么 |
|---|---|
| `_projection_core.load_pair` 的 `lag` 参数 | 未来扩展到 lag=2,3,... 直接改 (只要数据够) |
| `compute_vectors` 的 `cols` 拼接 | 同上 |
| `build_result_df` 的 prev 列插入位置 | 未来增列时改这里 |
| `FILE_PREFIX` 切换 | 想区分模式输出改这里 |

**不需要改**:
- `project_u_onto_v`(维度无关)
- `compute_projections`(维度无关)
- `find_resi_positive.py`(语义保持)

---

## 7. Verification(契约不变性)

### 7.1 2-D 模式(回归测试)

跑下列命令,验证输出与改动前**逐字节一致**:

```bash
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_2d.py --days 240
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_batch.py --input data/projection/stocks.csv --limit 5
```

期望:
- `data/projection/projection_<IX>_<ST>.csv` 与改动前**diff 为空**
- 6 个 HTML 文件结构不变
- `find_resi_positive.py` 输出不变

### 7.2 4-D 模式(新功能)

```bash
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_2d.py --code 002475.SZ --two-day-vec --days 240
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_batch.py --input data/projection/stocks.csv --limit 5 --two-day-vec
```

检查:
- CSV 列数 = 27
- 第一行 `Vol_<ix>_prev_raw` 等于原 2-D CSV 第二行 `Vol_<ix>_raw`(隔日对齐)
- `Proj_Coeff` ∈ ℝ(可以为负;正交性保证 `|Dot_After_Proj| < 1e-12`)
- `find_resi_positive.py` 仍能跑(语义不变)

### 7.3 边界

- 数据只有 1 行时 → `load_pair` 抛 `ValueError(--two-day-vec 需要 ≥2 行)`,`projection_batch.py` 把异常捕获后写入 `status='failed: ...'`
- 数据 2 行 → 正常输出 1 行
- 任意一行 `Vol_prev=NaN`(不应发生,已 `dropna`)→ 投影结果含 NaN,沿用现有 Min-Max 安全处理

---

## 8. Out of scope(本文档不覆盖)

- `lag` 参数化到 2, 3, ... (本次只做 lag=0 / lag=1)
- 4-D 散点图(投影仍只画前 2 轴)
- 下游 `find_resi_positive.py` 区分 2-D / 4-D CSV(本次 CSV 共用目录,文件名相同,**会互相覆盖**)
- 把 `Proj_Price` / `Resi_Price` 拆成 today/yesterday 两个分量(本次保持原语义)
- HTML 输出文件名加 `_4d` 之外的更多区分(如 `--output-suffix`)

---

## 9. 相关引用

- 源码:[backtrace/projection/_projection_core.py](../../../backtrace/projection/_projection_core.py)
- 源码:[backtrace/projection/projection_batch.py](../../../backtrace/projection/projection_batch.py)
- 源码:[backtrace/projection/projection_2d.py](../../../backtrace/projection/projection_2d.py)
- 上游 spec:[2026-08-02-projection-2d-spec.md](2026-08-02-projection-2d-spec.md)
- 下游消费:[backtrace/projection/find_resi_positive.py](../../../backtrace/projection/find_resi_positive.py)
- 父项目说明:[CLAUDE.md](../../../CLAUDE.md)