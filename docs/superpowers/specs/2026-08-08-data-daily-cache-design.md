# `data/` 本地日线缓存 — 设计 spec

> 写于 2026-08-08。目标:建立仓库根 `data/` 目录作为 TQ 日线数据的本地镜像,**修复当前已失效的本地 CSV 回退路径**,让 `backtrace/` 下现有脚本在 TQ 客户端未启动时仍可运行。
> 读者:实现本功能的人,以及未来需要改动本地缓存布局的人。

---

## 1. Why this exists(动机)

### 1.1 现存缺陷:本地回退是坏的

[backtrace/common/tsfresh_pipeline.py](../../../backtrace/common/tsfresh_pipeline.py) 第 50-55 行:

```python
def _try_local_csv(code):
    """回退路径:从 backtrace/{code}_daily.csv 读"""
    p = os.path.join(C.BACKTRACE_DIR, f'{code.replace(".", "_")}_daily.csv')
```

它读 `backtrace/{code}_daily.csv`,但仓库里仅有的两个日线 CSV 位于 `backtrace/outputs/`:

```
backtrace/outputs/000001_SH_daily.csv
backtrace/outputs/002475_SZ_daily.csv
```

**路径对不上,回退分支永远返回 `None`。** `load_ohlcva` / `load_sector` 在 TQ 拉取失败时看似有兜底,实际直接吐 `None`,下游脚本报的是无关的 `NoneType` 错误,掩盖了真实原因(TQ 客户端没启动)。

根因是**写数据的代码和读数据的代码各自持有一份路径知识**,改一处没同步另一处。本设计要从结构上消除这种可能。

### 1.2 目标

- 建立 `data/` 作为沪深全市场日线的本地镜像
- 修复 `_try_local_csv`,使离线运行真正可用
- 路径/schema 知识收敛到单一模块,杜绝同类 bug 复发

---

## 2. 已确认的决策

| 决策点 | 选定 | 理由 |
|---|---|---|
| 主要用途 | 修回退 + 离线跑现有脚本 | 决定了「一票一 CSV」而非合并面板 |
| 股票范围 | 沪深 A 股,排除 ST/退市 | 回测通常用不上 ST;少 ~200 只且避开数据残缺样本 |
| 行业板块 | 申万二级 128 行业指数 | CLAUDE.md 标注其为「干净的真行业」 |
| 历史长度 | **500 个交易日** | 留缓冲,以后窗口放到 500 日不用重拉 |
| 目录位置 | **仓库根 `data/`** | 区别于已被占用的 `backtrace/gp_factor_mining/data/` |
| 模块方案 | 新建 `data_store.py`,读写同源 | 见 §3 |

### 2.1 方案选型

考虑过三条路:

- **A. 独立脚本 + 给 `_try_local_csv` 打补丁** — 改动最小,但磁盘布局知识仍散在两个文件,正是当前 bug 的成因,等于把结构缺陷复制一遍。
- **B. 落盘逻辑塞进 `tsfresh_pipeline.py`** — 不新增文件,但该模块已 300+ 行、身兼拉数/特征/标签/筛选四职,再加一项更难拆。
- **C. 新建 `backtrace/common/data_store.py`,读写同源** — **选定**。

选 C 的理由不是「更整洁」,而是本次要修的 bug 恰恰源于读写路径分家。C 让 `_try_local_csv` 退化为对 `data_store.load_daily` 的一行委托,读写共用同一个 `csv_path`,路径不一致在结构上不再可能发生。

---

## 3. 模块边界

### 3.1 `backtrace/common/data_store.py`(新增)

**唯一职责:定义本地日线缓存长什么样、放在哪、怎么读写。** 纯文件 IO,**不 import TQ** —— 因此离线读取无需 TQ 客户端在场,也使该模块可独立测试。

公开 API:

| 函数 | 签名 | 说明 |
|---|---|---|
| `csv_path` | `(code, kind='stocks') -> str` | **路径的唯一真相**。`kind ∈ {'stocks','sectors','indices'}` |
| `save_daily` | `(code, df, kind='stocks') -> str` | 原子写(先 `.tmp` 后 `os.replace`),返回落盘路径 |
| `load_daily` | `(code) -> DataFrame \| None` | 按 `stocks → sectors → indices` 顺序查找;全都没有返回 `None` |
| `has_daily` | `(code) -> bool` | 供 manifest / 断点续传判断 |

`load_daily` 跨三个子目录查找,是为了让调用方**不需要知道**某个代码属于个股还是指数 —— `_try_local_csv(code)` 只有 code,没有 kind。

### 3.2 `backtrace/data_fetch/fetch_daily.py`(新增)

**唯一职责:拉取编排** —— 建 universe、分批、重试、写 manifest。落盘一律通过 `data_store.save_daily`,自身不拼路径。

放在新目录 `backtrace/data_fetch/` 而非 `backtrace/data/`,避免与仓库根 `data/` 数据目录同名混淆。

### 3.3 `tsfresh_pipeline._try_local_csv`(修改)

```python
def _try_local_csv(code):
    """回退路径:委托给 data_store(仓库根 data/)"""
    from common import data_store
    return data_store.load_daily(code)
```

对 `load_ohlcva` / `load_sector` 的调用契约零改动 —— 仍是「有数据返回 DataFrame,没有返回 `None`」。

---

## 4. 磁盘布局

```
data/                                   # 仓库根;.gitignore 的全局 *.csv 已覆盖
├── stocks/
│   ├── 000001_SZ_daily.csv             # 沪深 A 股(去 ST/退市),~5200 只
│   └── ...
├── sectors/
│   ├── 880xxx_SH_daily.csv             # 申万二级 128 行业指数
│   └── ...
├── indices/
│   ├── 000001_SH_daily.csv             # 上证综合指数
│   └── 399001_SZ_daily.csv             # 深证成分指数
└── manifest.json
```

### 4.1 文件名与 schema

沿用仓库现有约定,**不发明新格式**:

- 文件名:`{code}_daily.csv`,`.` 替换为 `_`(如 `000001.SH` → `000001_SH_daily.csv`)
- schema:无名日期索引 + `Open,High,Low,Close,Volume,Amount`
- 编码 `utf-8`,`index=True`,索引为 `YYYY-MM-DD`

因此现有 `backtrace/outputs/*_daily.csv` 可直接移入 `data/indices/` 与 `data/stocks/` 复用。

### 4.2 `manifest.json`

```json
{
  "generated_at": "2026-08-08T21:30:00",
  "trading_days": 500,
  "entries": {
    "000001.SZ": {"kind": "stocks", "rows": 500,
                  "start": "2024-07-15", "end": "2026-08-07",
                  "fetched_at": "2026-08-08T21:12:03", "status": "ok"},
    "880472.SH": {"kind": "sectors", "name": "半导体", "rows": 500, "...": "..."},
    "300xxx.SZ": {"status": "failed", "reason": "TQ 返回空列"}
  }
}
```

行业指数额外记 `name`(来自 `get_stock_list('11', list_type=1)` 的 `it['Name']`),使 `880xxx` 可读。

---

## 5. 拉取编排

### 5.1 四个步骤

**1) 建 universe**

- 个股:见下方「universe 来源」
- 行业:`tq.get_stock_list('11', list_type=1)` → 128 条 `{'Code','Name'}`;`Code` 可直接喂 `get_market_data`(已由 [tsfresh_top1_industry.py:46-56](../../../backtrace/tsfresh/tsfresh_top1_industry.py) 跑通)
- 指数:硬编码 `['000001.SH', '399001.SZ']`

**universe 来源(主 + 备,消除未知):**

`tq.get_stock_list` 取沪深两市所需的 `market` / `list_type` 实参值当前未知(仅已知 `'11'` 对应申万二级)。实现时按以下顺序,**不猜实参**:

1. **主路径** — 先探测并打印 `get_stock_list` 在若干候选实参下的返回结构,确认哪组给出沪深全市场,再据实写死
2. **备路径** — 若主路径无法确定,改用 **128 个申万二级行业的成分股并集**:`get_stock_list_in_sector(code)` 对这批 `Code` 已验证可用([tsfresh_top1_industry.py:69](../../../backtrace/tsfresh/tsfresh_top1_industry.py)),覆盖面接近全市场,且顺带拿到行业归属

两条路径都产出 `List[str]` 代码列表,后续步骤无差异。

**ST/退市过滤:** 剔除 `Name` 中(不区分大小写)含 `ST`(涵盖 `*ST`/`SST`)或含 `退` 的标的。

**2) 分批拉**

批大小 **250**,约 21 批。依据:CLAUDE.md 记录 6000 只会 timeout、~600 只可行,250 留足余量。每批拉完立即落盘,不在内存攒全量。

字段 `['Open','High','Low','Close','Volume','Amount']`,`dividend_type='front'`,`period='1d'`,`fill_data=True`。

**3) 每批健康检查(关键)**

CLAUDE.md 明确记载:TQ 客户端未启动时 `get_market_data` 会**「假装成功」返回空数据**。因此每批必须验:

```python
if df is None or 'Close' not in df or df['Close'].shape[1] == 0:
    raise RuntimeError("TQ 返回空列 —— 客户端可能未启动")
```

**为空立即中止整轮并非零退出**,而非继续跑。否则会用空 CSV 覆盖掉上一轮的好数据 —— 静默的数据损坏比显式崩溃危险得多。

**4) manifest + 断点续传**

- 重跑默认跳过 manifest 中当天已 `status == "ok"` 的代码
- `--force` 忽略 manifest 全量重拉
- 单批失败:重试 1 次;仍失败则记 `status: "failed"` + 原因,继续下一批(区别于步骤 3 的全局中止 —— 个别代码拉不到属正常,全部为空则是环境问题)

### 5.2 时间窗口

500 个交易日需向 TQ 请求约 **780 个自然日**,拉回后取尾部 500 行:

```python
df = df.sort_index().tail(500)
```

> 780 的来历:实测 `backtrace/outputs/000001_SH_daily.csv` 为 181 行 / 270 自然日,交易日占比 **0.670**。500 / 0.670 ≈ 746,取 780 留约 5% 余量。请求多余天数的成本几乎为零(TQ 按区间返回),少拉却要整轮重来。

不足 500 行的(次新股等)照落盘,manifest 如实记录 `rows`,不补齐、不丢弃。

### 5.3 原子写

先写 `{path}.tmp` 再 `os.replace(tmp, path)`。`os.replace` 在同文件系统内是原子的,中途 Ctrl-C 不会留下半截 CSV 被下次当成有效缓存读走。

---

## 6. 规模预估

| 项 | 数量 | 磁盘 |
|---|---|---|
| 沪深 A 股(去 ST/退市) | ~5200 | ~160 MB |
| 申万二级行业指数 | 128 | ~4 MB |
| 上证综指 + 深证成指 | 2 | ~60 KB |

单文件 500 行 × 7 列 ≈ 30 KB。全量一轮预计 20-40 分钟,建议 `run_in_background=true`。

---

## 7. 错误处理汇总

| 场景 | 行为 |
|---|---|
| TQ 客户端未启动(`tq.initialize` 抛 `RuntimeError`) | 立即退出,提示先启动客户端 |
| TQ「假装成功」返回空列 | **中止整轮**,非零退出(§5.1 步骤 3) |
| 单批超时 / 异常 | 重试 1 次 → 仍失败记 `failed` 继续 |
| 个别代码无数据 | manifest 记 `failed`,不写 CSV |
| 行情不足 500 行 | 照写,manifest 记实际 `rows` |
| 板块指数无 `Amount` 列 | 该列留空,其余字段照写(运行时验证) |
| 中途 Ctrl-C | 已落盘文件完好(原子写);manifest 保留已完成项,下次续传 |

---

## 8. 验证方式

本项目无测试框架,依下列可执行检查确认:

1. **`data_store` 往返(无需 TQ)** — 造一个 DataFrame → `save_daily` → `load_daily` → 断言 schema 与数值一致
2. **路径一致性(无需 TQ)** — 断言 `csv_path` 产出与 `load_daily` 查找路径同源;这是防 bug 复发的核心断言
3. **回退真的修好了(无需 TQ)** — `P.load_ohlcva('000001.SH', use_tq=False)` 必须返回 DataFrame 而非 `None`(修复前必然是 `None`,可作前后对照)
4. **小批量实跑(需 TQ)** — 先用 `--limit 20` 跑通全流程,验证 CSV 落盘 + manifest 正确,再放全量
5. **断点续传(需 TQ)** — 跑到一半 Ctrl-C,重跑确认跳过已完成项且无半截文件

---

## 9. 范围外(YAGNI)

明确**不做**:

- 合并面板 / parquet 输出(`gp_factor_mining/data/panel.parquet` 另有其道)
- 增量追加(每轮重拉 500 日全量;A 股日线量级下省不出多少)
- 分钟线 / 复权因子 / 财务数据
- 股票↔行业归属映射表(本次无消费方)
- 自动定时拉取
- 清理 `tsfresh_top1_industry.py:44` 的死代码(与本功能无关)
- 迁移 `backtrace/outputs/` 下的历史 CSV(可手工复制;不写迁移脚本)
