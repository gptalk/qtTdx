# `backtrace/` API 参考

> **单源真相**。本文档覆盖 [backtrace/](../backtrace/) 下所有 35 个 `.py` 文件的公开 API。
> [CLAUDE.md](../CLAUDE.md) 仅保留导航 + 已知陷阱,API 表格全部搬到这里。
> **修改公开函数时,请同步更新本文档。**

---

## 如何读本文档

每节对应一个模块,先列一句话模块说明,再用 markdown 表格列公开函数 / 常量。
表格列含义:
- **签名**:函数/常量原貌(行内 `code` 块)
- **一句话**:做什么
- **参数 / 返回**:关键参数和返回 schema
- **备注**:副作用、坑、跨脚本注意点

所有源码路径相对于仓库根 `c:/Users/yellow/mcp/qtTdx/`。

---

## 快速开始

```bash
# 1. Windows GBK 终端必加 — 中文 print 不报 UnicodeEncodeError
PYTHONIOENCODING=utf-8 python backtrace/<subdir>/<script>.py

# 2. TQ 客户端必须先启动 — 否则 tq.initialize() 抛 RuntimeError

# 3. 导入约定 — 把 backtrace 加到 sys.path 后 from common import ...
from common import tsfresh_config as C, tsfresh_pipeline as P, jhzq_fees as F

# 4. 后台跑 — 长任务(3 小时+)用 run_in_background=true,监控用 TaskOutput
```

详见 [CLAUDE.md §跑脚本的最快姿势](../CLAUDE.md)。

---

## `backtrace/common/` — 共享模块

3 个文件,所有研究脚本都从这里 import。

### [`backtrace/common/tsfresh_pipeline.py`](../backtrace/common/tsfresh_pipeline.py)

11 个公开函数;tqfresh pipeline 的核心流水线(拉数据 → 提特征 → 打标签 → 训模型)。

| 函数 | 一句话 | 参数 | 返回 | 备注 |
|---|---|---|---|---|
| [`init_tq_path()`](../backtrace/common/tsfresh_pipeline.py) | 解析 `tq.initialize` 所需的脚本路径 | — | `str` | 兜底链:`__file__` → `sys.argv[0]` → `cwd` |
| [`load_ohlcva(code, lookback_years=None, use_tq=True, verbose=False, include_amount=True)`](../backtrace/common/tsfresh_pipeline.py) | TQ 优先 → 本地 CSV 回退,拉单只票 OHLCV(+Amount) | `code`:`'600118.SH'`;`include_amount=False` 时 CSV 无 Amount 时用 | `DataFrame`(DatetimeIndex × Open/High/Low/Close/Volume[/Amount]) | `data_store.load_daily` 跨 stocks/sectors/indices 查找(失败返回 None) |
| [`load_sector(sector_name=None, lookback_years=None, use_tq=True, verbose=False, include_amount=True)`](../backtrace/common/tsfresh_pipeline.py) | TQ 拉整个板块 → 回退 `LOCAL_FALLBACK_CODES` | `sector_name`:默认 `C.SECTOR_NAME` (`'通达信88'`) | `Dict[code, DataFrame]` | 板块拉空时抛 RuntimeError |
| [`to_long_format(ohlcv_df, channels=None, id_value=0)`](../backtrace/common/tsfresh_pipeline.py) | OHLCV → tsfresh long format | `channels`:`None`=Open/High/Low/Close/Volume;`id_value`:单只票可固定 0 | `DataFrame[id, time, kind, value]` | 内部 `pd.to_numeric(..., errors='coerce')`,NaN 由 tsfresh `impute` 兜底 |
| [`extract_window_features(long_df, window=None, use_kind=False, roll=True, verbose=True)`](../backtrace/common/tsfresh_pipeline.py) | tsfresh 滑窗提特征(强制 impute) | `window`:`None`=`C.WINDOW`(30);`roll=False` 时整段历史当 1 样本 | `DataFrame`(id 或 `(id, end_t)` × N 特征) | Windows 下 `n_jobs=0` 强制单进程 |
| [`make_labels(X, close_arr, horizon=None, ref_arr=None, verbose=True)`](../backtrace/common/tsfresh_pipeline.py) | 窗口结束 → `horizon` 日后的二分类标签 | `ref_arr`:`None`=绝对涨跌,否则"跑赢大盘" | `(y: Series, X_filtered: DataFrame)` | 尾部越界样本被丢弃 |
| [`select_relevant(X, y, fdr_level=None, verbose=True)`](../backtrace/common/tsfresh_pipeline.py) | FDR 多重检验校正 | `fdr_level`:`None`=`C.FDR_LEVEL`(0.05) | `DataFrame`(显著特征子集) | 0 特征时自动放宽到 0.20 |
| [`fit_logreg(X, y, verbose=True)`](../backtrace/common/tsfresh_pipeline.py) | StandardScaler + 平衡 LogisticRegression | 用所有 `C.LR_*` 配置 | `(scaler, clf)` | predict 时记得先 `scaler.transform(X)` |
| [`align_window_features(X_win, ref_cols, fill_value=0.0)`](../backtrace/common/tsfresh_pipeline.py) | 单行特征 reindex 到训练列结构 | 缺失列填 `fill_value` | `DataFrame`(列对齐) | 不同股票窗口长度可能不同 / tsfresh 跳过短序列 |
| [`csv_path(kind, code)`](../backtrace/common/tsfresh_pipeline.py) | `backtrace/outputs/tsfresh_{kind}_{code}.csv` | — | `str` 路径 | 同 code 会覆盖 |
| [`timestamped_csv_path(kind, ext='csv')`](../backtrace/common/tsfresh_pipeline.py) | `backtrace/outputs/tsfresh_{kind}_{YYYYMMDD_HHMMSS}.{ext}` | — | `str` 路径 | 不覆盖历史 |

---

### [`backtrace/common/jhzq_fees.py`](../backtrace/common/jhzq_fees.py)

江海证券 A 股真实费率;5 个常量 + 4 个函数。**关键:输出 schema 是中文列/键名**。

#### 费率常量

| 常量 | 值 | 说明 |
|---|---|---|
| `STOCK_COMMISSION` | `0.000085` | 双向 万 0.85,**免 5**(无最低收费);vbt 内部只按双边 fee 收,我们要补完整 |
| `STAMP_TAX_SELL` | `0.0005` | 仅卖出 万 5 |
| `TRANSFER_SH` | `0.00001` | 仅沪市双边 万 0.1;深市为 0 |
| `ETF_COMMISSION` | `0.00005` | ETF(预留,未用) |
| `BOND_COMMISSION` | `0.00005` | 转债(预留,未用) |

#### 函数

| 函数 | 一句话 | 参数 | 返回 |
|---|---|---|---|
| [`is_sh(code)`](../backtrace/common/jhzq_fees.py) | 是否沪市 | `code`:`str` | `bool`(以 `.SH` 结尾) |
| [`calc_single_fee(amount, is_sell, is_sh_market)`](../backtrace/common/jhzq_fees.py) | 单笔成交完整费用 | 金额 / 是否卖出 / 是否沪市 | `(总费用, 佣金, 印花税, 过户费)` 4 元组 |
| [`adjust_trades_pnl(trades_df, stock_code)`](../backtrace/common/jhzq_fees.py) | 修正 VBT trades 表,补完整手续费 → 净盈亏 | `trades_df`:`pf.trades.records_readable` | 见下方 Schema |
| [`summary_after_fees(trades_df, stock_code)`](../backtrace/common/jhzq_fees.py) | 汇总扣费后整体指标 | 同上 | 见下方 Dict Keys |

#### `adjust_trades_pnl` 输出 Schema(新增 5 列)

| 列名 | 类型 | 说明 |
|---|---|---|
| `佣金_实扣` | `float` | 买卖双边佣金之和 |
| `印花税_实扣` | `float` | 仅卖出端的印花税 |
| `过户费_实扣` | `float` | 仅沪市双边,深市为 0 |
| `总手续费` | `float` | 上述三项合计 |
| `净盈亏_扣费后` | `float` | VBT 原 PnL 扣减总手续费后的净 PnL |

#### `summary_after_fees` 返回 dict Keys

| Key | 类型 | 说明 |
|---|---|---|
| `trades` | `int` | 成交笔数 |
| `gross_pnl` | `float` | 扣费前 PnL 合计(VBT 原 PnL) |
| `total_stamp` | `float` | 印花税合计 |
| `total_transfer` | `float` | 过户费合计 |
| `net_pnl` | `float` | 净盈亏合计 |
| `avg_net_per_trade` | `float` | 单笔平均净盈亏 |

---

### [`backtrace/common/tsfresh_config.py`](../backtrace/common/tsfresh_config.py)

14 个常量;改这里等于改所有脚本。

| 常量 | 值 | 说明 |
|---|---|---|
| `TQ_PLUGINS_DIR` | `'C:/new_tdx_mock/PYPlugins/user'` | `tqcenter.py` 所在目录,`sys.path.insert` 用 |
| `LOCAL_FALLBACK_CODES` | `['000001.SH', '002475.SZ']` | TQ 失败时回退的本地 CSV 代码 |
| `SECTOR_NAME` | `'通达信88'` | 默认板块(88 个通达信行业) |
| `LOOKBACK_YEARS` | `5` | 默认回看年数,实际取 `LOOKBACK_YEARS*365+30` 天 |
| `WINDOW` | `30` | tsfresh 滚动窗口大小(bar 数) |
| `HORIZON` | `5` | 标签 = 窗口结束后 N 日的收益 |
| `FDR_LEVEL` | `0.05` | Benjamini-Hochberg FDR 阈值;0 特征时 `select_relevant` 自动放宽到 0.20 |
| `TSFRESH_N_JOBS` | `0` | **0 = 单进程,Windows 下 multiprocessing 卡死,生产环境必须保持 0** |
| `LR_C` | `0.5` | LR 正则强度倒数 |
| `LR_MAX_ITER` | `2000` | LR 最大迭代次数 |
| `LR_CLASS_WEIGHT` | `'balanced'` | 自动平衡正负样本 |
| `LR_RANDOM_STATE` | `42` | 复现实验结果 |
| `BACKTRACE_DIR` | `<abs>/backtrace/` | 自动解析 |
| `OUTPUTS_DIR` | `<abs>/backtrace/outputs/` | gitignored |
| `DATA_DIR` | 仓库根 `data/` 绝对路径 | 本地日线缓存根目录 |

---

### `backtrace/common/data_store.py`

本地日线缓存的唯一真相源。纯文件 IO,不依赖 TQ。

| 函数 | 签名 | 说明 |
|---|---|---|
| `csv_path` | `(code, kind='stocks') -> str` | 路径唯一真相;`kind ∈ ('stocks','sectors','indices')`,非法值抛 `ValueError` |
| `save_daily` | `(code, df, kind='stocks') -> str` | 原子写(.tmp + os.replace),返回落盘路径 |
| `load_daily` | `(code) -> DataFrame \| None` | 按 stocks→sectors→indices 查找,都没有返回 None |
| `has_daily` | `(code) -> bool` | 任一 kind 下存在该 code |
| `manifest_path` | `() -> str` | `data/manifest.json` |
| `load_manifest` | `() -> dict` | 不存在时返回 `{'generated_at':None,'trading_days':None,'entries':{}}` |
| `save_manifest` | `(man) -> str` | 原子写 |

模块级 `DATA_DIR`(默认 `tsfresh_config.DATA_DIR`)可 monkeypatch 用于测试。

---

### `backtrace/data_fetch/fetch_daily.py`

日线批量拉取编排。落盘经由 `data_store`,自身不拼路径。

| 函数 | 签名 | 说明 |
|---|---|---|
| `filter_st` | `(items) -> list[str]` | 剔除 ST/*ST/退市;跳过 None 与缺 Code 条目 |
| `chunked` | `(seq, size=250) -> Iterator[list]` | 分批 |
| `calendar_days_for` | `(trading_days=500) -> int` | 交易日→自然日(占比 0.670,余量 5%) |
| `trim_tail` | `(df, n=500) -> DataFrame` | 排序取尾;不足不补齐 |
| `build_sector_universe` | `(tq) -> (list[str], dict)` | 128 申万二级行业码 + 中文名 |
| `build_stock_universe` | `(tq, sector_codes) -> list[str]` | 行业成分股并集去 ST |
| `fetch_batch` | `(tq, codes, start, end) -> dict` | 一批 OHLCVA;**空返回抛 RuntimeError** |
| `main` | `() -> int` | CLI 入口,退出码 |

CLI:`--limit N` 冒烟 / `--force` 忽略 manifest / `--probe` 探测 TQ 列表接口。

常量:`TRADING_DAYS=500`、`BATCH_SIZE=250`、`INDEX_CODES=['000001.SH','399001.SZ']`。

---

## `backtrace/gp_factor_mining/` — GP 因子挖掘子项目

9 个 `.py` + 1 个 [README.md](../backtrace/gp_factor_mining/README.md)。运行顺序:`00 → 01 → 02 → 03 → 05 → 06 → 07`(04 是共用模块)。`small_test.py` 是冒烟测试。

### [`backtrace/gp_factor_mining/00_config.py`](../backtrace/gp_factor_mining/00_config.py)

~45 个常量,9 个 banner 分组。**集中改这块,不要散落到各模块**。

| 分组 | 常量示例 | 说明 |
|---|---|---|
| 路径 | `PROJECT_ROOT` / `GP_DIR` / `DATA_DIR` / `FACTOR_DIR` / `LOG_DIR` | 自动 mkdir;`panel.parquet` 落 `DATA_DIR` |
| 数据源 | `USE_TQ=True` / `TQ_SECTOR="沪深A股"` / `TQ_INIT_PATH` | TQ 优先,失败回退 CSV |
| 时间窗 | `TRAIN_START/END` / `TEST_START/END` / `DATA_FETCH_START` | 训练 vs 测试;fetch 要往前推 60 天做 warm-up |
| 股票池 | `MIN_LIST_DAYS=60` / `MIN_PRICE=2.0` / `EXCLUDE_ST=True` / `EXCLUDE_SUSPEND=True` | 上市天数 / 壳股 / ST / 停牌 过滤 |
| 标签 | `HOLD_PERIOD=20` / `LABEL_NAME="fwd_ret_20d"` | 持有期 20 个交易日 ≈ 1 个月 |
| GP | `POP_SIZE=2000` / `N_GENERATIONS=30` / `PARSIMONY_COEFFICIENT=0.001` | gplearn SymbolicRegressor 参数;parsimony 防膨胀 |
| 残差 | `N_RESIDUAL_ROUNDS=5` / `MIN_IMPROVE_IC=0.005` / `RESIDUAL_TOP_K=5` | 多轮残差挖掘 |
| 入库门槛 | `IN_SAMPLE_ICIR_MIN=1.5` / `OUT_SAMPLE_IC_MIN=0.04` / `MAX_TURNOVER=1.0` / `MAX_CORR_WITH_POOL=0.70` | 因子进正式因子库的硬门槛 |
| 回测 | `BACKTEST_TOP_N=50` / `BACKTEST_REBAL_FREQ="M"` / `INIT_CASH=1_000_000` / `FEE_RATE=0.0003` / `SLIPPAGE=0.001` / `SIZE_GRANULARITY=100` | vbt 回测参数;A 股最小 100 股 |

---

### [`backtrace/gp_factor_mining/01_data_prep.py`](../backtrace/gp_factor_mining/01_data_prep.py)

TQ 取数 → 清洗 → 截面标准化 → 落 `DATA_DIR/panel.parquet`。

| 函数 | 一句话 | 返回 |
|---|---|---|
| [`fetch_from_tq(codes, start, end)`](../backtrace/gp_factor_mining/01_data_prep.py) | 批量拉日线(500/批);逐票 try/except 不抛 | `dict[code] → DataFrame` |
| [`fetch_panel()`](../backtrace/gp_factor_mining/01_data_prep.py) | 主入口:拉数 → 过滤 → 标签 → 截面标准化 → 落盘 | `DataFrame`(内存 + `panel.parquet`) |

---

### [`backtrace/gp_factor_mining/02_primitive_set.py`](../backtrace/gp_factor_mining/02_primitive_set.py)

GP 算子集(时序 + 截面)。**时序特征预计算当 Terminal,不入 gplearn**(原生不支持 group/rolling)。

| 函数 | 一句话 | 输入 | 输出 |
|---|---|---|---|
| [`add_timeseries_primitives(panel)`](../backtrace/gp_factor_mining/02_primitive_set.py) | 加 ma/std/min/max/delta/ret/delay/rsi/macd/atr/bbi/bb/vwap/vol/mom 等列 | panel | panel + ~40 列 |
| [`add_crosssection_primitives(panel)`](../backtrace/gp_factor_mining/02_primitive_set.py) | 加 `cs_*` 截面 rank + 偏离度截面分位 | panel | panel |
| [`build_xy(panel, label)`](../backtrace/gp_factor_mining/02_primitive_set.py) | 切 (X, y, meta, feat_cols),丢掉标签/特征 NaN | panel, label | `(X, y, meta, feat_cols)` |
| [`make_function_set()`](../backtrace/gp_factor_mining/02_primitive_set.py) | 11 个保护算子(add/sub/mul/div/abs/neg/sqrt/log/max/min/sign) | — | `List[Function]` |

私有:`_rsi(close, N=14)`、`_atr(panel, N=14)`(Wilder 平滑)。

---

### [`backtrace/gp_factor_mining/03_neutralize.py`](../backtrace/gp_factor_mining/03_neutralize.py)

截面中性化(行业 + 市值),OLS `factor ~ 1 + log(size) + C(industry)`,取残差。

| 函数 | 一句话 | 说明 |
|---|---|---|
| [`add_size_proxy(panel, lookback=20)`](../backtrace/gp_factor_mining/03_neutralize.py) | 用过去 20 日均成交额做规模代理 | 加 `size_proxy` / `log_size` 列 |
| [`add_industry_proxy(panel, n_clusters=20)`](../backtrace/gp_factor_mining/03_neutralize.py) | 无真实行业时,按"过去 20 日 ret/vol 分布"做简易聚类 | 加 `ind_proxy` 列 |
| [`neutralize(panel, factor_col, size_col='size_proxy', ind_col='ind_proxy')`](../backtrace/gp_factor_mining/03_neutralize.py) | 每日横截面 OLS → 残差列 `{factor_col}_neu` | 原 `factor_col` 保留 |

私有:`_neutralize_one_day(df_day, factor_col, size_col, ind_col)`。

---

### [`backtrace/gp_factor_mining/04_ic_metrics.py`](../backtrace/gp_factor_mining/04_ic_metrics.py)

因子评估指标(RankIC、ICIR、分组收益、月度换手);7 个函数。

| 函数 | 一句话 | 返回 |
|---|---|---|
| [`daily_rankic(df, factor_col, label_col)`](../backtrace/gp_factor_mining/04_ic_metrics.py) | 每日 Spearman → 时序 | `Series`(index=date) |
| [`ic_summary(ic_ts)`](../backtrace/gp_factor_mining/04_ic_metrics.py) | IC 时序 → 摘要 | `dict`(7 keys: n_days / ic_mean / ic_std / icir / ic_t / ic_pos / ic_abs_mean) |
| [`ic_decay(df, factor_col, label_col, horizons=(1,5,10,20,40,60))`](../backtrace/gp_factor_mining/04_ic_metrics.py) | 不同 horizon 的 RankIC | `DataFrame`(horizon, rank_ic) |
| [`quantile_returns(df, factor_col, label_col, n_quantiles=10)`](../backtrace/gp_factor_mining/04_ic_metrics.py) | 每日分位数 → 平均未来收益 | `DataFrame`(列含 long_short) |
| [`quantile_summary(qret, periods_per_year=250)`](../backtrace/gp_factor_mining/04_ic_metrics.py) | 分组收益 → 年化摘要 | `DataFrame`(group / mean / std / ann_ret / ann_vol / sharpe) |
| [`monthly_turnover(df, factor_col)`](../backtrace/gp_factor_mining/04_ic_metrics.py) | 月度因子截面 Spearman | `float`(换手率 = 1 - 相邻月 Spearman) |
| [`full_evaluate(panel, factor_col, label_col, name='factor')`](../backtrace/gp_factor_mining/04_ic_metrics.py) | 一键全套评估 | `DataFrame`(一行摘要) |

---

### [`backtrace/gp_factor_mining/05_gp_mine.py`](../backtrace/gp_factor_mining/05_gp_mine.py)

GP 多轮残差因子挖掘(核心);`gplearn` 可选(未装时降级到 sklearn)。

| 函数 | 一句话 | 返回 |
|---|---|---|
| [`load_panel()`](../backtrace/gp_factor_mining/05_gp_mine.py) | 载入 `01` 落盘的 panel | `DataFrame` |
| [`make_xy(panel)`](../backtrace/gp_factor_mining/05_gp_mine.py) | 时序+截面特征 + 切训练/测试 | `(X_tr, y_tr, m_tr, X_te, y_te, m_te, feat_cols)` |
| [`train_one_round(X, y, feat_cols, round_idx, sample_frac=0.30)`](../backtrace/gp_factor_mining/05_gp_mine.py) | 单轮 SymbolicRegressor;>50k 行时抽样 | 训练结果 dict |
| [`multi_round_residual_mine(...)`](../backtrace/gp_factor_mining/05_gp_mine.py) | 多轮残差挖掘 | 各轮产出 |
| [`save_results(rounds_out, m_tr, m_te, X_tr, X_te, feat_cols)`](../backtrace/gp_factor_mining/05_gp_mine.py) | 落 `factor_summary_*.csv` + `factor_formulas_*.json` + 每轮 `factor_r*_*.parquet` | — |

---

### [`backtrace/gp_factor_mining/06_factor_pool.py`](../backtrace/gp_factor_mining/06_factor_pool.py)

因子库管理:入库 / 去重 / 体检;6 个函数。

| 函数 | 一句话 |
|---|---|
| [`latest_run()`](../backtrace/gp_factor_mining/06_factor_pool.py) | 找最新的 `factor_summary_*.csv` → 返回 `(summary_df, formulas_path, timestamp)` |
| [`load_panel()`](../backtrace/gp_factor_mining/06_factor_pool.py) | 载入 `01` 落盘的 panel |
| [`evaluate_one(factor_df, panel, label, name)`](../backtrace/gp_factor_mining/06_factor_pool.py) | 合并因子值 + 跑 `full_evaluate` |
| [`passes_gate(row)`](../backtrace/gp_factor_mining/06_factor_pool.py) | 入库门槛(IN_SAMPLE_ICIR_MIN / OUT_SAMPLE_IC_MIN / OUT_SAMPLE_ICIR_MIN / MAX_TURNOVER) |
| [`decorrelate(accepted, pool_df)`](../backtrace/gp_factor_mining/06_factor_pool.py) | 与已入库因子月度相关 > MAX_CORR_WITH_POOL → 剔除 |

---

### [`backtrace/gp_factor_mining/07_backtest.py`](../backtrace/gp_factor_mining/07_backtest.py)

vectorbt 多因子选股回测;`vectorbt` 可选(无 vbt 时给等权组合近似净值)。

| 函数 | 一句话 | 返回 |
|---|---|---|
| [`load_pool()`](../backtrace/gp_factor_mining/07_backtest.py) | 读 `06` 产出的主表 + 每入选因子的 test parquet 目录 | `(pool_df, pool_dir)` |
| [`combine_factors(pool, pool_dir, method="ic_weighted")`](../backtrace/gp_factor_mining/07_backtest.py) | `ic_weighted` / `equal` / `zscore_avg` 三种合成 | `DataFrame[date, code, composite]` |
| [`monthly_topn(composite, top_n=50)`](../backtrace/gp_factor_mining/07_backtest.py) | 每月末取截面 top N → 0/1 持仓标记 | `DataFrame`(wide: index=date, cols=code) |
| [`get_price_panel(codes, start, end)`](../backtrace/gp_factor_mining/07_backtest.py) | 拉入选因子的价格面板(给 vbt 用) | `DataFrame` |

---

### [`backtrace/gp_factor_mining/small_test.py`](../backtrace/gp_factor_mining/small_test.py)

端到端冒烟测试(无需 TQ);30 只合成股 × 1000 日,注入 `fwd_ret_20 ≈ 0.5*ret_5 + 噪声` 让 GP 有机会挖出来。

| 函数 | 一句话 |
|---|---|
| [`gen_synthetic_panel(n_stocks=30, n_days=1000, seed=42)`](../backtrace/gp_factor_mining/small_test.py) | 合成 panel + 已知 alpha 注入 |
| [`run_small_gp(panel)`](../backtrace/gp_factor_mining/small_test.py) | 复用 05 的核心但在小参数上跑(POP=200, GEN=5) |
| [`main()`](../backtrace/gp_factor_mining/small_test.py) | 入口:合成 → 训 → 评估 |

---

## `backtrace/tsfresh/` — tsfresh 系列(10 个文件)

线性脚本,文件头注释 + 顶部常量。每个文件配 `common.tsfresh_pipeline` 使用。

| 文件 | 一句话 | 输出 |
|---|---|---|
| [`tsfresh_features_002457.py`](../backtrace/tsfresh/tsfresh_features_002457.py) | tsfresh 提特征(整段历史当 1 样本) | `tsfresh_features_<code>.csv` |
| [`tsfresh_select_002457.py`](../backtrace/tsfresh/tsfresh_select_002457.py) | 滑窗 → 显著特征筛选(FDR) | `tsfresh_selected_<code>.csv` |
| [`tsfresh_classify_002457.py`](../backtrace/tsfresh/tsfresh_classify_002457.py) | walk-forward 时序回测 + 最新窗口打分 | `tsfresh_model_<code>.csv` |
| [`tsfresh_pick_stocks.py`](../backtrace/tsfresh/tsfresh_pick_stocks.py) | 对板块内所有票打分 → 选股 CSV | `tsfresh_pick_stocks_YYYYMMDD_HHMMSS.csv` |
| [`tsfresh_multichannel_pick.py`](../backtrace/tsfresh/tsfresh_multichannel_pick.py) | 个股+大盘双通道,"相对大盘"标签 | `tsfresh_multichannel_<code>_vs_<idx>.csv` |
| [`tsfresh_eval_indicators.py`](../backtrace/tsfresh/tsfresh_eval_indicators.py) | 11 个技术指标的板块 IC / 胜率 | `tsfresh_indicator_ic_<sector>_<start>_<end>.csv` + `_summary.csv` |
| [`tsfresh_vbt_combo.py`](../backtrace/tsfresh/tsfresh_vbt_combo.py) | tsfresh 信号当 vbt entry/exit,4 组合网格 | `tsfresh_vbt_grid_<code>_<start>_<end>.csv` |
| [`tsfresh_with_ma_channel.py`](../backtrace/tsfresh/tsfresh_with_ma_channel.py) | 把 ma5/10/20/偏离度 当额外 tsfresh 通道 | `tsfresh_with_ma_<code>_<start>_<end>.csv` |
| [`tsfresh_with_ma_grid_sector.py`](../backtrace/tsfresh/tsfresh_with_ma_grid_sector.py) | 跨通达信88 板块验证 with_ma 方案稳定性 | `tsfresh_with_ma_grid_<code>_<start>_<end>.csv` |
| [`tsfresh_top1_industry.py`](../backtrace/tsfresh/tsfresh_top1_industry.py) | 双重跑赢(>大盘 & >板块)→ tsfresh → TQ 推送 | `tsfresh_top1_industry_<start>_<end>.csv` |

`tsfresh_top1_industry.py` 内嵌函数:`get_members` / `lead_lag_signal` / `vbt_with_real_fees`。

---

## `backtrace/vbt/` — vbt 系列(2 个文件)

| 文件 | 一句话 | 输出 |
|---|---|---|
| [`vbt_simple_backtest.py`](../backtrace/vbt/vbt_simple_backtest.py) | MA 交叉 demo(零摩擦上限) | `backtrace/vbt_backtest_<code>.html` |
| [`vbt_jhzq_backtest.py`](../backtrace/vbt/vbt_jhzq_backtest.py) | vbt + `F.summary_after_fees` 真实扣费 | `vbt_jhzq_<code>_<start>_<end>_trades.csv` |

注意:vbt 0.27 的 `from_signals` **不支持** `short_enabled`;**某些 dtype** 上 `MACD.run`/`STOCH.run`/`ADX.run` 会报 `'ewms' int → PyBool`,必须用 pandas 手算绕过。

---

## `backtrace/alpha/` — 双层 α 选股(3 个文件)

| 文件 | 一句话 | 输出 |
|---|---|---|
| [`two_layer_industry_strength.py`](../backtrace/alpha/two_layer_industry_strength.py) | 双层:880xxx 行业 vs 大盘 → 行业里 Top 票 | `two_layer_industry_strong_<idx>_<start>_<end>.csv` |
| [`two_layer_industry_strength_live.py`](../backtrace/alpha/two_layer_industry_strength_live.py) | 1/5/10 日窗口横向对比 + vbt 真实扣费 | `two_layer_industry_live_<idx>_<start>_<end>.csv` |
| [`two_layer_relative_strength.py`](../backtrace/alpha/two_layer_relative_strength.py) | "跑赢大盘 + 跑赢板块"双层 + walk-forward | `two_layer_strong_stocks_<sector>_<start>_<end>.csv` |

---

## `backtrace/talib/` — K 线形态(2 个文件)

| 文件 | 一句话 | 输出 |
|---|---|---|
| [`talib_pattern_verify.py`](../backtrace/talib/talib_pattern_verify.py) | 直接算每形态后第 2 日命中/未命中/胜率 | `backtrace/talib_pattern_verify.csv` |
| [`talib_pattern_backtest.py`](../backtrace/talib/talib_pattern_backtest.py) | vbt portfolio 模拟验证(单只票) | `backtrace/talib_pattern_verify_<code>.csv` |

---

## `backtrace/legacy/` — 过时模板(5 个文件,**已不推荐**)

| 文件 | 一句话 |
|---|---|
| [`kline_chart.py`](../backtrace/legacy/kline_chart.py) | K 线 plotly HTML(单只票);早期可视化,改用 `vbt_simple_backtest` |
| [`ma_cross_signals.py`](../backtrace/legacy/ma_cross_signals.py) | MA 交叉 → TQ 推送(信号推送基础模板);改用 `tsfresh_top1_industry` |
| [`price_rise_monitor.py`](../backtrace/legacy/price_rise_monitor.py) | 实时涨幅 > 5% 推送预警(订阅板块);唯一带 `__main__` 守卫的 legacy |
| [`projection_2d.py`](../backtrace/projection/projection_2d.py) | 2-D 投影 / 正交性可视化;支持 `--two-day-vec` 开启 lag=1 的 4-D 模式(Volume/Amount 各 +prev);7 个 HTML 落 backtrace/ 根 |
| [`stock_picker.py`](../backtrace/legacy/stock_picker.py) | 连续上涨 N 日选股 → 推 TQ 自定义板块 |

---

## 如何新增脚本(boilerplate 模板)

放在 `backtrace/<新子目录>/` 下,从仓库根运行:

```python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

import sys, os
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import numpy as np
import pandas as pd
import vectorbt as vbt
from datetime import datetime
from tqcenter import tq

from common import tsfresh_config as C, tsfresh_pipeline as P, jhzq_fees as F

tq.initialize(__file__)


def main():
    """一句话说明这脚本做什么"""
    # 1. 拉数
    df = P.load_ohlcva('600118.SH', verbose=True)
    # 2. 业务逻辑 ...
    # 3. vbt + 真实扣费
    pf = vbt.Portfolio.from_signals(...)
    trades = pf.trades.records_readable
    summary = F.summary_after_fees(trades, '600118.SH')
    # 4. 推送
    tq.send_user_block('TQI_TOP', ['600118.SH'])
    print(summary)


if __name__ == '__main__':
    main()
    tq.close()
```

要点:
- 加 `if __name__ == '__main__'` 守卫 — 现有 `tsfresh/`/`vbt/`/`alpha/`/`talib/` 多数缺,新脚本请补
- 中文 print 必须 `PYTHONIOENCODING=utf-8`(Windows 必加)
- 输出 CSV 默认 `backtrace/outputs/`(gitignored)
- 公开函数写好 docstring → 同步更新 [docs/api.md](api.md)

---

## `backtrace/projection/` — 2-D 投影(单股 + 批量 + 动力学层)

3 个脚本 + 1 个共享核心模块;`projection_2d.py` 单股 HTML+CSV,`projection_batch.py` 批量 CSV,`_projection_core.py` 纯数学。**动力学层(`--dynamics`,2026-08-16 新增)** 在运动投影之上叠加离散系统指标。

### [`backtrace/projection/_projection_core.py`](../backtrace/projection/_projection_core.py)

共享数学;无 plotly / 无 HTML / 无文件写。所有 `projection_*` 脚本都从这里 import。

| 函数 | 一句话 | 参数 | 返回 | 备注 |
|---|---|---|---|---|
| [`MARKET_TO_INDEX`](../backtrace/projection/_projection_core.py) | dict | `{'SZ':('399001.SZ','深证成指'),'SH':('000001.SH','上证综指')}` | — | 后缀→大盘 |
| [`resolve_index(stock_code)`](../backtrace/projection/_projection_core.py) | 个股后缀→大盘指数 | `stock_code`:`'002475.SZ'` | `(code, name)` | 未知后缀抛 ValueError |
| [`INDUSTRY_MAP`](../backtrace/projection/_projection_core.py) | dict | 由 `data/sw2/members.csv` 启动时构建的 code→(sector_code, sector_name) | — | 个股→申万二级行业 |
| [`resolve_industry(stock_code)`](../backtrace/projection/_projection_core.py) | 个股→申万二级行业 | 同 `resolve_index` | `(code, name)` | 新股/非 A 股抛 ValueError |
| [`project_u_onto_v(u, v)`](../backtrace/projection/_projection_core.py) | 2-D 向量投影 | `u, v`:ndarray shape (2,) | ndarray | `v·v=0` 返回零向量 |
| [`_safe_ratio(num, den, default=NaN)`](../backtrace/projection/_projection_core.py) | 安全除法 | num / den | float | 0/NaN/Inf → `default` |
| [`load_pair(stock_code, days, pipeline, prefer_industry=False, index_code=None, lag=0)`](../backtrace/projection/_projection_core.py) | 本地 `data/` 缓存拉 (stock_df, index_df) 共同交易日 | `lag=1` 时附 Vol_prev / Amt_prev | dict | 基线优先级:`index_code` > `prefer_industry=True` 申万二级 > 大盘 |
| [`compute_vectors(stock_df, index_df, index_tag, stock_tag, lag=0)`](../backtrace/projection/_projection_core.py) | Min-Max 归一化 Vol/Amt(及可选 prev) | `lag=0`:2-D;`lag=1`:4-D | `(vec_index, vec_stock, norms_ix, norms_st, norm_params_str)` | 仅 lag ∈ {0, 1} |
| [`compute_projections(vec_stock, vec_index)`](../backtrace/projection/_projection_core.py) | 朴素 2-D 投影 9 个指标 | ndarray (T, k) | dict | 不输出 `state_resi_price`(2-D 退化,选股无效,2026-08-16 删) |
| [`compute_movement_projection(stock_df, index_df)`](../backtrace/projection/_projection_core.py) | 运动向量投影(ΔV, ΔA)→ β / proj / resi | 首行因 diff 丢 | dict 含 16 keys(stock_move/index_move/proj/residual/proj_mag/resi_mag/dot_after/proj_coeff/proj_prices/resi_prices/4 magnitudes 等) | 末行与 caller `common_idx[1:]` 对齐 |
| [`build_movement_result_df(common_idx, mv, index_tag, stock_tag)`](../backtrace/projection/_projection_core.py) | 组装 18 列运动投影 CSV | `common_idx[1:]` | DataFrame | `Move_` 前缀,与 state 区分 |
| [`build_movement_intermediate_df(common_idx, mv, stock_df, index_df, index_tag, stock_tag)`](../backtrace/projection/_projection_core.py) | 组装 25 列「逐日复核」CSV(原始 Vol·Ama + Δ + β 分子分母 + 3 个 price) | 同上 | DataFrame | 落 `data/projection/intermediate/`,人工核对公式用 |
| [`compute_dynamics(mv, lambda_q)`](../backtrace/projection/_projection_core.py) | **动力学层** 9 指标(基于 `mv` dict) | `mv` = `compute_movement_projection` 输出;`lambda_q=None` 走 median 自适应 | dict 含 13 keys(q_t/theta/R/v_*/E_*/a_*/lambda_q_used) | 详见 [superpowers/specs/2026-08-16-market-stock-dynamics-design.md](superpowers/specs/2026-08-16-market-stock-dynamics-design.md) |
| [`classify_states(R, theta, E_self, thresholds)`](../backtrace/projection/_projection_core.py) | **状态分类** 7 标签 + none | `thresholds=(R_low, R_high, theta_following_rad, theta_against_rad)` | list[str],长度 T-1 | 优先级:against > resonance > accelerating > returning > independent > weak_div > follow > none |
| [`build_dynamics_df(common_idx, dyn, states, index_tag, stock_tag)`](../backtrace/projection/_projection_core.py) | 组装 14 列动力学 CSV | `common_idx[1:]` | DataFrame | `Dyn_` 前缀,加速度列右补 NaN(末行 NaN) |
| [`compute_forces(dyn, mv, k_restore=0.0, c_damp=0.0)`](../backtrace/projection/_projection_core.py) | **力分解** a_S = β·a_M - k·d - c·u + F_self(用户 prompt §14-17) | `dyn`+`mv`;默认 k=c=0(纯残差基线) | dict 含 8 keys(F_market/F_restore/F_damp/F_self/d_mag/u_mag + k/c_used) | F_* 输出标量模长;末行 NaN(加速度末行 NaN 衍生) |
| [`build_forces_df(common_idx, frc, index_tag, stock_tag)`](../backtrace/projection/_projection_core.py) | 组装 8 列力分解 CSV | `common_idx[1:]` | DataFrame | `Frc_` 前缀;含 d_mag/u_mag/√(F_M²+F_S²) 三辅助列 |
| `STATE_LABELS` / `STATE_COLORS` / `STATE_LABELS_CN` | 7 状态标签 / 配色 / 中文映射 | — | list / dict / dict | projection_2d.py 的 HTML band 与中文 print 共用 |

### [`backtrace/projection/projection_2d.py`](../backtrace/projection/projection_2d.py)

单股可视化;`--dynamics` 在 `--movement` 之上叠加动力学 4 子图 HTML + 14 列 CSV。

新增 CLI(2026-08-16):

| Flag | 默认 | 说明 |
|---|---|---|
| `--dynamics` | False | 启用动力学层;自动开启 `--movement` |
| `--lambda-q` | `-1` | 锚定强度系数 λ_q;传 `-1` 走 median(‖ΔM‖) 自适应;传 `0` 等价无阻尼(q_t=1) |
| `--classify-thresholds` | `0.10,0.50,30,90` | `R_low,R_high,theta_following_deg,theta_against_deg` |
| `--k-restore` | `0.0` | 恢复力系数 k。F_restore = -k·d;0 = 无均值回复力 |
| `--c-damp` | `0.0` | 阻尼系数 c。F_damp = -c·u;0 = 无阻尼 |

动力学产物:

- HTML: `backtrace/outputs/dynmv_trajectory.html`(5 子图:速度 / 能量 / R+θ / 状态分类 / 力分解)
- CSV: `data/projection/dynamics_<INDEX_TAG>_<STOCK_TAG>.csv`(14 列,见 `build_dynamics_df`)
- CSV: `data/projection/forces_<INDEX_TAG>_<STOCK_TAG>.csv`(8 列,见 `build_forces_df`;`--dynamics` 启用时自动产出)

### [`backtrace/projection/projection_batch.py`](../backtrace/projection/projection_batch.py)

批量跑(只产 CSV,不画 HTML)。`--dynamics` 在 `--movement` 之上叠加,每只票额外产
`dynamics_*.csv` + `forces_*.csv`(2026-08-16 新增)。

新增 CLI(2026-08-16,与 `projection_2d.py` 一致):

| Flag | 默认 | 说明 |
|---|---|---|
| `--dynamics` | False | 启用动力学层;自动开启 `--movement` |
| `--lambda-q` | `-1` | 锚定强度系数 λ_q;传 `-1` 走 median(‖ΔM‖) 自适应 |
| `--classify-thresholds` | `0.10,0.50,30,90` | `R_low,R_high,theta_following_deg,theta_against_deg` |
| `--k-restore` | `0.0` | 恢复力系数 k |
| `--c-damp` | `0.0` | 阻尼系数 c |

动力学层失败**不阻塞**主路径(失败时 status=`ok (dynamics failed: <ExcType>: <msg>)`),
`projection_*.csv` 与 `movement_*.csv` 仍正常写入。

Manifest schema(`data/projection/batch_manifest.csv`,10 列):

| 列 | 含义 |
|---|---|
| `code` / `name` | 个股代码 / 名称 |
| `index_code` / `index_name` | 实际基线(申万二级 / 大盘 / `--index` 显式) |
| `rows` | 写入 CSV 的行数(扣除 4-D 模式首日) |
| `date_start` / `date_end` | 第一行 / 最后一行日期 |
| `csv_path` | state CSV 路径(必有) |
| `dyn_csv_path` | dynamics CSV 路径(`--dynamics` 启用时填,否则空) |
| `frc_csv_path` | forces CSV 路径(同上) |
| `status` | `ok` / `ok (dynamics failed: ...)` / `failed: ...` |

CLI 示例:

```bash
# 默认:state CSV × N
python backtrace/projection/projection_batch.py --limit 50

# 运动 + 动力学(产 3 组 CSV × N)
python backtrace/projection/projection_batch.py --movement --dynamics --limit 100

# 全市场基线 + 自定义 λ_q + 弱回复/阻尼
python backtrace/projection/projection_batch.py --market-baseline --dynamics \
    --lambda-q 1e6 --k-restore 0.1 --c-damp 0.05 --days 120

# 自定义分类阈值
python backtrace/projection/projection_batch.py --dynamics --classify-thresholds 0.15,0.60,20,100
```

### [`backtrace/projection/parameter_fit.py`](../backtrace/projection/parameter_fit.py)

**闭式 OLS 估计每只票的 (k̂, ĉ)**(2026-08-16 新增)。不依赖网格搜索 — 模型
`a_S = β·a_M − k·d − c·u + F_self` 对 k/c 严格线性,2N×2 线性系统单次 `lstsq` 即可。

输入: `data/projection/movement_*.csv`(由 `--movement` / `--dynamics` batch 跑产出)
输出: `data/projection/kc_estimates.csv`(10 列)

| 列 | 含义 |
|---|---|
| `code` / `name` / `index_code` / `index_tag` / `stock_tag` | 票与基线标识 |
| `k_hat` | 估计的恢复力系数;正值=均值回复,负值=反回复(趋势强化) |
| `c_hat` | 估计的阻尼系数;正值=阻尼(系统耗散),负值=反阻尼 |
| `f_self_loss` | `F_self = a_S − β·a_M + k̂·d + ĉ·u` 的均方范数;代表「模型无法解释的自主驱动力」 |
| `n_valid_days` | 参与 OLS 的有效观测天数(默认要求 ≥ 20) |
| `status` | `ok (restoring, damping)` / `singular` / `too_few_days` / `extreme` / `solve_failed` |

CLI:

```bash
# 默认扫描 data/projection/movement_*.csv 全部 fit
python backtrace/projection/parameter_fit.py

# 自定义股票列表(列:code;可选 name/index_code)
python backtrace/projection/parameter_fit.py --input data/projection/stocks.csv

# 冒烟 + 降低有效天数门槛
python backtrace/projection/parameter_fit.py --limit 10 --min-valid-days 5

# 截幅阈值(默认 |k|,|c| ≤ 10,超过则标 extreme)
python backtrace/projection/parameter_fit.py --clip-extreme 5.0
```

后续用法:`kc_estimates.csv` 直接喂回 `projection_batch.py --k-restore <k_hat> --c-damp <c_hat>`
或外层脚本按 k̂/ĉ 分布批量设参(均值 / 中位数 / 分位)。

## 已知陷阱(踩过无数次的)

1. **Windows GBK 终端**:中文 print 报错 → 用 `PYTHONIOENCODING=utf-8` 或 print 全 ASCII
2. **TQ 客户端**必须启动,否则 `tq.initialize` 直接 `RuntimeError`
3. **TQ 批量拉** `get_market_data(stock_list=[6000 只])` 会 timeout;`vbt.MACD.run` 等在某些 dtype 上会报 `'ewms' int → PyBool` 错 → 用 pandas `ewm/rolling` 手算
4. **tsfresh 默认 `n_jobs=-1`** 在 Windows 下 multiprocessing 卡死 → **强制 `n_jobs=0`**(单进程),已写死到 `C.TSFRESH_N_JOBS`
5. **NaN 在 tsfresh long format 中不被允许**;`to_long_format` 已强制 `pd.to_numeric(..., errors='coerce')`,下游用 `impute` 兜底
6. **回测数据集时间要 `df.loc[start:end]` 切片**,否则 vbt 会用全期数据
7. **vbt 0.27** 的 `from_signals` **不支持** `short_enabled` 参数
8. **`get_market_data` 返回 wide 格式**(`Dict[field, DataFrame(wide)]`)— 不是 MultiIndex,常见误解。`df['Close'][code]` 是两层访问
9. **`send_user_block` 内部**会调 `convert_or_validate` 把代码改写成 `'1#688318|0#002475'`,**别自己提前拼**
10. **`adjust_trades_pnl` / `summary_after_fees` 输出 schema 是中文列/键**,跨脚本读要小心(已显式记入 [§jhzq_fees 输出 Schema](#adjust_trades_pnl-输出-schema新增-5-列))
11. **TQ 板块** `get_stock_list_in_sector(code, block_type=0)` 必须传 `'880xxx.SH'` 带 `.SH` — 不带返回 0
12. **`get_stock_list('11', list_type=1)` 返回 128 个申万二级真行业** — 比 `get_sector_list(list_type=1)` 的 588 个混合板块干净

---

## tqcenter 跨引用(外部)

`backtrace/` **不包含** `tqcenter`,它位于 `C:/new_tdx_mock/PYPlugins/user/tqcenter.py`(另一仓库,本地安装)。
本项目用到的方法(列出即可,详情见 tqcenter):

| 方法 | 用途 | 返回 |
|---|---|---|
| `tq.initialize(path)` | 启动 TQ 客户端连接 | — |
| `tq.close()` | 关闭连接 | — |
| `tq.get_market_data(field_list, stock_list, ...)` | 批量拉 K 线 | `Dict[field, DataFrame wide]` |
| `tq.get_sector_list(list_type)` | 板块列表 | `List[dict]`(128 SW2 / 588 混合) |
| `tq.get_stock_list_in_sector(block_code, block_type=0)` | 板块成分股 | `List[str]`(必须 `.SH`) |
| `tq.get_stock_list(market, list_type)` | 全市场股票列表 | `List[dict]` |
| `tq.create_sector(block_code, block_name)` | 创建自定义板块 | raw str |
| `tq.send_user_block(block_code, stocks)` | 推股票到板块 | `Dict` |
| `tq.send_message(msg_str)` | 推字符串到策略管理器 | `Dict` |
| `tq.send_warn(...)` | 推预警信号 | `Dict` |

> 注:基于截至 2026-08 观察的 tqcenter;字段若变请同步更新本节。