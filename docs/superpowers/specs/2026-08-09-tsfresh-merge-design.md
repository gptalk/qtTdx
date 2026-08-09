# tsfresh 脚本合并设计

> 把 `backtrace/tsfresh/` 下 3 个 vbt 回测脚本共享的代码抽到 `backtrace/common/`,3 个原脚本保留为薄壳(只保留策略配置 + 编排)。消除 ~70% 重复代码,把近期零散修复统一到 common 的黑盒行为里。

**Goal:** 抽 2 个 common 模块(`tsfresh_walkforward.py` + `vbt_jhzq_backtest.py`),3 个原脚本瘦身到只保留「策略定义 + 编排循环」,所有近期修复(INIT_TRAIN_SIZE FDR 限制、bfill、80% 拒单 warning、friction_loss_pp 符号检查)作为 common 模块的默认行为。

**Architecture:**

```
backtrace/
├── common/
│   ├── tsfresh_walkforward.py     ← 新增:walk-forward proba 生成 + MA 通道 + 通道构成报告
│   └── vbt_jhzq_backtest.py       ← 新增:vbt + jhzq_fees 单次回测 + 80% 拒单 warning + 股数/格式化
├── tsfresh/
│   ├── tsfresh_vbt_combo.py       ← 瘦身后:1 stock, 4 策略网格, 调 W.proba() + B.run_one()
│   ├── tsfresh_with_ma_channel.py ← 瘦身后:1 stock, 4 策略(2 通道 + 复合), 调 W.proba() + B.run_one()
│   └── tsfresh_with_ma_grid_sector.py ← 瘦身后:88 票循环, 每只调 W.proba() + B.run_one(), 跨票聚合
```

**Tech Stack:** 现有依赖不动(`tsfresh_pipeline.py` / `jhzq_fees.py` / `data_store.py`),只新增 2 个 pure-utility 模块。

---

## 1. 新模块:`backtrace/common/tsfresh_walkforward.py`

### 1.1 公共 API

```python
def add_ma_channels(ohlcv_df, windows=(5, 10, 20), add_rel=True) -> pd.DataFrame:
    """原地加 ma5/ma10/ma20 + rel_ma5(Close 相对 ma5 的偏离度)。
       复制 df 后修改,避免污染上游调用方。
       使用 bfill(替代原 grid_sector 的 fillna(0.0))——见 with_ma_channel.py:96-98 注释。"""

def report_channel_composition(X_sel: pd.DataFrame, label: str = "") -> None:
    """统计 X_sel 各通道入选特征数。若 ma*/rel_ma5 占比 > 33% 打印 [WARN] 冗余风险。
       从 with_ma_channel.py:_report_channel_composition 搬过来,任何调用方都受益。"""

def tsfresh_walkforward_proba(
    ohlcv_df: pd.DataFrame,
    channels: list[str],
    *,
    init_train_size: int = 200,
    step: int = 50,
    fillna: str = "bfill",   # "bfill" / "zero";grid_sector 原本用 zero,改 bfill 后效果同 with_ma_channel
    id_value: str | None = None,
    verbose: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    """跑 tsfresh 全流程(转 long → 提特征 → 打标签 → FDR 筛选 → walk-forward 训 LR)→ 返回 (proba, X_sel)。

       FDR 筛选严格限制在 X.iloc[:init_train_size] 段(防止 init_train_size 之后用「未来特征」反推)。
       该限制从 vbt_combo.py:67-72 + with_ma_channel.py:104-114 统一;grid_sector 当前在 X_all 上跑(泄漏!)——迁移后会自然修好。

       walk-forward:初始训练 init_train_size,之后每 step 重训一次。
       proba 在 date_index[end_t] 当日计算,后续回测脚本 shift(1) 视作次日开盘成交。

       返回 proba:Series(索引 DatetimeIndex,dropna 后只保留有效预测点)。
       返回 X_sel:DataFrame(供调用方做通道构成报告或调试)。"""
```

### 1.2 行为约定(写入模块 docstring 顶部)

1. **bfill 而非 fillna(0.0)** — 避免序列开头 0 → 实值跳变被 tsfresh 当成「突变」特征(具体理由见 with_ma_channel.py:96-98)
2. **FDR 严格在 init_train_size 段筛** — 防止特征选择时用上 walk-forward 期内的未来样本
3. **`X_sel` 列对齐** — 在全期 `X_all[selected_cols]` 上做,索引仍覆盖全期,这样 walk-forward 循环按 `pos` 切片不会因早期样本特征列缺失而崩

### 1.3 与 `tsfresh_pipeline.py` 的边界

| 职责 | tsfresh_pipeline.py (已有) | tsfresh_walkforward.py (新增) |
|---|---|---|
| `load_ohlcva` / `load_sector` | ✅ | — |
| `to_long_format` / `extract_window_features` / `select_relevant` / `fit_logreg` | ✅ | 调用这些 |
| `make_labels` | ✅ | 调用 |
| MA 通道计算 | — | ✅ (新) |
| 通道构成报告 | — | ✅ (新) |
| walk-forward predict 循环 | — | ✅ (新) |

---

## 2. 新模块:`backtrace/common/vbt_jhzq_backtest.py`

### 2.1 公共 API

```python
def compute_shares_per_trade(init_cash: float, max_pos_pct: float, init_open: float) -> int:
    """每笔固定股数 = floor(init_cash * max_pos_pct / open0 / 100) * 100。
       返回 0 表示价格/仓位下没有 100 股整手(调用方应跳过该票)。"""

def run_vbt_backtest(
    ohlcv_df: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    stock_code: str,
    *,
    init_cash: float = 100_000,
    max_pos_pct: float = 0.95,
    upon_long_conflict: str = "exit",
    print_rejection_warning: bool = True,
) -> dict:
    """跑 vbt + jhzq_fees 真实扣费的单次回测。
       返回 summary dict(包含 zero_friction_ret + net_ret + friction_loss_pp 等 12 列)。

       80% 拒单 warning(从 vbt_combo.py:212-222 统一):当实际成交笔数 < 信号数 * 0.8,
       打印 [WARN] + 原因(MAX_POS_PCT 太高 + 股价上涨后资金不足)。

       friction_loss_pp 符号检查(从 vbt_combo.py:234-235 统一):
       zero_friction_ret - net_ret 应该恒 ≥ 0(扣费后收益 ≤ 零摩擦收益),负值说明 zero/net_ret 口径不一致或费率 bug。

       关键不变量:复用 pf_zero 的 trades 算 jhzq_fees(pf_zero 本身已经是 fees=0 + slippage=0,
       signal 按价格穿越触发,扣费独立计算,见 vbt_combo.py:174-176 caveat 注释)。"""

def fmt_money(x) -> str: ...
def fmt_pct(x) -> str: ...
def fmt_pp(x) -> str: ...
"""3 个格式化函数从 vbt_combo.py:237-242 + with_ma_channel.py:246-251 统一。"""

def build_proba_signals(
    proba: pd.Series,
    bar_index: pd.DatetimeIndex,
    *,
    entry_th: float,
    exit_th: float,
    shift_for_next_open: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """proba reindex 到 bar_index 上,生成 (entries, exits) 布尔 Series,shift(1) 视作次日开盘成交。
       替代 3 个脚本里重复的:
         aligned = proba.reindex(close_full.index)
         entries = (aligned > th_entry).shift(1).fillna(False).astype(bool)
         exits   = (aligned < th_exit).shift(1).fillna(False).astype(bool)
       边界:aligned 全 NaN → 返 (全 False, 全 False),不会崩。"""
```

### 2.2 与 `jhzq_fees.py` 的边界

| 职责 | jhzq_fees.py (已有) | vbt_jhzq_backtest.py (新增) |
|---|---|---|
| 费率常量 (COMMISSION/STAMP/TRANSFER_SH/TRANSFER_SZ) | ✅ | — |
| `adjust_trades_pnl` (单笔扣费) | ✅ | 调用 |
| `summary_after_fees` (汇总) | ✅ | 调用 |
| vbt Portfolio 包装 + shares 计算 + 80% warning | — | ✅ (新) |
| 格式化输出 | — | ✅ (新) |

---

## 3. 3 个原脚本瘦身映射

### 3.1 `tsfresh_vbt_combo.py`

**保留:** 配置(STOCK_CODE, TARGET_*, INIT_CASH, MAX_POS_PCT, INIT_TRAIN_SIZE, STEP, ENTRY_TSF, EXIT_TSF)+ STRATEGIES 列表(4 个)+ 输出 CSV 路径。

**删除:** walk-forward 循环(45 行)→ 调 `W.tsfresh_walkforward_proba(df, channels=['OHLCV'])`(5 行)。`build_signals()` 走 `B.build_proba_signals()`。`run_vbt_backtest()` 整个删,改调 `B.run_vbt_backtest()`。

**预计:** 274 行 → ~140 行。**输出 CSV schema 完全不变**(zero_friction_ret / gross_pnl / total_stamp / total_transfer / net_pnl / net_ret / win_rate / profit_factor / friction_loss_pp 9 列)。

### 3.2 `tsfresh_with_ma_channel.py`

**保留:** 配置 + MA 通道调用(`add_ma_channels(df)`)+ 2 套 PROBA_CACHE(basic + with_ma)+ STRATEGIES 列表(含 'mode': 'pure'/'confirmed')。

**删除:** `build_tsfresh_proba()` 内部实现(50 行)→ 调 `W.tsfresh_walkforward_proba(df, channels=...)`。`_report_channel_composition()` 删,改调 `W.report_channel_composition()`。`build_signals()` 简化:ma 模式本地写;tsfresh 模式走 `B.build_proba_signals()`;'confirmed' 模式(MA5 AND tsfresh)本地算 1 行。

**预计:** 273 行 → ~140 行。**输出 CSV schema 完全不变**(strategy / trades / zero_friction_ret / gross_pnl / total_stamp / total_transfer / net_pnl / net_ret / win_rate / profit_factor / friction_loss_pp)。

### 3.3 `tsfresh_with_ma_grid_sector.py`

**保留:** 配置(SECTOR_NAME + 2 套 CHANNELS_BASIC / CHANNELS_MA)+ `run_backtest` 内部(去掉 shares 计算,改调 `B.compute_shares_per_trade`)+ 88 票循环 + 跨票聚合(with_ma_wins / alpha_diff_pp / 胜率统计)。

**删除:** `tsfresh_walkforward_proba()` 内部实现(40 行)→ 调 `W.tsfresh_walkforward_proba()`。MA 通道计算(4 行循环)→ 调 `W.add_ma_channels(df)`。`run_backtest()` 内 vbt 调用 → 调 `B.run_vbt_backtest()`。

**预计:** 195 行 → ~120 行。**输出 CSV schema 完全不变**(stock / n_days / basic_* / with_ma_* / alpha_diff_pp)。

**重要副作用:** grid_sector 当前 FDR 在 X_all 上跑(泄漏!),迁移后会通过 `W.tsfresh_walkforward_proba` 自动获得 INIT_TRAIN_SIZE 限制 — 这是免费修好的潜在 bug。

---

## 4. 全局约束(Global Constraints)

1. **不修改** `tsfresh_pipeline.py` / `jhzq_fees.py` / `data_store.py` / `tsfresh_config.py`(它们已经稳定,改动会牵连所有脚本)
2. **不修改** `tsfresh_vbt_combo.py` / `tsfresh_with_ma_channel.py` / `tsfresh_with_ma_grid_sector.py` 的输出 CSV schema(用户已经基于历史输出做对比)
3. **TQ 客户端不在 common 模块 import** — 跟现有 `tsfresh_pipeline.py` 一致(懒加载 + 委托给调用方 `tq.initialize(__file__)`)
4. **保留 vbt_combo.py:174-176 的 pf_zero reuse caveat 注释** — 在 `B.run_vbt_backtest` 的 docstring 里再写一遍
5. **保留 with_ma_channel.py:7-15 的阈值偏差警告** — 在 `W.tsfresh_walkforward_proba` 的 docstring 里再写一遍
6. **PYTHONIOENCODING=utf-8** 仍然必须在命令行加(Windows GBK 终端)— 见 README
7. **新增模块不引入新依赖**(vectorbt / sklearn / tsfresh 都已经在 requirements 里)

---

## 5. 测试计划

### 5.1 单元测试(无 TQ 依赖,纯函数层)

| 测试 | 命令 | 期望 |
|---|---|---|
| `add_ma_channels` 加 4 列 | `python -c "from common.tsfresh_walkforward import add_ma_channels; ..."` | df 多出 ma5/ma10/ma20/rel_ma5 |
| `compute_shares_per_trade` 边界 | `compute_shares_per_trade(100000, 0.95, 50.0)` | 返回 0(< 100 股) |
| `compute_shares_per_trade` 正常 | `compute_shares_per_trade(100000, 0.95, 30.0)` | 返回 3100(100000*0.95/30/100*100) |
| `build_proba_signals` 边界 | 全 NaN proba | (全 False, 全 False) |
| `report_channel_composition` 阈值 | X_sel 中 ma* 占 50% | 打印 [WARN] 冗余风险 |

### 5.2 集成测试(需 TQ 客户端启动)

| 测试 | 命令 | 期望 |
|---|---|---|
| 3 个脚本冒烟跑 1 只票(limit 1) | `python backtrace/tsfresh/tsfresh_vbt_combo.py`(改 STOCK_CODE=600118.SH) | 输出 CSV, schema 不变 |
| 88 板块冒烟跑 5 只 | `python backtrace/tsfresh/tsfresh_with_ma_grid_sector.py`(改 SECTOR_NAME + limit 5) | 输出 CSV, schema 不变 |
| 数字一致性 | 跟迁移前最后一次输出的 CSV 行/列对比 | 数字差异 < 1e-6(只来自 fillna bfill 引入的细微差别) |

### 5.3 验证修复已统一

- [ ] vbt_combo / with_ma_channel / grid_sector 三个脚本跑同段数据,signal 数都打印「拒单 warning」当 MAX_POS_PCT=0.95 时
- [ ] grid_sector 输出 CSV 的特征筛选日志显示「前 200 样本筛」(而不是「全期筛」)
- [ ] grid_sector 输出 CSV 在 ma20 早期段不再有 0 值(改 bfill)

---

## 6. 不在范围(YAGNI)

- 不合并 vbt_combo / with_ma_channel(用户已选定「轻量」路径)
- 不重写 88 票循环用 multiprocessing / joblib(单进程跑够用,改架构超出本次范围)
- 不给 common 模块加 CLI(只做函数,CLI 在调用方)
- 不引入新依赖(arrow / polars 都不动)
- 不改 `tsfresh_features_002457.py` / `tsfresh_select_002457.py`(它们已经走 select_relevant,跟本次抽出的 walk-forward 解耦)

---

## 7. 风险与回滚

| 风险 | 概率 | 影响 | 回滚 |
|---|---|---|---|
| `pf_zero` reuse 在新 common 里行为不一致 | 低 | 输出数字漂移 | git revert 单 commit |
| bfill 改 fillna 影响 grid_sector 数字 | 中 | 早期 ma20 段特征变,数字会略变 | 在 commit message 里标注 |
| FDR INIT_TRAIN_SIZE 限制意外引入 bug | 低 | grid_sector 特征数变,数字变 | 同上,预期行为 |
| `build_proba_signals` shift(1) 在边界样本漏 1 根 | 低 | entry 信号少 1 个 | 单元测试覆盖 |

---

## 8. 提交策略

预计 4 个 commit,每个独立可回滚:

1. `feat(common): 新增 tsfresh_walkforward.py — walk-forward proba + MA 通道 + 通道构成报告`
2. `feat(common): 新增 vbt_jhzq_backtest.py — vbt + jhzq_fees 单次回测 + 80% 拒单 warning + 格式化`
3. `refactor(tsfresh): tsfresh_vbt_combo.py 改用 common 模块,删除重复代码`
4. `refactor(tsfresh): tsfresh_with_ma_channel.py + tsfresh_with_ma_grid_sector.py 同步迁移`

每次 commit 后跑「冒烟测试」(改 STOCK_CODE='600118.SH' 跑 1 只票),确保数字 schema 不变。