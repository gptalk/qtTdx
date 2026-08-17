# `backtrace/dynamics` — v4.10 时序 SI 的 lagged IC 评估

> 2026-08-18 写。v4.7 (`c63e783`) + v4.8 (`dbd367d`) + v4.9 (`f2178a3`) 已合。本 spec 闭环 v4.8 IC ≈ 0 结论:用 lagged IC(时序 SI(t) vs future forward return)测"今日 SI 能否预测未来收益",与 v4.8 contemporaneous IC(同时点 SI(t) vs forward return(t, h))形成对照。

## 1. 背景与动机

v4.7/v4.8/v4.9 三轮迭代脉络:

| 版 | 主题 | 关键发现 |
|---|---|---|
| v4.7 | SI 单一指标 | 银行 / 公用事业 SI 高,半导体 / 医疗器械 SI 低 |
| v4.8 | SI × forward return rolling IC | **contemporaneous IC ≈ 0** → SI 不是预测性指标 |
| v4.9 | SI 时序 + 漂移检测 | 行业 SI 随时间漂移,drift event 可预警 |
| **v4.10** | **时序 SI 的 lagged IC** | **今日 SI(t) 能否预测未来 forward return?** |

**v4.8 的局限性**:contemporaneous IC = `Spearman(SI_i(t), r_i(t, h))`,**同一时点** t 的 SI 与 forward 收益对比。这反映"行业长期稳 vs 该日收益"的相关性,但**不是预测力测试**——SI 和 forward return 都观察自 t 时刻,无时间先后。

**v4.10 的改进**:lagged IC = `Spearman(SI_i(t), r_i(t+h, h))`,**不同时点**:
- t 时刻观察到 SI(t) 排名
- t+h 时刻观察到 forward return(实际未来收益)
- 这才是真正的预测力测试

**3 种结果含义**:
| lagged IC | 含义 |
|---|---|
| ≈ 0 | 行业层**纯描述性**,SI 用于报告 / 风险标签,不作选股信号 |
| > 0.05 显著 | 行业 SI(t) 是预测性指标,可作轮动信号(下一步:v4.12 行业轮动策略) |
| 接近 v4.8 | 时序 SI 没新信息(早 asof 漂移是真实信号) |

## 2. 范围(Scope)

**In scope**:
1. **新 CLI** `backtrace/dynamics/dynamics_si_lagged_ic.py` (~300 行,独立,不修改 v4.8 / v4.9 任何文件)
2. 4 个新单元测试(`tests/test_dynamics_eigen.py` 末尾)
3. 更新 `backtrace/dynamics/README.md` §3.10

**Out of scope(冻结,本轮显式不做)**:
- 多维 SI dict — v4.11+ 候选
- 交易所层 SI (SH/SZ/BJ) — v4.11+ 候选
- 漂移预警 → 行业轮动策略 — v4.12+ 候选
- 受迫系统 + G(ω) 频率响应 — v4.12+ / v5 工作包
- 修改 `dynamics_si_ic.py` (v4.8) — 新文件独立 CLI
- 修改 `dynamics_si_timeseries.py` (v4.9) — 只读 sector_si_timeseries.csv
- 修改 `compute_sector_stability_timeseries` (v4.9) — 只读
- 修改 `_dynamics_core.py` / 3 caller / v4.7 `compute_sector_stability`

## 3. Lagged IC 计算定义

### 3.1 Lagged IC 与 Contemporaneous IC 的形式区别

```
Contemporaneous IC (v4.8):
  IC(t, h) = SpearmanRankCorr( {SI_i(t)}, {r_i(t, h)} )
              ↑                       ↑
              同一时点 t 的 SI 排名    同一时点 t 看到的 forward 收益(t → t+h)

Lagged IC (v4.10, 本 spec):
  IC(t, h) = SpearmanRankCorr( {SI_i(t)}, {r_i(t+h, h)} )
              ↑                       ↑
              t 时刻的 SI 排名          t+h 时刻看到的 forward 收益(t+h → t+2h)
```

**时间结构**:
- v4.8:同时点观察(SI 与收益都已知)— 描述性 / 同因
- v4.10:不同时点观察(SI 已知,收益要等到未来)— 真正的预测性

### 3.2 Forward return(同 v4.8)

$$r_i(t, h) = \frac{P_{\text{median},i}(t + h) - P_{\text{median},i}(t)}{P_{\text{median},i}(t)}$$

行业成员股票**中位数收盘价**法,避免极端值。$h \in \{20, 60\}$ 日。

### 3.3 滚动窗口(沿用 v4.8)

- 窗口大小:60 日(交易日)
- 步长:20 日
- 起点: `max(sector_si_timeseries.csv 的 max(asof_date), 数据首日) + 60` — 留出 forward 60 日所需窗口
- 终点: 数据末日 - 60
- 每窗口 IC = 窗口内 60 个**逐日 lagged IC 的算术平均**(避免 pool 重算)

### 3.4 Lagged 偏移

```
每日 lagged IC(t, h) 计算:
  1. 取行业 SI 在 (t - h) 时刻的排名(即 SI 排名领先 forward 收益 h 日)
  2. 取行业 forward return 在 t 时刻的排名(实际未来收益,看到的是 t → t+h)
  3. Spearman(SI 排名, forward 收益排名)
```

**实现**:在 `compute_industry_forward_returns` 输出 `forward_returns_by_date` 基础上,**对齐 SI 时序向后偏移 h 日**:

```python
# 对每个 horizon h:
si_aligned = si_ts[si_ts['asof_date'] <= (eval_date - h)]    # SI 只能用到 (t - h)
forward_aligned = forward_returns[forward_returns['asof_date'] == eval_date]   # forward 收益在 t 时刻
# 然后跨截面 Spearman
```

### 3.5 跨期汇总(沿用 v4.8)

输出 5 个数(per horizon):
- `ic_mean` = 跨窗口 IC 算术平均
- `ic_std` = 跨窗口 IC 标准差
- `ic_ir` = `ic_mean / ic_std` (信息比)
- `p_value_mean` = 跨窗口 p-value 算术平均
- `n_windows` = 窗口数

## 4. 数据流与文件 IO

### 4.1 输入

| 来源 | 用途 |
|---|---|
| `data/dynamics/sector_si_timeseries.csv` | v4.9 产出(11 列,long format,asof_date × industry) |
| `data/projection/kc_estimates.csv` | 回查 code → industry_l1(同 v4.8) |
| `data/sw2/members.csv` | code → industry_l2 中文名 |
| `data/daily/<code>.csv` | 每只票日线,算 forward return |
| `data/stock_basic.csv` | 可选,过滤停牌 / ST |

**注意**:v4.10 的 SI 输入是**时序的**(每行业 N 个 asof_date 的 SI),与 v4.8 单值 SI 不同。这是 v4.10 的关键差异。

### 4.2 输出(全 gitignored)

| 路径 | 内容 |
|---|---|
| `data/dynamics/si_lagged_ic_summary.csv` | **新增** — 跨期汇总 2 行(forward 20d / 60d),列: `horizon, ic_mean, ic_std, ic_ir, p_value_mean, n_windows` |
| `data/dynamics/si_lagged_ic_timeseries.csv` | **新增** — per-window detail,列: `window_end_date, horizon, ic, p_value, n_industries` |
| `backtrace/outputs/dynsys_si_lagged_ic.html` | **新增** — 3 子图 plotly |
| `backtrace/outputs/dynsys_si_lagged_ic_summary.txt` | **新增** — UTF-8 中文汇总 |

### 4.3 脚本结构

新文件 `backtrace/dynamics/dynamics_si_lagged_ic.py` (~300 行,独立 CLI,不 import 同目录 sibling 模块):
- `load_sector_si_timeseries(path)` — 读 v4.9 sector_si_timeseries.csv(11 列)
- `load_industry_membership(kc_path, sw2_path)` — 回查 code → industry_l1(同 v4.8)
- `compute_industry_forward_returns(stocks_df_by_industry, dates, horizon)` — 中位数收盘价法(同 v4.8)
- `compute_lagged_cross_sectional_ic(si_ts_df, forward_returns_df, horizon)` — 滞后对齐 + Spearman
- `rolling_lagged_ic(si_ts_df, forward_returns_df, window, step, horizon)` — 滚动窗口
- `write_si_lagged_ic_summary(timeseries_df, output_path)` — 跨期汇总
- `build_si_lagged_ic_html(timeseries_df, summary_df, v4_8_summary_path, output_path)` — 3 子图 plotly
- `main()` — 端到端串起来

代码增量: **~300 行 + 4 tests**

## 5. HTML 布局

```
┌──────────────────────────────────────────────────┐
│ (1,1) Lagged IC 时序 (20d / 60d 双线 + IC=0 红虚线)│
│ (1,2) v4.10 lagged vs v4.8 contemporaneous IC 对比 │
│ (2,1, 全宽) IC 统计汇总(ic_mean / std / ir / p / n)│
└──────────────────────────────────────────────────┘
```

子图 (1,2) 读取 `data/dynamics/si_ic_summary.csv`(v4.8 产出,若存在)做对比散点图:v4.8 IC_mean vs v4.10 lagged IC_mean per horizon。**若 v4.8 CSV 不存在**,此子图退化为"v4.8 数据未生成"提示文字。

## 6. 测试设计

4 个新单元测试,放在 `tests/test_dynamics_eigen.py` 末尾:

| # | 名称 | 断言 |
|---|---|---|
| 1 | `test_si_lagged_ic_synthetic_perfect` | 构造 5 行业 × 100 日,SI(t) 与 forward_return(t+20) 完美正相关 → lagged IC > 0.5 |
| 2 | `test_si_lagged_ic_synthetic_random` | 构造 5 行业 × 100 日,SI 与 forward 完全独立 → |lagged IC| < 0.3 |
| 3 | `test_si_lagged_ic_temporal_shift` | t=0 SI 排名 + t=20 收益排名相关,验证时间偏移正确 |
| 4 | `test_si_lagged_ic_summary_schema` | `write_si_lagged_ic_summary` 写出 2 行 × 6 列 |

**总测试数**: 44 (v4.9) + 4 (v4.10) = **48 tests pass**

## 7. 与现有代码的关系

| 现有 | 关系 |
|---|---|
| `compute_sector_stability_timeseries` (v4.9) | **只读** — `load_sector_si_timeseries` 消费其 CSV |
| `dynamics_state_backtest.py:275-304` IC 模式 | **参考** — 复制其 `scipy.stats.spearmanr` 用法 |
| `dynamics_si_ic.py` (v4.8) | **不动** — 新文件独立 CLI,逻辑相似但输入是时序 SI |
| `dynamics_si_timeseries.py` (v4.9) | **不动** — 只读 sector_si_timeseries.csv |
| `parameter_fit.py --rolling-time` (v4.9) | **不动** — 不再追溯到 (k̂, ĉ),只读 SI |
| 3 caller (`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`) | **0 行修改** |
| `_dynamics_core.py` | **不动** |

## 8. 已知风险

| 风险 | 缓解 |
|---|---|
| 行业 member 数小(< 10 只)中位数收益不稳 | `compute_industry_forward_returns` 用 `dropna()`,N < 5 行业时该窗口 IC 标 NaN |
| `sector_si_timeseries.csv` 缺失(刚清缓存 / v4.9 未跑) | 函数开头 `if not os.path.exists(...): raise FileNotFoundError` + 友好提示 |
| `data/daily/<code>.csv` 缺失 / 停牌 | `compute_industry_forward_returns` graceful skip,N < 5 跳过该窗口 |
| v4.10 IC ≈ 0(类似 v4.8 现象) | 文本汇总明确报告"lagged IC ≈ 0,行业层纯描述性",不强行包装 |
| `scipy.stats.spearmanr` 在 2 行业时退化为 ±1 | `n_industries >= 5` 才计算 IC |
| 滚动窗口 60 日 + 步长 20 日,需要 ~6 个月数据 | README 标注 "需要 ≥ 6 个月历史" |
| asof_date 月度,horizon=20/60 日可能不对齐 asof_date 网格 | 内部用日期序数对齐(forward return 在每个交易日都可算,SI 在每月 asof_date) |

## 9. 后续(本轮不做)

- **v4.11+**: 多维 SI dict(替代单一指标,UI tooltip)
- **v4.11+**: 交易所层 SI (SH/SZ/BJ,N=3 噪声大)
- **v4.12+**: 漂移预警 + 时序 SI 的 lagged IC 显著 → 行业轮动策略
- **v4.12+ / v5**: 受迫系统 + G(ω) 频率响应

## 10. 关键文件

- 新增: `backtrace/dynamics/dynamics_si_lagged_ic.py` (~300 行)
- 新增: `tests/test_dynamics_eigen.py` 末尾追加 4 测试 (~60 行)
- 修改: `backtrace/dynamics/README.md` §3.10 (~30 行)
- spec: `docs/superpowers/specs/2026-08-18-dynamics-v4-10-si-lagged-ic.md` (本文件)
- plan: `docs/superpowers/plans/2026-08-18-dynamics-v4-10-si-lagged-ic.md` (下一步)

## 11. 验证清单

- [ ] 4 个新测试通过 (48/48 total)
- [ ] `dynsys_si_lagged_ic.html` 3 子图正常渲染
- [ ] `si_lagged_ic_summary.csv` 2 行(20d / 60d),ic_mean 在合理范围 [-0.5, 0.5]
- [ ] `si_lagged_ic_timeseries.csv` 至少 5 个窗口的 detail
- [ ] 0 行修改:`backtrace/dynamics/_dynamics_core.py` / `dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py` / `dynamics_si_ic.py` (v4.8) / `dynamics_si_timeseries.py` (v4.9) / `dynamics_eigen_analysis.py:compute_sector_stability_timeseries`
- [ ] 端到端: 跑 CLI exit 0,产出 4 个新文件