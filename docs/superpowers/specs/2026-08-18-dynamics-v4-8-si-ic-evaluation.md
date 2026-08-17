# `backtrace/dynamics` — v4.8 SI 与 forward return 的 IC 评估

> 2026-08-18 写。v4.7 (`c63e783`) 已合,产出 SI 单一指标但**未验证业务价值**。本 spec 闭环:用 cross-sectional Spearman IC 评估 SI 排名 vs 行业 forward return 排名的相关性,回答"稳定的行业未来是否真的跑得稳"。

## 1. 背景与动机

v4.7 行业稳定性指数 SI ∈ [0,1] 回答"哪些行业整体最稳定 / 最分裂",但**没有回答"稳定有没有用"**。本 spec 把 SI 接入跨截面 IC 评估,看 SI 排名与行业 forward return 排名是否相关。

**业务问题**:
- 高 SI 行业(银行、公用事业)→ forward 20d/60d 收益是否更低方差 / 更稳?
- 低 SI 行业(半导体、医疗器械)→ forward 收益是否更高方差 / 风险更大?
- SI 的 IC 跨期是否稳定(滚动 60 日)?还是均值回归后失效?

如果 SI 与 forward return 的 IC ≈ 0(类似 v3 README §3.4 state_prop 的现象),则 SI 是**描述性指标**而非**预测性指标** — 用于报告 / 风险标签,不是选股信号。如果 IC 显著,则 SI 可作为行业轮动辅助。

## 2. 范围(Scope)

**In scope**:
1. 新 CLI: `backtrace/dynamics/dynamics_si_ic.py`
   - 读 `data/dynamics/sector_si.csv` (v4.7 产出) + `data/projection/kc_estimates.csv` (回查 code → industry_l1 映射)
   - 读 `data/daily/<code>.csv` 算各行业 forward 20d / 60d 收益
   - 滚动 IC: 60 日窗口,步长 20 日,跨截面 Spearman(SI 排名 vs forward return 排名)
   - 输出: `data/dynamics/si_ic_summary.csv` (N 窗口 IC 汇总) + `data/dynamics/si_ic_timeseries.csv` (per-window detail) + 1 个 HTML
2. 3 个新单元测试(`tests/test_dynamics_eigen.py` 末尾)
3. 更新 `backtrace/dynamics/README.md` §3.8 + spec footnote

**Out of scope(冻结,本轮显式不做)**:
- 行业 SI 时序(滚动 60 日)+ 漂移检测 — v4.9 候选
- 多维 SI dict 替代单一指标 — v4.9 候选
- 交易所层 SI — v4.9 候选
- 行业轮动策略(基于 IC 的实际交易信号)— v4.10+ 候选
- 修改 `analyze_eigenvalues` / `simulate_trajectory` 数学
- 修改 3 个现有 caller:`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`
- 修改 `compute_sector_stability` (v4.7) — 只读取
- 修改 `dynamics_state_backtest.py` (现成 IC 模式)— 只参考其 Spearman 计算方式

## 3. IC 计算定义

### 3.1 Forward return

对每个行业 $i$, 在时点 $t$, forward return 定义为:
$$r_i(t, h) = \frac{P_{\text{median},i}(t + h) - P_{\text{median},i}(t)}{P_{\text{median},i}(t)}$$

其中 $P_{\text{median},i}(t)$ 是行业 $i$ 在 $t$ 时刻**成员股票中位数收盘价**(避免极端值),$h \in \{20, 60\}$ 日。

### 3.2 跨截面 Spearman IC

对每个评估日 $t$, 计算 $N_{\text{industries}}$ 个行业的 SI 排名 vs forward return 排名:
$$\text{IC}(t, h) = \text{SpearmanRankCorr}(\{SI_i\}_{i=1}^{N}, \{r_i(t, h)\}_{i=1}^{N})$$

### 3.3 滚动窗口

- 窗口大小: 60 日
- 步长: 20 日
- 起点: `max(kc_estimates.csv 的 max(asof_date), 数据首日) + 60` — 留出 forward 60 日所需窗口
- 终点: 数据末日 - 60
- 每窗口 IC = 窗口内 60 个**逐日 IC 的算术平均**(不是把所有 (SI, r) 对 pool 起来重算一次 Spearman,避免重复计权重) 日

### 3.4 跨期汇总

输出 3 个数:
- `ic_mean` = 跨窗口 IC 的算术平均
- `ic_std` = 跨窗口 IC 的标准差
- `ic_ir` = `ic_mean / ic_std` (信息比, 类似 Sharpe)
- `p_value_mean` = 跨窗口 p-value 的算术平均
- `n_windows` = 窗口数

## 4. 数据流与文件 IO

### 4.1 输入

| 来源 | 用途 |
|---|---|
| `data/dynamics/sector_si.csv` | v4.7 产出(9 列,128 行业 × SI 单值) |
| `data/projection/kc_estimates.csv` | 回查 `code → industry_l1` 映射 |
| `data/sw2/members.csv` | 回查 `industry_l1 → industry_l2`(中文名) |
| `data/daily/<code>.csv` | 每只票日线,算 forward return |
| `data/stock_basic.csv` | 可选,过滤停牌 / ST |

**注意**:`sector_si.csv` 是**单值**的(每行业 1 个 SI)— 不带时间戳。要做 IC 评估,需要重新估**时序 SI**。

**简化方案**(本轮范围):
- 不重做时序 SI 估计
- 用 v4.7 单一 SI 值(基于当前 kc_estimates 跑批结果)作为**整个评估期恒定的行业评分**
- 每个评估日,跨截面 Spearman 仍是"SI 排名 vs forward return 排名",只是 SI 不变

**这意味着 IC 反映"行业长期稳不稳定 vs 该日收益"的相关性**,而非"日级 SI 变化"的相关性。
符合 v4.7 spec §9 候选 2 的简化版。

### 4.2 输出(全部 gitignored)

| 路径 | 内容 |
|---|---|
| `data/dynamics/si_ic_summary.csv` | **新增** — 跨期汇总 2 行(forward 20d / 60d),列: `horizon, ic_mean, ic_std, ic_ir, p_value_mean, n_windows` |
| `data/dynamics/si_ic_timeseries.csv` | **新增** — per-window detail,列: `window_end_date, horizon, ic, p_value, n_industries` |
| `backtrace/outputs/dynsys_si_ic.html` | **新增** — 1 个 plotly HTML,含 2 子图(滚动 IC 时序 + 行业 SI 散点 vs 累计 forward 60d 收益) |
| `backtrace/outputs/dynsys_si_ic_summary.txt` | **新增** — UTF-8 文本汇总 |

### 4.3 脚本结构

新文件 `backtrace/dynamics/dynamics_si_ic.py`(~150 行):
- `load_sector_si(path)` — 读 sector_si.csv
- `load_industry_membership(kc_path, sw2_path)` — 反查 code → industry_l1
- `compute_industry_forward_returns(stocks_df_by_industry, dates, horizon)` — 中位数收盘价法
- `rolling_cross_sectional_ic(si_by_industry, forward_returns_by_date, window, step)` — 滚动 IC
- `write_si_ic_summary(timeseries_df, output_path)` — 跨期汇总
- `build_si_ic_html(timeseries_df, summary_df, output_path)` — 1 HTML 2 子图
- `main()` — 端到端串起来

代码增量: **~150 行 + 3 tests**

## 5. HTML 布局

```
┌─────────────────────────────────────────────────────────┐
│  1. (1, 1) Rolling IC(20d / 60d)时序 + IC=0 红虚线     │
│  2. (1, 2) 行业 SI vs 累计 forward 60d 收益 散点        │
│                                                          │
│  (2, 1, 全宽) IC 统计汇总(ic_mean / ic_std / ic_ir)    │
└─────────────────────────────────────────────────────────┘
```

## 6. 测试设计

3 个新单测,放在 `tests/test_dynamics_eigen.py` 末尾:

| # | 名称 | 断言 |
|---|---|---|
| 1 | `test_si_ic_synthetic_perfect` | 构造 5 行业 × 100 日,SI 与 forward return 完美正相关 → IC ≈ 1.0 |
| 2 | `test_si_ic_synthetic_random` | 构造 5 行业 × 100 日,SI 与 forward return 完全独立 → IC ≈ 0.0 (±0.3) |
| 3 | `test_si_ic_summary_schema` | `write_si_ic_summary` 写出 2 行(20d / 60d) × 6 列 |

**总测试数**: 35 (v4.7) + 3 (v4.8) = **38 tests pass**

## 7. 与现有代码的关系

| 现有 | 关系 |
|---|---|
| `compute_sector_stability` (v4.7) | **只读** — `load_sector_si` 消费其 CSV |
| `dynamics_state_backtest.py:275-304` IC 模式 | **参考** — 复制其 `scipy.stats.spearmanr` 用法 |
| `dynamics_eigen_analysis.py` | **不动** |
| 3 caller (`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`) | **0 行修改** — 新文件是独立 CLI |
| `parameter_fit.py` (projection) | **不动** — 只读取 kc_estimates.csv |

## 8. 已知风险

| 风险 | 缓解 |
|---|---|
| 行业 member 数小(< 10 只)中位数收益不稳 | 文本汇总列 `n_industries`,行内说明每窗口有效行业数 |
| `sector_si.csv` 缺失(刚清缓存) | 函数开头 `if not os.path.exists(...): raise FileNotFoundError` |
| `data/daily/<code>.csv` 缺失 / 停牌 | `compute_industry_forward_returns` 用 `dropna()`,N < 5 行业时该窗口 IC 标 NaN |
| IC ≈ 0(类似 state_prop 现象, README §3.4) | 文本汇总明确报告"预测力 ≈ 0,SI 是描述性而非预测性",不强行包装 |
| `scipy.stats.spearmanr` 在 2 行业时退化为 ±1 | `n_industries >= 5` 才计算 IC,否则该窗口 NaN |
| 滚动窗口 60 日 + 步长 20 日,需要 ~6 个月数据 | README 标注 "需要 ≥ 6 个月历史" |

## 9. 后续(本轮不做)

- v4.9 候选 1: 行业 SI 时序(滚动 60 日)+ 漂移检测(本轮的"用单值 SI 评估"的扩展)
- v4.9 候选 2: 多维 SI dict 替代单一指标
- v4.9 候选 3: 交易所层 SI
- v4.10+: 受迫系统 + G(ω) 频率响应(独立 v5 工作包)
- v4.10+: 基于 IC 的行业轮动策略(若 IC 显著才有意义)

## 10. 关键文件

- 新增: `backtrace/dynamics/dynamics_si_ic.py` (~150 行)
- 新增: `tests/test_dynamics_eigen.py` 末尾追加 3 测试 (~50 行)
- 修改: `backtrace/dynamics/README.md` §3.8 (v4.8 节,~30 行)
- spec: `docs/superpowers/specs/2026-08-18-dynamics-v4-8-si-ic-evaluation.md` (本文件)
- plan: `docs/superpowers/plans/2026-08-18-dynamics-v4-8-si-ic-evaluation.md` (下一步)

## 11. 验证清单

- [ ] 3 个新测试通过(38/38 total)
- [ ] `dynsys_si_ic.html` 2 子图正常渲染
- [ ] `si_ic_summary.csv` 2 行(20d / 60d),ic_mean 在合理范围 [-0.5, 0.5]
- [ ] `si_ic_timeseries.csv` 至少 5 个窗口的 detail
- [ ] 0 行修改:`backtrace/dynamics/_dynamics_core.py` / `dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py` / `dynamics_eigen_analysis.py` / `_dynamics_core.py`
- [ ] 端到端: 跑 CLI exit 0,产出 4 个新文件
