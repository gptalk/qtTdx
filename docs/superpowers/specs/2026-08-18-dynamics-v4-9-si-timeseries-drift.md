# `backtrace/dynamics` — v4.9 行业 SI 时序 + 漂移检测

> 2026-08-18 写。v4.7 (`c63e783`) + v4.8 (`dbd367d`) 已合,v4.8 IC ≈ 0 揭示 SI 是描述性而非预测性。本 spec 把"描述性"扩展到时序:行业 SI 是否随时间漂移?漂移是否能预警风险?

## 1. 背景与动机

v4.7 答"哪些行业最稳",v4.8 IC ≈ 0 答"稳定对未来收益无预测力"。但**没答**:
- 行业稳定性是**持续**的还是漂移的?
- 行业从稳 → 不稳,是预警信号吗?
- 滚动 SI vs 单点 SI:哪个对风险预警更有用?

**业务问题**:
- 银行 SI 长期 0.85,但 2024-Q3 突然跌到 0.6 — 是改善还是风险信号?
- 半导体 SI 从 0.3 → 0.5 → 0.2 反复,均值约 0.3 — 稳定性是否就是行业属性?
- 哪些行业在最近 6 个月出现过漂移事件?

## 2. 范围 (Scope)

**In scope**:
1. **扩展 `parameter_fit.py`** 添加 `--rolling-time` 模式 — 每只票在每月末用最近 N 天 OLS 估 (k̂, ĉ),产出 `kc_estimates_time.csv` (long format: asof_date × code × k̂ × ĉ)
2. **新函数** `compute_sector_stability_timeseries(kc_long_df, ...)` 在 `dynamics_eigen_analysis.py` 末尾追加 (~50 行) — 复用 v4.7 公式,加 asof_date 轴
3. **新 CLI** `backtrace/dynamics/dynamics_si_timeseries.py` (~250 行,独立)
4. **漂移检测**: rolling 60 日 z-score < -2 → drift event
5. 5 个新单元测试 (`tests/test_dynamics_eigen.py` 末尾)
6. 更新 `backtrace/dynamics/README.md` §3.9

**Out of scope(冻结,本轮显式不做)**:
- 时序 SI 的 lagged IC 评估 — v4.10 候选
- 多维 SI dict — v4.11+ 候选
- 交易所层 SI (SH/SZ/BJ) — v4.11+ 候选
- 漂移预警 → 行业轮动 / 风险对冲策略 — v4.12+ 候选
- 修改 `analyze_eigenvalues` / `simulate_trajectory` / `compute_sector_stability` 数学
- 修改 3 个现有 caller:`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`
- 修改 `_dynamics_core.py`
- 修改 `dynamics_si_ic.py` (v4.8) — 时序 IC 评估是 v4.10

## 3. SI 时序计算

### 3.1 数据源

`parameter_fit.py --rolling-time` 输出 `data/projection/kc_estimates_time.csv`:

```csv
asof_date,code,name,index_code,index_tag,stock_tag,k_hat,c_hat,f_self_loss,n_valid_days,status
2024-01-31,600519.SH,贵州茅台,000001.SH,SH,S,0.85,1.05,1.2e-4,500,ok
2024-02-29,600519.SH,贵州茅台,...,...,...,0.83,1.07,...
...
```

**asof_date 频率**: 月末(每月最后一个交易日),~60 个 asof_date / 5 年数据
**rolling window**: 默认 240 日(可调 `--rolling-time-window`),保证稳定 OLS

### 3.2 公式(同 v4.7,只是加了 asof_date 轴)

对每个 (asof_date, industry_l1):
- 收集该行业所有成员的 (k̂, ĉ)
- `ρ_member = sqrt(|k̂² - k̂·ĉ + ĉ²|)` (同 v4.7 谱半径公式)
- `ρ_med = median(ρ_member)`, `c_med = median(ĉ_member)`
- `in_wedge_pct = (#ρ, c 在 Schur 楔形) / N_members`
- `ρ_health = clip(1 - ρ_med / 2, 0, 1)`
- `damping_health = clip(1 - |c_med - 1| / 2, 0, 1)`
- `wedge_health = clip(in_wedge_pct, 0, 1)`
- `SI = 0.5·ρ_health + 0.2·damping_health + 0.3·wedge_health`

**关键**:公式与 v4.7 `compute_sector_stability` 完全一致,共享 `SI_WEIGHTS = (0.5, 0.2, 0.3)` 常量。**不重复实现聚合 helper** — 直接调 `_industry_name_lookup` 和 Schur 楔形判定。

### 3.3 输出 schema (long format)

`data/dynamics/sector_si_timeseries.csv`:
```csv
asof_date,industry_l1,sector_name,n_stocks,rho_median,c_median,
in_wedge_pct,rho_health,damping_health,wedge_health,SI
2024-01-31,801010,银行,42,0.85,1.05,0.92,0.575,0.975,0.92,0.871
2024-01-31,801080,半导体,38,2.85,1.45,0.21,0.075,0.725,0.21,0.279
...
```

行业筛选(同 v4.3 / v4.7): `n_stocks >= 50` 强 / `n_stocks >= 30` 弱;都 < 5 时标 `low-confidence`。

## 4. 漂移检测算法

对每个行业 i 在时点 t (`t > first_asof_date + 60` 才计算):

```
rolling_mean_i(t) = mean(SI_i over [t-60, t))    # 60 日历史均值 (不含 t)
rolling_std_i(t)  = std(SI_i over [t-60, t))      # 60 日历史标准差 (不含 t)
z_score_i(t)      = (SI_i(t) - rolling_mean_i(t)) / rolling_std_i(t)

drift event: z_score_i(t) < -2                    # 2σ 突降,触发预警
```

**输出 schema** (long format):
```csv
asof_date,industry_l1,sector_name,SI,rolling_mean,rolling_std,z_score
2024-08-31,801080,半导体,0.18,0.45,0.08,-3.4
```

**注意**:
- rolling mean/std 用**闭区间 [t-60, t)** (不含 t) — 避免 leak
- 前 60 日 (warmup 期) → N < 60,标记 `warmup`,不计算 z-score
- `z_threshold=-2` 是默认值,可调 `--z-threshold`

## 5. 数据流与文件 IO

### 5.1 输入

| 来源 | 用途 |
|---|---|
| `data/projection/kc_estimates_time.csv` (new) | parameter_fit.py --rolling-time 输出 |
| `data/projection/eigen_summary.csv` 或 `kc_estimates.csv` | v4.3 产出 (回查 industry_l1 / sector_name) |
| `data/sw2/members.csv` | code → industry_l1 (已有) |
| `data/stock_basic.csv` | code → 中文名 (已有) |

### 5.2 输出(全 gitignored)

| 路径 | 内容 |
|---|---|
| `data/dynamics/sector_si_timeseries.csv` | **新增** — long format, asof_date × industry × 11 列 |
| `data/dynamics/si_drift_events.csv` | **新增** — drift event list |
| `backtrace/outputs/dynsys_si_timeseries.html` | **新增** — 4 子图 plotly |
| `backtrace/outputs/dynsys_si_timeseries_summary.txt` | **新增** — UTF-8 中文文本汇总 |

### 5.3 脚本结构

**修改** `backtrace/projection/parameter_fit.py` (新增 `--rolling-time` 模式,~60 行):
- `main_rolling_time(targets, window=240, ...)` — 月末 asof_date 列表,每只票 × 每个月末 → OLS
- 复用现有 `fit_one()` 函数(每个月末用最近 N 天)
- 输出 `data/projection/kc_estimates_time.csv`

**修改** `backtrace/dynamics/dynamics_eigen_analysis.py` (末尾追加 ~50 行):
- `compute_sector_stability_timeseries(kc_long_df, n_stocks_threshold=50)` — 复用 SI_WEIGHTS / Schur 楔形判定
- **不修改** `compute_sector_stability` (v4.7)

**新增** `backtrace/dynamics/dynamics_si_timeseries.py` (~250 行,独立 CLI):
- `load_kc_long(rolling_time_path)` — 读 `kc_estimates_time.csv`
- `detect_si_drift(si_ts_df, window=60, z_threshold=-2)` — 漂移检测
- `write_si_timeseries_summary(si_ts_df, drift_events_df, output_path)` — 文本汇总
- `build_si_timeseries_html(si_ts_df, drift_events_df, output_path)` — 4 子图 plotly
- `main()` — 端到端串起来

**代码增量**: ~360 行 (parameter_fit ~60 + eigen_analysis ~50 + new CLI ~250) + 5 tests

## 6. HTML 布局

```
┌──────────────────────────────────────────────────┐
│ (1,1) Top 6 SI 行业 时序 + 漂移事件红点          │
│ (1,2) Bottom 6 SI 行业 时序 + 漂移事件红点       │
│ (2,1) z-score 热力图(industry × date)            │
│ (2,2) 漂移事件频次 top 10 行业直方图              │
└──────────────────────────────────────────────────┘
```

子图细节:
- (1,1) / (1,2):x 轴 asof_date, y 轴 SI ∈ [0,1],漂移事件画红点
- (2,1):y 轴 industry (按 mean SI 排序),x 轴 asof_date,颜色 = z-score (RdBu 反转)
- (2,2):x 轴 drift_event_count, y 轴 industry

## 7. 测试设计

5 个新单元测试,放在 `tests/test_dynamics_eigen.py` 末尾:

| # | 名称 | 断言 |
|---|---|---|
| 1 | `test_si_timeseries_basic_shape` | 5 行业 × 100 日 → 500 行,SI ∈ [0,1] |
| 2 | `test_si_timeseries_stable_industry` | k̂, ĉ 恒定 → SI 几乎不变,无 drift event |
| 3 | `test_si_timeseries_sudden_drop` | 构造 SI(t=50) 从 0.8 → 0.2 → 触发 drift event |
| 4 | `test_si_timeseries_drift_zscore` | z-score 计算正确(均值 0 / 标准差 1) |
| 5 | `test_si_timeseries_summary_text` | summary txt 包含 "漂移事件" + 中文行业名 |

**总测试数**: 38 (v4.8) + 5 (v4.9) = **43 tests pass**

## 8. 与现有代码的关系

| 现有 | 关系 |
|---|---|
| `compute_sector_stability` (v4.7) | **复用公式** — 新函数是其时序版本,共享 SI_WEIGHTS |
| `parameter_fit.py --rolling-fit` | **共存** — 新模式 `--rolling-time` 是平行选项,不修改既有 `--rolling-fit` |
| `analyze_eigenvalues` (dynamics) | **不动** |
| 3 caller (`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`) | **0 行修改** — v4.9 不触及 |
| `_dynamics_core.py` | **不动** |
| `dynamics_si_ic.py` (v4.8) | **不动** — 时序 IC 是 v4.10 |
| `_industry_name_lookup` / Schur wedge 判定 | **复用** — 不重新实现 |

## 9. 已知风险

| 风险 | 缓解 |
|---|---|
| `parameter_fit.py --rolling-time` 输出 schema 不固定 | 函数开头 schema check,缺失列抛 FileNotFoundError |
| 月末 OLS 重跑 ~5000 stocks × 60 asof_dates × 240 days ≈ 200 万次 lstsq | 用 numpy 向量化(每只票独立 lstsq),单进程 < 5 分钟 |
| 行业 member 数 < 10 → SI 估计噪声大 | n_stocks_threshold=50 沿用 v4.7 |
| 60 日 warmup 期 → 前 60 日无 z-score | `warmup` 标记,summary 注明 |
| 漂移检测 z < -2 是任意阈值 | `--z-threshold` 可调,summary 注明"经验阈值" |
| `kc_estimates_time.csv` 大文件 (~5000 × 60 × 11 列 ≈ 300K 行) | pandas 分块读,只读必要列 |
| v4.7 / v4.8 数学层/Caller 0 修改约束 | 严格遵守 — v4.9 只在 projection + dynamics_eigen_analysis 末尾追加 |

## 10. 后续(本轮不做)

- **v4.10**: 时序 SI 的 lagged IC 评估(今日 SI(t) vs forward 20d 收益),闭环 v4.8 IC ≈ 0 结论
- **v4.11+**: 多维 SI dict(替代单一指标,UI tooltip)
- **v4.11+**: 交易所层 SI (SH/SZ/BJ, N=3 噪声大)
- **v4.12+**: 漂移预警 → 行业轮动 / 风险对冲策略(若 v4.10 IC 显著才有意义)

## 11. 关键文件

- 实现:
  - `backtrace/projection/parameter_fit.py` (新增 `--rolling-time` 模式 ~60 行)
  - `backtrace/dynamics/dynamics_eigen_analysis.py` (末尾追加 ~50 行)
  - `backtrace/dynamics/dynamics_si_timeseries.py` (new, ~250 行)
- 测试: `tests/test_dynamics_eigen.py` (末尾追加 5 测试, ~80 行)
- 文档: `backtrace/dynamics/README.md` §3.9 (~30 行)
- spec: `docs/superpowers/specs/2026-08-18-dynamics-v4-9-si-timeseries-drift.md` (本文件)
- plan: `docs/superpowers/plans/2026-08-18-dynamics-v4-9-si-timeseries-drift.md` (下一步)

## 12. 验证清单

- [ ] 5 个新测试通过 (43/43 total)
- [ ] `sector_si_timeseries.csv` 多行业 × 多 asof_date,SI ∈ [0,1]
- [ ] `si_drift_events.csv` 至少 1 个 drift event(若有数据)
- [ ] `dynsys_si_timeseries.html` 4 子图正常渲染
- [ ] `dynsys_si_timeseries_summary.txt` 中文可读,top 漂移行业清晰
- [ ] 0 行修改:`backtrace/dynamics/_dynamics_core.py` / `dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py` / `dynamics_si_ic.py` / `dynamics_eigen_analysis.py:compute_sector_stability`
- [ ] 端到端:跑 CLI exit 0,产出 4 个新文件