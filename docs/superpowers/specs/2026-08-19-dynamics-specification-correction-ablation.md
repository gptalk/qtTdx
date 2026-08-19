# V0.1 — Dynamics Specification Correction & Ablation

> Date: 2026-08-19
> Base: V0 audit `75a7b2b` (full-market 5211 stocks, well_conditioned 99.8%, R² median 1.58%)
> Branch: main
> Theme: **Spec correction, not parameter tuning** — Model 0/1/2/3 ablation + placebo

## 1. Context (为什么做)

V0 audit 排除了一种怀疑:**OLS 是否病态**。5211 只股票 full-market 结果:

```
Identification Status:  Well conditioned 99.8% / Singular 0.2%
Fit Quality:            Good 11.7% / Weak 46.6% / Poor 41.5%
cond(X):        median 7.65 (p75 10.5) — 数值上 (k̂, ĉ) 稳定可识别
R²:             median 1.58% (p75 4.97%) — 模型解释力几乎为零
```

这意味着 **IC≈0 不是 OLS 数学问题,而是动力学方程本身的 specification 问题**。

### 1.1 现行模型的两个 specification issue

代数约束:
```
v_S = β·v_M + u
```

对时间求导:
```
a_S = β·a_M + β̇·v_M + u̇     ← 数学恒等式
```

**现行 Model 0**(隐式假设 `q=1`、忽略 β̇·v_M):
```
a_S = β·a_M − k·d − c·u + F_self
```

代入恒等式得 `u̇` 的隐式模型:
```
u̇ = −β̇·v_M − k·d − c·u + F_self
```

问题:**β̇·v_M 项被塞进 F_self**。当 β 真实随时间漂移(quantile β(t) 不是常数)时,F_self 必然大 → R² 必然低。

**Issue 2**:Model 0 假设 q=1(市场推力 100% 传导)。真实 q 可能显著偏离 1。

### 1.2 V0.1 的研究问题

> **动力学方程本身是不是写错了?** — 而不是 OLS 病态、不是数据问题。

具体三个 sub-question:
- **V0.1a**:加 `β̇·v_M` 项后,R² 是否上升?(β drift 是否是主因)
- **V0.1b**:放开 q(OLS 估计)后,|q̂−1| 中位是多少?(q=1 假设是否合理)
- **V0.1c**:两者联合后,ΔIC 是否显著 > placebo IC?(模型是否捕捉到物理结构)

## 2. 模型定义 (写死的数学)

**所有 4 个模型共享同一组离散时间状态**(关键 — user explicitly required 时间索引一致):

```
t ∈ {0, 1, ..., T−2}                ← 有效预测索引(末行 NaN drop)
v_S(t), v_M(t) ∈ ℝ²                  ← t 时刻个股/大盘速度
β(t) ∈ ℝ                              ← t 时刻回归系数
d(t), u(t) ∈ ℝ²                       ← t 时刻位置偏离 + 速度差
a_S(t) := v_S(t+1) − v_S(t)          ← t→t+1 个股加速度
a_M(t) := v_M(t+1) − v_M(t)          ← t→t+1 大盘加速度
β̇(t) := β(t+1) − β(t)               ← t→t+1 β 漂移
```

**所有 4 个模型的残差即 F_self**。禁止修改 F_self 的定义。

### Model 0 (baseline / status quo)

```
a_S(t) = β(t)·a_M(t) − k·d(t) − c·u(t) + F_self(t)
```

OLS design matrix(2-D stacking,2T × 2):
```
Y = [a_S(t) − β(t)·a_M(t)]_{t=0..T−2}     长度 2(T−1)
X = [−d(t); −u(t)]_{t=0..T−2}              shape (2(T−1), 2)
θ = (k, c)                                  2 free params
```

### Model 1 (add β̇·v_M)

```
a_S(t) = β(t)·a_M(t) + β̇(t)·v_M(t) − k·d(t) − c·u(t) + F_self(t)
```

OLS design matrix(2T × 2):
```
Y = [a_S(t) − β(t)·a_M(t) − β̇(t)·v_M(t)]_{t=0..T−2}     长度 2(T−1)
X = [−d(t); −u(t)]_{t=0..T−2}                              shape (2(T−1), 2)
θ = (k, c)                                                  2 free params
```

**关键点**:`β̇·v_M` 项的系数是数学恒等式要求的 1.0,**不作为 free param 估计**(否则会与 q 共线)。它直接进 Y 侧作为 known offset。

### Model 2 (q as OLS parameter, no β̇·v_M)

```
a_S(t) = q·β(t)·a_M(t) − k·d(t) − c·u(t) + F_self(t)
```

OLS design matrix(2T × 3):
```
Y = [a_S(t)]_{t=0..T−2}                            长度 2(T−1)
X = [β(t)·a_M(t); −d(t); −u(t)]_{t=0..T−2}         shape (2(T−1), 3)
θ = (q, k, c)                                       3 free params
```

### Model 3 (joint: q + β̇·v_M)

```
a_S(t) = q·β(t)·a_M(t) + β̇(t)·v_M(t) − k·d(t) − c·u(t) + F_self(t)
```

OLS design matrix(2T × 3):
```
Y = [a_S(t) − β̇(t)·v_M(t)]_{t=0..T−2}             长度 2(T−1)
X = [β(t)·a_M(t); −d(t); −u(t)]_{t=0..T−2}         shape (2(T−1), 3)
θ = (q, k, c)                                       3 free params
```

β̇·v_M 项同样是 known offset(coef=1.0),不进 X 列。

## 3. OOS 边界(写死)

每个 stock 划分:
- **train** = 前 70%(t ∈ [0, floor(0.7·(T−2))])
- **test** = 后 30%(t ∈ [floor(0.7·(T−2)), T−2])

**禁止 train/test 重叠**;禁止未来信息泄漏进 train;禁止 shuffle within split。

**回归目标(Y)**:每只票 `a_S(t) ∈ ℝ²`,每个有效 t 一个 2-D 观测。

**IC 定义**(per stock):test 期间预测 a_S(t) 与实际 a_S(t) 的 Spearman 秩相关。

```
IC_real(s) = Spearman(â_S^test(s), a_S^test(s)) ∈ [−1, +1]
```

**Aggregate IC**:cross-sectional median across stocks(per model)。

## 4. Placebo Test (核心 ablation 工具)

**目的**:把"模型真的有预测信息"与"参数拟合产生的随机相关"区分开。

**对每只股票 × 每个 model**:
1. **Real fit**:用真实时序 β(t)·a_M(t), β̇(t)·v_M(t), d(t), u(t) → OLS → 预测 → IC_real
2. **Permuted fit**:把上述 4 个 regressor 的行索引**随机置换**(破坏时序耦合),但保留每个 regressor 的 marginal 分布;保持 a_S 不动 → OLS → 预测 → IC_null

**置换方法**(写死,禁止调参):
- 对每只股票 `np.random.default_rng(seed=42)`(seed 固定)
- 生成 4 个独立 permutation index(per regressor)
- 把每个 regressor 的行按 permuted index 重排
- a_S 行序保持不变

**对比规则**:
```
ΔIC_real = median_s IC_real(s)
ΔIC_null = median_s IC_null(s)
如果 ΔIC_null > 0 且 ΔIC_real ≈ ΔIC_null  →  模型无真信息(纯噪声)
如果 ΔIC_null ≈ 0 且 ΔIC_real > ΔIC_null  →  模型有真信息(即使绝对 IC 小)
如果 ΔIC_real < ΔIC_null                   →  反向信号(模型破坏了边际分布)
```

**禁止**:用 train/test split 不同来作弊 placebo;禁止 permute a_S(那是 leakage);禁止用 seed 调参找最优。

## 5. 指标表(per model × per stock)

每个 model 输出以下 11 列(`kc_estimates_<model>.csv` 每行一只票):

| 字段 | 类型 | 说明 |
|---|---|---|
| `code` | str | 股票代码 |
| `name` | str | 股票名称 |
| `index_code` | str | 大盘代码 |
| `index_tag` | str | 大盘 tag(000001/399001) |
| `stock_tag` | str | 个股 tag(代码 6 位) |
| `n_train` | int | train 集有效天数 |
| `n_test` | int | test 集有效天数 |
| `condition_number` | float | cond(X),不取 cond(XᵀX) |
| `regressor_corr` | float | X 第一列与其余列的相关中位(3 列版本) |
| `r2` | float | 1 − SS_res/SS_tot,SS_tot≈0 → NaN |
| `identification_status` | str | well/ill/unidentifiable/singular |
| `fit_quality` | str | good/weak/poor/uninformative |
| `q_hat` | float | q 值:Model 0/1 固定 1.0,Model 2/3 由 OLS 估计 |
| `k_hat` | float | k 估计值 |
| `c_hat` | float | c 估计值 |
| `f_self_loss` | float | ‖F_self‖² 平均 |
| `ic_real` | float | test 集 Spearman IC |
| `ic_null` | float | placebo IC |

**Cross-stock 汇总**(`kc_summary_<model>.csv`,每行一个 metric):
| metric | model_0 | model_1 | model_2 | model_3 |
|---|---|---|---|---|
| median R² | ✓ | ✓ | ✓ | ✓ |
| p25 R² | ✓ | ✓ | ✓ | ✓ |
| p75 R² | ✓ | ✓ | ✓ | ✓ |
| median cond(X) | ✓ | ✓ | ✓ | ✓ |
| median IC_real | ✓ | ✓ | ✓ | ✓ |
| median IC_null | ✓ | ✓ | ✓ | ✓ |
| median ΔIC (=IC_real − IC_null) | — | ✓ | ✓ | ✓ |
| median \|q̂ − 1\| | — | — | ✓ | ✓ |
| **ΔR² vs Model 0** | — | ✓ | ✓ | ✓ |
| **ΔIC vs Model 0** | — | ✓ | ✓ | ✓ |

## 6. 输出文件

```
data/projection/
├── kc_estimates_model0.csv      # 18 列(含 q_hat=1.0 固定列)
├── kc_estimates_model1.csv      # 18 列(含 q_hat=1.0 固定列)
├── kc_estimates_model2.csv      # 18 列(q_hat=OLS 估计)
├── kc_estimates_model3.csv      # 18 列(q_hat=OLS 估计)
├── kc_ablation_summary.csv      # 4×10 metric 矩阵
├── kc_ablation_recommendation.txt  # UTF-8 中文决策摘要
└── ablation_distribution.html   # 4-panel plotly: R² / IC / ΔIC / q̂

backtrace/outputs/
└── ablation_<model>_per_stock/  # 诊断 HTML(可选,scope gate)
```

## 7. CLI 接口(写死)

```bash
# 单个 model 跑
PYTHONIOENCODING=utf-8 python backtrace/projection/ablation_fit.py \
    --model {0|1|2|3} --limit 0

# 全 ablation + placebo + 汇总(默认 --placebo-samples 1,seed 42)
PYTHONIOENCODING=utf-8 python backtrace/projection/ablation_fit.py \
    --all --limit 0

# 冒烟
PYTHONIOENCODING=utf-8 python backtrace/projection/ablation_fit.py \
    --all --limit 10

# 单 model 干跑(跳过 placebo,快速冒烟)
PYTHONIOENCODING=utf-8 python backtrace/projection/ablation_fit.py \
    --model 3 --no-placebo --limit 100
```

**禁止**:
- ❌ 写 `--pick-best-ic` / `--maximize-ic` 之类调参 flag
- ❌ 写 `--feature-engineering` / `--add-feature` 之类加新 regressor flag
- ❌ 写 `--reverse-select-stocks-from-ic`

## 8. 实现边界(模块结构)

新增/修改文件:
- **新增** `backtrace/projection/ablation_fit.py` — 主 CLI + 4 个 model 调度 + placebo + 汇总
- **复用** `parameter_fit.py` 的 `_load_movement`、`_build_kinematics`(纯读取/重建,不动数学)
- **复用** `parameter_fit.py` 的 `build_identifiability_distribution_html`(可视化复用)
- **不动**:`prediction_ode.py`、`dynamics_*.py`、`gp_factor_mining/*`、v3-v6 所有产出

`_solve_ols` 扩展为接受不同 design matrix 的统一接口(类似 `ols_fit(X, Y, fixed_offsets=None) → (θ, diagnostics)`)。

## 9. 测试策略

### 9.1 单元测试(每 model 1 个)

每个 model 的 OLS 在 synthetic 数据上能精确恢复 `(q, k, c)`(已知噪声方差)。

### 9.2 Placebo test sanity

- 用纯噪声 `a_S`(无预测信息)→ `IC_real ≈ IC_null ≈ 0`
- 用 perfect linear 数据(`a_S = q·β·a_M + β̇·v_M − k·d − c·u`) → `IC_real > IC_null`

### 9.3 OOS 边界

- train/test split 不重叠(单元测试 assert)
- permutation index 不动 `a_S`(单元测试 assert)
- seed=42 固定(单元测试 assert)

### 9.4 Integration test

`--model 3 --limit 5` 冒烟:
- 4 个 CSV 输出存在
- HTML/TXT 输出存在
- exit 0

## 10. 决策树(归因优先级)

**跑完全市场 ablation 后,按以下顺序解读**:

```
Step 1: 看 Model 1 vs Model 0
   ΔR²_M1 = median_s (R²_M1,s − R²_M0,s)  > 0.005 ?     ← per-stock median delta
     ↓ YES
   → β drift 是 specification 问题的主要来源
   → Model 3 比 Model 2 更值得看

Step 2: 看 Model 2 vs Model 0
   median_s |q̂_M2,s − 1| > 0.1 ?                          ← per-stock median |q̂ − 1|
     ↓ YES
   → q=1 假设显著错,q 是 specification 问题的次要来源
   → Model 3 比 Model 1 更值得看

Step 3: 看 Model 3 的 ΔIC vs placebo
   median(IC_real_M3) − median(IC_null_M3) > 0.02 ?         ← difference of medians
     ↓ YES
   → 模型有真预测信息(即使 R² 仍低)
   → 结论:V0.2 = 接 Model 3 进 V6

     ↓ NO
   → 模型无真预测信息(纯 specification 错误 → 边际分布相关)
   → 结论:V0.2 = 收口动力学方法论

Step 4: 兜底
   如果 Step 1 和 Step 2 都是 NO
   → 两个 correction 都不解决问题
   → 结论:模型假设的更深结构错了(不是 β drift,不是 q=1)
   → V0.2 = 重新审视 F_self 定义(此 spec 之外)
```

**ΔR² / |q̂ − 1| / ΔIC 定义写死**:
- `ΔR²` = **median of per-stock deltas**(`median_s (R²_new,s − R²_old,s)`),NOT `median(R²_new) − median(R²_old)`。前者代表"典型股票的改善",后者受样本量/异常值影响。
- `|q̂ − 1|` = **median of per-stock |q̂ − 1|**(per-stock),NOT `|median(q̂) − 1|`。
- `ΔIC` = **difference of medians**(`median(IC_real) − median(IC_null)`),与 ΔR²/ΔIC 是不同的统计量(spec 写死后不改)。

**禁止**:
- ❌ 用 Step 1 + Step 2 同时 YES 但 ΔIC 不显著 → 推 Model 3 为最终模型(那只是过拟合)
- ❌ 把 `ΔR² > 0` 但 IC 不变 当作"成功"(模型只增加了 in-sample 拟合,无 OOS 价值)
- ❌ Mid-run 调整 ΔR² 阈值 / |q̂ − 1| 阈值 / ΔIC 阈值
- ❌ 看到 Model 1/2/3 结果后改 verdict 逻辑

## 11. 范围(Scope)与禁止(Non-Goals)

### 允许

- ✅ 加 `β̇·v_M` 项(Models 1/3)
- ✅ q 作为 OLS 参数估计(Models 2/3)
- ✅ 4 个 model ablation
- ✅ Placebo test(permutation baseline)
- ✅ 复用 V0 的 diagnostics(cond/regressor_corr/R²/identification/fit_quality)
- ✅ 输出 R²、cond、IC、ΔIC、|q̂−1|

### 禁止(YAGNI)

- ❌ **重新设计 d_vec**(锚定问题留 v0.2+)
- ❌ **修改 residual 定义**(F_self 始终 = Y − X·θ)
- ❌ **修改 prediction target**(仍 Δv_S)
- ❌ **改交易策略**
- ❌ **引入新 regressor**(除 β·a_M, β̇·v_M, −d, −u 之外)
- ❌ **调参寻找最高 IC**
- ❌ **根据 V6 IC 结果反向选择股票**
- ❌ **重写动力学方程的其他形式**(如非线性 k,时变 c 等)
- ❌ **修改 `_solve_ols` 现有 math**(只允许扩展为统一接口)
- ❌ **修改 F_self_predictor**(rolling mean / constant / oracle 都保留)

## 12. 风险

| 风险 | 缓解 |
|---|---|
| Model 3 的 `q̂` 与 `β̇·v_M` 共线(若 β 是常数) | β 是 quantile regression 估计的,实际随时间漂移;`regressor_corr` 监控共线 |
| Placebo 用 `np.random.default_rng(seed=42)` 固定 → 不调参 | seed 写死在 constants,代码 review 检查 |
| OOS test 集太短(≤ 30 天)→ IC 噪声大 | `n_test < 30` 的票在 summary 中单独标注,但仍报告 |
| 4 个 model 全市场跑 4 次 OLS,计算量大 | 复用 train OLS θ,test 阶段只算 `â_S_test = X_test · θ`(无需重新 fit) |
| V6 端用 `prediction_ode.py` 不感知 model 1/2/3 | 本 spec 不动 `prediction_ode.py`;后续若需用 model 3 预测,在 V0.2 再做 |

## 13. 交付物清单

- [ ] `backtrace/projection/ablation_fit.py` — 主 CLI
- [ ] `data/projection/kc_estimates_model{0,1,2,3}.csv` — 4 个 model 的 per-stock 估计
- [ ] `data/projection/kc_ablation_summary.csv` — 4×10 metric 矩阵
- [ ] `data/projection/kc_ablation_recommendation.txt` — UTF-8 中文决策摘要
- [ ] `backtrace/outputs/ablation_distribution.html` — 4-panel plotly
- [ ] `tests/test_dynamics_eigen.py` — 新增 8-10 个测试(per-model OLS + placebo sanity + OOS boundary + integration smoke)

## 14. 与 V0 / V6 / v3-v5.12 的关系

| 版 | commit | 主题 |
|---|---|---|
| V0 (audit) | `75a7b2b` | 加 diagnostics,不修数学 |
| **V0.1 (本 spec)** | (待) | 修数学 specification(Model 1/2/3)+ ablation + placebo |
| V0.2 (后续) | (待) | 若 V0.1 通过 → 接进 V6;若失败 → 收口 |
| V6 (factor validation) | `2971633` | IC + Q1-Q5 评估,基于 V0 模型 |

V0.1 不直接覆盖 V6。V6 仍跑 Model 0(现状),结果不重写。V0.1 的输出仅供 V0.2 决策。

## 15. 不在范围 / 故意不做

- ❌ 把 `predict_next_state` (dynamics v3) 升级为 Model 3 兼容(留 V0.2)
- ❌ 重写 `parameter_fit.py` 的现有 API(只加新文件)
- ❌ 修 v3-v5.12 任何模型(`dynamics_*` 全不动)
- ❌ 把 β 估计从 OLS 改成 Kalman / Bayesian
- ❌ 把 `a_u` 时间索引从 `t→t+1` 改成其他(本 spec §2 写死)