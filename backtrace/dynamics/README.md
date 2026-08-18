# `backtrace/dynamics` — 离散动力系统入口

> 2026-08-16 新建。把 24 节「市场—个股耦合动力系统」中属于**动力系统**的部分
> (1 步预测、N 步轨迹模拟、状态分类后的力分解)收口成可调用 API + CLI 入口。
> 数学源头 = [backtrace/projection/](../projection/)(`compute_dynamics` / `compute_forces` / `classify_states`),
> 本目录**不重写**任何数学,只做"用户面向动力系统"的封装。

## 1. 目录结构

```
dynamics/
├── _dynamics_core.py       数学 re-export + 6 个新增 API
│                           (predict_next_state / simulate_trajectory / build_simulation_df
│                            + analyze_eigenvalues  ← v4
│                            + F_self 预测器 ×3 + forecast helper ×5)
├── dynamics_system.py      单股端到端 CLI (load → describe → simulate → HTML/CSV)
├── dynamics_batch.py       批量 CLI (读 stocks.csv → 全跑 → manifest)
├── dynamics_1step_oos.py   OOS 1 步预测(纯动力学基线,F_self 滚动均值)
├── dynamics_state_backtest.py  状态分组 + vbt basket 回测 + IC 评估
├── dynamics_eigen_analysis.py  (k̂, ĉ) → 特征值 + 11 类稳定性分类 + HTML(v4)
└── README.md               本文件
```

## 2. 数学分层

| 层 | 位置 | 职责 |
|---|---|---|
| **数据加载** | `projection._projection_core.load_pair` | 共同交易日对齐,本地 `data/` 缓存 |
| **运动投影** | `projection._projection_core.compute_movement_projection` | Δu / Δv / β / proj / residual |
| **描述层** | `projection._projection_core.compute_dynamics` + `classify_states` | q_t / θ / R / E_market / E_self / state |
| **力模型** | `projection._projection_core.compute_forces` | F_market / F_restore / F_damp / F_self |
| **预测(新)** | `dynamics._dynamics_core.predict_next_state` | 1 步: a_pred = β·a_M - k·d - c·u + F_self |
| **模拟(新)** | `dynamics._dynamics_core.simulate_trajectory` | N 步前向积分 v_{t+1} = v_t + a_t |
| **F_self 预测器(新)** | `make_rolling_mean_f_self_predictor` / `make_constant_f_self_predictor` / `make_ar1_f_self_predictor` | 把残差外推从"末日瞬时值"升级到"滚动均值" / "AR(1) 自回归" |
| **Forecast 模式(新)** | `forecast_v_M_random_walk` / `forecast_v_M_last_value` / `forecast_beta_*` / `forecast_q_t_constant` | 无未来大盘观测时,合成 v_M_seq / beta_seq / q_t_seq |
| **OOS 1 步预测 CLI(新)** | `dynamics_1step_oos.py` | 用 `predict_next_state` 跑 1 步 OOS,产预测 CSV + summary |
| **状态分组 + vbt 回测 CLI(新)** | `dynamics_state_backtest.py` | 按 dominant_state 分组 + basket 回测 + IC |
| **特征值稳定性分析(v4)** | `analyze_eigenvalues(k, c)` + `dynamics_eigen_analysis.py` | 从 2D 离散系统 `A=[[1,1],[-k,1-c]]` 求 λ₁/λ₂/ρ,11 类稳定性分类 |
| **HTML/CSV 落盘** | `dynamics_system.py` / `dynamics_batch.py` / `dynamics_eigen_analysis.py` | 数据组装 + 文件落地 |

## 3. CLI 用法

### 3.1 单股

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_system.py
# 默认 002475.SZ / 立讯精密 / 240 日 / 大盘基线 / N=5 / k=c=0

PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_system.py \
    --code 600519.SH --name 贵州茅台 --days 240 --horizon 10
# 10 步模拟

PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_system.py \
    --code 002475.SZ --horizon 5 --k-restore 0.1 --c-damp 0.05
# 加弱恢复力 + 弱阻尼

PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_system.py --k-from-fit --c-from-fit
# 从 data/projection/kc_estimates.csv 自动加载 OLS 拟合的 k̂/ĉ
```

**F_self 模式选择**(全部 CLI 都支持):
- `rolling`(默认)— 末日滚动均值,常数预测器;`--f-self-window N` 控制窗口
- `constant` — 末日瞬时值,常数预测器
- `oracle` — 末日观测残差恒定外推(N 次复制)— 调试 / description 验证用
- `ar1`(2026-08-17 新)— **AR(1) 自回归**,每步 `F_self(t) = μ + ρ^t · (F_self(0) - μ)`;ρ/μ 估自历史残差(per-dim);数据不足(< `--f-self-window`)自动退化到常数

**输出**(单股):
- `backtrace/outputs/dynsys_simulation.html` — 5 子图(实际 v_S → 模拟 v_S / 能量 / R+θ / 状态 / 力分解)
- `data/dynamics/dynamics_<idx>_<stk>.csv` — 描述层 14 列(与 `projection_2d --dynamics` 共享 schema)
- `data/dynamics/forces_<idx>_<stk>.csv` — 力分解 8 列
- `data/dynamics/simulation_<idx>_<stk>.csv` — 模拟 19 列(18 + Date)

### 3.2 批量

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_batch.py --limit 50
# 跑前 50 只

PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_batch.py \
    --input data/my_stocks.csv --horizon 10 --k-restore 0.1
# 自定义股票列表

PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_batch.py --k-from-fit --c-from-fit
# 用 OLS 拟合的 k̂/ĉ 跑全 A 股(前提:parameter_fit.py 跑过)
```

**输出**(批量):
- 每只票:`data/dynamics/{dynamics,forces,simulation}_<idx>_<stk>.csv`
- 清单:`data/dynamics/batch_manifest.csv`,列:
  `code, name, index_code, index_name, rows, horizon, k_restore, c_damp,
   desc_csv_path, frc_csv_path, sim_csv_path,
   sim_mean_R, sim_max_E_self, sim_state_dist, status`

### 3.3 OOS 1 步预测

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_1step_oos.py --limit 50
# 默认滚动均值 W=10,跑前 50 只

PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_1step_oos.py \
    --input data/my_stocks.csv --f-self-window 20
# W=20 滚动均值

PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_1step_oos.py --k 0.5 --c 0.1
# 加弱恢复 + 弱阻尼(给模型一点拟合自由度)
```

**输出**:
- 每只票:`data/dynamics/prediction_<idx>_<stk>.csv` (11 列,日级预测 vs 实际)
- 汇总:`data/dynamics/prediction_summary.csv` (跨股票命中率 + RMSE)

**重要陷阱**:默认 F_self 用 `a_S_now` 在 k=c=0 下会退化成恒等式(命中率虚高 100%),
本脚本已用 `--f-self-window` 滚动均值兜底;设 `--f-self-window 0` 会触发警告。
**`--f-self-mode ar1`** 让 F_self 按 AR(1) 自回归外推,比 rolling 更进一步(捕捉残差自相关)。

### 3.4 状态分组 + vbt basket 回测 + IC

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_state_backtest.py --limit 50 --use-vbt
# 跑前 50 只,每只算 state 分布 → dominant_state 分组 → basket 回测 + IC

PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_state_backtest.py \
    --input data/my_stocks.csv --target-states resonance,against,independent,follow
# 自定义目标状态

PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_state_backtest.py --min-prop 0.10
# 只纳入 state 占比 ≥ 10% 的股票
```

**输出**:
- `data/dynamics/state_distribution.csv` — 每只票 7 状态占比 + dominant_state
- `data/dynamics/backtest_per_state.csv` — 每组 basket 总收益 / Sharpe / MaxDD
- `data/dynamics/state_ic.csv` — 每状态 IC(Spearman,p-value)

**已知现象**:在大盘震荡期(2024–2026 这段数据),state_prop 与 240 日 forward return 的
跨截面 IC ≈ 0(state 分布对该窗口 forward return 几乎无预测力)。这说明模型更适合做**短期
N 步轨迹**(动力学层),而不是长期截面预测(α 选股层)。

## 4. Python API

```python
import sys
sys.path.insert(0, 'backtrace')
from dynamics import (
    # 描述层(来自 projection._projection_core)
    compute_dynamics, classify_states, build_dynamics_df,
    compute_forces, build_forces_df,
    STATE_LABELS, STATE_COLORS, STATE_LABELS_CN,
    # 新增
    predict_next_state, simulate_trajectory, build_simulation_df,
)
from projection._projection_core import (
    load_pair, compute_movement_projection,
)
from common import tsfresh_pipeline as P

# 1. 加载 + 运动投影
loaded = load_pair('002475.SZ', 240, P)
mv = compute_movement_projection(loaded['stock_df'], loaded['index_df'])
dyn = compute_dynamics(mv, lambda_q=None)
states = classify_states(dyn['R'], dyn['theta'], dyn['E_self'],
                          (0.10, 0.50, np.deg2rad(30), np.deg2rad(90)))

# 2. 1 步预测(2026-08-17 v3 — u / a_M 内部派生,防 caller 飘移)
v_S_now = mv['stock_move'][-1]               # (2,)
v_M_now = mv['index_move'][-1]               # (2,)
v_M_next = mv['index_move'][-2] if len(mv['index_move']) >= 2 else v_M_now  # 下一日(若可取)
beta_now = mv['proj_coeff'][-1]
beta_next = mv['proj_coeff'][-2] if len(mv['proj_coeff']) >= 2 else beta_now
# d_now 重建(同 dynamics_system.py)
u_full = mv['stock_move'] - mv['proj_coeff'][:, None] * mv['index_move']
d_full = np.zeros_like(mv['stock_move'])
d_full[1:] = np.cumsum(u_full[:-1], axis=0)
# q_t 锚定强度:从 description 层拿
q_now = float(dyn['q_t'][-1])
# v3:caller 不再传 a_M_now / u_now(内部从 v_M_now/v_M_next/β 派生);
# 也不再传 a_S_now(F_self_now 由外部给定,默认 None = 0)
a_pred, v_pred, d_pred, u_pred = predict_next_state(
    v_S_now=v_S_now,
    v_M_now=v_M_now,
    v_M_next=v_M_next,
    beta_now=beta_now,
    beta_next=beta_next,
    d_now=d_full[-1],
    F_self_now=np.zeros(2),     # (2,) 残差;None = 0
    k=0.0, c=0.0, q_now=q_now,
)
# 1 步预测:下日个股 ΔS = v_pred;d_pred/u_pred 是下一状态
# 注:v3 返回 4 元组 (a_pred, v_pred, d_pred, u_pred);只看前 2 个也兼容旧 caller

# 3. N 步模拟(Oracle 模式:已知未来大盘;2026-08-17 v3 — u_init 删除)
N = 5
v_S_init = mv['stock_move'][-1]
# v_M_seq / β_seq 用 N+1 个状态(v_M_seq[t+1]-v_M_seq[t] 产生 N 个 a_M,无 NaN)
v_M_seq = mv['index_move'][-(N + 1):]        # (N+1, 2)
beta_seq = mv['proj_coeff'][-(N + 1):]       # (N+1,)
F_self_seq = np.tile(np.array([0.0, 0.0]), (N, 1))  # 残差序列是步长量 (N, 2)
sim = simulate_trajectory(
    v_S_init=v_S_init,
    v_M_seq=v_M_seq, beta_seq=beta_seq, F_self_seq=F_self_seq,
    d_init=d_full[-1],                  # d[t+1]=d[t]+u[t] 的自然初值
    # v3:u_init 删除(派生量,simulate_trajectory 在 t=0 自动派生)
    k=0.0, c=0.0,
)
sim_df = build_simulation_df(sim, dates=None, index_tag='399001', stock_tag='002475')

# 4. N 步模拟(Forecast 模式:无未来大盘,用预测器生成)
from dynamics import (
    forecast_v_M_random_walk, forecast_beta_rolling_mean,
    forecast_q_t_constant, make_rolling_mean_f_self_predictor,
)
# v_M 用随机游走(噪声 std 估自历史 diff);返 (N+1, 2)
v_M_last = mv['index_move'][-1]
diff_std = float(np.nanstd(np.diff(mv['index_move'], axis=0), axis=0).mean())
v_M_seq = forecast_v_M_random_walk(v_M_last, N, sigma_per_step=diff_std, random_state=42)
# β 用末日滚动均值;返 (N+1,)
beta_seq = forecast_beta_rolling_mean(mv['proj_coeff'], N, window=10)
# q_t 用末日观测(没有未来 ‖ΔM‖,沿用);q_t 是步长量,返 (N,)
q_t_seq = forecast_q_t_constant(float(dyn['q_t'][-1]), N)
# F_self 用滚动均值预测器(避免末日瞬时值过拟合)
F_self_pred = make_rolling_mean_f_self_predictor(F_self_full, window=10)
sim = simulate_trajectory(
    v_S_init=v_S_init, v_M_seq=v_M_seq, beta_seq=beta_seq,
    F_self_predictor=F_self_pred,
    d_init=d_full[-1],
    # v3:u_init 删除(派生量)
    k=0.0, c=0.0, q_t_seq=q_t_seq,
)
# 注意:sim['F_self_predictor_used'] 会回放这个 predictor
```

## 5. 关键参数建议

| 参数 | 含义 | 默认 | 建议 |
|---|---|---|---|
| `--days` | 描述层回看天数 | 240 | ≥ horizon + 50(模拟起点需足够历史估 d / u) |
| `--horizon` | 模拟步数 N | 5 | 5-20;超过 20 残差外推失效严重 |
| `--lambda-q` | 锚定强度系数 | median(‖ΔM‖) 自适应 | 设 0 = 无阻尼(朴素投影);设大 = 大盘运动被压扁 |
| `--k-restore` | 恢复系数 k | 0.0 | 0.1-1.0 弱恢复;≥ 5 强恢复(立即拉回 d=0) |
| `--c-damp` | 阻尼系数 c | 0.0 | 0.05-0.2 弱阻尼;≥ 1 强阻尼(速度差瞬间消除) |
| `--classify-thresholds` | R_low, R_high, θ_following°, θ_against° | 0.10, 0.50, 30, 90 | 改 θ_against=120 收紧逆势判定 |

## 6. 已知坑

| 现象 | 原因 | 解决 |
|---|---|---|
| `R = 1.0` 全程 | **2026-08-17 时间轴重构后已修复**:改用沿 v_M(t) 方向的 Gram-Schmidt 真正交投影,R 严格 ∈ [0, 1] 不需 clip | n/a |
| `E_market = 0` | 同上 | n/a |
| 模拟 F_self 巨大 | F_self 末日残差本身就大(变 β + 原始量纲) | 用 4-D lag 向量 / 走归一化空间 |
| `state=none` 频繁 | 前 2 步斜率不够;或 θ NaN(末行) | 设 `--horizon ≥ 3` |
| 批量速度慢(单只 ≈ 0.3s) | `compute_movement_projection` 在描述层 + 模拟层各跑一次 | 后续可缓存 mv dict |
| **predict_next_state 现在返 2 元组不是 3** | delta_S_pred 已被删除(冗余 ≡ v_pred) | 旧 caller 改成 `a, v = predict_next_state(...)` |
| **d_init 必须 = `d_full[-1]`(2026-08-17 v2)** | `simulate_trajectory` 改用 N+1 个 v_M/β 状态,让所有 N 个 a_M 都有效;d 递推用 spec 的 `d[t+1]=d[t]+u[t]`,d_init 取自然初始值 `d_full[-1]` | 已自动在 batch/system 脚本里改好 |
| **v_M_seq / beta_seq 长度 N+1(2026-08-17 v2)** | 大盘未来 N 天 → 个股未来 N 天,需要 N 个 a_M;v_M/β 是状态量,故长度 N+1 | caller 切片用 `[-(N+1):]`;旧 (N,) 切片会报错 |
| **sim['state'] 末项从 'none' 变真状态(2026-08-17 v2)** | t=N 现在有有效 v_S/v_M,R/theta/E 全有限,状态分类覆盖末行 | state 分布计数会偏移(7 状态而非 7+none);manifest 已用 `[:horizon]` 切片不受影响 |

## 6.5 时间轴重构(2026-08-17)

为消除以下 7 处一致性问题:

1. `predict_next_state` 与 `simulate_trajectory` 不共享同一动力学方程 → 现统一为 `a = q·β·a_M - k·d - c·u + F_self`
2. `predict_next_state` 不应用 q_t → 现新增 `q_now` 参数(默认 1.0,向后兼容)
3. `predict_next_state` 多返冗余 `delta_S_pred` → 现返 2 元组
4. `simulate_trajectory` 的 `E_market` 不是真正交 → 改用沿 v_M(t) 的 Gram-Schmidt 投影
5. `d_seq` 递推差一(实现 `d[t+1]=d[t]+u[t+1]`、spec `d[t+1]=d[t]+u[t]`)→ 改回 spec 写法
6. `F_restore[0]` 的 `hasattr(k * d_init, '__len__')` 死分支 → 改用 `np.linalg.norm(k * d_init)`
7. `forecast_v_M_random_walk` docstring 噪声索引不准 → 改 `noise[t-1]`

## 6.5 F_self 自回归预测器(2026-08-17)

新增 `make_ar1_f_self_predictor(F_self_history, min_history=10)` — 把"末日残差复制 N 次"升级到"AR(1) 自回归外推"。

**模型**(per-dim):
```
F_self_d(t+1) = μ_d + ρ_d · (F_self_d(t) - μ_d)
⇒ 闭式:    F_self_d(t) = μ_d + ρ_d^t · (F_self_d(0) - μ_d)
```

**ρ/μ 估计**(OLS on history):
```
μ_d = mean(F_self_d[:])
ρ_d = Σ (F_d[t]-μ_d)(F_d[t-1]-μ_d) / Σ (F_d[t-1]-μ_d)²
```
ρ 截断到 [-1, 1] 避免数值发散。

**退化路径**:
- 无有效样本 → 零预测器
- 有效样本 < `min_history` → 常数预测器(用 μ)
- ρ 估不出来(分母 ≈ 0) → ρ = 0,等同于常数预测器

**实证**:在 000001.SZ 残差序列上,`ρ ≈ -0.07 / +0.005`(基本无自相关)— AR(1) 退化为均值常数预测。这与"动力学方程残差近似白噪声"的假设一致,意味着动力学模型已经吸收了残差里大部分系统结构。

**API**:与现有 `predictor(t, hist=None) -> (2,)` 完全兼容,直接替换 rolling/constant 预测器即可。

**CLI**:`--f-self-mode ar1`(`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`),`--f-self-window N` 控制 `min_history`。

## 6.6 时间轴彻底重构(2026-08-17 v2)

为消除"`simulate_trajectory` 主循环最末步无 a_M"的隐藏状态污染:

**问题**:上一版(6.5)`v_M_seq.shape == (N, 2)`,`a_M_seq[t] = v_M_seq[t+1] - v_M_seq[t]` 只产生 N-1 个有效加速度;末步 `a_M_seq[N-1] = NaN`,代码特判"末步 a_pred = -k·d - c·u + F_self",**第 N 步完全没有市场驱动力 `q·β·a_M`**。名义上 N 步模拟,实际只有 N-1 步真正遵循动力方程。

**修复**(4 处):

1. `v_M_seq` shape `(N, 2)` → `(N+1, 2)`(t=0..N 共 N+1 个状态)
2. `beta_seq` shape `(N,)` → `(N+1,)`(β 是状态量,需覆盖 t=0..N)
3. `forecast_*` helper 全部按 `n_steps+1` 个状态生成(`forecast_q_t_constant` 除外 — q_t 是步长量)
4. `d_init = d_full[-1]`(自然初值,无 `u` 补偿)— 因为 `d(1) = d(0) + u(0) = d_full[-1] + u_full[-1]`,这是最直观的物理定义

**主循环不再有 NaN 跳步**:每天都严格遵循同一方程 `a(t) = q(t)·β(t)·a_M(t) - k·d(t) - c·u(t) + F_self(t)`。

**CSV schema 不变**(19 列);但 `sim['state']` 末项从 `'none'` 变成真状态,`sim['v_M_seq_used']` 从 (N, 2) 变成 (N+1, 2)(末行不再是 NaN pad)。**数值再次偏移**(API 破坏性变更)— 旧 simulation_*.csv 与新结果不会逐行相等。

`compute_dynamics` / `compute_forces`(description 层)保留原 `v_proj = q·β·v_M` 投影不动 —
那个 β 是回归斜率,语义独立,与 simulate 层的"严格正交"是两件事。

## 6.7 派生量统一(2026-08-17 v3)

为消除"`u` 作为独立输入但其实是代数约束"和"`predict_next_state` 让 caller 重复造轮子"的两处 API 表面与技术债:

### 6.7.1 问题

```python
# simulate_trajectory 旧签名:caller 必须计算 u_init
u_init = u_full[-1]                                  # day T-1
sim = simulate_trajectory(..., u_init=u_init, ...)
# 但 u_seq[1] 立刻被覆盖:
u_seq[1] = v_seq[1] - beta_seq[1] * v_M_seq[1]      # 派生值
# u_init 在 t=1 之后永远不再被使用
```

```python
# predict_next_state 旧签名:caller 同时持有 v_M_now / v_M_next 才能算 a_M_now
a_M_now = v_M_next - v_M_now                        # caller 算
u_now = v_S_now - beta_now * v_M_now                 # caller 算
a_pred, v_pred = predict_next_state(v_S_now, a_M_now, beta_now, d_now, u_now, ...)
# 传错任何一个就飘移
```

`u` 由代数约束 `u(t) = v_S(t) − β(t)·v_M(t)` 唯一决定 — 系统真正的状态只有 `X(t) = (d(t), v_S(t)) ∈ R⁴`,`u` 是导出量。caller 传 `u_init` 是**误导性的 footgun**(内部立即覆盖,传错没人发现)。

### 6.7.2 修复

**(1)** `predict_next_state` 新签名:

```python
# v3:caller 只传 v_M_now / v_M_next / β_now / β_next,内部派生 u_now / a_M_now
a_pred, v_pred, d_pred, u_pred = predict_next_state(
    v_S_now, v_M_now, v_M_next, beta_now, beta_next, d_now,
    F_self_now=None,         # None = 0;旧版 a_S_now 参数删除
    k=0.0, c=0.0, q_now=1.0,
)
# 内部:
#   u_now   = v_S_now - beta_now * v_M_now          # 代数约束
#   a_M_now = v_M_next - v_M_now                    # 前向差
#   a_S     = q_now * beta_now * a_M_now - k * d_now - c * u_now + F_self_now
#   v_pred  = v_S_now + a_S
#   u_pred  = v_pred - beta_next * v_M_next         # 代数约束
#   d_pred  = d_now + u_now                         # spec 写法
```

**(2)** `simulate_trajectory` 删除 `u_init` 参数:

```python
# v3:不再传 u_init
sim = simulate_trajectory(
    v_S_init=v_S_init, d_init=d_init,
    v_M_seq=v_M_seq, beta_seq=beta_seq,
    k=0.0, c=0.0, q_t_seq=q_t_seq,
    F_self_seq=F_self_seq,    # 或 F_self_predictor
)
# 内部在 t=0 自动派生:
#   u_seq[0] = v_S_init - beta_seq[0] * v_M_seq[0]
```

### 6.7.3 影响

| 项 | 行为 |
|---|---|
| 旧 caller 传 `u_init=...` | `TypeError: unexpected keyword argument 'u_init'` |
| 旧 caller 用 `a_M_now=...` / `u_now=...` kwarg | `TypeError` |
| 旧 2 元组解构 `a, v = predict_next_state(...)` | **仍兼容**(新 4 元组的前 2 元素不变) |
| 模拟数值 | u[0] 变化(v2 用 u_init=v_full[-1] 是 day T-1,v3 用 β[0]·v_M[0] 是 day T-1-N)。其后 v/u/d 全部按新 u[0] 重新递推,与 v2 数值**略有偏移**。API 破坏性,数值在合理范围内。 |
| `dynamics_1step_oos.py` 的 1 步预测 | **零数值变化**(旧 API 内部用的就是同一组方程) |
| `dynamics_1step_oos.py` 的返回值解构 | 改成 `a_pred, v_pred, _d_pred, _u_pred = ...` |

### 6.7.4 状态空间的明确化

v3 后,系统的状态空间变得**显式且不可误用**:

```
真状态 X(t) = (d(t), v_S(t)) ∈ R⁴
派生量 u(t) = v_S(t) − β(t)·v_M(t)       ← 代数约束(不递推)
外部输入 v_M(t), β(t), q(t), F_self(t)
市场加速度 a_M(t) = v_M(t+1) − v_M(t)
动力学 a(t) = q(t)·β(t)·a_M(t) − k·d(t) − c·u(t) + F_self(t)
递推 v_S(t+1) = v_S(t) + a(t)
     u(t+1)   = v_S(t+1) − β(t+1)·v_M(t+1)        ← 派生
     d(t+1)   = d(t) + u(t)                       ← spec 写法
```

## 6.8 特征值稳定性分析(2026-08-17 v4)

把 2D 离散线性系统 `(d, u)` 写成标准状态转移形式:

```
d(t+1) = d(t) + u(t)               ⇒   X(t+1) = A · X(t),X = (d, u)ᵀ
u(t+1) = (1 − c) · u(t) − k · d(t)

            | 1     1   |
      A  =  |          |        trace = 2 − c,det = 1 − c + k
            |−k  1 − c |
```

注:`simulate_trajectory` 里的 `v_S(t+1) = v_S(t) + a(t)` 已经隐含了这个 2D 转移
(把 `q·β·a_M` 和 `F_self` 当作外部驱动),`A` 矩阵只依赖 `(k, c)` — 是 LTI 线性时不变系统。

### 6.8.1 `analyze_eigenvalues(k, c)` — 11 类稳定性分类

```python
from dynamics import analyze_eigenvalues
eig = analyze_eigenvalues(k=0.145, c=1.112)
# {'k': 0.145, 'c': 1.112,
#  'A': [[1, 1], [-0.145, -0.112]],
#  'eigenvalues': [0.849+0j, 0.849+0j],   # 实重根
#  'spectral_radius': 0.849, 'trace': 0.888, 'determinant': 0.033,
#  'discriminant': 1.236 - 0.580 = 0.656 > 0,
#  'mode': 'real', 'stability': 'stable', 'classification': 'stable_overdamped',
#  'schur_stable': True, 'in_wedge': True, 'distance_to_unit_circle': 0.151, ...}
```

### 6.8.2 11 类分类法(v4.1 ρ-primary)

v4.1 起分类逻辑**以谱半径 ρ 作主信号**,边界判定仅作辅助(避免把"在楔形
边界外但 ρ≈1"误判成 critical):

| 分类 | 触发条件 | 物理含义 |
|---|---|---|
| `stable_oscillatory` | ρ<1, D<0 | 共振稳定(衰减振荡) |
| `stable_overdamped` | ρ<1, D>0 | 过阻尼稳定(单调回归) |
| `stable_critical_damping` | ρ<1, D≈0 | 临界阻尼(最快的非振荡回归) |
| `oscillatory_divergent` | ρ>1, D<0, k≥0 | 振幅发散(共振本质 — 危险!) |
| `monotonic_divergent` | ρ>1, D≥0, k≥0 | 单调发散 |
| `anti_restoring` | k<0 | 反回复力(趋势强化,必发散) |
| `critical_periodic` | ρ≈1, D<0, 无 λ=±1 | 周期-N 振荡边界 |
| `critical_period2` | ρ≈1, ∃ λ≈-1 | 周期-2 边界(隔日反向) |
| `critical_real_unit` | ρ≈1, 有 λ=+1(或实根兜底) | 实根单位圆边界 |
| `marginal_const` | k≈0, c>0 | 边界常数模(纯阻尼,无恢复) |
| `jordan_drift` | k≈0, c≈0 | Jordan 漂移(`x(t) ~ t·x(0)` 多项式漂移) |

### 6.8.3 v4.1 关键边界修正

**之前误判**:把整条 `c = 2 + k/2` 都标 `critical_period2`,把整条 `c = k` 都按
ρ=1 标 `critical_periodic` / `critical_real_unit`。这是错的——这些边界只是
Jury 准则的几何位置,但**不等于** ρ=1。

**反例**:
- `c = 2 + k/2` 当 k>4:λ₁=-1,λ₂=1-k/2,**|λ₂|>1**(实际发散,不是临界)
- `c = k` 当 k>4:实根,|λ₁| = k-√(k²-4) > 1(实际发散,不是临界)

新逻辑用 ρ 作主信号:
- ρ<1 ⇒ schur_stable(自动覆盖楔形内)
- ρ>1 ⇒ unstable,再按 k 符号 / D 符号细分
- ρ≈1 ⇒ critical,按"是否存在 λ=±1"细分

**测试覆盖**:`tests/test_dynamics_eigen.py::test_boundary_fix_*` 锁住这两个
反例作为回归测试。

### 6.8.4 Schur 楔形(完整稳定性条件)

只有 `c > k` 是**不完整**的。完整 Schur 稳定性(2nd-order Jury 准则):
- `1 + a₁ + a₀ > 0`:⇒  `k > 0`
- `1 − a₁ + a₀ > 0`:⇒  `c < 2 + k/2`
- `|a₀| < 1`:⇒  `|1 − c + k| < 1` ⇒  `0 < c − k < 2`

合起来:`k > 0` 且 `k < c < 2 + k/2`(**楔形区域**),这是 2D 离散线性系统
ρ<1 的**充分条件**(楔形内 ⇒ ρ<1,但楔形外不一定 ρ>1,见 6.8.3 反例)。

### 6.8.5 楔形距离几何(v4.2)

为度量"一只股票距离稳定边界多远",定义三个距离量:

```
distance_lower_boundary  = c - k                # > 0 表示在 c=k 上方
distance_upper_boundary  = 2 + k/2 - c          # > 0 表示在 c=2+k/2 下方
distance_to_wedge        = min(k, c-k, 2+k/2-c) # > 0 楔形内,=0 边界,<0 楔形外
```

`distance_to_wedge` 是一个**有符号**度量:正值越大越稳定,负值越大越发散。
比单纯的 `ρ` 更有解释力——`ρ=0.98` 可以是"接近稳定边界"也可以是"远离稳定边界
但本身模态慢衰减",后者其实更安全。楔形距离把"接近边界"和"远在内部"区分开。

### 6.8.6 完整 R⁴ 状态空间(v4.1 澄清)

本目录的 2×2 `A = [[1, 1], [-k, 1-c]]` 是"单个 Vol 或 Amt 分量"对应的核心
动力矩阵。完整的 2-D 状态(Vol, Amt)各自的偏离和速度构成 4-D 状态:

```
X(t) = [d_Vol, d_Amt, u_Vol, u_Amt]^T ∈ R⁴
X(t+1) = A_4x4 · X(t) + 外部驱动
A_4x4 = [[I_2,    I_2  ],
         [-kI_2, (1-c)I_2]]
```

4×4 A 的特征值集合 = 2×2 A 的特征值各重复一次:`{λ₁, λ₁, λ₂, λ₂}`,**谱半径相同**。
所以"2×2 A 的 ρ"等价于"4×4 系统的 ρ"——用 2×2 矩阵做分析数学上完全成立。

### 6.8.7 Gram-Schmidt 能量恒等式(v4.1 自检)

`simulate_trajectory` 用沿 v_M(t) 方向的严格 Gram-Schmidt 投影分解
`v_S = v_∥ + v_⊥`,因此严格满足:

```
E_total(t) = E_market(t) + E_self(t)
           = 0.5·‖v_∥(t)‖² + 0.5·‖v_⊥(t)‖²
           ≡ 0.5·‖v_S(t)‖²
```

`simulate_trajectory` 返 `sim['energy_error'] = E_total - E_market - E_self`
作为数值健康检查;数值误差 ~ 1e-15(纯浮点精度)。若 > 1e-10 表明正交分解实现
有 bug,应触发警告。

测试:`tests/test_dynamics_eigen.py::test_simulate_trajectory_energy_identity`
断言 `nanmax(|energy_error|) < 1e-10`。

### 6.8.4 经验分布(2026-08-17 kc_estimates 4 票)

| code | k̂ | ĉ | λ₁ | λ₂ | ρ | 分类 |
|---|---|---|---|---|---|---|
| 000001.SZ | 0.145 | 1.112 | 0.849 | 0.849 | 0.849 | stable_overdamped |
| 000002.SZ | −0.044 | 8.292 | 7.30+0j | 0.99+0j | **7.30** | anti_restoring |
| 000006.SZ | −0.060 | 3.978 | 2.99+0j | 0.99+0j | **2.99** | anti_restoring |
| 000007.SZ | 0.111 | 2.244 | 1.194+0j | 0.05+0j | **1.194** | monotonic_divergent |

**Schur 稳定率:25%(1/4)**;**ρ>1(发散):75%**;ρ 中位数 ≈ 1.4;ρ 均值 ≈ 3.0。

**解读**:A 股个股系统的"2D 投影稳定性"在经验上偏弱 — 多数股票要么 k<0(趋势强化),
要么 c 过大(过度阻尼 = 速度差瞬变)。这与"个股相对大盘的偏离会被市场拉回"的
朴素直觉**不一致** — OLS 在短窗口(240 日)估出的 (k̂, ĉ) 反映的是**统计共变**而非
**因果回拉力**(详见 projection/state_kc_analysis.py 的状态关联分析)。

### 6.8.5 `dynamics_eigen_analysis.py` — 批量经验分析

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_eigen_analysis.py
# 读 data/projection/kc_estimates.csv,产 eigen_summary.csv + HTML

PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_eigen_analysis.py --limit 500
# 取前 500 只

PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_eigen_analysis.py \
    --input data/my_kc.csv --output backtrace/outputs/my_eigen.html
# 自定义输入输出
```

**输出**:
- `data/dynamics/eigen_summary.csv` — 每只票 18 列(基础 14 + v4.2 楔形距离 3 + 其它)
- `backtrace/outputs/dynsys_eigen.html` — **6 子图 plotly**(v4.2 起):
  1. **(k̂, ĉ) 散点 + 楔形背景**(颜色 = 11 类分类)
  2. **ρ 直方图 + ρ=1 红虚线**
  3. **11 类分类柱状图**(颜色按 CLASS_COLORS)
  4. **(k̂, ĉ) 散点按楔形距离上色**(RdYlGn 红→黄→绿)
  5. **楔形距离分布直方图**(0 = 边界)
  6. **ρ vs 楔形距离**(越靠右越稳,越靠下越稳)

### 6.8.6 与 `simulate_trajectory` 集成(v4 增量)

`simulate_trajectory` 现在在返回 dict 里**直接附带** `analyze_eigenvalues(k, c)` 结果:

```python
sim = simulate_trajectory(...)
print(sim['spectral_radius'], sim['classification'])
# 0.849 stable_overdamped

sim['schur_stable']  # True
sim['in_wedge']      # True
```

`build_simulation_df` 在 simulation CSV 里加 10 列(整个 trajectory **常量**,因 k/c 在 sim 内固定):

| 列 | 含义 |
|---|---|
| `Sim_Lambda1_Real` | λ₁ 实部 |
| `Sim_Lambda1_Imag` | λ₁ 虚部 |
| `Sim_Lambda2_Real` | λ₂ 实部 |
| `Sim_Lambda2_Imag` | λ₂ 虚部 |
| `Sim_SpectralRadius` | ρ(A) = max(|λ|) |
| `Sim_DynamicClass` | 11 类稳定性分类之一 |
| `Sim_DistanceLowerBoundary` | c - k(到下边界距离) |
| `Sim_DistanceUpperBoundary` | 2+k/2 - c(到上边界距离) |
| `Sim_DistanceToWedge` | min(k, c-k, 2+k/2-c)(楔形距离,有符号) |
| `Sim_EnergyError` | E_total - E_market - E_self(数值健康检查)|

零数值开销(`analyze_eigenvalues` 只是 2×2 矩阵特征值),但把"模型参数稳定性"和"轨迹形状"绑在一张表里 — 后续 IC / basket 回测可以直接按 `Sim_DynamicClass` 分组,看"稳定 vs 发散股票的未来收益是否真有差异"。

## 7. 与其他目录的关系

- **`backtrace/projection/`** — 数学源头。本目录 `from projection._projection_core import ...`。
  - `_projection_core.py` — 描述层 + 力模型(state classification 在此)
  - `prediction_ode.py` — OOS 1 步评估(给定 (k̂, ĉ) 算命中率/RMSE);本目录做"已知 k/c + 已知未来 → N 步"
  - `parameter_fit.py` — 估 k̂/ĉ;本目录消费其结果(`--k-from-fit` / `--c-from-fit`)
- **`backtrace/common/tsfresh_pipeline.py`** — 数据加载(load_ohlcva)
- **`backtrace/outputs/`** — HTML 落点
- **`data/dynamics/`** — CSV 落点(与 `data/projection/` 隔离)
- **`docs/superpowers/specs/`**:
  - `2026-08-16-market-stock-dynamics-design.md` — 描述层 spec(本目录的上游)
  - `2026-08-16-dynamics-system-design.md` — 本目录 spec(N 步模拟增量)

### 3.5 全市场经验分布 (v4.3, 2026-08-17)

把 `kc_estimates.csv` 从 4 只 smoke-test 扩张到全 A 股 (~5000 只),回答经验问题:
"动力系统参数 (k̂, ĉ) 在全市场到底呈什么分布?"

**跑全市场**:

```bash
# 1. (前置) 生成全 A 股 movement 文件 (~20-40 分钟)
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_batch.py --movement \
    --input data/stock_basic.csv --limit 0 --days 240

# 2. 跑全 A 股 parameter_fit (~20-40 分钟)
PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --limit 0

# 3. 跑 v4.3 报告
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_eigen_analysis.py
```

**数据源**:

| 文件 | 用途 |
|---|---|
| `data/projection/kc_estimates.csv` | 主输入(全 A 股 ~5000 只 (k̂, ĉ)) |
| `data/stock_basic.csv` | 反查 `exchange`(`market` 字段 = SH/SZ/BJ) |
| `data/sw2/members.csv` | 反查 `industry_l1`(sector_code) / `industry_l2`(sector_name) |

**输出**:

| 路径 | 内容 |
|---|---|
| `data/dynamics/eigen_summary.csv` | 21 列(18 + industry_l1/l2/exchange) |
| `backtrace/outputs/dynsys_eigen.html` | **2×4 网格 8 子图 plotly** (~2-4 MB) |
| `backtrace/outputs/dynsys_eigen_summary.txt` | 纯文本汇总(便于 CI/grep) |
| `data/dynamics/v43_eigen_top_industries.csv` | 行业聚合表(申万二级) |
| `data/dynamics/v43_eigen_by_exchange.csv` | 交易所聚合表 |

**HTML 8 子图布局**:

| (行, 列) | 子图 |
|---|---|
| (1, 1) | (k̂, ĉ) 散点 + 楔形(分类着色) |
| (1, 2) | ρ 直方图 + ρ=1 红虚线 |
| (1, 3) | 11 类分类柱状 |
| **(1, 4)** | **行业 ρ 中位数 top10**(误差棒 [p25, p75]) |
| (2, 1) | (k̂, ĉ) 散点(楔形距离着色) |
| (2, 2) | 楔形距离直方图 |
| (2, 3) | ρ vs 楔形距离 |
| **(2, 4)** | **交易所 ρ 中位数对比(SH vs SZ vs BJ)** |

**关键决策**:

- 聚合用 **median** 而非 mean(ρ 分布偏态,mean 被极端值拉飞)
- 行业筛选 `n_stocks >= 50` 硬阈值取 top 10,不足则降级 `n >= 30`
- 行业用申万二级 sector_name(sw2/members.csv),非申万一级
- 交易所从 stock_basic.csv 的 market 字段读取(SH/SZ/BJ)
- 文本汇总是 UTF-8 中文,Windows `PYTHONIOENCODING=utf-8` 下能直接 `cat`

**v4.3 显式不做**(留 v4.4 / v4.5 / v5):

- G(ω) 频率响应函数(独立 v5 工作包)
- 行业稳定性指数 SI(v4.4+)
- (k, c) 相图 + 7 状态颜色叠加(v4.4)
- 状态转移矩阵(v4.5)
- 任何 IC / basket / 交易信号(明确不做)

**测试**: `tests/test_dynamics_eigen.py` 加 3 个测试,总 **26 passed**(23 旧 + 3 新)。

**v4.4 (2026-08-17)**: (1,4) bar chart x-axis 也升级到 `电力(881459.SH)` 格式(文本汇总原已支持),通过抽出 `_industry_name_lookup` helper 复用 lookup。

### 3.6 v4.5 phase plot (2026-08-17)

新增独立 HTML `dynsys_eigen_phase.html`,画 (k̂, ĉ) 散点 + 11 类离散着色 + 楔形稳定区边界 overlay(浅绿背景 + 3 段虚线)。
启用:`--phase-plot` 标志(默认 off,不影响 v4.3 2x4 行为)。

11 类分类:`CLASS_COLORS` 字典定义(11 种颜色 + 11 中文标签)。
楔形 boundary 由 `wedge_boundary_polygon(k_max, n)` helper 提供。

用法:
```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_eigen_analysis.py --limit 50 --phase-plot
# 产出 backtrace/outputs/dynsys_eigen_phase.html
```

输出示例(实证发现,N=4972):
- 大多数票落在 `monotonic_divergent` (橙) + `anti_restoring` (棕) + `stable_overdamped` (蓝) 三大区域
- `jordan_drift` / `marginal_const` 在 N=4972 样本中 0 只 → 该 trace 跳过(代码已处理)

### 3.7 v4.7 — 行业稳定性指数 SI (Sector Stability Index)

`dynamics_eigen_analysis.py` 默认运行后追加产出:
- `data/dynamics/sector_si.csv` (9 列)
- `backtrace/outputs/dynsys_sector_si.html` (4 子图 plotly)
- `backtrace/outputs/dynsys_sector_si_summary.txt` (UTF-8 中文 top 12 强/弱)

SI 定义: `SI = 0.5·ρ_health + 0.2·damping_health + 0.3·wedge_health`,权重集中在 `SI_WEIGHTS = (0.5, 0.2, 0.3)`(`dynamics_eigen_analysis.py` 顶部常量)。

行业筛选(沿用 v4.3): `n_stocks >= 50` 强 / `n_stocks >= 30` 弱;两者都 < 5 行业时上层标 `low-confidence`。

### 3.8 v4.8 — SI × Forward Return Rolling IC

新 CLI: `backtrace/dynamics/dynamics_si_ic.py`,评估 v4.7 SI 指数与行业 forward 20d/60d 收益的滚动 cross-sectional Spearman IC。

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_si_ic.py
# 默认: window=60 日, step=20 日, horizons=20,60
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_si_ic.py --window 30 --step 10
# 短窗口实验
```

**输出**:
- `data/dynamics/si_ic_summary.csv` — 跨期汇总 2 行(20d / 60d),列: `horizon, ic_mean, ic_std, ic_ir, p_value_mean, n_windows`
- `data/dynamics/si_ic_timeseries.csv` — per-window detail
- `backtrace/outputs/dynsys_si_ic.html` — 1 HTML 3 子图(rolling IC 时序 + 行业 SI vs 累计 forward 60d 散点 + 跨期统计表)
- `backtrace/outputs/dynsys_si_ic_summary.txt` — UTF-8 文本汇总

**forward return 估算**: 行业成员股票中位数收盘价(`P_median`),forward h 日收益 = `(P_median(t+h) - P_median(t)) / P_median(t)`。

**每窗口 IC**: 60 日窗口内逐日 Spearman IC 的算术平均(避免 pool 重算造成大 N 日重复计权重)。`n_industries < 5` 跳过该日。

**已知陷阱**: 
- 行业 member 数 < 3 的跳过该行业
- v4.7 `sector_si.csv` 是单值(整期恒定),本 IC 反映"行业长期稳不稳 vs 该日 forward 收益"的相关性
- 若 IC ≈ 0(类似 v3 README §3.4 state_prop 现象),SI 是描述性而非预测性指标

### 3.9 v4.9 — SI 时序 + 漂移检测 (Sector Stability Timeseries + Drift)

v4.7 SI 单值答"哪些行业最稳",v4.8 IC ≈ 0 答"稳定对未来收益无预测力"。v4.9 把 SI 扩展到时序:
行业稳定性是否随时间漂移?漂移能否预警风险?

**数据流**:
1. `parameter_fit.py --rolling-time` (新增) — 每月末用最近 240 天 OLS 估 (k̂, ĉ)
   产出 `data/projection/kc_estimates_time.csv` (long format)
2. `compute_sector_stability_timeseries` (eigen_analysis 末尾追加) — 复用 v4.7 SI 公式,
   按 (asof_date, industry_l1) 聚合
3. `detect_si_drift` — rolling 60 日 z-score < -2 → drift event

**输出** (全 gitignored):
- `data/dynamics/sector_si_timeseries.csv` — 11 列 long format
- `data/dynamics/si_drift_events.csv` — drift event list
- `backtrace/outputs/dynsys_si_timeseries.html` — 4 子图 plotly
- `backtrace/outputs/dynsys_si_timeseries_summary.txt` — UTF-8 中文汇总

**漂移检测**: rolling window = 3 asof_dates (≈ 60 交易日 ≈ 3 个月)。
对每个行业 SI(t):
  rolling_mean = mean(SI over [t-60d, t))
  rolling_std = std(SI over [t-60d, t))
  z_score = (SI(t) - rolling_mean) / rolling_std
  drift event: z_score < -2.0

**CLI**:
```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_si_timeseries.py
# 默认: window=3, z_threshold=-2.0, ramp-up-min-n-valid=192
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_si_timeseries.py --window 6 --z-threshold -1.5
# 调参
```

**已知陷阱**:
- 月末 asof_date 列表依赖 daily data 完整性,数据 < 60 天则该 asof_date 跳过
- 行业 member 数 < 10 → SI 噪声大,n_stocks_threshold=50 沿用 v4.7
- drift event 是经验性信号,不是预测性 — v4.10 lagged IC 验证
- **ramp-up filter (reviewer finding #2)**: Task 1 `--rolling-time` 早期 asof_date 用 expanding
  window(非固定 240),(k̂, ĉ) 估计单调漂移易被误判为行业 SI 漂移。`--ramp-up-min-n-valid 192`
  (= 240 × 0.8) 会在 `main()` 入口剔除 `n_valid_days < 192` 的行,使其从 SI 时序消失,
  漂移检测不会被 ramp-up artifact 污染。`detect_si_drift` 还内置 0.01 noise floor + 二次防御层
  (若 si_ts 含 `n_valid_days_min` 列,跳过低于阈值的历史点)

### 3.10 v4.10 — 时序 SI 的 lagged IC 评估

v4.8 contemporaneous IC ≈ 0 揭示 SI 不是预测性指标。v4.10 闭环:用 lagged IC(时序 SI(t) vs future forward return)测"今日 SI 能否预测未来收益排名"。

**关键差异(vs v4.8)**:
- v4.8 contemporaneous:`Spearman(SI_i(t), r_i(t, h))` 同时点 SI vs forward return — 描述性
- v4.10 lagged:`Spearman(SI_i(t), r_i(t+h, h))` 不同时点 — 真正预测性测试

**数据流**:
1. 输入: `data/dynamics/sector_si_timeseries.csv` (v4.9 产出,11 列 long format)
2. 输入: `data/daily/<code>.csv` 算各行业 forward return(同 v4.8 中位数法)
3. lagged 对齐:对每个 eval_date t,取 SI 在 (t - horizon) 的排名,forward return 在 t 的排名
4. 跨截面 Spearman + 60 日 rolling window / 20 日 step
5. 输出 4 个文件(全 gitignored)

**输出**:
- `data/dynamics/si_lagged_ic_summary.csv` — 跨期汇总 2 horizons × 6 列
- `data/dynamics/si_lagged_ic_timeseries.csv` — per-window detail
- `backtrace/outputs/dynsys_si_lagged_ic.html` — 3 子图 plotly
  - (1,1) Lagged IC 时序 + IC=0 红虚线
  - (1,2) v4.10 lagged vs v4.8 contemporaneous 对比(若 v4.8 CSV 存在)
  - (2,1) IC 统计汇总表
- `backtrace/outputs/dynsys_si_lagged_ic_summary.txt` — UTF-8 中文汇总

**CLI**:
```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_si_lagged_ic.py
# 默认: window=60, step=20, horizons=20,60
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_si_lagged_ic.py --window 30 --step 10
# 短窗口实验
```

**已知陷阱**:
- SI 时序是月度 asof_date,horizon=20/60 日可能不对齐 — 用 `asof_date <= t - horizon` 的最新 SI
- 行业 member 数 < 5 → 该 eval_date 跳过
- v4.8 CSV 缺失时,对比子图 (1,2) 退化为 annotation 提示
- 截面行业数太少(< 10)时 Spearman 抽样噪声本身就有 E|IC| ≈ 0.3,别把它当信号
- 若 lagged IC ≈ 0 → 行业层**纯描述性**,SI 用于报告 / 风险标签,不作选股信号(确认 v4.8 结论)
- 若 lagged IC > 0.05 显著 → 行业 SI(t) 是预测性指标,v4.12 行业轮动策略有基础

### §4 v5 — 受迫系统 + G(ω) 频率响应

v4.7-v4.10 把 SI 当成被动的"稳定性指标"。v5 扩展到受迫:用 sinusoidal β(t) 主动驱动,测系统的复频响应 H(jω),把 SI 与频域行为耦合。

**核心公式**(离散 z 域):
```
H(jω) = V_M0·[k + c·(z-1)] / [(z-1)² + c·(z-1) + k]    其中 z = e^(jω)
```

(本目录实现忽略 V_M0 标量 — 它只放大常数倍,不影响 |H| 形状。)

**关键性质**:
- **DC gain**:H(j0) = 1(任意 k, c > 0)
- **共振**:Schur 楔形外(k > c)→ |H| 在 ω_n = arctan(√(4k-c²)/(2-c)) 处爆炸
- **滚降**:Schur 楔形内(k < c)→ |H| 单调滚降,无峰值
- **Schur wedge**:本线性化系统的稳定性边界为 k = c(不是 v4.7 的 c² = 4k,后者是不同 2×2 系统的边界)
- **抗阻尼**:k < 0 → 低频 |H| 爆炸,系统无界

**输出**:
- `data/dynamics/transfer_function_grid.csv` — ω × (|H|, arg H) 200 点
- `data/dynamics/transfer_function_stability.csv` — 60×60 (k, c) 网格
- `backtrace/outputs/dynsys_forced_response.html` — Bode 图 2 子图
  - (1,1) |H(jω)| 半对数 + ω_n 红虚线 + |H|=1 灰虚线
  - (1,2) arg H(jω) 度数 + -180° 红虚线
- `backtrace/outputs/dynsys_forced_response_stability.html` — 2D 热图
  - (1,1) |H(jω_n)| 颜色 + Schur 边界 c = k 黑虚线(本系统的稳定性边界)
- `backtrace/outputs/dynsys_forced_response_summary.txt` — UTF-8 中文汇总

**CLI**:
```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_forced_response.py
# 默认: k=2.0, c=1.5 (Schur 外,有共振)
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_forced_response.py --k 2.0 --c 4.0
# Schur 内:稳定,无共振
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_forced_response.py --k 4.0 --c 0.5
# 欠阻尼:共振爆炸
```

**与现有代码的关系**:
- 推导基于 `_dynamics_core.predict_next_state` (v3) 但**不调用**
- 与 `dynamics_1step_oos.py` 互补:时域 OOS vs 频域解析
- 与 v4.7 SI 计算互补:SI 给出 (k̂, ĉ) → 本 spec 给出该点的频率响应

**已知陷阱**:
- ω=0 时 `e^(jω)-1 = 0`,但分母 = k ≠ 0(无 DC 奇异);omega_grid 从 0.001 起避免数值噪声
- |H| 爆炸(超 1e10)时 log-scale 显示,数值上 `np.clip(..., 1e-12, None)`
- 离散 Bode ≠ 连续 Bode — Nyquist 在 ω=π(不是 ∞)
- 5 测试用合成 (k, c),非真实数据 — 真实 SI 频率响应是 v5.3 候选

### 与 v4.7-v4.10 的关系

| 版 | commit | 主题 | 数学层 |
|---|---|---|---|
| v4.7 | `c63e783` | SI 单一指标(描述性) | Schur 楔形 |
| v4.8 | `dbd367d` | SI × forward return IC ≈ 0(SI 不是预测性) | Spearman IC |
| v4.9 | `f2178a3` | SI 时序 + 漂移检测 | 时序 + z-score |
| v4.10 | `d002a0e` | 时序 SI 的 lagged IC 评估 | Lagged Spearman IC |
| **v5** | **(本版)** | **受迫系统 + G(ω) 频率响应** | **z 域 H(jω)** |

### §4.1 v5.1 — Industry G(ω) Frequency Response Comparison

**多对 (k, c) Bode plot 叠加对比**,回答业务问题"哪个行业对 β 强迫最敏感 / 哪个是低通过滤器 / 哪个危险"。

#### 新增 CLI flag

| flag | 类型 | 说明 |
|---|---|---|
| `--overlay` | str | 多对 (k, c) 字符串:`"k1,c1,label1; k2,c2,label2; ..."` |
| `--overlay-html` | path | overlay HTML 输出(默认 `backtrace/outputs/dynsys_bode_overlay.html`) |
| `--overlay-summary-txt` | path | overlay UTF-8 汇总(默认 `backtrace/outputs/dynsys_bode_overlay_summary.txt`) |

**触发条件**:传 `--overlay` 时进入 overlay-only 模式,跳过单对 main() 逻辑,**不写** 单对 `grid_csv` / `stability_csv` / `bode_html` / `heatmap_html` / `summary_txt`。不传 `--overlay` 时与 v5 单对模式完全一致(向后兼容)。

#### 端到端示例

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_forced_response.py \
    --overlay "0.5,2.0,Strong damping; 2.0,1.5,Mild damping; 2.01,2.0,Near boundary; 4.0,0.5,Weak damping"
# 期待:
  - backtrace/outputs/dynsys_bode_overlay.html (4 条曲线叠加 + 每对 ω_n 标记)
  - backtrace/outputs/dynsys_bode_overlay_summary.txt (4 对业务解读)
```

#### 输出(overlay-only 模式)

| 路径 | 内容 |
|---|---|
| `backtrace/outputs/dynsys_bode_overlay.html` | 2 子图:上 |H(jω)| dB + 下 arg H(jω) degrees。每对 (k, c) 一条曲线 + ω_n 处 × marker(只在 ω_n 有限时)|
| `backtrace/outputs/dynsys_bode_overlay_summary.txt` | UTF-8 中文汇总,每对一行 + 业务解读 |

#### 中文汇总字段(每对一行)

- 响应类型(`overdamped` / `critical` / `underdamped` / `anti_damped`,从 `classify_response_type(k, c)`)
- Schur 楔形内/外(`is_in_schur_wedge(k, c)`)
- ω_n + |H(jω_n)|(若有)
- |H(j0)| DC 增益 + |H(jπ)| Nyquist
- 业务解读:
  - 共振风险高(楔形外):β 强迫会在 ω_n 处放大 N 倍
  - 低通过滤器(过阻尼 + 楔形内):β 强迫不会引发共振,稳定
  - 临界阻尼:边界 case
  - 标准响应(其他)

#### 解析规则(`parse_overlay_pairs`)

- 分号 `;` 分隔不同对
- 逗号 `,` 分隔 k / c / label(只 split 前 2 个逗号,label 可含逗号)
- label 可含空格(trim 后)
- 错误格式 → `ValueError`("格式错误" / "k 必须" / "c 必须" / "overlay 字符串为空")

推荐 ≤ 10 对(plotly 默认 10 色),> 10 对 label 需手动分组。

#### 与 v5 的关系

v5.1 是 v5 的**纯可视化层扩展**,不动数学层(`transfer_function` / `natural_frequency` / `magnitude_phase` 0 修改)。v5.2 候选:与 `parameter_fit` 集成,自动从历史 (k̂, ĉ) 序列选 top-N 行业画 overlay。

#### 已知陷阱

- 单对模式 `main()` 函数体**一行未动**,仅在最开头加 `if args.overlay: ... return` 分支
- 现有 53 v5 测试 + 8 v5.1 测试 = 61 tests pass,所有 v5 单对模式行为向后兼容
- 边界:复极点区域 (k<c) 才有有限 ω_n;实极点区域 (k>>c) ω_n 不显示 marker

### §4.1.1 v5.2 — parameter_fit Integration (数据驱动 overlay)

把 `parameter_fit.py` 的 `kc_estimates.csv` 数据接到 v5.1 overlay,从"对比框架"升级到"真实行业 G(ω) 对比"。

#### 新增 CLI flag

| flag | 类型 | 说明 |
|---|---|---|
| `--from-kc-estimates` | path | parameter_fit kc_estimates.csv 路径(与 `--overlay` 互斥) |
| `--top-n` | int | 选 top-N 行业(默认 5) |
| `--industry-agg` | str | 行业聚合方法:`median` / `mean`(默认 median) |
| `--select-criterion` | str | 排序标准:`by_n_stocks` / `by_c_over_k` / `by_k_over_c`(默认 by_n_stocks) |
| `--industry-pairs-csv` | path | 选中行业 CSV 输出(默认 `backtrace/outputs/dynsys_industry_overlay_pairs.csv`) |

#### 端到端示例

```bash
# 前提:parameter_fit.py 已跑过,data/projection/kc_estimates.csv 存在
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_forced_response.py \
    --from-kc-estimates data/projection/kc_estimates.csv \
    --top-n 5 \
    --select-criterion by_n_stocks
# 期待:3 个 gitignored 输出
#   backtrace/outputs/dynsys_bode_overlay.html
#   backtrace/outputs/dynsys_bode_overlay_summary.txt
#   backtrace/outputs/dynsys_industry_overlay_pairs.csv
```

#### 排序标准

| criterion | 含义 | 业务用途 |
|---|---|---|
| `by_n_stocks` | 按行业股票数降序 | 默认:成分股最多的行业(覆盖广) |
| `by_c_over_k` | 按 c/k 比降序 | 最过阻尼 / 最稳 / 低通过滤器 |
| `by_k_over_c` | 按 k/c 比降序 | 最欠阻尼 / 共振风险高 / 危险行业 |

#### 与 v5.1 的关系

v5.2 是 v5.1 的**数据接入层** — v5.1 提供"对比框架",v5.2 提供"真实数据 → 框架输入"转换。两者组合 = 业务可决策的行业 G(ω) 对比。

#### 与 parameter_fit 的接口契约(只读)

```python
# v5.2 期望 kc_estimates.csv 的列:
# code: str — 股票代码
# index_code: str — 申万二级代码
# k_hat: float — OLS 拟合恢复系数
# c_hat: float — OLS 拟合阻尼系数
# status: str — "ok" / "fail" (过滤 fail 行)
#
# 其他列可选。**不调任何 parameter_fit 函数** — CSV 是 stable 接口
```

#### 已知陷阱

- `kc_estimates.csv` 必须先存在(跑 `parameter_fit.py`),否则 FileNotFoundError
- 必需列缺失 → ValueError(列出缺失列名)
- `--from-kc-estimates` 与 `--overlay` 互斥,不能同时传
- `select_top_n_industries` 只取实际存在的行业数,如果少于 `--top-n` 会 print 警告

### §4.1.2 v5.3 — Real SI Frequency Response (时序动画)

**动机**:v5.2 数据驱动 overlay 只画**单帧**(一个 asof_date 的行业 G(ω) 对比)。v5.3 把这层补上:**多 asof_date 的 Bode overlay 通过 plotly 动画 slider 联动**,业务可拖时间轴看行业频率响应如何漂移。

**新文件**:`backtrace/dynamics/dynamics_si_freq_response.py`(独立文件,不动 `dynamics_forced_response.py`)

**端到端示例**:
```bash
# 前置:v4.9 parameter_fit --rolling-time 已跑过,data/projection/kc_estimates_time.csv 存在
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_si_freq_response.py
# 期待:3 个 gitignored 输出
#   backtrace/outputs/dynsys_si_freq_response.html (plotly 动画 slider)
#   backtrace/outputs/dynsys_si_freq_response_summary.txt (中文业务解读)
#   data/dynamics/si_freq_response_pairs.csv (审计)
```

**CLI flags**:

| Flag | 默认 | 说明 |
|---|---|---|
| `--kc-time-csv PATH` | `data/projection/kc_estimates_time.csv` | v4.9 rolling 时序输出 |
| `--top-n-industries N` | 5 | 每个 asof_date 选 top-N industries |
| `--industry-selection` | `by_n_stocks` | `by_n_stocks` / `by_c_over_k` / `by_k_over_c` |
| `--max-dates N` | 12 | 最多取最近 N 个 asof_date(避免动画过慢) |
| `--html-output PATH` | `backtrace/outputs/dynsys_si_freq_response.html` | |
| `--summary-output PATH` | `backtrace/outputs/dynsys_si_freq_response_summary.txt` | |
| `--pairs-csv-output PATH` | `data/dynamics/si_freq_response_pairs.csv` | |

**与 v5 / v5.1 / v5.2 / v4.9 的关系**:

| 版 | commit | 主题 |
|---|---|---|
| v5 | `0ce3014` | 受迫系统 + G(ω) 单对频率响应 |
| v5.1 | `e990fb3` | 多对 (k, c) overlay 对比 |
| v5.2 | `fce9532` | 数据驱动 overlay(单帧) |
| v4.9 | `f2178a3` | SI(t) 时序 + 漂移检测 |
| **v5.3** | **(本次)** | **时序动画 G(ω)(t) overlay** |

v5.3 是**时序维度**的扩展:v5 单对 → v5.1 多对 overlay → v5.2 行业 overlay → v5.3 时序动画 overlay。

**已知陷阱**:

- `kc_estimates_time.csv` 不存在(v4.9 没跑过)→ `load_kc_time_series` raise `FileNotFoundError`,CLI 给清晰错误提示用户跑 `parameter_fit.py --rolling-time`
- asof_date 数 > `--max-dates` 12 → 自动截断到最近 N 个,print 警告
- 同行业在某 asof_date 整段 fail(无 `status='ok'`)→ 该 date 该行业跳过,top-N 不足时按实际数
- 动画 HTML 大 → plotly CDN 渲染,数据 ≤ 12 帧 × 5 industries × 200 ω points = 12k 点(~200KB)

### §4.1.3 v5.4 — Dual-Pane Bode (|H(jω)| + ∠H(jω))

v5.4 把 v5.3 的单子图 (|H(jω)| dB) 扩成**双子图 Bode**:
- 上子图 |H(jω)| dB vs ω
- 下子图 ∠H(jω) deg vs ω(共享 x 轴)

实现: `build_animated_overlay_html` 内部用 `plotly.subplots.make_subplots(2, 1, shared_xaxes=True)`,每帧 2 × N traces。

CLI/输出/签名 0 变化,只影响 HTML 渲染(从单图 → 双图,size ~400KB)。

### §4.1.4 v5.5 — Regime Color Coding (regime 颜色编码)

v5.5 在 v5.4 双子图基础上**按阻尼 regime 给曲线着色**,业务一眼区分稳定/共振:

| Regime | 颜色 | 业务语义 |
|---|---|---|
| overdamped (k<c) | 🟢 绿 `#2ca02c` | Schur 内,稳定 |
| critical (k≈c) | 🟠 橙 `#ff7f0e` | Schur 边界,临界 |
| underdamped (k>c) | 🔴 红 `#d62728` | Schur 外,共振风险 |
| anti_damped (k<0) | 🟣 紫 `#9467bd` | 病态(负恢复系数) |

实现: `build_animated_overlay_html` 内 `_regime_color(k, c)` 闭包 → 复用 v5 `classify_response_type`(`dynamics_forced_response.py:130`)。

**业务可读性升级**: 拖 slider 时同 industry 颜色随 (k̂, ĉ) 漂移变化 — 业务可直观看到"哪些 industry 从绿(稳定)漂到红(共振)"。

CLI/输出/签名 0 变化,只影响 HTML 颜色 + 右上角 inline 颜色 ↔ regime 注释。

### §4.1.5 v5.6 — Static 2D Grid PNG (matplotlib 导出)

v5.6 给同一份数据加**静态 PNG 导出** — 业务写报告 / 嵌 PDF 用:

- 函数: `build_static_bode_grid(pairs_per_date, omega_grid, output_path, dpi=100)`
- 布局: 2D 网格 (rows = unique asof_date, cols = |H|/∠H)
- 颜色: 与 v5.5 一致 (4 种 regime hex)
- 0 新依赖 (matplotlib 3.10.6 已装)
- CLI: `--static-output PATH` (默认 `backtrace/outputs/dynsys_si_freq_response_static.png`)

**v5.5 vs v5.6 关系**:
- v5.5 HTML: 交互, 浏览器拖 slider
- v5.6 PNG: 静态, 嵌 PDF / PPT
- 两者共用 `magnitude_phase` + `classify_response_type` + REGIME_COLORS → 0 重复, 0 不一致
