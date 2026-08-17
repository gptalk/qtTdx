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
