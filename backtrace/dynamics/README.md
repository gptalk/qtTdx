# `backtrace/dynamics` — 离散动力系统入口

> 2026-08-16 新建。把 24 节「市场—个股耦合动力系统」中属于**动力系统**的部分
> (1 步预测、N 步轨迹模拟、状态分类后的力分解)收口成可调用 API + CLI 入口。
> 数学源头 = [backtrace/projection/](../projection/)(`compute_dynamics` / `compute_forces` / `classify_states`),
> 本目录**不重写**任何数学,只做"用户面向动力系统"的封装。

## 1. 目录结构

```
dynamics/
├── _dynamics_core.py       数学 re-export + 5 个新增 API
│                           (predict_next_state / simulate_trajectory / build_simulation_df
│                            + F_self 预测器 ×2 + forecast helper ×5)
├── dynamics_system.py      单股端到端 CLI (load → describe → simulate → HTML/CSV)
├── dynamics_batch.py       批量 CLI (读 stocks.csv → 全跑 → manifest)
├── dynamics_1step_oos.py   OOS 1 步预测(纯动力学基线,F_self 滚动均值)
├── dynamics_state_backtest.py  状态分组 + vbt basket 回测 + IC 评估
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
| **F_self 预测器(新)** | `make_rolling_mean_f_self_predictor` / `make_constant_f_self_predictor` | 把残差外推从"末日瞬时值"升级到"滚动均值" |
| **Forecast 模式(新)** | `forecast_v_M_random_walk` / `forecast_v_M_last_value` / `forecast_beta_*` / `forecast_q_t_constant` | 无未来大盘观测时,合成 v_M_seq / beta_seq / q_t_seq |
| **OOS 1 步预测 CLI(新)** | `dynamics_1step_oos.py` | 用 `predict_next_state` 跑 1 步 OOS,产预测 CSV + summary |
| **状态分组 + vbt 回测 CLI(新)** | `dynamics_state_backtest.py` | 按 dominant_state 分组 + basket 回测 + IC |
| **HTML/CSV 落盘** | `dynamics_system.py` / `dynamics_batch.py` | 数据组装 + 文件落地 |

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

# 2. 1 步预测
v_S_now = mv['stock_move'][-1]               # (2,)
a_M_now = np.diff(mv['index_move'][-2:], axis=0)[0]   # 最近有效 a_M
a_S_recent = np.diff(mv['stock_move'][-2:], axis=0)[0]  # 最近有效 a_S
beta_now = mv['proj_coeff'][-1]
# d_now / u_now 重建(同 dynamics_system.py)
u_full = mv['stock_move'] - mv['proj_coeff'][:, None] * mv['index_move']
d_full = np.zeros_like(mv['stock_move'])
d_full[1:] = np.cumsum(u_full[:-1], axis=0)
# q_t 锚定强度:从 description 层拿(2026-08-17 时间轴重构新增)
q_now = float(dyn['q_t'][-1])
a_pred, v_pred = predict_next_state(
    v_S_now, a_M_now, beta_now, d_full[-1], u_full[-1],
    a_S_now=a_S_recent, k=0.0, c=0.0, q_now=q_now,
)
# 1 步预测: 下日个股 ΔS = v_pred
# 注:predict_next_state 现在只返 (a_pred, v_pred);原 3 元组的第 3 个 delta_S_pred 已删除(冗余)

# 3. N 步模拟(Oracle 模式:已知未来大盘)
N = 5
v_S_init = mv['stock_move'][-1]
v_M_seq = mv['index_move'][-N:]              # (N, 2)
beta_seq = mv['proj_coeff'][-N:]             # (N,)
F_self_seq = np.tile(np.array([0.0, 0.0]), (N, 1))  # 假设无残差(或用末日 a_S 推)
sim = simulate_trajectory(
    v_S_init=v_S_init,
    v_M_seq=v_M_seq, beta_seq=beta_seq, F_self_seq=F_self_seq,
    d_init=d_full[-1] - u_full[-1],  # NEW(2026-08-17):d[t+1]=d[t]+u[t] 递推,前推一格
    u_init=u_full[-1],
    k=0.0, c=0.0,
)
sim_df = build_simulation_df(sim, dates=None, index_tag='399001', stock_tag='002475')

# 4. N 步模拟(Forecast 模式:无未来大盘,用预测器生成)
from dynamics import (
    forecast_v_M_random_walk, forecast_beta_rolling_mean,
    forecast_q_t_constant, make_rolling_mean_f_self_predictor,
)
# v_M 用随机游走(噪声 std 估自历史 diff)
v_M_last = mv['index_move'][-1]
diff_std = float(np.nanstd(np.diff(mv['index_move'], axis=0), axis=0).mean())
v_M_seq = forecast_v_M_random_walk(v_M_last, N, sigma_per_step=diff_std, random_state=42)
# β 用末日滚动均值
beta_seq = forecast_beta_rolling_mean(mv['proj_coeff'], N, window=10)
# q_t 用末日观测(没有未来 ‖ΔM‖,沿用)
q_t_seq = forecast_q_t_constant(float(dyn['q_t'][-1]), N)
# F_self 用滚动均值预测器(避免末日瞬时值过拟合)
F_self_pred = make_rolling_mean_f_self_predictor(F_self_full, window=10)
sim = simulate_trajectory(
    v_S_init=v_S_init, v_M_seq=v_M_seq, beta_seq=beta_seq,
    F_self_predictor=F_self_pred,
    d_init=d_full[-1] - u_full[-1],  # NEW(2026-08-17):前推一格
    u_init=u_full[-1],
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
| **d_init 必须 = `d_full[-1] - u_full[-1]`** | `simulate_trajectory` 改用 `d[t+1]=d[t]+u[t]`,要使 d_seq[1] 与 description 层 d_full[-1] 一致需前推一格 | 已自动在 batch/system 脚本里改好;手动调用时注意 |

## 6.5 时间轴重构(2026-08-17)

为消除以下 7 处一致性问题:

1. `predict_next_state` 与 `simulate_trajectory` 不共享同一动力学方程 → 现统一为 `a = q·β·a_M - k·d - c·u + F_self`
2. `predict_next_state` 不应用 q_t → 现新增 `q_now` 参数(默认 1.0,向后兼容)
3. `predict_next_state` 多返冗余 `delta_S_pred` → 现返 2 元组
4. `simulate_trajectory` 的 `E_market` 不是真正交 → 改用沿 v_M(t) 的 Gram-Schmidt 投影
5. `d_seq` 递推差一(实现 `d[t+1]=d[t]+u[t+1]`、spec `d[t+1]=d[t]+u[t]`)→ 改回 spec 写法
6. `F_restore[0]` 的 `hasattr(k * d_init, '__len__')` 死分支 → 改用 `np.linalg.norm(k * d_init)`
7. `forecast_v_M_random_walk` docstring 噪声索引不准 → 改 `noise[t-1]`

CSV schema / sim dict keys / manifest 字段全部向后兼容;旧 batch CSV 列数与列名不变。
**数值会偏移**(d_init 改、真正交改)— 旧 simulation_*.csv 与新结果不会逐行相等。

`compute_dynamics` / `compute_forces`(description 层)保留原 `v_proj = q·β·v_M` 投影不动 —
那个 β 是回归斜率,语义独立,与 simulate 层的"严格正交"是两件事。

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
