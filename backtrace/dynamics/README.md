# `backtrace/dynamics` — 离散动力系统入口

> 2026-08-16 新建。把 24 节「市场—个股耦合动力系统」中属于**动力系统**的部分
> (1 步预测、N 步轨迹模拟、状态分类后的力分解)收口成可调用 API + CLI 入口。
> 数学源头 = [backtrace/projection/](../projection/)(`compute_dynamics` / `compute_forces` / `classify_states`),
> 本目录**不重写**任何数学,只做"用户面向动力系统"的封装。

## 1. 目录结构

```
dynamics/
├── _dynamics_core.py       数学 re-export + 2 个新增 API (predict_next_state / simulate_trajectory)
├── dynamics_system.py      单股端到端 CLI (load → describe → simulate → HTML/CSV)
├── dynamics_batch.py       批量 CLI (读 stocks.csv → 全跑 → manifest)
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
a_pred, v_pred, dS_pred = predict_next_state(
    v_S_now, a_M_now, beta_now, d_full[-1], u_full[-1],
    a_S_now=a_S_recent, k=0.0, c=0.0,
)
# 1 步预测: 下日个股 ΔS = v_pred

# 3. N 步模拟(Oracle 模式:已知未来大盘)
N = 5
v_S_init = mv['stock_move'][-1]
v_M_seq = mv['index_move'][-N:]              # (N, 2)
beta_seq = mv['proj_coeff'][-N:]             # (N,)
F_self_seq = np.tile(np.array([0.0, 0.0]), (N, 1))  # 假设无残差(或用末日 a_S 推)
sim = simulate_trajectory(
    v_S_init=v_S_init,
    v_M_seq=v_M_seq, beta_seq=beta_seq, F_self_seq=F_self_seq,
    d_init=d_full[-1], u_init=u_full[-1],
    k=0.0, c=0.0,
)
sim_df = build_simulation_df(sim, dates=None, index_tag='399001', stock_tag='002475')
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
| `R = 1.0` 全程 | 变 β 下 v_resi ≥ v_S(正交分解退化) | 已 clip [0, 1];真要解决需用动态 Gram-Schmidt |
| `E_market = 0` | 同上,v_resi² ≥ v_S² | 同上 |
| 模拟 F_self 巨大 | F_self 末日残差本身就大(变 β + 原始量纲) | 用 4-D lag 向量 / 走归一化空间 |
| `state=none` 频繁 | 前 2 步斜率不够;或 θ NaN(末行) | 设 `--horizon ≥ 3` |
| 批量速度慢(单只 ≈ 0.3s) | `compute_movement_projection` 在描述层 + 模拟层各跑一次 | 后续可缓存 mv dict |

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
