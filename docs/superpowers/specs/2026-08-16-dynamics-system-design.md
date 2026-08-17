# `backtrace/dynamics` — 离散动力系统入口

> 写于 2026-08-16,在 [`2026-08-16-market-stock-dynamics-design.md`](2026-08-16-market-stock-dynamics-design.md)
> 之上加一层「动力系统入口」:复用其数学,但对外暴露一组**面向动力系统**的 API + 脚本,
> 突出 `v_{t+1} = v_t + a_t·Δt` 的离散积分链(用户 prompt §19)与多步轨迹模拟。

---

## 1. 范围(Scope)

**In scope**:
1. 公开一个统一入口模块 `backtrace/dynamics/_dynamics_core.py`,re-export 描述层 + 状态分类 + 力模型,
   并新增两个**真正属于"动力系统"**的函数:
   - `predict_next_state(...)` — 1 步预测(扩展 `prediction_ode.py` 的散装函数为可调用 API)
   - `simulate_trajectory(...)` — N 步前向模拟(在已知未来大盘输入下,个股系统怎么演化)
2. 两个 CLI 入口:
   - `dynamics_system.py` — 单股端到端(load → describe → predict → simulate → state → HTML/CSV)
   - `dynamics_batch.py` — 批量(给 manifest 累加 simulation summary)
3. 一份 README,串起整个 `dynamics/` 目录与 projection/ 关系。

**Out of scope(已由其他脚本覆盖,本目录不重写)**:
- 描述层 `compute_dynamics / classify_states / build_dynamics_df` — 在 `_projection_core.py`
- 力模型 `compute_forces / build_forces_df` — 在 `_projection_core.py`
- OOS 1 步预测 + 命中率/RMSE — `prediction_ode.py`(CLI 形态,本目录包成可调用函数)
- 参数拟合 k̂/ĉ — `parameter_fit.py`
- 批量投影底座 — `projection_batch.py`(本目录不重抄)

---

## 2. 复用策略(避免重复)

| 本目录需要 | 来源 | 复用方式 |
|---|---|---|
| `load_pair` (数据加载) | `projection._projection_core` | `from projection._projection_core import load_pair` |
| `compute_movement_projection` (mv dict) | 同上 | `from projection._projection_core import compute_movement_projection` |
| `compute_dynamics` (9 指标) | 同上 | `from projection._projection_core import compute_dynamics` |
| `classify_states` (7 状态) | 同上 | `from projection._projection_core import classify_states` |
| `build_dynamics_df` (14 列 CSV) | 同上 | `from projection._projection_core import build_dynamics_df` |
| `compute_forces` (F_market/R/D/S) | 同上 | `from projection._projection_core import compute_forces` |
| `build_forces_df` (8 列 CSV) | 同上 | `from projection._projection_core import build_forces_df` |
| `STATE_LABELS / STATE_COLORS / STATE_LABELS_CN` | 同上 | re-export |
| OLS k̂/ĉ 拟合值 | `data/projection/kc_estimates.csv` | `parameter_fit.py` 已生成;`dynamics_batch` 读 |
| OOS 1 步预测细节 | `prediction_ode.predict_one` | 复算逻辑(避免导入散装 CLI 入口) |

**新增**(本目录的真正贡献):
- `predict_next_state()` — 1 步状态预测(给 `simulate_trajectory` 当砖)
- `simulate_trajectory()` — N 步前向模拟
- `build_simulation_df()` — 模拟结果 DataFrame
- 完整 CLI 入口

---

## 3. 新 API 设计(`_dynamics_core.py`)

### 3.1 `predict_next_state`

```python
def predict_next_state(
    v_S_now: np.ndarray,    # (2,) 当前个股速度
    a_M_now: np.ndarray,    # (2,) 当前大盘加速度
    beta_now: float,        # 当前 β
    d_now: np.ndarray,      # (2,) 当前位置偏离
    u_now: np.ndarray,      # (2,) 当前速度偏离
    F_self_now: np.ndarray | None = None,  # (2,) 当前残差;None 时按 F_self = a_S - q·β·a_M + k·d + c·u 推
    a_S_now: np.ndarray | None = None,     # (2,) 当前个股加速度(F_self_now=None 时必传)
    k: float = 0.0,         # 恢复系数
    c: float = 0.0,         # 阻尼系数
    q_now: float = 1.0,     # 锚定强度 q_t;默认 1.0(无阻尼)
) -> tuple[np.ndarray, np.ndarray]:
    """a_pred = q_now · β · a_M - k · d - c · u + F_self
       v_pred = v_S + a_pred
       返回 (a_pred, v_pred),都是 (2,) ndarray。
    """
```

**注**:用当前观测的 a_S 算 F_self = a_S - q·β·a_M + k·d + c·u,这就是
"残差项";然后预测下日 a。如果 k=c=0 且 F_self=None,F_self 完全吸收模型误差。

**2026-08-17 变化**:
- 新增 `q_now` 参数(默认 1.0,向后兼容)
- 删除返回值的第 3 个元素 `delta_S_pred`(冗余 ≡ v_pred);现返 2 元组
- 旧 caller 若解构 3 元组会 `ValueError: too many values to unpack`
- 与 `simulate_trajectory` 共享同一方程 `a = q·β·a_M - k·d - c·u + F_self`

### 3.2 `simulate_trajectory`

```python
def simulate_trajectory(
    v_S_init: np.ndarray,           # (2,) 起点速度(取末日真实 v_S)
    M_future: np.ndarray,           # (N, 2) 未来 N 天大盘速度输入
    beta_future: np.ndarray,        # (N,) 未来 N 天 β
    d_init: np.ndarray,             # (2,) 起点位置偏离
    u_init: np.ndarray,             # (2,) 起点速度偏离
    F_self_seq: np.ndarray,         # (N+1, 2) 残差序列(0..N);末日残差外推
    k: float = 0.0,
    c: float = 0.0,
    q_t_seq: np.ndarray | None = None,   # (N,) 锚定强度;None = 默认 1(无阻尼)
) -> dict:
    """N 步前向模拟(Oracle 模式:未来大盘已知)。

    时间轴约定(全篇):
      v_M(t)     第 t 步的大盘速度
      a_M(t)     = v_M(t+1) - v_M(t),代表"step t→t+1 发生的市场速度变化"(前向差)
      v_S(t)     个股速度
      u(t)       = v_S(t) - β(t)·v_M(t)
      d(t+1)     = d(t) + u(t)(递推:在 step t 内加 step t 的 u)

    链:
      for t in range(N):
        a_M(t) = v_M(t+1) - v_M(t)                 # 市场变化,前向差(末步 NaN)
        a_t = q_t · β_t · a_M(t) - k · d_t - c · u_t + F_self(t)
        v_{t+1} = v_t + a_t
        u_{t+1} = v_{t+1} - β_{t+1} · v_M(t+1)     # 速度偏离
        d_{t+1} = d_t + u_t                         # 位置偏离累计(用 step t 的 u)

    能量分解(沿 v_M(t) 方向的真正 Gram-Schmidt 投影):
      v_proj(t) = (v_S(t) · v_M(t) / |v_M(t)|²) · v_M(t)
      v_res(t)  = v_S(t) - v_proj(t)               ← 严格 ⊥ v_M(t)
      E_market(t) = 0.5 · |v_proj(t)|²
      E_self(t)   = 0.5 · |v_res(t)|²
      E_total(t)  = 0.5 · |v_S(t)|²
      R(t)        = |v_res|² / |v_S|²  ∈ [0, 1] 严格成立(不需 clip)
      θ(t)        = arccos( v_S(t) · v_M(t) / (|v_S|·|v_M|) )

    注:本 spec 模拟层用「沿 v_M(t) 方向的严格正交」与 description 层 `compute_dynamics` 的
    「v_proj = q·β·v_M」不同——后者是 β 回归投影(设计选择),前者是 Gram-Schmidt 正交。

    d_init 取值约定:要使 d_seq[1] 与 description 层 d_full[-1] 一致,
    需 d_init = d_full[-1] - u_full[-1](「前推一格」消去累积 u)。

    Returns:
        dict with keys:
          v_seq:  ndarray (N+1, 2)  v_0=init, v_1..v_N 模拟
          a_seq:  ndarray (N, 2)    a_0..a_{N-1}
          d_seq:  ndarray (N+1, 2)  d_0=init, d_1..d_N 累计
          u_seq:  ndarray (N+1, 2)  u_0=init, u_1..u_N
          E_total / E_market / E_self:  ndarray (N+1,)
          R:                        ndarray (N+1,)
          theta:                    ndarray (N+1,) 弧度
          state:                    list[str] (N+1,)
    """
```

### 3.3 `build_simulation_df`

```python
def build_simulation_df(sim: dict, index_tag: str, stock_tag: str) -> pd.DataFrame:
    """组装 12 列模拟结果 DataFrame(长度 N+1)。"""
```

### 3.4 重新导出的便利

```python
from dynamics._dynamics_core import (
    # 复用
    compute_dynamics, classify_states, build_dynamics_df,
    compute_forces, build_forces_df,
    STATE_LABELS, STATE_COLORS, STATE_LABELS_CN,
    # 新增
    predict_next_state, simulate_trajectory, build_simulation_df,
)
```

---

## 4. CLI 入口

### 4.1 `dynamics_system.py` — 单股

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_system.py \
    --code 002475.SZ --name 立讯精密 --days 240 \
    --horizon 5 --k-restore 0.1 --c-damp 0.05
```

**流程**:
1. `load_pair` + `compute_movement_projection` + `compute_dynamics` + `classify_states`
   (与 `projection_2d --dynamics` 共享同一段;下面 4 行 CSV 落到 `data/dynamics/`)
2. `compute_forces(k, c)` → `forces_<idx>_<stk>.csv`(8 列,Frc_ 前缀)
3. 调 `simulate_trajectory(...)`:
   - v_S_init = 末日真实 v_S
   - M_future / beta_future = 末日往后 N 日的"实际大盘"(可走真实末日窗口;
     默认 --horizon=5 取末日往后 5 个共同交易日)
   - F_self_seq = 末日残差 + 后续 ... (本批次只复制末日值;后续迭代用 autoregressive)
4. `build_simulation_df` → `simulation_<idx>_<stk>.csv`(12 列,Sim_ 前缀)
5. HTML 报告 `backtrace/outputs/dynsys_simulation.html`:
   - Row 1: 实际 v_S vs 模拟 v_S 叠加(2 维 X=Volume, Y=Amount)
   - Row 2: 模拟 E_market / E_self / E_total 时序
   - Row 3: 模拟 R / θ 双轴时序
   - Row 4: 模拟状态分类带
   - Row 5: 末日力分解对比(实际 F_market/R/D/S vs 模拟对应的力)

### 4.2 `dynamics_batch.py` — 批量

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_batch.py \
    --input data/projection/stocks.csv --days 240 --horizon 5 --limit 50
```

**输出**:
- 每只票:`data/dynamics/simulation_<idx>_<stk>.csv` + `forces_<idx>_<stk>.csv` + `dynamics_<idx>_<stk>.csv`
- 清单:`data/dynamics/batch_manifest.csv`,列:
  `code, name, index_code, index_name, rows, sim_horizon, sim_mean_R, sim_max_E_self, sim_state_dist, status`

**与 `projection_batch.py --dynamics` 的关系**:
- 那个只跑描述层(q/θ/R/E/state/forces),不模拟
- 这个在描述层之上**追加** simulation 步骤;若 CSV 已存在(由 `projection_batch` 预生成)就复用

---

## 5. 目录结构

```
backtrace/
├── dynamics/                          ← 新建
│   ├── __init__.py
│   ├── _dynamics_core.py              ← 数学 re-export + 2 个新函数
│   ├── dynamics_system.py             ← 单股 CLI(load → describe → simulate → HTML/CSV)
│   ├── dynamics_batch.py              ← 批量 CLI
│   └── README.md                      ← 目录说明(动力学速查、参数建议、已知坑)
├── projection/                        ← 既有,数学源头(本目录不重写)
│   ├── _projection_core.py            ← compute_dynamics / compute_forces / classify_states
│   ├── prediction_ode.py              ← 1 步 OOS 评估(本目录不重抄)
│   └── ...
└── outputs/                            ← HTML 输出落点
```

---

## 6. Edge cases & guard rails

| 情况 | 处理 |
|---|---|
| 数据 < horizon+2 行 | 报错并指出"需要 ≥ horizon+2 共同交易日,实际 X 行" |
| M_future 末日超出数据 | 报"horizon=X 超过可用末日 N 日,自动缩到 N" |
| k=0, c=0 | 模拟退化为"无恢复无阻尼" — 残差完全吸收,只走 β·a_M 驱动 |
| k 或 c 巨大 | 模拟立即回到 d=0(过度恢复)/ 速度差瞬间消失(过度阻尼);脚本会打警告 |
| λ_q 同 description 层 | 一致;`--lambda-q` 默认 median(‖ΔM‖) |
| `state_kc_analysis.py` 联动 | 若 k̂/ĉ 已估,`--k-from-fit / --c-from-fit` 复用,行为同 `projection_2d` |

---

## 7. 决策记录

| 选择 | 替代 | 理由 |
|---|---|---|
| 复用 `_projection_core` 全部数学 | 在本目录重抄 | 单一来源真理;bug fix 改一处生效 |
| `dynamics_system` 走 Oracle 模式(用真实末日大盘) | 走 forecast(随机生成市场) | Oracle 给出"模型极限",先验证再扩 |
| `simulate_trajectory` 不重算 F_self(用末日残差复制) | autoregressive F_self | 第一版简化;F_self 的 evolution 留作后续 |
| 新建 `data/dynamics/`(不与 projection 共用) | 放 `data/projection/` | simulation 输出粒度不同,分开更清晰 |
| HTML 走 5 行 5 主题 | 拆 5 个单图 | 共享 X 轴便于对照,眼扫式诊断 |
| 不引入"质量 m" | 设 m=1 | 能量比 R = ‖v_resi‖²/‖v‖² 与 m 无关 |

---

## 8. 验证路径

```bash
# 冒烟(单股)
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_system.py \
    --code 002475.SZ --name 立讯精密 --days 240 --horizon 5
# 验收:1 个 simulation CSV(12 列) + 1 个 forces CSV(8 列) + 1 个 HTML(5 子图)

# 验收 1:simulation CSV 长度 = horizon + 1
# 验收 2:Sim_v_*_init (row 0) 等于末日真实 v_S
# 验收 3:--horizon 0 → 模拟只有 1 行(初始)
# 验收 4:--k 0 --c 0 → 残差 = a_S - β·a_M(对照 forces CSV 末行 Frc_Self)
# 验收 5:批量 50 只 < 2 分钟(单只 ≈ 0.3s)

# 批量
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_batch.py --limit 50
# 验收:50 个 simulation CSV + 1 个 manifest
```

---

## 9. 与现有文档的关系

- **数学源头**:[`docs/superpowers/specs/2026-08-16-market-stock-dynamics-design.md`](2026-08-16-market-stock-dynamics-design.md)
- **本 spec 增量**:把 §9 列为 out-of-scope 的"预测 ODE"和"批量跑"两个方向,合并到
  新的 `dynamics/` 目录,补上多步 trajectory simulation。
- **`prediction_ode.py`**:不替换;它做 OOS 1 步评估(给定真实 (k̂, ĉ) 算命中率),
  本目录做"已知 k/c + 已知未来大盘 → N 步轨迹"。
- **`parameter_fit.py`**:不替换;它估 k̂/ĉ,本目录消费其结果。
