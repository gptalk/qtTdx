# `backtrace/dynamics` — 离散动力系统入口

> 写于 2026-08-16,在 [`2026-08-16-market-stock-dynamics-design.md`](2026-08-16-market-stock-dynamics-design.md)
> 之上加一层「动力系统入口」:复用其数学,但对外暴露一组**面向动力系统**的 API + 脚本,
> 突出 `v_{t+1} = v_t + a_t·Δt` 的离散积分链(用户 prompt §19)与多步轨迹模拟。

---

## 1. 范围(Scope)

**In scope**:
1. 公开一个统一入口模块 `backtrace/dynamics/_dynamics_core.py`,re-export 描述层 + 状态分类 + 力模型,
   并新增三个**真正属于"动力系统"**的函数:
   - `predict_next_state(...)` — 1 步预测(扩展 `prediction_ode.py` 的散装函数为可调用 API)
   - `simulate_trajectory(...)` — N 步前向模拟(在已知未来大盘输入下,个股系统怎么演化)
   - `analyze_eigenvalues(k, c)` — (2026-08-17 v4) 从 2D 离散系统 `A=[[1,1],[-k,1-c]]` 求特征值 + 11 类稳定性分类
2. 三个 CLI 入口:
   - `dynamics_system.py` — 单股端到端(load → describe → predict → simulate → state → HTML/CSV)
   - `dynamics_batch.py` — 批量(给 manifest 累加 simulation summary)
   - `dynamics_eigen_analysis.py` — (2026-08-17 v4) 批量读 kc_estimates,产出 eigen_summary CSV + HTML
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

### 3.1 `predict_next_state` (2026-08-17 v3 派生量统一版)

```python
def predict_next_state(
    v_S_now: np.ndarray,           # (2,) 当前个股速度
    v_M_now: np.ndarray,           # (2,) 当前大盘速度
    v_M_next: np.ndarray,          # (2,) 下一天大盘速度
    beta_now: float,               # 当前 β
    beta_next: float,              # 下一天 β
    d_now: np.ndarray,             # (2,) 当前位置偏离
    F_self_now: np.ndarray | None = None,  # (2,) 当前残差;None = 0
    k: float = 0.0,                # 恢复系数
    c: float = 0.0,                # 阻尼系数
    q_now: float = 1.0,            # 锚定强度 q_t;默认 1.0(无阻尼)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """1 步预测下一个交易日完整状态。

    **内部派生**(消除 caller 重复造轮子,防飘移):
        u_now    = v_S_now - beta_now * v_M_now       # 代数约束
        a_M_now  = v_M_next - v_M_now                 # 前向差

    动力学:
        a_S     = q_now * beta_now * a_M_now - k * d_now - c * u_now + F_self_now
        v_pred  = v_S_now + a_S
        u_pred  = v_pred - beta_next * v_M_next       # 代数约束
        d_pred  = d_now + u_now                       # spec 写法

    Returns:
        (a_pred, v_pred, d_pred, u_pred) — 全 (2,) ndarray
    """
```

**v3 主要变化**(2026-08-17):
- **删除** `a_M_now` / `u_now` 参数 — 内部派生,消除 caller 飘移源
- **删除** `a_S_now` 参数 — 残差由外部 F_self_now 直接给(默认 None = 0)
- **返回 4 元组** (a_pred, v_pred, d_pred, u_pred),前 2 个与 v2 兼容
- 旧 2 元组 caller `a, v = predict_next_state(...)` 仍兼容(前 2 元素不变)

**v2 旧版**(2026-08-17):
- 新增 `q_now` 参数(默认 1.0,向后兼容)
- 删除返回值的第 3 个元素 `delta_S_pred`(冗余 ≡ v_pred);现返 2 元组
- 与 `simulate_trajectory` 共享同一方程 `a = q·β·a_M - k·d - c·u + F_self`

### 3.2 `simulate_trajectory` (2026-08-17 v3 派生量统一版)

```python
def simulate_trajectory(
    v_S_init: np.ndarray,           # (2,) 起点速度(末日真实 v_S)
    v_M_seq: np.ndarray,            # (N+1, 2) t=0..N 大盘速度(2026-08-17 v2:N+1 个状态)
    beta_seq: np.ndarray,           # (N+1,)   t=0..N 回归系数(2026-08-17 v2:状态量)
    d_init: np.ndarray,             # (2,) 起点位置偏离
    # v3:删除 u_init 参数(派生量,在 t=0 由 v_S_init - β[0]·v_M[0] 派生)
    k: float = 0.0,
    c: float = 0.0,
    q_t_seq: np.ndarray | None = None,   # (N,) 锚定强度;None = 默认 1(无阻尼)
    F_self_seq: np.ndarray | None = None,         # (N, 2) 残差序列
    F_self_predictor: callable | None = None,     # 残差预测器
) -> dict:
    """N 步前向模拟(Oracle/Forecast 模式;2026-08-17 v3 + 时间轴彻底重构)。

    状态空间:
        真状态 X(t) = (d(t), v_S(t)) ∈ R⁴
        派生量 u(t) = v_S(t) - β(t)·v_M(t)        ← 代数约束(不递推)
        外部输入 v_M(t+1), β(t+1), q(t), F_self(t)

    时间轴约定(全篇):
      v_M(t)     第 t 步的大盘速度
      a_M(t)     = v_M(t+1) - v_M(t),代表"step t→t+1 发生的市场速度变化"(前向差)
      v_S(t)     个股速度
      u(t)       = v_S(t) - β(t)·v_M(t)           ← 派生
      d(t+1)     = d(t) + u(t)(递推:在 step t 内加 step t 的 u)

    链:
      for t in range(N):
        a_M(t) = v_M(t+1) - v_M(t)                 # 市场变化,前向差(N 个,无 NaN)
        u(t)   = v_S(t) - β(t)·v_M(t)              # 派生
        a_t    = q_t · β_t · a_M(t) - k · d_t - c · u_t + F_self(t)
        v(t+1) = v(t) + a_t
        u(t+1) = v(t+1) - β(t+1)·v_M(t+1)          # 派生
        d(t+1) = d(t) + u(t)                       # 位置偏离累计(用 step t 的 u)

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

    d_init 取值约定(2026-08-17 v2):`d(0) = d_full[-1]`(自然初始条件,无 u 补偿)。
    旧 v1 的 `d_init = d_full[-1] - u_full[-1]` 已被删除。

    u[0] 取值约定(2026-08-17 v3):`u(0) = v_S_init - β(0)·v_M(0)`,派生自代数约束。
    旧 caller 传的 `u_init=u_full[-1]` 已被删除(simulate_trajectory 在 t=0 自动派生)。

    Returns:
        dict with keys:
          v_seq:  ndarray (N+1, 2)  v_0=init, v_1..v_N 模拟
          a_seq:  ndarray (N, 2)    a_0..a_{N-1}
          d_seq:  ndarray (N+1, 2)  d_0=init, d_1..d_N 累计
          u_seq:  ndarray (N+1, 2)  u_0=派生, u_1..u_N
          E_total / E_market / E_self:  ndarray (N+1,)
          R:                        ndarray (N+1,)
          theta:                    ndarray (N+1,) 弧度
          state:                    list[str] (N+1,)
    """
```

### 3.3 `build_simulation_df`

```python
def build_simulation_df(sim: dict, index_tag: str, stock_tag: str) -> pd.DataFrame:
    """组装 19 列模拟结果 DataFrame(长度 N+1)。"""
```

### 3.5 `analyze_eigenvalues` (2026-08-17 v4.1 + v4.2 — ρ-primary 分类 + 楔形距离)

把 `simulate_trajectory` 的核心动力学方程(在 `F_self=0`、忽略外部驱动时)
写成标准 2D 状态转移形式:

```
d(t+1) = d(t) + u(t)               ⇒   X(t+1) = A · X(t),X = (d, u)ᵀ
u(t+1) = (1 − c) · u(t) − k · d(t)

            | 1     1   |
      A  =  |          |        trace = 2 − c,det = 1 − c + k
            |−k  1 − c |
```

`A` 只依赖 (k, c),是 LTI 系统 → 特征值 / 谱半径 / 稳定性分类全是 (k, c) 的纯函数。

**R⁴ 状态空间澄清(v4.1)**:本 2×2 A 是"单个 Vol 或 Amt 分量"对应的核心动力矩阵。
完整 2-D 状态 (Vol, Amt) 各自的偏离和速度构成 4-D 状态:

```
X(t) = [d_Vol, d_Amt, u_Vol, u_Amt]^T ∈ R⁴
A_4x4 = [[I_2,    I_2  ],
         [-kI_2, (1-c)I_2]]
```

4×4 A 的特征值集合 = 2×2 A 的特征值各重复一次:`{λ₁, λ₁, λ₂, λ₂}`,**谱半径相同**。
所以"2×2 A 的 ρ"等价于"4×4 系统的 ρ"——用 2×2 矩阵做分析数学上完全成立。

```python
def analyze_eigenvalues(k: float, c: float, tol: float = 1e-8) -> dict:
    """对 2D 离散系统 [[1,1],[-k,1-c]] 求特征值 + 11 类稳定性分类。

    v4.1 ρ-primary 分类逻辑:
      - k=0,c=0 → jordan_drift
      - k=0,c>0 → marginal_const
      - ρ < 1 → schur_stable(按 D 划分 stable_oscillatory/overdamped/critical_damping)
      - ρ > 1 → unstable(按 k 符号 / D 划分 anti_restoring/oscillatory_divergent/monotonic_divergent)
      - ρ ≈ 1 → critical(按 λ≈±1 划分 critical_period2/periodic/real_unit)

    Returns:
        dict with keys:
          # 原始参数 + 矩阵
          k, c:                 float
          A:                    2×2 ndarray
          # 特征系统
          eigenvalues:          list[complex] (length 2)
          spectral_radius:      ρ(A) = max(|λ|)
          trace:                2 − c
          determinant:          1 − c + k
          discriminant:         c² − 4k
          mode:                 'real_distinct' / 'real_double' / 'complex_conjugate'
          # 稳定性 + 分类
          stability:            'schur_stable' / 'unstable' / 'critical'
          classification:       11 类字符串(见下表)
          schur_stable:         bool
          # 几何距离(v4.2)
          in_wedge:             bool
          distance_to_unit_circle:  1 − ρ
          distance_lower_boundary:  c - k
          distance_upper_boundary:  2 + k/2 - c
          distance_to_wedge:        min(k, c-k, 2+k/2-c)(有符号)
    """
```

**11 类分类法(v4.1 ρ-primary)**:

| 分类 | 触发 | 物理 |
|---|---|---|
| `stable_oscillatory` | ρ<1, D<0 | 共振稳定 |
| `stable_overdamped` | ρ<1, D>0 | 过阻尼稳定 |
| `stable_critical_damping` | ρ<1, D≈0 | 临界阻尼 |
| `oscillatory_divergent` | ρ>1, D<0, k≥0 | 振幅发散(共振本质) |
| `monotonic_divergent` | ρ>1, D≥0, k≥0 | 单调发散 |
| `anti_restoring` | k<0 | 反回复力(趋势强化) |
| `critical_periodic` | ρ≈1, D<0, 无 λ=±1 | 周期-N 振荡边界 |
| `critical_period2` | ρ≈1, ∃ λ≈-1 | 周期-2 边界(隔日反向) |
| `critical_real_unit` | ρ≈1, 有 λ=+1 | 实根单位圆边界 |
| `marginal_const` | k≈0, c>0 | 纯阻尼 |
| `jordan_drift` | k≈0, c≈0 | Jordan 漂移 |

**Schur 楔形(完整稳定性,ρ<1 的充分条件)**:

```
            c
            │
       2+k/2├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
            │   ╱ 楔形(Schur 稳定) ╱
            │  ╱                ╱
            │ ╱              ╱
            │╱            ╱
   ─────────●────────────────── k
            0
              ↘ c = k(下界)
```

完整 Schur 条件:`k > 0` ∧ `k < c < 2 + k/2`。
只看 `c > k` 是不完整的 — `c=3, k=1` 满足 `c>k` 但 ρ=1.5,实际发散。

**v4.1 关键边界修正**:
- `c = 2 + k/2` 当 k>4:λ₁=-1,λ₂=1-k/2,**|λ₂|>1** — 之前误判 critical_period2,现归 unstable
- `c = k` 当 k>4:实根 |λ₁|>1 — 之前误判 critical_real_unit,现归 unstable

**楔形距离几何(v4.2)**:三个距离量衡量"距离稳定边界多远":
```
distance_lower_boundary  = c - k
distance_upper_boundary  = 2 + k/2 - c
distance_to_wedge        = min(k, c-k, 2+k/2-c)  # 有符号
```
正值越大越稳定,负值越大越发散。比 ρ 多一层几何解释力(接近边界 vs 远离边界)。

**与 `simulate_trajectory` 集成**:

`simulate_trajectory` 在返回 dict 里**附带** `analyze_eigenvalues(k, c)` 结果
+ `energy_error = E_total - E_market - E_self`(v4.1 Gram-Schmidt 数值自检,误差 ~ 1e-15):

```python
sim = simulate_trajectory(...)
sim['spectral_radius']        # 0.849
sim['classification']         # 'stable_overdamped'
sim['schur_stable']           # True
sim['in_wedge']               # True
sim['distance_to_wedge']      # 0.145
sim['energy_error']           # ndarray ~ 1e-15(数值健康检查)
```

`build_simulation_df` 在原 19 列基础上**加 10 列**(v4 + v4.2):
- 6 eigenvalue 列(Lambda1/2_Real/Imag, SpectralRadius, DynamicClass)
- 3 楔形距离列(DistanceLower/Upper/ToWedge)
- 1 能量误差列(EnergyError)

**新 CLI**:`dynamics_eigen_analysis.py`(batch)

读 `data/projection/kc_estimates.csv` → 对每行 (k̂, ĉ) 调 `analyze_eigenvalues` → 输出:
- `data/dynamics/eigen_summary.csv`(18 列:基础 + v4.2 楔形距离)
- `backtrace/outputs/dynsys_eigen.html`(**6 子图 plotly**,v4.2 起):
  1. (k̂, ĉ) 散点 + 楔形背景(颜色 = 分类)
  2. ρ 直方图 + ρ=1 红虚线
  3. 11 类分类柱状图
  4. (k̂, ĉ) 散点按楔形距离上色(RdYlGn)
  5. 楔形距离分布直方图
  6. ρ vs 楔形距离

**测试覆盖**:`tests/test_dynamics_eigen.py`(23 个测试,2026-08-17 v4.1 起)
- 11 个分类全分支测试
- 2 个边界 bug 回归测试(c=2+k/2 k>4, c=k k>4)
- 4 个 wedge distance 字段测试
- 3 个数值一致性测试(λ+λ=trace, λ·λ=det, 复共轭)
- 3 个集成测试(energy_error, v4 字段齐全, v4.2 列齐全)

### 3.4 重新导出的便利

```python
from dynamics._dynamics_core import (
    # 复用
    compute_dynamics, classify_states, build_dynamics_df,
    compute_forces, build_forces_df,
    STATE_LABELS, STATE_COLORS, STATE_LABELS_CN,
    # 新增
    predict_next_state, simulate_trajectory, build_simulation_df,
    # 2D 离散系统特征值分析(v4)
    analyze_eigenvalues,
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

### 3.6 v4.3 全市场经验分布(2026-08-17)— 仅数据层扩张

数学层**完全不动**。`dynamics_eigen_analysis.py` 加 3 列读入 + 2 个聚合子图 + 1 个文本汇总:

**输入扩张**:
- `parameter_fit.py --limit 0` 跑全 A 股(~5000 只,~20-40 分钟;前置依赖: `projection_batch.py --movement` 先跑 ~20-40 分钟)
- `data/sw2/members.csv`(列: `sector_code, sector_name, member_code`)反查 `industry_l1`(sector_code)/ `industry_l2`(sector_name)
- `data/stock_basic.csv`(列: `code, market, name, status`)反查 `exchange`(`market` 字段 = SH/SZ/BJ)
- 缺文件 / 缺列 → 行业列 / exchange 留空,流程不致命

**聚合统计量**(用 median,不用 mean):
- 行业(申万二级 sector_name)top10: `n_stocks` / `rho_median` / `rho_p25` / `rho_p75` / `k_hat_median` / `c_hat_median` / `schur_stable_pct` / `in_wedge_pct` / `dist_wedge_median`
- 交易所 SH / SZ / BJ: 同 9 个统计量

**HTML 2×4**(现 2×3 + 新 2):
- (1,4) 行业 ρ 中位数 top10(误差棒 [p25, p75])
- (2,4) 交易所 ρ 中位数对比(SH / SZ / BJ)

**新增输出**:
- `data/dynamics/eigen_summary.csv` 18 → 21 列
- `backtrace/outputs/dynsys_eigen.html` 2×4
- `backtrace/outputs/dynsys_eigen_summary.txt` 纯文本汇总
- `data/dynamics/v43_eigen_top_industries.csv` 行业聚合
- `data/dynamics/v43_eigen_by_exchange.csv` 交易所聚合

**显式不做**(留 v4.4 - v5):
- G(ω) 频率响应
- 行业稳定性指数 SI
- (k, c) 相图 + 7 状态颜色
- 状态转移矩阵
- 任何 IC / basket / 交易信号

**测试**:`tests/test_dynamics_eigen.py` 加 3 个测试,总 **26 passed**(23 旧 + 3 新)。

**关键修正(2026-08-17 实施时发现)**:
- `data/stock_basic.csv` 不含 `industry_l1` / `industry_l2` 列,实际只含 `code, market, name, status`
- 行业映射全部来自 `data/sw2/members.csv`,不是 stock_basic
- 原 v4.3 spec 误以为 stock_basic 含 3 列,实施时已纠正(见 [`2026-08-17-dynamics-v4-3-full-market-distribution.md`](2026-08-17-dynamics-v4-3-full-market-distribution.md) §3.1)
