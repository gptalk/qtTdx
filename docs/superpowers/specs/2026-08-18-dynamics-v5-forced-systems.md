# `backtrace/dynamics` — v5 受迫系统 + G(ω) 频率响应

> 2026-08-18 写。v4.7 (`c63e783`) + v4.8 (`dbd367d`) + v4.9 (`f2178a3`) + v4.10 (`d002a0e`) 已合。本 spec 把"被动响应"扩展到"主动驱动" — 用 sinusoidal forcing 测系统的复频响应 H(jω),把 SI(稳定性)与频率域行为耦合起来。

## 1. 背景与动机

v4.7-v4.10 四轮迭代脉络:

| 版 | 主题 | 关键发现 |
|---|---|---|
| v4.7 | SI 单一指标 | 银行 / 公用事业 SI 高,半导体 / 医疗器械 SI 低 |
| v4.8 | SI × forward return rolling IC | **contemporaneous IC ≈ 0** → SI 不是预测性指标 |
| v4.9 | SI 时序 + 漂移检测 | 行业 SI 随时间漂移,drift event 可预警 |
| v4.10 | 时序 SI 的 lagged IC | 闭环 v4.8:lagged IC 真预测性测试 |
| **v5** | **受迫系统 + G(ω)** | **sinusoidal β-forcing → 复频响应 \|H(jω)\| / arg H(jω),SI ↔ 频率域耦合** |

**v4.7-v4.10 局限**:动力学系统一直被当成**被动**的 — β(t) 是 OLS 拟合的"市场回归系数",系统响应市场而无主动驱动。

**v5 的改进**:把系统当成**可控对象** — 用 sinusoidal β(t) = β₀ + A·cos(ω·t+φ) 主动驱动,测稳态响应的幅值 \|H(jω)\| 和相位 arg H(jω)。这给出 **Bode 图**,直接连接:
- v4.7 SI 稳定性指标 ↔ 频率响应峰值(共振 / 阻尼 / 截止频率)
- v4.7 Schur 楔形 (k, ĉ) ↔ \|H(jω)\| 在共振处是否爆炸
- v4.9 时序 SI ↔ 频率响应随时间的演化

**3 种典型频率响应**:
| 系统 | \|H(jω)\| 行为 | 物理含义 |
|---|---|---|
| 强阻尼 (k=2, c=4, Schur 内) | 单调滚降,无峰值 | 系统不"记住"过去的强迫,只跟当前 |
| 临界阻尼 (k=4, c=4) | 微凸峰 | 边界 case,刚好不振荡 |
| 欠阻尼 (k=4, c=0.5, Schur 外) | 共振峰 → ∞ | 不稳定,强迫放大,系统"打摆" |
| 抗阻尼 (k=-1, c=0.5) | 高频滚降但低频爆炸 | 反向弹簧,无界 |

## 2. 范围 (Scope)

**In scope**:
1. **新 CLI** `backtrace/dynamics/dynamics_forced_response.py` (~400 行,独立,不修改 v4.7-v4.10 任何文件)
2. 解析推导的复频响应 H(jω) 公式(基于离散时间 z 域)
3. Bode 图 HTML 输出(幅值 + 相位)
4. 2D (k, c) 热图可视化 \|H(jω_resonance)\|,直接画在 Schur 楔形上
5. 5 个新单元测试(`tests/test_dynamics_eigen.py` 末尾)
6. 更新 `backtrace/dynamics/README.md` §4

**Out of scope(冻结,本轮显式不做)**:
- 时变 β(t) 的非正弦驱动(脉冲、白噪声) — v5.1+ 候选
- 受迫系统的 1-step 预测下游应用(rotation strategy) — v5.2+ 候选
- 修改 `predict_next_state` / `simulate_trajectory` / `compute_sector_stability` / `analyze_eigenvalues` 数学
- 修改 7 个 v4.7-v4.10 文件(`_dynamics_core.py` / 3 caller / `dynamics_si_ic.py` / `dynamics_si_timeseries.py` / `compute_sector_stability_timeseries`)
- 真实数据的 G(ω) 拟合 — 本轮纯解析 + 数值扫描
- v4.10 lagged IC 真实数据 verdict — 已知未跑,等用户 E2E

## 3. 受迫系统的数学

### 3.1 离散时间系统(从 `_dynamics_core.predict_next_state` 出发)

```
a_S(t) = q·β(t)·a_M(t) - k·d(t) - c·u(t) + F_self(t)
v_S(t+1) = v_S(t) + a_S(t)
u(t) = v_S(t) - β(t)·v_M(t)
d(t+1) = d(t) + u(t)
```

**简化假设**(纯受迫分析,无市场驱动 + 无 F_self):
- a_M(t) = 0(无市场加速度)
- F_self(t) = 0(无自作用力)
- v_M(t) = V_M₀ = const(常市场水平)
- β(t) = β₀ + β_f·cos(ω·t)(正弦强迫)

代入:
```
a_S(t) = -k·d(t) - c·(v_S(t) - (β₀ + β_f·cos(ωt))·V_M₀)
v_S(t+1) = v_S(t) + a_S(t)
d(t+1) = d(t) + v_S(t) - (β₀ + β_f·cos(ωt))·V_M₀
```

**关键观察**:β(t)·V_M₀ 进入 a_S 项为 `c·β(t)·V_M₀`;进入 d 递推为 `β(t)·V_M₀`。

### 3.2 线性化与 z 域

设稳态时 v_S(t) = V·e^(jωt), d(t) = D·e^(jωt),令 z = e^(jω)(单位步长):

```
z·V = V - k·D + c·V_M₀·β(z)        [a_S 方程]
z·D = D + V - V_M₀·β(z)            [d 递推]
```

其中 β(z) 在 z 域是其正弦分量的 z 变换。整理:

```
(z - 1)·V + k·D = c·V_M₀·β(z)      ...(1)
-(z - 1)·D + V = V_M₀·β(z)          ...(2)
```

由 (2):`V = V_M₀·β(z) + (z-1)·D`,代入 (1):

```
(z-1)·[V_M₀·β(z) + (z-1)·D] + k·D = c·V_M₀·β(z)
(z-1)·V_M₀·β(z) + (z-1)²·D + k·D = c·V_M₀·β(z)
[(z-1)² + k]·D = V_M₀·β(z)·[c - (z-1)]
D = V_M₀·β(z)·[c - (z-1)] / [(z-1)² + k]
```

代入 (2):
```
V = V_M₀·β(z) + (z-1)·V_M₀·β(z)·[c - (z-1)] / [(z-1)² + k]
V = V_M₀·β(z)·[1 + (z-1)·(c - (z-1)) / ((z-1)² + k)]
V = V_M₀·β(z)·[((z-1)² + k) + (z-1)·c - (z-1)²] / [(z-1)² + k]
V = V_M₀·β(z)·[k + c·(z-1)] / [(z-1)² + k]
```

**复频响应(从 β 强迫到 v_S)**:
```
H(jω) = [k + c·(e^(jω) - 1)] / [(e^(jω) - 1)² + k]
       = [k + c·(z - 1)] / [(z - 1)² + k]   其中 z = e^(jω)
```

### 3.3 极限行为

| ω | H(jω) | 物理含义 |
|---|---|---|
| ω → 0 | k / (k) = 1 | DC gain,稳态放大 = 1 |
| ω → π | k - 2c / (4 + k) | Nyquist,有限值 |

### 3.4 共振条件

对欠阻尼系统 `c² < 4k`,特征方程 `(z-1)² + k = 0` 在 z 域:`z = 1 ± j·√k`。对应**离散时间自然频率** ω_n = arctan(√k / 1) = arctan(√k)。

**共振峰条件**:`d|H|/dω = 0` 在 ω ≈ ω_n 处;在 Schur 楔形内 (c ≥ 2√k),峰被阻尼压平。

## 4. 数值实现

### 4.1 频率扫描

`omega_grid = np.linspace(0.001, np.pi, 200)`(避免 DC 奇异,扫到 Nyquist)

`H_grid = [compute_H(omega, k, c) for omega in omega_grid]` → 复数数组

`magnitude_grid = |H_grid|`(线性尺度)
`phase_grid = np.angle(H_grid)`(弧度)

### 4.2 网格扫描 (k, c)

为画 Schur 楔形上的频率响应热图:
- `k_grid = np.linspace(0.1, 6.0, 60)`
- `c_grid = np.linspace(0.1, 6.0, 60)`
- 对每对 (k, c),固定 ω = ω_n = arctan(√k),算 |H(jω_n)|
- 输出 60×60 矩阵

## 5. 数据流与文件 IO

### 5.1 输入

| 来源 | 用途 |
|---|---|
| 无外部输入 — 纯解析 | CLI 接受 `--k` / `--c` / `--k-grid` / `--c-grid` |

### 5.2 输出(全 gitignored)

| 路径 | 内容 |
|---|---|
| `data/dynamics/transfer_function_grid.csv` | **新增** — ω × \|H\| × arg H,固定 (k, c) |
| `data/dynamics/transfer_function_stability.csv` | **新增** — 60×60 网格,每格 (k, c, ω_n, \|H(jω_n)\|, in_wedge) |
| `backtrace/outputs/dynsys_forced_response.html` | **新增** — Bode 图(幅值 + 相位 2 子图) |
| `backtrace/outputs/dynsys_forced_response_stability.html` | **新增** — 2D (k, c) 热图 + Schur 楔形边界 |
| `backtrace/outputs/dynsys_forced_response_summary.txt` | **新增** — UTF-8 中文汇总 |

### 5.3 脚本结构

新文件 `backtrace/dynamics/dynamics_forced_response.py` (~400 行,独立 CLI):
- `transfer_function(omega, k, c)` → 复数 ndarray(核心公式)
- `magnitude_phase(omega_array, k, c)` → (|H|, arg H)
- `natural_frequency(k)` → 离散 ω_n = arctan(√k)
- `classify_response_type(k, c)` → 'overdamped' / 'critical' / 'underdamped' / 'anti_damped'
- `bode_plot(k, c, omega_grid)` → 2 子图(幅值 log-scale + 相位)
- `stability_heatmap(k_grid, c_grid)` → 60×60 \|H(jω_n)\| 矩阵 + Schur 边界
- `main()` — 端到端

代码增量:**~400 行 + 5 tests**

## 6. HTML 布局

### 6.1 `dynsys_forced_response.html`(Bode 图)

```
┌──────────────────────────────────────────────────┐
│ (1,1) |H(jω)| 半对数图(给定 k, c)              │
│       - 横轴:ω ∈ [0.001, π]                      │
│       - 纵轴:log10|H|                            │
│       - 红线:ω_n = arctan(√k) 标记               │
│       - 灰虚线:|H|=1 单位增益                     │
│ (2,1) arg H(jω) 度数                             │
│       - 横轴:ω                                    │
│       - 纵轴:arg H in degrees [-180, 180]        │
│       - 红虚线:-180° 反相边界                     │
└──────────────────────────────────────────────────┘
```

### 6.2 `dynsys_forced_response_stability.html`(Schur 热图)

```
┌──────────────────────────────────────────────────┐
│ (1,1) 2D (k, c) 热图 \|H(jω_n)\|                 │
│       - 蓝=稳定(值小),红=共振放大(值大)           │
│       - 黑色虚线:Schur 楔形上界 c = 2√k           │
│       - 黑色实线:c = 2√k (Schur boundary)        │
│       - 颜色:|H(jω_n)| (log scale)               │
└──────────────────────────────────────────────────┘
```

## 7. 测试设计

5 个新单元测试,放在 `tests/test_dynamics_eigen.py` 末尾:

| # | 名称 | 断言 |
|---|---|---|
| 1 | `test_transfer_function_dc_gain` | `H(jω=0)` ≈ 1.0(任意 k, c)— DC 增益验证 |
| 2 | `test_transfer_function_stable_rolloff` | (k=2, c=4) Schur 内 → \|H(jπ)\| < \|H(j0.1)\|(高频滚降) |
| 3 | `test_transfer_function_resonance_peak` | (k=4, c=0.5) 欠阻尼 → \|H(jω)\| 在 ω ≈ arctan(2) ≈ 1.107 处有局部峰值 |
| 4 | `test_transfer_function_unstable_blowup` | (k=4, c=0.05) 严重欠阻尼 → \|H(jω_peak)\| > 5(共振爆炸) |
| 5 | `test_classify_response_type` | (k=4, c=4) critical / (k=4, c=0.5) underdamped / (k=2, c=4) overdamped / (k=-1, c=0.5) anti_damped |

**总测试数**: 48 (v4.10) + 5 (v5) = **53 tests pass**

## 8. 与现有代码的关系

| 现有 | 关系 |
|---|---|
| `_dynamics_core.predict_next_state` (v3) | **理论入口** — 复频响应公式从此推导,但不调此函数(纯解析) |
| `dynamics_1step_oos.py` (1-step OOS predictor) | **配套** — 1-step OOS 是时域实验,本 spec 是频域解析;两者互补 |
| `analyze_eigenvalues` / `compute_sector_stability` (v4.7) | **关联** — (k, c) Schur 楔形 ↔ H(jω) 共振峰抑制 |
| `dynamics_si_ic.py` (v4.8) | **不动** |
| `dynamics_si_timeseries.py` (v4.9) | **不动** |
| `dynamics_si_lagged_ic.py` (v4.10) | **不动** |
| `compute_sector_stability_timeseries` (v4.9) | **不动** |
| 3 caller (`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`) | **0 行修改** |

## 9. 已知风险

| 风险 | 缓解 |
|---|---|
| ω=0 时 `e^(jω) - 1 = 0`,分母 = k 不为零,但仍可能数值不稳 | `omega_grid = np.linspace(0.001, π, 200)`,避免 ω=0 |
| 极端 (k, c) → \|H\| 爆炸(超 1e10) | log-scale 显示,数值上 `np.clip(np.log10(\|H\|), -2, 4)` |
| 离散 vs 连续 Bode 区别 | README 注明 "离散时间 Bode,ω ∈ [0, π],Nyquist 在 π" |
| `classify_response_type` 与 Schur wedge 边界 | 用 `c² vs 4k`:c² < 4k 欠阻尼,c² = 4k 临界,c² > 4k 过阻尼,与 v4.7 Schur 一致 |
| 5 个测试用合成 (k, c) 而非真实数据 | 本 spec 是数学验证,不依赖真实数据;README 注明 "需 v4.7 SI 数据可加 §5.3 真实 (k̂, c) 频率响应" |

## 10. 后续(本轮不做)

- **v5.1+**: 时变 β(t) 非正弦驱动(脉冲、白噪声) → 时域 vs 频域对照
- **v5.2+**: 受迫系统的 1-step 预测下游应用(sector rotation strategy,基于 Bode peak 信号)
- **v5.3+**: 真实数据 (k̂, ĉ) 频率响应 — 读 v4.7 SI 时序,对每个行业画 G(ω) 演化
- **v5.4+**: 多维强迫(同时驱动 β + a_M + F_self)→ MIMO 频率响应
- **v6+**: 非线性系统频率响应(describing function)

## 11. 关键文件

- 新增: `backtrace/dynamics/dynamics_forced_response.py` (~400 行)
- 新增: `tests/test_dynamics_eigen.py` 末尾追加 5 测试 (~80 行)
- 修改: `backtrace/dynamics/README.md` §4 (~40 行)
- spec: `docs/superpowers/specs/2026-08-18-dynamics-v5-forced-systems.md` (本文件)
- plan: `docs/superpowers/plans/2026-08-18-dynamics-v5-forced-systems.md` (下一步)

## 12. 验证清单

- [ ] 5 个新测试通过 (53/53 total)
- [ ] `transfer_function_grid.csv` 200 行 × 多列
- [ ] `transfer_function_stability.csv` 60×60 网格(可能含 in_wedge 列)
- [ ] `dynsys_forced_response.html` Bode 图 2 子图正常渲染
- [ ] `dynsys_forced_response_stability.html` 2D 热图 + Schur 边界
- [ ] `dynsys_forced_response_summary.txt` UTF-8 中文可读
- [ ] **0 行修改**:`backtrace/dynamics/_dynamics_core.py` / `dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py` / `dynamics_si_ic.py` (v4.8) / `dynamics_si_timeseries.py` (v4.9) / `dynamics_si_lagged_ic.py` (v4.10) / `compute_sector_stability_timeseries` (v4.9)
- [ ] 端到端: 跑 CLI exit 0,产出 5 个新文件