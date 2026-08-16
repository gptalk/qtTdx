# `projection` — 大盘↔个股耦合动力学 spec

> 写于 2026-08-16,在 `backtrace/projection/_projection_core.py` 既有运动投影
> (`compute_movement_projection`) 之上,加一层「离散动力学系统」指标:
> 锚定强度 q_t、速度分解、加速度、动能 E_market/E_self、耦合度 R_i、
> 偏离角 θ_i、状态分类标签。
>
> **现有 2-D / 运动投影行为保持不变**(`--dynamics` 默认关闭;开启后增量产 HTML + CSV)。

---

## 1. Why this exists(动机)

运动投影已经能给出:
- `Δu`/`Δv`(个股/大盘运动向量)
- `proj`(β·Δv,大盘方向投影)、`residual`(Δu − proj,正交分量)
- `proj_coeff`(β)、`proj_mag`、`resi_mag`、`relative_move`(R=‖u‖/‖v‖)
- `proj_price` / `resi_price`(运动向量 Amount/Volume 比)

但**这些只是「描述」**。要回答 "下一时刻个股会怎么动",需要把它当一个
**离散动力系统**:
- 大盘运动 → 系统驱动力
- 个股残差 → 个股自身摆动
- 速度 / 加速度 / 能量 → 系统状态
- 偏离角 θ、耦合度 R → 个股独立性度量

并且大盘运动 `‖ΔM‖` 接近 0 时,朴素投影 `β = u·v / v·v` 会爆掉。需要引入
**市场锚定强度 q_t** = `‖ΔM‖ / (‖ΔM‖ + λ_q)` 做阻尼。

本 spec 只落地「测量层 + 状态分类」(理论 §1-§13、§23);力模型
(F_restore / F_damp / F_self,理论 §14-§22) 留待后续 spec。

---

## 2. Inputs contract

| 项 | 来源 | 备注 |
|---|---|---|
| 数据源 | `P.load_ohlcva(..., use_tq=False)` 走 `data/` 缓存 | 沿用 `projection_2d.py` / `projection_batch.py` |
| 个股 `Δu` / `Δv` | `compute_movement_projection` 输出 | 已丢首行,T-1 行 |
| 运动投影残差 / β / 幅度 | 同上 `mv` dict | `mv['proj_coeff']` / `mv['proj']` / `mv['residual']` / `mv['resi_mag']` |
| λ_q | `--lambda-q` CLI 或 `median(‖ΔM‖)` | 默认中位数,自适应窗口 |
| 分类阈值 | `--classify-thresholds "r_low,r_high,theta_following_deg,theta_against_deg"` | 默认 `0.10,0.50,30,90` |
| 时间步 Δt | 1 个交易日 | 隐含 `v = ΔS`、`a = Δv`(量纲直接用 Vol/Amt 步长) |

**前置依赖**:本层必须开 `--movement`(因依赖 `Δu` / `Δv`)。CLI 启用
`--dynamics` 时自动 `--movement=True`,无需用户重复传。

---

## 3. Outputs contract

### 3.1 CSV — `data/projection/dynamics_<INDEX_TAG>_<STOCK_TAG>.csv`

长度 `T-1` 行(与运动投影 CSV 一致,共享同一时间轴)。列:

| 列 | 类型 | 公式 | 备注 |
|---|---|---|---|
| `Date` | str | `common_idx[1:]` | 与 `movement_*.csv` 对齐 |
| `Dyn_q_<idx>` | float | `‖ΔM‖ / (‖ΔM‖ + λ_q)` | 锚定强度 ∈ [0, 1) |
| `Dyn_Theta_<st>` | float (rad) | `arccos(clip(Δu·ΔM / (‖Δu‖·‖ΔM‖), -1, 1))` | 偏离大盘运动方向的角度 |
| `Dyn_Coupling_<st>` | float | `‖v_resi‖² / ‖v_S‖²` | 摆动能量占比 R ∈ [0, 1] |
| `Dyn_E_Market` | float | `0.5 · ‖v_proj‖²` | 系统动能 |
| `Dyn_E_Self` | float | `0.5 · ‖v_resi‖²` | 特异动能 |
| `Dyn_E_Total` | float | `E_Market + E_Self` | = `0.5 · ‖v_S‖²` |
| `Dyn_V_Mag_<st>` | float | `‖v_S‖` | 个股速度模长 |
| `Dyn_V_Mag_<idx>` | float | `‖v_M‖` | 大盘速度模长 |
| `Dyn_V_Proj_Mag` | float | `‖v_proj‖` | 沿大盘方向分量 |
| `Dyn_V_Resi_Mag` | float | `‖v_resi‖` | 正交分量 |
| `Dyn_A_Mag_<st>` | float | `‖a_S‖`(末行 NaN) | 个股加速度模长(右补 NaN 保持矩形) |
| `Dyn_A_Mag_<idx>` | float | `‖a_M‖`(末行 NaN) | 大盘加速度模长 |
| `Dyn_State_<st>` | str | 见 §4 | 状态分类标签 |

列数:**14**(`Dyn_` 前缀,与既有 `Move_` / `State_` 区分)。

### 3.2 HTML — `backtrace/outputs/dynmv_trajectory.html`

单文件 4 子图(共享 X 轴 = 日期):

1. **Row 1 — 速度对比**:`‖v_M‖`(大盘,蓝) + `‖v_S‖`(个股,红) 时序
2. **Row 2 — 能量拆分**:`E_Market` + `E_Self` 堆叠面积(Stacked Area)
3. **Row 3 — 耦合度 + 偏离角**:R(左轴 [0,1],绿) + θ(右轴 0°-180°,橙)双 Y 轴
4. **Row 4 — 状态分类带**:7 种状态各自配色,文本标签 + 背景条

```
y4: 跟随 | 弱偏离 | 加速偏离 | 独立 | 逆势 | 回归 | 共振
     绿      黄         红         紫      棕    青    粉
```

模板 `plotly_dark`,与既有 M1-M7 风格一致。

---

## 4. State classification(状态分类)

7 个互斥标签 + `none`(都不命中)。优先级从高到低:

| 优先级 | 状态 | 条件 |
|---|---|---|
| 1 | 逆势 (against) | θ ≥ θ_against |
| 2 | 共振 (resonance) | R ≥ R_high AND θ < θ_following |
| 3 | 加速偏离 (accelerating) | R 上升 AND dE_Self/dt > 0(3 日线性斜率) |
| 4 | 回归 (returning) | R 下降 AND dE_Self/dt < 0(3 日线性斜率) |
| 5 | 独立 (independent) | R ≥ R_high AND θ < θ_against |
| 6 | 弱偏离 (weak_div) | R_low ≤ R < R_high AND θ < θ_against |
| 7 | 跟随 (follow) | R < R_low AND θ < θ_following |
| - | none | 都不命中(罕见,主要是数据缺失的边缘日) |

**斜率定义**:对 R 序列和 E_Self 序列分别用 `np.polyfit(np.arange(k), x[-k:], 1)[0]` 估
3 日斜率(`k = min(3, len(x))`)。前 2 天没有足够窗口 → 该日跳过 3/4 优先级,
回退到 5/6/7。

**阈值默认值**(CLI 可覆盖):
- `R_low = 0.10`
- `R_high = 0.50`
- `θ_following = 30°`
- `θ_against = 90°`

---

## 5. CLI 集成(`projection_2d.py`)

新增 3 个 flag:

| Flag | 默认 | 说明 |
|---|---|---|
| `--dynamics` | False | 启用动力学层(自动开启 `--movement`) |
| `--lambda-q` | `median(‖ΔM‖)` | 锚定强度系数(浮点);`-1` 走默认 |
| `--classify-thresholds` | `0.10,0.50,30,90` | 逗号分隔 4 个浮点 |

`parse_args()` 末尾追加:
```python
if args.dynamics and not args.movement:
    args.movement = True   # 自动联动
if args.lambda_q < 0:
    args.lambda_q = None   # 走默认 median
```

**输出文件**(仅 `--dynamics=True` 时):
- HTML: `backtrace/outputs/dynmv_trajectory.html`
- CSV: `data/projection/dynamics_<INDEX_TAG>_<STOCK_TAG>.csv`

**不影响**:6 个 2-D HTML、1 个 state CSV、4 个 movement HTML、1 个 movement CSV
全部照旧。`--dynamics` 启用后**额外**产出 1 个 HTML + 1 个 CSV。

---

## 6. Math layer(`_projection_core.py` 新增)

```python
def compute_dynamics(mv: dict, lambda_q: float) -> dict:
    """基于运动投影的输出,计算 9 个动力学指标 + 状态分类。

    Returns:
        dict with keys:
          q_t:        ndarray (T-1,)        # 锚定强度
          theta:      ndarray (T-1,)        # 偏离角(弧度)
          R:          ndarray (T-1,)        # 耦合度
          v_S_mag:    ndarray (T-1,)        # 个股速度模长
          v_M_mag:    ndarray (T-1,)        # 大盘速度模长
          v_proj_mag: ndarray (T-1,)        # 沿大盘方向分量
          v_resi_mag: ndarray (T-1,)        # 正交分量
          E_market:   ndarray (T-1,)        # 0.5·‖v_proj‖²
          E_self:     ndarray (T-1,)        # 0.5·‖v_resi‖²
          E_total:    ndarray (T-1,)        # E_market + E_self
          a_S_mag:    ndarray (T-1,) 末行 NaN  # 个股加速度模长
          a_M_mag:    ndarray (T-1,) 末行 NaN  # 大盘加速度模长
          state:      list[str] (T-1,)      # 状态分类标签

    Caller slices:
      velocity / energy / R / theta / q_t / state: common_idx[1:] (T-1)
      acceleration: common_idx[2:] (T-2), 右补 NaN 保持 CSV 矩形

    Note:
      v_proj = q_t · β · Δv      (用 mv['proj'] = β·Δv,再逐元素乘 q_t)
      v_resi = Δu - v_proj        (Δu 即 v_S,Δt=1)
    """

def classify_states(R, theta, E_self, thresholds) -> list[str]:
    """按 §4 优先级规则,逐日打标签。R/theta/E_self 都是 T-1 长。
    thresholds = (R_low, R_high, theta_following_rad, theta_against_rad)
    内部把度转弧度。前 2 天斜率无法估,跳过 3/4 优先级。"""

def build_dynamics_df(common_idx, dyn, index_tag, stock_tag) -> pd.DataFrame:
    """组装 14 列 CSV(common_idx 取 common_idx[1:],T-1 行;
    末行的加速度列填 NaN)。"""
```

---

## 7. Edge cases & guard rails

| 情况 | 处理 |
|---|---|
| `‖ΔM‖ = 0` | `q_t = 0` → `v_proj = 0` → `v_resi = Δu`,E_market = 0,E_self 全归特异 |
| `‖Δu‖ = 0` 或 `‖ΔM‖ = 0` | θ 算 `arccos(0/0)` → 设 NaN;状态降级为 `none` |
| `‖v_S‖ = 0` | R = 0/0 → 设 0(耦合度物理上 = 0,个股无速度时谈不上耦合) |
| 数据 < 2 行 | `compute_movement_projection` 已 raise,本层不重复检查 |
| 数据 = 2 行(最小) | T-1 = 1,加速度列全 NaN;状态分类无法估斜率 → 强制 5/6/7 路径 |
| λ_q = 0 | `q_t = ‖ΔM‖ / (‖ΔM‖ + 0) = 1`,退化为无阻尼朴素投影 |
| λ_q 巨大 | `q_t ≈ 0`,大盘运动被压扁成 0 → 个股残差 = Δu(全部算特异) |
| 阈值非法(R_low ≥ R_high 等) | 解析时 ValueError,告诉用户哪条不合法 |

---

## 8. Why these choices(决策记录)

| 选择 | 替代 | 理由 |
|---|---|---|
| λ_q 默认 median(‖ΔM‖) | 硬编码 e.g. 1e5 | 自适应窗口,不用用户调 |
| 不引入质量 m | 设 m=1 | 能量比 R = ‖v_resi‖²/‖v‖² 与 m 无关,设了浪费 |
| 加速度列右补 NaN(不左补) | 左补 / 截断 | 保留「最新一天没加速度」的物理事实 |
| `Dyn_` 前缀(而非 `DynM_`) | 复 `Move_` 前缀 | 新维度层,与运动投影并列;不与 `Move_` / `State_` 混 |
| 单文件 `dynmv_trajectory.html` | 拆 4 个子图 | 4 子图共享 X 轴,放一起便于眼扫对照 |
| 状态分类不带回归方向(±) | 加 R 升/降二分 | 用户 prompt §23 表给的是 7 状态(单向);加减号合并会让「加速偏离 / 回归」消失 |

---

## 9. Out of scope(下一阶段 spec 候选)

- **力模型层**:β·a_M + F_restore + F_damp + F_self。需要估 k/c 参数,
  候选方案 — 从 R/θ 历史自相关衰减拟合,或网格搜索。独立 spec。
- **预测 ODE**:`v_{t+1} = v_t + a_t · Δt`,用历史 Δt=1 滚动,生成
  "下一交易日个股速度预测",回测命中率。独立 spec。
- **批量跑**:`projection_batch --dynamics` 跑全 A 股,输出
  `dist_manifest.csv`(R/θ 分布 + 状态标签分布)。独立 spec。
- **共振/逆势选股信号**:用 §4 状态分布找全市场「共振」「独立」个股,
  跑 vbt 回测。独立 spec,需 GP 因子挖掘 / tsfresh 系列配合。

---

## 10. 验证路径(本 spec 验收)

```bash
# 冒烟(单股,默认大盘基线,动力学 + 运动)
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_2d.py \
    --code 002475.SZ --name 立讯精密 --days 240 --movement --dynamics

# 验收 1:CSV 14 列,T-1 行,首行 q_t 合理(0-1)
# 验收 2:CSV 末行 a_S_mag / a_M_mag = NaN
# 验收 3:HTML 4 子图能开,R/θ 双轴不重叠
# 验收 4:--lambda-q 1e8(巨大)→ q_t ≈ 0 → v_proj_mag ≈ 0 → E_market ≈ 0
# 验收 5:--lambda-q 0 → q_t ≈ 1 → 与朴素投影一致(对照无 --dynamics 的 M3)
# 验收 6:不动既有 — 不传 --dynamics 时输出与改动前完全一致(字节级比对 HTML)
```