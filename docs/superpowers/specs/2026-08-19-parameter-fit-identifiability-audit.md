# Spec v0 — Parameter Fit Identifiability Audit

> **Date:** 2026-08-19
> **Base:** v6 factor validation `2971633` (post v6.0.1 fix)
> **Branch:** `main`
> **Theme:** 模型审计 (不堆 dashboard,不动数学)

## 1. 问题

V6 factor validation 在 4972 只股票上 cross-sectional Spearman IC 普遍 ≈ 0(spec §13 决策树)。

**未排除的替代解释**:**OLS 求出的 (k̂, ĉ) 本身可能是病态数学解,不是有意义的动力学参数。**

具体 4 个怀疑点(用户诊断):
1. `d_vec` 全样本累计 → rolling k̂ 受历史 origin 污染
2. β̇·v_M 缺失 → 残差包含 β drift,F_self 解释力过强
3. `q` 强制 = 1 → `(q-1)·β·a_M` 进 F_self
4. **没有 condition number / R² 检验** → 现有 (k̂, ĉ) 中混有"OLS 病态补偿"

**业务核心问题**:**现有 (k̂, ĉ) 到底有多少是"可识别动力学参数",有多少是"病态数学噪声"？**

答案决定 V6 (re-run) / V7 (model restructure) / 收口 (model) 三条路。

## 2. 目标

**核心**:不改任何动力学方程,只加诊断层,回答"现有 (k̂, ĉ) 在数值上是否可识别"。

**业务价值**:回答 V6 IC≈0 的 3 种可能解释:
- (a) 动力学模型描述性而非预测性 → 归档 v3-v5.11
- (b) **模型参数本身不可识别** → 重做数学 → v0.2+ 修复(本 spec 回答)
- (c) (a) + (b) 混合 → 重新定义可识别子集,在该子集重跑 V6

**非目标 (YAGNI / 严格冻结数学)**:
- ❌ 不修改 `_solve_ols` / `fit_one` / `fit_rolling` / `main_rolling_time` 任何**数学**分支
- ❌ 不改变 `d_vec` 累计方式
- ❌ 不加 β̇·v_M 项
- ❌ 不加 q 回归
- ❌ 不重跑 V6 / OOS / projection
- ❌ 不动 11 个 protected files
- ❌ 不引入新依赖

## 3. 修改清单

### 3.1 `parameter_fit.py::_solve_ols` 扩展

**改动**:返回值扩展 1→3 新字段。**OLS 解本身一字不改**。

```python
def _solve_ols(a_u_vec, a_v_vec, d_vec, u_vec, beta, valid):
    """核心 OLS 解(内部函数,fit_one 和 fit_rolling 复用)。

    输入:从 movement CSV 重建的 2-D 向量 + valid mask。
    输出: 
        (k_hat, c_hat, f_residual_loss, n_valid, rank, condition_number, regressor_corr, r2)
    """
    # ... 现有 OLS 代码一字不改 ...
    theta, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    k_hat, c_hat = float(theta[0]), float(theta[1])
    F_resid = Y - X @ theta
    f_residual_loss = float(np.mean(F_resid ** 2))

    # === 新增诊断(纯后处理,不动 OLS) ===

    # cond(X),不要 cond(X.T @ X) — 后者人为放大病态
    # κ(X^T X) ≈ κ(X)^2,会失真
    condition_number = float(np.linalg.cond(X)) if X.size > 0 else np.nan

    # regressor 列相关系数(不是参数 correlation)
    # X 列 0 = -d,列 1 = -u,计算 ρ = corr(-d, -u) = corr(d, u)
    if X.shape[0] >= 2 and X.shape[1] == 2:
        col0, col1 = X[:, 0], X[:, 1]
        if np.std(col0) > 1e-12 and np.std(col1) > 1e-12:
            regressor_corr = float(np.corrcoef(col0, col1)[0, 1])
        else:
            regressor_corr = np.nan
    else:
        regressor_corr = np.nan

    # R² = 1 - SS_res / SS_tot
    # SS_tot 用 Y 自己的均值(不是 0) — 标准定义
    y_mean = float(np.mean(Y))
    ss_tot = float(np.sum((Y - y_mean) ** 2))
    ss_res = float(np.sum(F_resid ** 2))
    if ss_tot <= 1e-12:
        r2 = np.nan  # Y 几乎常数 → R² 未定义
    else:
        r2 = 1.0 - ss_res / ss_tot

    return (k_hat, c_hat, f_residual_loss, n_valid, rank,
            condition_number, regressor_corr, r2)
```

**关键设计**:
- `condition_number` 用 `cond(X)`(不是 `cond(X.T @ X)`)
- `regressor_corr`(不是 `param_correlation`)— 计算 X 列 = -d, -u 的相关系数
- `R²` 标准定义,`SS_tot ≈ 0` 时返 NaN
- 旧 `f_self_loss` 字段**保留**(向后兼容),但**新增** `f_residual_loss` 字段与之相同(后续版本可统一)

### 3.2 `parameter_fit.py::fit_one` 扩展

```python
def fit_one(movement_csv, stock_tag, index_tag, min_valid_days=20, clip_extreme=10.0):
    """对一只股票的全样本做闭式 OLS,返回 dict。

    新增诊断字段:
      - identification_status: 数值可识别性(基于 rank + condition_number)
      - fit_quality: 模型解释力(基于 R²)
      - regressor_corr: 设计矩阵 X 两列的相关系数
      - r2: 1 - SS_res/SS_tot
      - condition_number: cond(X)
    """
    # ... 现有代码一字不改 ...
    # 旧调用:
    #   k_hat, c_hat, f_self_loss, _, rank = _solve_ols(...)
    # 新调用:
    k_hat, c_hat, f_residual_loss, n_valid, rank, \
        condition_number, regressor_corr, r2 = _solve_ols(...)

    # classification (现有逻辑保留,作为 identification_status 的输入)
    finite = np.isfinite(k_hat) and np.isfinite(c_hat)
    extreme = abs(k_hat) > clip_extreme or abs(c_hat) > clip_extreme

    # === 新增:identification_status (中性,仅看 rank + cond) ===
    if not finite:
        identification_status = 'singular'
    elif rank < 2:
        identification_status = 'singular'
    elif condition_number >= 1e5:
        identification_status = 'unidentifiable'
    elif condition_number >= 1e3:
        identification_status = 'ill_conditioned'
    else:
        identification_status = 'well_conditioned'

    # === 新增:fit_quality (中性,仅看 R²) ===
    if not np.isfinite(r2):
        fit_quality = 'uninformative'  # Y 几乎常数 → R² 未定义
    elif r2 < 0.01:
        fit_quality = 'poor'         # 模型解释力极弱
    elif r2 < 0.1:
        fit_quality = 'weak'         # 弱解释力
    else:
        fit_quality = 'good'         # 有意义解释力

    # 保留旧 status(向后兼容业务读取)
    if not finite:
        status = 'solve_failed'
    elif rank < 2:
        status = 'singular'
    elif extreme:
        status = f'extreme (|k| or |c| > {clip_extreme:g})'
    else:
        sign_k = 'restoring' if k_hat >= 0 else 'anti-restoring'
        sign_c = 'damping' if c_hat >= 0 else 'anti-damping'
        status = f'ok ({sign_k}, {sign_c})'

    return {
        'k_hat': k_hat, 'c_hat': c_hat,
        'f_self_loss': f_residual_loss,    # 旧字段,保留向后兼容
        'f_residual_loss': f_residual_loss,  # 新字段,语义更准确
        'n_valid_days': n_valid,
        'rank': rank,
        'condition_number': condition_number,
        'regressor_corr': regressor_corr,
        'r2': r2,
        'identification_status': identification_status,
        'fit_quality': fit_quality,
        'status': status,  # 旧业务字段保留
    }
```

### 3.3 `parameter_fit.py::main_fit_all` 输出列扩展

`kc_estimates.csv` 列(向后兼容,新列追加):

| # | 列 | 类型 | 新增 | 来源 |
|---|---|---|---|---|
| 1 | code | str | | existing |
| 2 | name | str | | existing |
| 3 | index_code | str | | existing |
| 4 | index_tag | str | | existing |
| 5 | stock_tag | str | | existing |
| 6 | k_hat | float | | existing |
| 7 | c_hat | float | | existing |
| 8 | f_self_loss | float | | existing (alias) |
| 9 | n_valid_days | int | | existing |
| 10 | status | str | | existing (verbose,向后兼容) |
| **11** | **f_residual_loss** | float | ✓ | `_solve_ols` |
| **12** | **rank** | int | ✓ | `_solve_ols` |
| **13** | **condition_number** | float | ✓ | `_solve_ols` |
| **14** | **regressor_corr** | float | ✓ | `_solve_ols` |
| **15** | **r2** | float | ✓ | `_solve_ols` |
| **16** | **identification_status** | str | ✓ | `fit_one` |
| **17** | **fit_quality** | str | ✓ | `fit_one` |

`fit_rolling` 和 `main_rolling_time` 一并扩展(`(k̂, ĉ)` 之外追加 5 字段)。**rolling 输出 schema 改但文件名不变**(向后兼容列序)。

## 4. 决策阈值

### 4.1 `identification_status`(数值可识别性)

| 类别 | 触发条件 | 含义 |
|---|---|---|
| `well_conditioned` | `rank=2` 且 `cond(X) < 1e3` | 设计矩阵接近正交,OLS 解稳定 |
| `ill_conditioned` | `rank=2` 且 `1e3 ≤ cond(X) < 1e5` | 解存在但对数据噪声敏感 |
| `unidentifiable` | `rank=2` 且 `cond(X) ≥ 1e5` | d/u 高度共线,OLS 解几乎不可信 |
| `singular` | `rank < 2` 或 `not finite` | X^T X 不可逆,退化 |

### 4.2 `fit_quality`(模型解释力)

| 类别 | 触发条件 | 含义 |
|---|---|---|
| `good` | `r2 ≥ 0.1` | 模型解释 10% 以上方差,有意义 |
| `weak` | `0.01 ≤ r2 < 0.1` | 弱解释力,接近噪声 |
| `poor` | `r2 < 0.01` | 模型几乎不解释,Y 主要由噪声/未捕捉项驱动 |
| `uninformative` | `r2 = NaN`(SS_tot ≈ 0) | Y 几乎常数,信号不存 |

### 4.3 Gate(项目 go/no-go 参数)

| Well-conditioned 占比 | 决定 |
|---|---|
| `> 50%` | **(k̂, ĉ) 在多数股票上可识别** → 隔离 V6 在 well_conditioned 子集重跑(spec v0.2) |
| `10%–50%` | **半数可识别** → V6 因子降级,只看 well_conditioned 子集,降级使用 |
| `< 10%` | **几乎不可识别** → 收口。动力学模型作为方法论不可用,V6 IC≈0 根因找到 |

**注意**:这是"可识别股票覆盖率",**不是**"模型有效率"。即使 65% well_conditioned,V6 仍需独立验证 well-conditioned subset → forward IC。

## 5. 新输出

### 5.1 `data/projection/kc_estimates.csv`(扩展)

列定义见 §3.3。

### 5.2 `backtrace/outputs/kc_identifiability_distribution.html` (4 子图 plotly)

| (行,列) | 子图 | 含义 |
|---|---|---|
| (1,1) | R² 直方图 | 整体解释力分布 |
| (1,2) | cond(X) 直方图(对数 x 轴) | 病态程度分布 |
| (2,1) | R² vs \|k̂\| 散点 | 低 R² → \|k̂\| 爆炸?经典 OLS 病态证据 |
| (2,2) | (k̂, ĉ) 散点,颜色 = R² | 极端 (k̂, ĉ) 是否集中在低 R² 区域 |

颜色映射(R²):RdYlGn(红=低 R²,绿=高 R²)。

### 5.3 `data/projection/kc_identifiability_summary.txt` (UTF-8 中文)

```
===== Parameter Fit Identifiability Audit =====
  Run date:  2026-08-19
  Total stocks:    4972
  Finite outputs:  4972
  
  --- Identification Status ---
  Well conditioned:  2184 (43.9%)
  Ill conditioned:   2310 (46.5%)
  Unidentifiable:     478 (9.6%)
  Singular:             0 (0.0%)
  
  --- Fit Quality ---
  Good:              532 (10.7%)
  Weak:             1843 (37.1%)
  Poor:             2597 (52.2%)
  Uninformative:       0 (0.0%)
  
  --- R² Distribution ---
  median = 0.0134
  p25    = 0.0012
  p75    = 0.0821
  
  --- Condition Number Distribution ---
  median = 1.2e+03
  p25    = 4.5e+02
  p75    = 8.7e+03
  
  --- Recommendation ---
  Well-conditioned coverage 43.9% → 10-50% 降级区间
  V6 应在 well_conditioned 子集重跑才可靠
```

**说明**:以上数字仅为示例,实际值由 audit run 后填入。

## 6. CLI

**不改** `--help` / 现有 flags。新增隐式行为:跑 `parameter_fit.py` 默认会写新 HTML 和 TXT。

```bash
# 默认:全样本,产出 3 个新诊断 + HTML + TXT
PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --limit 0
# 跑 4972 只,15-20 分钟

# 冒烟
PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --limit 10
```

**零新 flag**——新输出是默认行为,业务不需要单独开关。

## 7. 测试

`tests/test_dynamics_eigen.py` 新增 5 个测试:

```python
def test_solve_ols_well_conditioned_synthetic():
    """Y = -0.5 d - 0.2 u + tiny_noise → 精确恢复 + good R² + well_conditioned。
    
    Regression test: 验证 audit 没改变 OLS 数学。
    """
    T = 100
    d = np.random.randn(T).cumsum()
    u = np.random.randn(T) * 0.5
    # Y = -0.5 d - 0.2 u + 1e-3 noise
    Y = -0.5 * d - 0.2 * u + np.random.randn(T) * 1e-3
    # X 拼成 2T × 2 形式,与 _solve_ols 同构
    X = np.zeros((2 * T, 2))
    X[:T, 0] = -d
    X[:T, 1] = -u
    X[T:, 0] = -np.random.randn(T).cumsum()  # 第二维用独立 d
    X[T:, 1] = -np.random.randn(T) * 0.5
    Y_full = np.concatenate([Y, Y])  # 简化:第二维复用
    k_hat, c_hat, f_res, n, rank, cond, rcorr, r2 = _solve_ols(
        Y_full, X, np.ones(2 * T, dtype=bool),  # 简化签名
    )
    assert abs(k_hat - 0.5) < 0.05
    assert abs(c_hat - 0.2) < 0.05
    assert r2 > 0.9
    assert cond < 1e3
    # 注:实际签名需匹配 _solve_ols 当前实现,见 plan

def test_solve_ols_ill_conditioned_high_cond():
    """X 列接近共线 → cond > 1e3,identification_status='ill_conditioned'。"""
    T = 100
    d = np.random.randn(T).cumsum()
    u = d * 0.999 + np.random.randn(T) * 1e-3  # u ≈ d,设计矩阵接近共线
    Y = -0.5 * d - 0.2 * u + np.random.randn(T) * 0.1
    X = np.zeros((2 * T, 2))
    X[:T, 0] = -d
    X[:T, 1] = -u
    X[T:, 0] = -np.random.randn(T).cumsum()
    X[T:, 1] = -np.random.randn(T) * 0.5
    Y_full = np.concatenate([Y, Y])
    _, _, _, _, _, cond, _, _ = _solve_ols(Y_full, X, np.ones(2 * T, dtype=bool))
    assert cond > 1e3
    # 注:r2 可能仍高(因 d,u 同步),但 cond 高 → 标记 ill_conditioned

def test_solve_ols_singular_zero_variance():
    """X 某列全 0 → X^T X 不可逆 → rank < 2 → singular。"""
    T = 100
    X = np.zeros((2 * T, 2))
    X[:T, 0] = -np.random.randn(T).cumsum()
    X[:T, 1] = 0  # 第二列全 0
    X[T:, 0] = -np.random.randn(T).cumsum()
    X[T:, 1] = 0
    Y = np.random.randn(2 * T)
    _, _, _, _, rank, _, _, _ = _solve_ols(Y, X, np.ones(2 * T, dtype=bool))
    assert rank < 2

def test_solve_ols_ss_tot_near_zero():
    """Y 几乎常数 → SS_tot ≈ 0 → r2 = NaN → fit_quality='uninformative'。"""
    T = 100
    X = np.zeros((2 * T, 2))
    X[:T, 0] = -np.random.randn(T).cumsum()
    X[:T, 1] = -np.random.randn(T) * 0.5
    X[T:, 0] = -np.random.randn(T).cumsum()
    X[T:, 1] = -np.random.randn(T) * 0.5
    Y = np.zeros(2 * T) + np.random.randn(2 * T) * 1e-15  # 几乎常数
    _, _, _, _, _, _, _, r2 = _solve_ols(Y, X, np.ones(2 * T, dtype=bool))
    assert np.isnan(r2)

def test_cli_smoke_audit_outputs():
    """CLI --limit 5 跑通 + CSV 含新列 + HTML 生成 + TXT 生成。
    
    注:CLI smoke 不写 timeout,实际限制 5 只 < 30 秒。
    """
    # 用临时 output dir 避免污染
    # 检查 exit 0 + 4 新列 (rank, condition_number, regressor_corr, r2)
    # + 2 新 status 列 (identification_status, fit_quality)
    # + 1 HTML + 1 TXT 生成
```

**测试注意**:`_solve_ols` 当前签名 `(a_u_vec, a_v_vec, d_vec, u_vec, beta, valid)`,plan 阶段会调整测试以匹配实际代码。

## 8. 已知陷阱

| 陷阱 | 说明 | 解决 |
|---|---|---|
| `condition_number = nan` 当 X 为空 | `_solve_ols` 早期 valid 数 < 3 → 不会调 cond | None 路径已经处理 |
| `ss_tot ≈ 0` → R² = NaN | Y 几乎常数时 np.sum((Y - mean)^2) 极小 | 显式 `if ss_tot <= 1e-12: r2 = NaN` |
| `regressor_corr` 在 u 或 d 全 0 时 NaN | std = 0 → corrcoef 失真 | 显式 std 检查 |
| `cond(X)` 大数 → inf | X 极病态时可能 overflow | 知道 cap 在 1e20 上限以内 |
| 旧 caller 读 `f_self_loss` 不是 `f_residual_loss` | **保留 `f_self_loss` 字段**作为 alias | 零 breaking change |
| 旧 caller 读 `status` 字段 | 仍可用(只是 verbose 形式) | 零 breaking change |
| `kc_estimates.csv` 列序变化 | 追加 7 列,旧列序不变 | 旧 groupby/order-by 仍 OK |
| `fit_rolling` per-stock CSV 列变化 | 追加 5 列 | 旧 read 仍 OK,但新 reader 需要 5 新列 |
| `main_rolling_time` long format CSV 列变化 | 追加 5 列 | 同上 |
| HTML 输出 5-10 MB | 4972 散点 + 4 子图 | 接受,plotly 自动 downsample |

## 9. 决策树(Gate)

```
跑 --limit 0 audit
  │
  ├─ kc_identifiability_summary.txt 生成
  │
  ├─ 看 well_conditioned 占比
  │
  ├─ > 50% → spec v0.2: well_conditioned subset + V6 re-run
  ├─ 10-50% → V6 因子降级,只看 well_conditioned 子集
  └─ < 10% → 收口:动力学模型作为方法论不可用
              V6 IC≈0 根因 = (k̂, ĉ) 不可识别
              不进 v7,不发新 PR
```

## 10. 不在范围 / 后续

### 10.1 本 spec 显式不做

- ❌ 修复 d_vec 全样本累计(留 v0.2 candidate)
- ❌ 加 β̇·v_M 项(留 v0.3 candidate)
- ❌ q 回归(留 v0.4 candidate)
- ❌ d / u 重新锚定 / 中心化(留 v0.5 candidate)
- ❌ V6 factor validation 重跑(等 audit 结果)
- ❌ R² 跨期 / 滚动(单期 audit 足够)
- ❌ 因子相关性矩阵(留 v0.6)
- ❌ 新 dashboard

### 10.2 待 audit 结果决定的事

| 占比 | 下一步 |
|---|---|
| > 50% well_conditioned | v0.2: (a) V6 re-run on well_conditioned subset, (b) 排除 RSS extreme 股票 |
| 10-50% | v0.2: V6 降级,只看 well_conditioned 子集 |
| < 10% | v0.3 candidate: 模型数学重构(re-derive 方程,加 β̇,加 q) |

## 11. Status: 📝 DRAFT — 2026-08-19

待 plan + implementer + 5 tests。
