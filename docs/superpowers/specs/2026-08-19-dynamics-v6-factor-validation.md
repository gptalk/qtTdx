# Spec v6 — Dynamics Factor Validation

> **Date:** 2026-08-19
> **Base:** v5.11 `kc_estimates` integration (`378d448`)
> **Branch:** new (from `main` HEAD = `378d448` — 实际 HEAD = `140a937` v5.12 revert 后等效 v5.11)
> **Theme:** 业务验证(不堆 dashboard)

## 1. 问题

v3-v5.11 投入了大量工程造出动力学变量:`k̂, ĉ, λ, ρ, θ, E_self, hit_rate, RMSE, MAE, state` 等。

但**从未在真数据上验证过这些变量是否有业务价值**(v5.12 regime 失败教训:不能先设计再找数据)。

**业务核心问题**:**哪个动力学变量对 forward return 有预测力?**

答案决定下一步方向:
- 若 hit_rate / RMSE 有 IC → OOS 预测 → α 信号(走 tsfresh/alpha 路径)
- 若 k̂ / ĉ / ρ 有 IC → 动力学结构 → 截面/行业 alpha(走 alpha 路径)
- 若 Δk / Δρ 有 IC → 动力学演化 → regime 切换信号(走 v5.14 rolling 路径)
- 若都 ≈ 0 → 模型描述性而非预测性,归档 v3-v5.11

## 2. 目标

**核心**:把现有所有动力学变量 + 预测质量变量,统一做一次 cross-sectional Spearman IC + Q1-Q5 quantile 评估,业务可一次性看到"哪些变量有用"。

**业务价值**:**一次性回答 v3-v5.11 是否值得继续投入**。

**非目标 (YAGNI)**:
- ❌ 不做 walk-forward IC(先做 contemporaneous IC,简单版)
- ❌ 不做因子相关性矩阵 / decorrelation / 因子合成
- ❌ 不做因子-因子回归
- ❌ 不做行业相对 IC / neutralization(直接 raw IC)
- ❌ 不做实时 α 信号生成
- ❌ 不动 protected files(11 个)
- ❌ 不重写任何 projection / dynamics 数学
- ❌ 不重跑 projection / dynamics / OOS batch(只读现有 CSV)

## 3. 数据来源

| 数据 | 来源文件 | 必须 / 可选 | 缺失行为 |
|---|---|---|---|
| k̂ / ĉ | `data/projection/kc_estimates.csv` | 必须 | FileNotFoundError + 提示跑 `parameter_fit.py` |
| rolling (k̂, ĉ, ρ) | `data/projection/kc_estimates_time.csv` | 可选 | 跳过 Δk / Δc / Δρ / Δθ factors |
| hit_rate / RMSE / MAE | `prediction_summary.csv` (v5.10 batch 输出) | 可选 | 跳过 prediction quality factors |
| state distribution | `data/dynamics/state_distribution.csv` | 可选 | 跳过 state factors |
| eigen_summary (ρ / θ) | `data/dynamics/eigen_summary.csv` | 可选 | 跳过 λ / ρ / θ factors (从 k̂, ĉ 现算也行) |
| daily prices | `data/stocks/<code>.csv` | 必须 | FileNotFoundError |
| industry | `data/sw2/members.csv` | 可选 | 跳过 by_industry 拆分 |
| index basic | `data/stock_basic.csv` | 必须 | name → industry_l1/l2 |

**关键设计**:**V6 是纯消费者,不写任何上游 CSV**。所有上游必须由用户在 V6 跑前生成。

## 4. 因子列表

### 4.1 预测质量(v5.9-v5.11 OOS)

| 因子 | 来源列 | 含义 |
|---|---|---|
| `hit_rate` | `prediction_summary.csv::hit_rate` | 1 步方向预测准确率 |
| `rmse` | `...::rmse` | 1 步预测 RMSE |
| `mae` | `...::mae` | 1 步预测 MAE |
| `direction_acc` | `...::direction_accuracy` | 同 hit_rate (alias) |

### 4.2 参数估计(kc_estimates)

| 因子 | 来源列 | 含义 |
|---|---|---|
| `k` | `kc_estimates.csv::k_hat` | 恢复系数 |
| `c` | `...::c_hat` | 阻尼系数 |
| `c_over_k` | computed: c_hat / k_hat | 阻尼-恢复比 |
| `log_c_over_k` | computed: log10(c_hat / k_hat) | log 比 |

### 4.3 特征值(从 k̂, ĉ 派生)

| 因子 | 来源 | 含义 |
|---|---|---|
| `rho` | `analyze_eigenvalues(k,c)['spectral_radius']` | 谱半径 |
| `theta` | `...['eigenvalues']` 主导角 | 振荡强度 |
| `dist_to_unit` | computed: 1 - rho | 稳定裕度 |
| `regime` | `classify_response_type(k,c)` | 4 类(overdamped/critical/underdamped/anti_damped) |

### 4.4 状态分布(可选,state_distribution.csv)

| 因子 | 来源列 | 含义 |
|---|---|---|
| `state_dominant` | `...::dominant_state` | 7 状态主导 |
| `state_p_resonance` | `...::resonance` | 共振占比 |
| `state_p_against` | `...::against` | 逆势占比 |
| `state_p_independent` | `...::independent` | 独立占比 |
| `state_p_follow` | `...::follow` | 顺势占比 |

### 4.5 动力学演化(可选,kc_estimates_time.csv)

| 因子 | 来源 | 含义 |
|---|---|---|
| `delta_k` | diff(k_hat) over rolling windows | k̂ 变化趋势 |
| `delta_c` | diff(c_hat) over rolling windows | ĉ 变化趋势 |
| `delta_rho` | diff(rho) over rolling windows | ρ 变化趋势 |
| `delta_theta` | diff(theta) over rolling windows | θ 变化趋势 |

## 5. Forward Returns

horizons = [1, 5, 10, 20] 日

`fwd_ret_h(code, date_t) = close[code][t+h] / close[code][t] - 1`

NaN 处理:t+h 超出 daily data 范围 → 该 (code, date, h) 跳过。

## 6. 评估方法

### 6.1 Cross-Sectional Spearman IC

对每个 (factor, horizon, date_t):
1. 取 date_t 该日所有 stock 的 factor 值 → factor_series
2. 取 date_t 该日所有 stock 的 fwd_ret_h → ret_series
3. `IC_t = spearmanr(factor_series, ret_series).correlation`
4. 剔除 `n_stocks < 10` 的日
5. 跨所有 date_t 求 IC mean / std / IR = mean / std / p-value (one-sample t-test H0: mean=0)

### 6.2 Quantile Returns (Q1-Q5)

对每个 (factor, horizon, date_t):
1. 按 factor 值分 5 等分位数(用 pd.qcut,允许 ties)
2. 计算每组平均 forward return → `q1_ret_t, ..., q5_ret_t`
3. 跨所有 date_t 求每组 mean
4. Q5 - Q1 spread → 多空收益

### 6.3 By-Year

按 `date_t.year` 分组,每个 year 单独算 IC 和 quantile。

### 6.4 By-Industry

按 industry_l1 分组(申万一级),每个 industry 单独算 IC。

**注意**:industry 内的 stock 数 < 30 时该 industry 跳过(样本太少)。

## 7. 输出

### 7.1 `data/dynamics/factor_validation.csv`

主输出,**每个 (factor, horizon) 一行**:

```csv
factor,horizon,n_obs,n_dates,ic_mean,ic_std,ic_ir,ic_pvalue,
q1_ret,q2_ret,q3_ret,q4_ret,q5_ret,q5_minus_q1,
top_year,top_year_ic,top_industry,top_industry_ic,
status
```

- `status`: `ok` / `insufficient_data`(n_obs<30) / `not_loaded`(上游 CSV 缺失)

### 7.2 `data/dynamics/factor_validation_by_year.csv`

```csv
factor,horizon,year,n_obs,n_dates,ic_mean,ic_std,status
```

### 7.3 `data/dynamics/factor_validation_by_industry.csv`

```csv
factor,horizon,industry_l1,n_obs,n_dates,ic_mean,ic_std,status
```

### 7.4 `backtrace/outputs/dynsys_factor_validation_summary.txt`

UTF-8 中文可读汇总:
- top 5 factors by |ic_mean| × |ic_ir| (across all horizons)
- 每个 factor × horizon 一行:IC / IR / p-value / Q5-Q1 spread
- 按年 / 按 industry 显著 IC 列表

### 7.5 `backtrace/outputs/dynsys_factor_validation.html` (可选)

plotly HTML(只展示 top-10 factors,避免信息爆炸):
- 子图 (1,1): IC mean across horizons (bar, color by |IR|)
- 子图 (1,2): Q5-Q1 spread across horizons (bar)
- 子图 (2,1): top factor 的 by-year IC 时序(line + scatter)
- 子图 (2,2): top factor 的 by-industry IC bar

## 8. CLI

```bash
# 默认:全 factor,全 horizon,~5000 stocks 全部 (限 limit 0 = 全部)
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_factor_validation.py

# 限定 stocks
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_factor_validation.py --limit 500

# 自选 factors
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_factor_validation.py \
    --factors hit_rate,k,c,rho,theta

# 自选 horizons
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_factor_validation.py \
    --horizons 5,10,20

# 自选 input 文件
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_factor_validation.py \
    --kc-csv data/projection/kc_estimates.csv \
    --oos-summary data/dynamics/prediction_summary.csv \
    --state-csv data/dynamics/state_distribution.csv \
    --kc-time-csv data/projection/kc_estimates_time.csv
```

## 9. 架构

新文件 1 个:`backtrace/dynamics/dynamics_factor_validation.py`

### 9.1 函数模块

```python
# === 加载层 ===
def load_kc_estimates(path) -> pd.DataFrame: ...
def load_oos_predictions_summary(path) -> pd.DataFrame: ...
def load_state_distribution(path) -> pd.DataFrame: ...
def load_kc_time_series(path) -> pd.DataFrame: ...
def load_daily_prices(stock_codes, repo_root) -> dict[str, pd.DataFrame]: ...
def load_industry_lookup(repo_root) -> pd.DataFrame: ...

# === 派生层 ===
def compute_eigen_factors(kc_df) -> pd.DataFrame: ...
    # 复用 _dynamics_core.analyze_eigenvalues (无修改)
def compute_kc_evolution_factors(kc_time_df) -> pd.DataFrame: ...

# === 评估层 ===
def compute_cross_section_ic(factor_series, ret_series) -> tuple[float, float, int]:
    """Spearman IC + p-value + n."""
def compute_quantile_returns(factor_series, ret_series, n_quantiles=5) -> dict:
    """Q1-Q5 mean ret + Q5-Q1 spread."""

# === 主流程 ===
def build_factor_panel(kc_df, oos_df, state_df, kc_time_df) -> pd.DataFrame:
    """(code × date) × factor value 长表。"""
def compute_forward_returns(daily_prices, codes, dates, horizons) -> pd.DataFrame:
    """(code × date) × fwd_ret_h。"""
def validate_all_factors(panel, fwd_rets, horizons) -> tuple[pd.DataFrame, ...]:
    """返回 (main, by_year, by_industry) 三个 DataFrame。"""

# === 输出层 ===
def write_main_csv(results, path): ...
def write_by_year_csv(results, path): ...
def write_by_industry_csv(results, path): ...
def write_summary_text(results, results_by_year, results_by_industry, path): ...
def build_factor_validation_html(results, results_by_year, results_by_industry, path): ...

# === CLI ===
def main(): ...
```

### 9.2 关键设计点

- **不重跑 projection/dynamics/OOS**:只读现有 CSV,缺失就跳过。
- **复用 `_dynamics_core.analyze_eigenvalues`**:import,不改。
- **复用 `dynamics_state_backtest.py` 输出格式**:同 schema。
- **数据合并靠 code join**:不重算 daily data。
- **缺失 graceful**:每个 factor set 独立 try/except,某个缺失不影响其他。

## 10. 性能预算

- 5000 stocks × 250 dates × 16 factors × 4 horizons ≈ 80M IC calculations
- Spearman O(n log n),5000 stocks ≈ 5000 log 5000 ≈ 60k ops × 250 dates × 4 horizons = 60M ops
- 实际:约 5-10 分钟(单进程,numpy 加速)
- 接受:跑全市场作为 `--limit 0`,默认 500

## 11. 测试

`tests/test_dynamics_eigen.py` 新增:

```python
def test_compute_cross_section_ic():
    """Perfect positive correlation → IC ≈ 1."""
    factor = pd.Series([1, 2, 3, 4, 5])
    ret = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    ic, pval, n = compute_cross_section_ic(factor, ret)
    assert ic > 0.99
    assert n == 5

def test_compute_cross_section_ic_negative():
    """Negative correlation → IC < 0."""
    factor = pd.Series([1, 2, 3, 4, 5])
    ret = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1])
    ic, _, _ = compute_cross_section_ic(factor, ret)
    assert ic < -0.99

def test_compute_quantile_returns_monotonic():
    """Q1 < Q2 < Q3 < Q4 < Q5."""
    factor = pd.Series(np.arange(100))
    ret = pd.Series(np.arange(100) * 0.01)  # 完美单调
    q = compute_quantile_returns(factor, ret, n_quantiles=5)
    assert q['q1_ret'] < q['q2_ret'] < q['q3_ret'] < q['q4_ret'] < q['q5_ret']

def test_compute_eigen_factors():
    """(k=0.145, c=1.112) → rho ≈ 0.849, regime=overdamped."""
    kc = pd.DataFrame({'code': ['x'], 'k_hat': [0.145], 'c_hat': [1.112]})
    out = compute_eigen_factors(kc)
    assert abs(out['rho'].iloc[0] - 0.849) < 0.01
    assert out['regime'].iloc[0] == 'overdamped'

def test_compute_forward_returns():
    """已知 close → 已知 fwd_ret."""
    prices = pd.DataFrame({'close': [10, 11, 12, 13]})  # +10% per day
    fwd = compute_forward_returns({'x': prices}, ['x'], [0], [1, 2])
    assert abs(fwd.loc[('x', 0), 1] - 0.10) < 0.01
    assert abs(fwd.loc[('x', 0), 2] - 0.20) < 0.01

def test_load_kc_estimates_missing():
    """missing path → FileNotFoundError with hint."""
    with pytest.raises(FileNotFoundError, match='parameter_fit.py'):
        load_kc_estimates('nonexistent.csv')

def test_cli_factor_validation_minimal():
    """CLI runs with required files, exits 0, writes 3 CSVs + 1 TXT."""
```

## 12. 已知陷阱

| 陷阱 | 说明 | 解决 |
|---|---|---|
| prediction_summary.csv 缺失 | V6 跳过 hit_rate/rmse/mae,但 main CSV 这些行 status='not_loaded' | OK,显式 status |
| kc_estimates_time.csv 缺失 | 跳过 Δk/Δc/Δρ/Δθ | OK |
| daily data 缺失部分日期 | forward return NaN → 跳过 | OK |
| industry l1 缺失 | by_industry 该行 status='not_loaded' | OK |
| 全市场 IC 算太久 | 接受 5-10 分钟 | 默认 limit=500,0=全 |
| pd.qcut ties 报 warning | 因子值大量重复(k̂ 离散) | `duplicates='drop'` 容忍 |
| sample_size 不够 | industry n<30 跳过 | 显式过滤 |

## 13. 决策标准(给业务)

跑完后业务看:

| 阈值 | 业务结论 |
|---|---|
| **无 factor \|IC_mean\| > 0.03** | 描述性 ≠ 预测性,归档 v3-v5.11 |
| **仅 hit_rate / RMSE 有 IC** | OOS 预测可作 α 因子,合并进 tsfresh/alpha |
| **k̂ / ĉ / ρ 有 IC,state 无 IC** | 动力学结构比状态分类更预测,聚焦 (k̂, ĉ) 因子化 |
| **Δk / Δρ 显著,静态 k̂ 无 IC** | regime 切换才是真信号,优先 v5.14 rolling 方向 |
| **多个 factor 联合有 IC** | 因子合成 / decorrelation,V7 候选 |

## 14. 不在范围 / 后续

- **V6.1**:因子相关性矩阵 / decorrelation
- **V6.2**:walk-forward IC(rolling eval_date)
- **V6.3**:因子合成(等权 / IC 加权)
- **V6.4**:行业 / 市值中性化
- **V6.5**:实时 α 信号生成 + backtrace/alpha/ 接入

## 15. Status: 📝 DRAFT — 2026-08-19

待 plan + implementer。
