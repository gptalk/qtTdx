# v0 — Parameter Fit Identifiability Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add diagnostics layer to `parameter_fit.py` (R², cond(X), regressor_corr, identification_status, fit_quality) + 4-panel HTML + summary TXT, without changing any OLS math.

**Architecture:** Single file modification (`backtrace/projection/parameter_fit.py`) + 3 new outputs (CSV extension, HTML, TXT). OLS coefficients and `d_vec`/`β̇`/`q` math frozen — only post-processing diagnostics added.

**Tech Stack:** Python 3 / numpy / pandas / plotly (already installed). No new dependencies.

## Global Constraints

- **0 modifications to 11 protected files** (per CLAUDE.md)
- **0 new dependencies** (numpy / pandas / plotly already installed)
- **Windows GBK compat**: `PYTHONIOENCODING=utf-8` required for all CLI runs
- **OLS math frozen**: `_solve_ols` core OLS call (`np.linalg.lstsq`) and `d_vec`/`u_vec`/`a_u_vec`/`a_v_vec` construction **unchanged**
- **New diagnostics only**: `condition_number`, `regressor_corr`, `r2`, `identification_status`, `fit_quality` — pure post-processing
- **Use `cond(X)` not `cond(X.T @ X)`** — `κ(X^T X) ≈ κ(X)^2` would falsely amplify ill-conditioning
- **Use `regressor_corr` not `param_correlation`** — name must reflect that it's correlation of X columns = -d and -u, not (k̂, ĉ)
- **Two status fields, not one**: `identification_status` (rank + cond) and `fit_quality` (R²) are conceptually orthogonal — split them
- **Backward compat**: `f_self_loss` field kept as alias for `f_residual_loss`; `status` verbose field kept; CSV columns only appended (no reorder)
- **Decisions from spec §4**: well_conditioned < 1e3, ill_conditioned 1e3-1e5, unidentifiable ≥ 1e5; fit_quality good ≥ 0.1, weak 0.01-0.1, poor < 0.01, uninformative NaN
- **SS_total ≈ 0 → R² = NaN**: explicit guard before division
- **default file naming**: `data/projection/kc_estimates.csv` (existing), `backtrace/outputs/kc_identifiability_distribution.html` (new), `data/projection/kc_identifiability_summary.txt` (new)

---

## File Structure

**Modified:**
- `backtrace/projection/parameter_fit.py` (~894 lines, ~80 line additions)
- `tests/test_dynamics_eigen.py` (~2150 lines, ~150 line additions for 5 tests)

**New outputs:**
- `backtrace/outputs/kc_identifiability_distribution.html` (4-panel plotly)
- `data/projection/kc_identifiability_summary.txt` (UTF-8 Chinese)

**Modified outputs:**
- `data/projection/kc_estimates.csv` (existing path, 7 columns appended: f_residual_loss / rank / condition_number / regressor_corr / r2 / identification_status / fit_quality)
- `data/projection/kc_rolling_<idx>_<stk>.csv` (existing, 5 columns appended)
- `data/projection/kc_rolling_summary.csv` (existing, 5 columns × N windows appended)
- `data/projection/kc_estimates_time.csv` (existing, 5 columns appended)

---

## Task 1: Diagnostics Layer — `_solve_ols` 8-tuple + `fit_one` 扩展

**Files:**
- Modify: `backtrace/projection/parameter_fit.py:155-185` (`_solve_ols` body)
- Modify: `backtrace/projection/parameter_fit.py:189-243` (`fit_one` body)
- Modify: `backtrace/projection/parameter_fit.py:245-329` (`fit_rolling` per-window dict)
- Modify: `backtrace/projection/parameter_fit.py:537-660` (`main_rolling_time` rows)
- Modify: `backtrace/projection/parameter_fit.py:427-486` (`main_rolling` summary columns)
- Modify: `backtrace/projection/parameter_fit.py:340-425` (`main_fit_all` output_df columns)

**Interfaces:**
- Consumes: existing `_solve_ols` signature `(a_u_vec, a_v_vec, d_vec, u_vec, beta, valid)` and `fit_one` signature unchanged
- Produces: 8-tuple from `_solve_ols` extended with `(condition_number, regressor_corr, r2)`; `fit_one` dict extended with 7 new fields

### Step 1.1: Write failing tests for `_solve_ols` 8-tuple

Create test file `tests/test_dynamics_eigen.py` new section at end (or appropriate location):

```python
# === v0 — Parameter Fit Identifiability Audit diagnostics ===
from backtrace.projection.parameter_fit import _solve_ols, _build_kinematics


def _make_ols_inputs(k_true=0.5, c_true=0.2, T=100, noise_std=1e-3, seed=42):
    """合成 Y = -k d - c u + noise,生成 2D 投影与 _solve_ols 兼容的输入。

    Returns: (a_u_vec, a_v_vec, d_vec, u_vec, beta, valid)
    """
    rng = np.random.default_rng(seed)
    # 2D 投影:Vol + Amt
    d_v = rng.standard_normal(T).cumsum() * 0.1
    d_a = rng.standard_normal(T).cumsum() * 0.1
    u_v = rng.standard_normal(T) * 0.5
    u_a = rng.standard_normal(T) * 0.5
    # β(t) 时变(2D 各自独立)
    beta_v = rng.uniform(0.5, 1.5, T)
    beta_a = rng.uniform(0.5, 1.5, T)

    # Δv = v_S - β v_M,生成 Δv 让 d = cumsum(u[:-1]) 一致
    delta_v = np.column_stack([u_v + beta_v * d_v, u_a + beta_a * d_a])
    # Δu = u(本定义)
    delta_u = np.column_stack([u_v, u_a])

    u_vec = np.column_stack([u_v, u_a])
    d_vec = np.zeros_like(u_vec)
    if T >= 2:
        d_vec[1:] = np.cumsum(u_vec[:-1], axis=0)

    # a_u, a_v 走原始 _build_kinematics 路径
    a_u_vec = np.full_like(delta_u, np.nan)
    a_v_vec = np.full_like(delta_v, np.nan)
    if T >= 2:
        a_u_vec[:-1] = np.diff(delta_u, axis=0)
        a_v_vec[:-1] = np.diff(delta_v, axis=0)

    # 注入 Y = -k d - c u + noise ⇒ a_S - β a_M = -k d - c u + noise
    # 让 noise 极小,验证 OLS 数学
    beta_arr = np.column_stack([beta_v, beta_a])
    valid = np.isfinite(a_u_vec).all(axis=1) & np.isfinite(a_v_vec).all(axis=1)
    valid &= np.isfinite(d_vec).all(axis=1) & np.isfinite(u_vec).all(axis=1)
    # 污染 a_u 使 Y = -k d - c u + 极小噪声
    noise_v = rng.normal(0, noise_std, T)
    noise_a = rng.normal(0, noise_std, T)
    a_u_vec[valid, 0] += noise_v[valid]
    a_u_vec[valid, 1] += noise_a[valid]
    return a_u_vec, a_v_vec, d_vec, u_vec, beta_arr, valid


def test_solve_ols_well_conditioned_synthetic():
    """Regression: well-conditioned 合成 OLS 精确恢复 (k, c) + R² 高 + cond 低。
    
    验证 audit 没改变 OLS 数学(用户最关心的 regression test)。
    """
    a_u, a_v, d, u, beta, valid = _make_ols_inputs(k_true=0.5, c_true=0.2, T=200, noise_std=1e-4)
    k_hat, c_hat, f_res, n, rank, cond, rcorr, r2 = _solve_ols(a_u, a_v, d, u, beta, valid)
    assert abs(k_hat - 0.5) < 0.05, f'k_hat={k_hat:.4f} 偏离 0.5 超过 tolerance'
    assert abs(c_hat - 0.2) < 0.05, f'c_hat={c_hat:.4f} 偏离 0.2 超过 tolerance'
    assert r2 > 0.9, f'R²={r2:.4f} 应 > 0.9'
    assert cond < 1e3, f'cond={cond:.2e} 应 < 1e3'
    assert rank == 2


def test_solve_ols_ill_conditioned_high_cond():
    """X 列接近共线 → cond > 1e3 → identification_status='ill_conditioned'。"""
    a_u, a_v, d, u, beta, valid = _make_ols_inputs(T=200, noise_std=1e-3)
    # 强加 d ≈ u,使 X 两列高度共线
    d[:, 0] = u[:, 0] * 0.999 + np.random.default_rng(0).standard_normal(200) * 1e-3
    d[:, 1] = u[:, 1] * 0.999 + np.random.default_rng(1).standard_normal(200) * 1e-3
    _, _, _, _, rank, cond, rcorr, _ = _solve_ols(a_u, a_v, d, u, beta, valid)
    assert cond > 1e3, f'cond={cond:.2e} 应 > 1e3'
    assert rcorr > 0.9, f'rcorr={rcorr:.4f} 应 > 0.9'


def test_solve_ols_singular_zero_variance():
    """X 某列全 0 → X^T X 不可逆 → rank < 2 → singular。"""
    a_u, a_v, d, u, beta, valid = _make_ols_inputs(T=100)
    # 让 d_vec 全 0
    d = np.zeros_like(d)
    u = np.zeros_like(u)
    _, _, _, _, rank, _, _, _ = _solve_ols(a_u, a_v, d, u, beta, valid)
    assert rank < 2


def test_solve_ols_ss_tot_near_zero():
    """Y 几乎常数 → SS_tot ≈ 0 → r2 = NaN → fit_quality='uninformative'。"""
    a_u, a_v, d, u, beta, valid = _make_ols_inputs(T=100, noise_std=1e-3)
    # 让 a_u 与 β·a_v 完全相等 → A_full = 0 → Y = 0
    a_u = beta[:, None] * a_v  # 让 A_full 全 0
    _, _, _, _, _, _, _, r2 = _solve_ols(a_u, a_v, d, u, beta, valid)
    assert np.isnan(r2), f'r2={r2} 应为 NaN'
```

**Run failing tests:**
```bash
cd C:/Users/yellow/mcp/qtTdx
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_eigen.py::test_solve_ols_well_conditioned_synthetic -v
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_eigen.py::test_solve_ols_ill_conditioned_high_cond -v
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_eigen.py::test_solve_ols_singular_zero_variance -v
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_eigen.py::test_solve_ols_ss_tot_near_zero -v
```
Expected: 4 failures ("too many values to unpack" — `_solve_ols` returns 5-tuple, not 8-tuple).

### Step 1.2: Extend `_solve_ols` to return 8-tuple

Modify `backtrace/projection/parameter_fit.py:155-185` (the `_solve_ols` function). Replace the existing function body with:

```python
def _solve_ols(a_u_vec: np.ndarray, a_v_vec: np.ndarray,
               d_vec: np.ndarray, u_vec: np.ndarray,
               beta: np.ndarray, valid: np.ndarray):
    """核心 OLS 解(内部函数,fit_one 和 fit_rolling 复用)。

    输入:从 movement CSV 重建的 2-D 向量 + valid mask。
    输出 (8-tuple):
        k_hat, c_hat, f_residual_loss, n_valid, rank,
        condition_number, regressor_corr, r2
    """
    n_valid = int(valid.sum())
    A_full = a_u_vec[valid] - beta[valid, None] * a_v_vec[valid]
    d_full = d_vec[valid]
    u_full = u_vec[valid]

    Y = np.concatenate([A_full[:, 0], A_full[:, 1]])
    X = np.zeros((2 * n_valid, 2))
    X[:n_valid, 0] = -d_full[:, 0]
    X[:n_valid, 1] = -u_full[:, 0]
    X[n_valid:, 0] = -d_full[:, 1]
    X[n_valid:, 1] = -u_full[:, 1]

    theta, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    k_hat, c_hat = float(theta[0]), float(theta[1])

    F_resid = Y - X @ theta
    f_residual_loss = float(np.mean(F_resid ** 2))

    # === v0 diagnostics (post-processing, 不动 OLS) ===

    # condition_number: cond(X), NOT cond(X.T @ X) — 后者 κ² 失真
    condition_number = float(np.linalg.cond(X)) if X.size > 0 else np.nan

    # regressor_corr: X 两列 = -d, -u 的相关系数
    if X.shape[0] >= 2 and X.shape[1] == 2:
        col0, col1 = X[:, 0], X[:, 1]
        std0, std1 = float(np.std(col0)), float(np.std(col1))
        if std0 > 1e-12 and std1 > 1e-12:
            regressor_corr = float(np.corrcoef(col0, col1)[0, 1])
        else:
            regressor_corr = np.nan
    else:
        regressor_corr = np.nan

    # R² = 1 - SS_res / SS_tot,SS_tot ≈ 0 → NaN
    y_mean = float(np.mean(Y))
    ss_tot = float(np.sum((Y - y_mean) ** 2))
    ss_res = float(np.sum(F_resid ** 2))
    if ss_tot <= 1e-12:
        r2 = np.nan
    else:
        r2 = 1.0 - ss_res / ss_tot

    return (k_hat, c_hat, f_residual_loss, n_valid, int(rank),
            condition_number, regressor_corr, r2)
```

**Run tests:**
```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_eigen.py::test_solve_ols_well_conditioned_synthetic tests/test_dynamics_eigen.py::test_solve_ols_ill_conditioned_high_cond tests/test_dynamics_eigen.py::test_solve_ols_singular_zero_variance tests/test_dynamics_eigen.py::test_solve_ols_ss_tot_near_zero -v
```
Expected: 4 PASS.

### Step 1.3: Update `fit_one` to consume 8-tuple and emit 7 new fields

Modify `backtrace/projection/parameter_fit.py:189-243` (`fit_one` body). Replace the `_solve_ols` call and return dict:

```python
def fit_one(movement_csv: str, stock_tag: str, index_tag: str,
            min_valid_days: int = 20, clip_extreme: float = 10.0):
    """对一只股票的全样本做闭式 OLS,返回 dict。

    新增 v0 字段: rank, condition_number, regressor_corr, r2,
                  identification_status, fit_quality, f_residual_loss
    """
    loaded, err = _load_movement(movement_csv, stock_tag, index_tag)
    if loaded is None:
        return {
            'k_hat': np.nan, 'c_hat': np.nan,
            'f_self_loss': np.nan, 'f_residual_loss': np.nan,
            'n_valid_days': 0,
            'rank': 0, 'condition_number': np.nan,
            'regressor_corr': np.nan, 'r2': np.nan,
            'identification_status': 'singular',
            'fit_quality': 'uninformative',
            'status': err,
        }
    df, delta_u, delta_v, beta = loaded
    u_vec, d_vec, a_u_vec, a_v_vec = _build_kinematics(delta_u, delta_v, beta)

    valid = (
        np.isfinite(a_u_vec).all(axis=1)
        & np.isfinite(a_v_vec).all(axis=1)
        & np.isfinite(d_vec).all(axis=1)
        & np.isfinite(u_vec).all(axis=1)
    )
    n_valid = int(valid.sum())
    if n_valid < max(3, min_valid_days):
        return {
            'k_hat': np.nan, 'c_hat': np.nan,
            'f_self_loss': np.nan, 'f_residual_loss': np.nan,
            'n_valid_days': n_valid,
            'rank': 0, 'condition_number': np.nan,
            'regressor_corr': np.nan, 'r2': np.nan,
            'identification_status': 'singular',
            'fit_quality': 'uninformative',
            'status': f'too_few_days ({n_valid} < {min_valid_days})',
        }

    k_hat, c_hat, f_residual_loss, _, rank, condition_number, regressor_corr, r2 = _solve_ols(
        a_u_vec, a_v_vec, d_vec, u_vec, beta, valid,
    )

    # === classification ===
    finite = np.isfinite(k_hat) and np.isfinite(c_hat)
    extreme = abs(k_hat) > clip_extreme or abs(c_hat) > clip_extreme

    # identification_status: 仅看 rank + cond
    if not finite or rank < 2:
        identification_status = 'singular'
    elif condition_number >= 1e5:
        identification_status = 'unidentifiable'
    elif condition_number >= 1e3:
        identification_status = 'ill_conditioned'
    else:
        identification_status = 'well_conditioned'

    # fit_quality: 仅看 R²
    if not np.isfinite(r2):
        fit_quality = 'uninformative'
    elif r2 < 0.01:
        fit_quality = 'poor'
    elif r2 < 0.1:
        fit_quality = 'weak'
    else:
        fit_quality = 'good'

    # 旧 status (verbose, 向后兼容)
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
        'f_self_loss': f_residual_loss,  # alias for backward compat
        'f_residual_loss': f_residual_loss,
        'n_valid_days': n_valid,
        'rank': rank,
        'condition_number': condition_number,
        'regressor_corr': regressor_corr,
        'r2': r2,
        'identification_status': identification_status,
        'fit_quality': fit_quality,
        'status': status,
    }
```

### Step 1.4: Update `fit_rolling` per-window dict

Modify `backtrace/projection/parameter_fit.py:245-329` (`fit_rolling`). Update the `_solve_ols` call unpacking and the dict construction. Specifically:

In the `n_valid < 3` early return:
```python
out.append({
    'window': w,
    'window_start': str(df['Date'].iloc[s])[:10] if s < T else '',
    'window_end': str(df['Date'].iloc[T - 1])[:10] if T > 0 else '',
    'k_hat': np.nan, 'c_hat': np.nan,
    'f_residual_loss': np.nan,
    'n_valid_days': n_valid,
    'rank': 0, 'condition_number': np.nan,
    'regressor_corr': np.nan, 'r2': np.nan,
    'identification_status': 'singular',
    'fit_quality': 'uninformative',
    'status': f'too_few_days ({n_valid})',
})
```

In the `solve_failed` exception handler:
```python
out.append({
    'window': w,
    'window_start': str(df['Date'].iloc[s])[:10],
    'window_end': str(df['Date'].iloc[T - 1])[:10],
    'k_hat': np.nan, 'c_hat': np.nan,
    'f_residual_loss': np.nan,
    'n_valid_days': n_valid,
    'rank': 0, 'condition_number': np.nan,
    'regressor_corr': np.nan, 'r2': np.nan,
    'identification_status': 'singular',
    'fit_quality': 'uninformative',
    'status': f'solve_failed: {type(e).__name__}: {e}',
})
```

In the success branch (after `_solve_ols` call):
```python
k_hat, c_hat, f_residual_loss, _, rank, condition_number, regressor_corr, r2 = _solve_ols(
    a_u_vec[sub], a_v_vec[sub], d_vec[sub], u_vec[sub], beta[sub], valid,
)
```

Then add classification after the existing `extreme` check:
```python
# identification_status
if not finite or rank < 2:
    identification_status = 'singular'
elif condition_number >= 1e5:
    identification_status = 'unidentifiable'
elif condition_number >= 1e3:
    identification_status = 'ill_conditioned'
else:
    identification_status = 'well_conditioned'

# fit_quality
if not np.isfinite(r2):
    fit_quality = 'uninformative'
elif r2 < 0.01:
    fit_quality = 'poor'
elif r2 < 0.1:
    fit_quality = 'weak'
else:
    fit_quality = 'good'
```

And the success-path dict:
```python
out.append({
    'window': w,
    'window_start': str(df['Date'].iloc[s])[:10],
    'window_end': str(df['Date'].iloc[T - 1])[:10],
    'k_hat': k_hat, 'c_hat': c_hat,
    'f_residual_loss': f_residual_loss,
    'n_valid_days': n_valid,
    'rank': rank,
    'condition_number': condition_number,
    'regressor_corr': regressor_corr,
    'r2': r2,
    'identification_status': identification_status,
    'fit_quality': fit_quality,
    'status': status,
})
```

### Step 1.5: Update `main_rolling_time` rows

Modify `backtrace/projection/parameter_fit.py:537-660` (`main_rolling_time`). In all 3 row-construction dictionaries (too_few_days, too_few_valid, solve_failed, success), add 7 new fields:

```python
# After 'n_valid_days': ..., add:
'rank': 0, 'condition_number': np.nan,
'regressor_corr': np.nan, 'r2': np.nan,
'identification_status': 'singular',
'fit_quality': 'uninformative',
```

For the success path, unpack the 8-tuple and run classification:
```python
k_hat, c_hat, f_residual_loss, _, rank, condition_number, regressor_corr, r2 = _solve_ols(...)
# ... classification logic (same as fit_one) ...
```

Update the row dict to include all 7 new fields.

### Step 1.6: Update `main_rolling` summary columns

Modify `backtrace/projection/parameter_fit.py:427-486` (`main_rolling`). Update `summary_cols` to append 5 new fields per window:

```python
for w in windows:
    summary_cols.extend([
        f'k_{w}', f'c_{w}', f'f2_{w}', f'n_{w}', f'status_{w}',
        f'cond_{w}', f'rcorr_{w}', f'r2_{w}', f'idstatus_{w}', f'fquality_{w}',
    ])
```

In the per-stock `srow` dict, populate:
```python
srow[f'cond_{w}'] = r['condition_number']
srow[f'rcorr_{w}'] = r['regressor_corr']
srow[f'r2_{w}'] = r['r2']
srow[f'idstatus_{w}'] = r['identification_status']
srow[f'fquality_{w}'] = r['fit_quality']
```

### Step 1.7: Update `main_fit_all` output columns

Modify `backtrace/projection/parameter_fit.py:340-425` (`main_fit_all`). Update the `out_df` column list to include 7 new fields AFTER the existing 10:

```python
out_df = pd.DataFrame(rows, columns=[
    'code', 'name', 'index_code', 'index_tag', 'stock_tag',
    'k_hat', 'c_hat', 'f_self_loss', 'n_valid_days', 'status',
    'f_residual_loss', 'rank', 'condition_number',
    'regressor_corr', 'r2',
    'identification_status', 'fit_quality',
])
```

Update the summary print lines to also print `R² / cond / identification_status / fit_quality` for at least one stock in the loop.

### Step 1.8: Update `main_rolling_time` output columns

Modify `backtrace/projection/parameter_fit.py:660-665` (the final `out = pd.DataFrame(rows, columns=[...])`):

```python
out = pd.DataFrame(rows, columns=[
    'asof_date', 'code', 'name', 'index_code', 'index_tag', 'stock_tag',
    'k_hat', 'c_hat', 'f_residual_loss', 'n_valid_days', 'status',
    'rank', 'condition_number', 'regressor_corr', 'r2',
    'identification_status', 'fit_quality',
])
```

### Step 1.9: Run smoke test (5 stocks)

```bash
cd C:/Users/yellow/mcp/qtTdx
PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --limit 5
```

Expected: existing CSV path written, 17 columns in `kc_estimates.csv`, all 4 tests still pass.

```bash
PYTHONIOENCODING=utf-8 head -2 data/projection/kc_estimates.csv
```
Expected: header line has 17 columns including `condition_number`, `regressor_corr`, `r2`, `identification_status`, `fit_quality`.

### Step 1.10: Commit

```bash
git add backtrace/projection/parameter_fit.py tests/test_dynamics_eigen.py
git commit -m "feat(parameter-fit): v0 diagnostics — R², cond(X), regressor_corr, identification_status, fit_quality"
```

---

## Task 2: HTML Distribution + TXT Summary + 5 Tests + CLI Smoke

**Files:**
- Modify: `backtrace/projection/parameter_fit.py` (add 2 functions: `build_identifiability_distribution_html` + `write_identifiability_summary_txt`; integrate call into `main_fit_all`)
- Modify: `tests/test_dynamics_eigen.py` (5 new tests including CLI smoke)

**Interfaces:**
- Consumes: `kc_estimates.csv` produced by `main_fit_all` (Task 1)
- Produces: `backtrace/outputs/kc_identifiability_distribution.html` (4-panel plotly) + `data/projection/kc_identifiability_summary.txt` (UTF-8 Chinese)

### Step 2.1: Write failing test for HTML distribution

Append to `tests/test_dynamics_eigen.py`:

```python
def test_build_identifiability_distribution_html_synthetic():
    """给定 100 行合成 kc_estimates,产出 4-panel HTML,文件存在 + plotly 加载。"""
    from backtrace.projection.parameter_fit import build_identifiability_distribution_html
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'code': [f'stk_{i:04d}' for i in range(100)],
        'k_hat': rng.normal(0, 1, 100),
        'c_hat': rng.normal(0, 1, 100),
        'r2': rng.uniform(0, 0.2, 100),
        'condition_number': np.exp(rng.uniform(2, 12, 100)),
        'identification_status': rng.choice(
            ['well_conditioned', 'ill_conditioned', 'unidentifiable', 'singular'], 100,
        ),
        'fit_quality': rng.choice(['good', 'weak', 'poor', 'uninformative'], 100),
    })
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, 'kc_id.html')
        build_identifiability_distribution_html(df, out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 5000  # plotly 最小 HTML 也不止 5k
        # 拆开 HTML 验证有 4 子图(找 subplot 关键字)
        with open(out_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'R²' in content or 'R^2' in content
        assert 'cond' in content or 'Condition' in content
```

### Step 2.2: Write failing test for TXT summary

```python
def test_write_identifiability_summary_txt_synthetic():
    """给定 100 行合成 kc_estimates,产出 TXT,关键字段全部出现。"""
    from backtrace.projection.parameter_fit import write_identifiability_summary_txt
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'code': [f'stk_{i:04d}' for i in range(100)],
        'r2': rng.uniform(0, 0.2, 100),
        'condition_number': np.exp(rng.uniform(2, 12, 100)),
        'identification_status': rng.choice(
            ['well_conditioned', 'ill_conditioned', 'unidentifiable', 'singular'], 100,
        ),
        'fit_quality': rng.choice(['good', 'weak', 'poor', 'uninformative'], 100),
    })
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, 'kc_id.txt')
        write_identifiability_summary_txt(df, out_path)
        assert os.path.exists(out_path)
        with open(out_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 关键字段
        for k in ['Total:', 'Well conditioned:', 'Ill conditioned:',
                  'Unidentifiable:', 'Singular:',
                  'Good:', 'Weak:', 'Poor:', 'Uninformative:',
                  'R²', 'Condition Number', 'median', 'p25', 'p75']:
            assert k in content, f'missing key: {k}'
```

### Step 2.3: Write failing test for CLI smoke

```python
def test_cli_smoke_audit_outputs(tmp_path_factory):
    """CLI --limit 5 跑通 + CSV 含 17 列 + HTML 生成 + TXT 生成。"""
    # 用 limit 5(限制 < 5 文件,既有 data/projection/movement_*.csv)
    import subprocess
    result = subprocess.run(
        [sys.executable, 'backtrace/projection/parameter_fit.py', '--limit', '5'],
        capture_output=True, text=True, timeout=120, encoding='utf-8', errors='replace',
    )
    assert result.returncode == 0, f'stderr: {result.stderr}'
    # CSV 17 列
    csv_path = 'data/projection/kc_estimates.csv'
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    assert len(df.columns) == 17
    expected_cols = {'condition_number', 'r2', 'regressor_corr',
                    'identification_status', 'fit_quality'}
    assert expected_cols.issubset(set(df.columns))
    # HTML
    html_path = 'backtrace/outputs/kc_identifiability_distribution.html'
    assert os.path.exists(html_path)
    assert os.path.getsize(html_path) > 5000
    # TXT
    txt_path = 'data/projection/kc_identifiability_summary.txt'
    assert os.path.exists(txt_path)
```

### Step 2.4: Run failing tests

```bash
cd C:/Users/yellow/mcp/qtTdx
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_eigen.py::test_build_identifiability_distribution_html_synthetic -v
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_eigen.py::test_write_identifiability_summary_txt_synthetic -v
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_eigen.py::test_cli_smoke_audit_outputs -v
```
Expected: 3 failures (import error: `build_identifiability_distribution_html` / `write_identifiability_summary_txt` not yet implemented).

### Step 2.5: Implement `build_identifiability_distribution_html`

Add to `backtrace/projection/parameter_fit.py` (after `plot_rolling_aggregate`):

```python
def build_identifiability_distribution_html(kc_df: pd.DataFrame, output_path: str) -> str:
    """4 子图 plotly:R² 直方图 / cond 直方图 / R² vs |k̂| / (k̂, ĉ) 散点按 R² 着色。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    r2 = pd.to_numeric(kc_df['r2'], errors='coerce').to_numpy()
    cond = pd.to_numeric(kc_df['condition_number'], errors='coerce').to_numpy()
    k_abs = np.abs(pd.to_numeric(kc_df['k_hat'], errors='coerce').to_numpy())
    k = pd.to_numeric(kc_df['k_hat'], errors='coerce').to_numpy()
    c = pd.to_numeric(kc_df['c_hat'], errors='coerce').to_numpy()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'R² 直方图(模型解释力)',
            'cond(X) 直方图(数值可识别性,对数轴)',
            'R² vs |k̂| 散点(低 R² → 参数爆炸?)',
            '(k̂, ĉ) 散点(颜色 = R²)',
        ),
        vertical_spacing=0.15, horizontal_spacing=0.10,
    )

    # (1,1) R² histogram
    r2_finite = r2[np.isfinite(r2)]
    fig.add_trace(go.Histogram(
        x=r2_finite, nbinsx=50, name='R²',
        marker_color='rgba(46, 204, 113, 0.7)',
    ), row=1, col=1)

    # (1,2) cond histogram (log scale)
    cond_finite = cond[np.isfinite(cond) & (cond > 0)]
    fig.add_trace(go.Histogram(
        x=np.log10(cond_finite), nbinsx=50, name='log10(cond)',
        marker_color='rgba(52, 152, 219, 0.7)',
    ), row=1, col=2)
    # 红虚线:1e3 / 1e5
    for boundary in [3, 5]:
        fig.add_trace(go.Scatter(
            x=[boundary, boundary], y=[0, 1], mode='lines',
            line=dict(color='red', dash='dash', width=1.5),
            showlegend=False, yaxis='y2',
        ), row=1, col=2)

    # (2,1) R² vs |k̂|
    valid_mask = np.isfinite(r2) & np.isfinite(k_abs)
    fig.add_trace(go.Scatter(
        x=r2[valid_mask], y=k_abs[valid_mask],
        mode='markers', name='|k̂|',
        marker=dict(size=5, color='rgba(155, 89, 182, 0.5)'),
        showlegend=False,
    ), row=2, col=1)

    # (2,2) (k̂, ĉ) scatter, color = R²
    valid_mask2 = np.isfinite(k) & np.isfinite(c) & np.isfinite(r2)
    fig.add_trace(go.Scatter(
        x=k[valid_mask2], y=c[valid_mask2],
        mode='markers', name='(k̂, ĉ)',
        marker=dict(
            size=5, color=r2[valid_mask2],
            colorscale='RdYlGn', cmin=0, cmax=0.2,
            colorbar=dict(title='R²', x=1.02, len=0.5, y=0.2),
            showscale=True,
        ),
        showlegend=False,
    ), row=2, col=2)

    fig.update_xaxes(title_text='R²', row=1, col=1)
    fig.update_xaxes(title_text='log10(cond(X))', row=1, col=2)
    fig.update_xaxes(title_text='R²', row=2, col=1)
    fig.update_xaxes(title_text='k̂', row=2, col=2)
    fig.update_yaxes(title_text='频数', row=1, col=1)
    fig.update_yaxes(title_text='频数', row=1, col=2)
    fig.update_yaxes(title_text='|k̂|', type='log', row=2, col=1)
    fig.update_yaxes(title_text='ĉ', row=2, col=2)

    fig.update_layout(
        template='plotly_dark', height=900, width=1400,
        title_text=f'Parameter Fit Identifiability Audit (N={len(kc_df)})',
    )
    fig.write_html(output_path)
    return output_path
```

### Step 2.6: Implement `write_identifiability_summary_txt`

```python
def write_identifiability_summary_txt(kc_df: pd.DataFrame, output_path: str) -> str:
    """UTF-8 中文汇总:分类计数 + 分布统计 + recommendation。"""
    from datetime import datetime
    n_total = len(kc_df)
    id_status = kc_df['identification_status'].fillna('singular')
    fq = kc_df['fit_quality'].fillna('uninformative')
    n_well = int((id_status == 'well_conditioned').sum())
    n_ill = int((id_status == 'ill_conditioned').sum())
    n_unid = int((id_status == 'unidentifiable').sum())
    n_sing = int((id_status == 'singular').sum())
    n_good = int((fq == 'good').sum())
    n_weak = int((fq == 'weak').sum())
    n_poor = int((fq == 'poor').sum())
    n_uninf = int((fq == 'uninformative').sum())

    r2 = pd.to_numeric(kc_df['r2'], errors='coerce').to_numpy()
    r2_finite = r2[np.isfinite(r2)]
    cond = pd.to_numeric(kc_df['condition_number'], errors='coerce').to_numpy()
    cond_finite = cond[np.isfinite(cond) & (cond > 0)]

    pct = lambda n: 100.0 * n / max(n_total, 1)
    well_pct = pct(n_well)
    if well_pct > 50:
        rec = 'well_conditioned 占比 > 50% → V6 在 well_conditioned 子集重跑 (spec v0.2)'
    elif well_pct >= 10:
        rec = 'well_conditioned 占比 10-50% → V6 因子降级,只看 well_conditioned 子集'
    else:
        rec = 'well_conditioned 占比 < 10% → 动力学模型作为方法论不可用,收口'

    lines = [
        '=' * 50,
        'Parameter Fit Identifiability Audit',
        '=' * 50,
        f'Run date:  {datetime.now().strftime("%Y-%m-%d")}',
        f'Total stocks:    {n_total}',
        '', '--- Identification Status ---',
        f'  Well conditioned:  {n_well} ({well_pct:.1f}%)',
        f'  Ill conditioned:   {n_ill} ({pct(n_ill):.1f}%)',
        f'  Unidentifiable:    {n_unid} ({pct(n_unid):.1f}%)',
        f'  Singular:          {n_sing} ({pct(n_sing):.1f}%)',
        '', '--- Fit Quality ---',
        f'  Good:              {n_good} ({pct(n_good):.1f}%)',
        f'  Weak:              {n_weak} ({pct(n_weak):.1f}%)',
        f'  Poor:              {n_poor} ({pct(n_poor):.1f}%)',
        f'  Uninformative:     {n_uninf} ({pct(n_uninf):.1f}%)',
        '', '--- R² Distribution ---',
    ]
    if len(r2_finite) > 0:
        lines.extend([
            f'  median = {np.median(r2_finite):.4f}',
            f'  p25    = {np.percentile(r2_finite, 25):.4f}',
            f'  p75    = {np.percentile(r2_finite, 75):.4f}',
        ])
    else:
        lines.append('  (no finite R²)')
    lines.extend([
        '', '--- Condition Number Distribution ---',
    ])
    if len(cond_finite) > 0:
        lines.extend([
            f'  median = {np.median(cond_finite):.2e}',
            f'  p25    = {np.percentile(cond_finite, 25):.2e}',
            f'  p75    = {np.percentile(cond_finite, 75):.2e}',
        ])
    else:
        lines.append('  (no finite condition number)')
    lines.extend([
        '', '--- Recommendation ---',
        f'  {rec}',
        '=' * 50,
    ])
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return output_path
```

### Step 2.7: Wire into `main_fit_all`

In `main_fit_all`, after the existing `out_df.to_csv(...)` line, add:

```python
# === v0: Identifiability Audit outputs ===
HTML_OUT_DIR = 'backtrace/outputs'
os.makedirs(HTML_OUT_DIR, exist_ok=True)
html_path = os.path.join(HTML_OUT_DIR, 'kc_identifiability_distribution.html')
build_identifiability_distribution_html(out_df, html_path)
txt_path = os.path.join(CSV_OUT_DIR, 'kc_identifiability_summary.txt')
write_identifiability_summary_txt(out_df, txt_path)
print(f'\n  v0 audit HTML: {html_path}')
print(f'  v0 audit TXT:  {txt_path}')
```

### Step 2.8: Run all 5 tests

```bash
cd C:/Users/yellow/mcp/qtTdx
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_eigen.py -v -k "solve_ols or build_identifiability or write_identifiability or cli_smoke_audit"
```
Expected: 7 PASS (4 OLS + 2 outputs + 1 CLI smoke).

### Step 2.9: Full smoke test (--limit 10)

```bash
PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --limit 10
```

Expected:
- stdout 中每只票多打印一行 diagnostics summary
- `data/projection/kc_estimates.csv` 17 列
- `backtrace/outputs/kc_identifiability_distribution.html` 4 子图存在
- `data/projection/kc_identifiability_summary.txt` 中文汇总存在

```bash
PYTHONIOENCODING=utf-8 cat data/projection/kc_identifiability_summary.txt
```
Expected:看到 4 个 identification_status 计数 + 4 个 fit_quality 计数 + R²/cond 中位数 + recommendation。

### Step 2.10: Run full test suite

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_eigen.py -v
```
Expected: 95+ tests PASS (90 baseline + 4 OLS + 2 outputs + 1 CLI smoke).

### Step 2.11: Commit

```bash
git add backtrace/projection/parameter_fit.py tests/test_dynamics_eigen.py
git commit -m "feat(parameter-fit): v0 — 4-panel HTML distribution + UTF-8 TXT summary + 5 tests"
```

---

## Self-Review

1. **Spec coverage:**
   - §3.1 `_solve_ols` 8-tuple → Step 1.2 ✓
   - §3.2 `fit_one` 7 new fields → Step 1.3 ✓
   - §3.3 CSV extension (17 cols) → Step 1.7 ✓
   - §4.1 identification_status thresholds → Step 1.3 ✓
   - §4.2 fit_quality thresholds → Step 1.3 ✓
   - §4.3 Gate thresholds → Step 2.6 rec logic ✓
   - §5.1 CSV extension → Step 1.7 ✓
   - §5.2 4-panel HTML → Step 2.5 ✓
   - §5.3 TXT summary → Step 2.6 ✓
   - §6 CLI no new flags → Step 1.9, 2.9 ✓
   - §7 5 tests → Step 1.1, 2.1, 2.2, 2.3 ✓

2. **Placeholder scan:** No TBDs.

3. **Type consistency:** 8-tuple unpacks consistent across `fit_one` / `fit_rolling` / `main_rolling_time`.

4. **Backward compat:** `f_self_loss` alias + `status` verbose preserved + CSV columns only appended.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-19-parameter-fit-identifiability-audit.md`. Two execution options:

1. **Subagent-Driven (recommended)** — Fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
