# V0.2-D — Dynamics OOS Reversal Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose why Model 2's free-q estimation improves in-sample R² 9× (0.019→0.171) but reverses OOS Spearman IC (+0.487 → −0.506), via 4 diagnostic groups (parameter stability, X-X collinearity, residual structure, OOS boundary) and a drift × collinearity scatter.

**Architecture:** Extend `backtrace/projection/ablation_fit.py` with 9 new diagnostic functions + a 33-column CSV schema for Models 2/3. Audit-fix Phase 0 ensures the 3 ΔIC statistics all persist. New CLI `backtrace/projection/v0_2_d_decompose.py` orchestrates the full diagnostic run. Decision gates are distribution reports (median / p25 / p75 / tail probability), not PASS/FAIL — interpretation routes to V0.2-E.

**Tech Stack:** Python 3.x, numpy, pandas, scipy.stats, plotly (existing); no new dependencies.

## Global Constraints

(Verbatim from spec §3, §4, §5, §6, §7, §11, §12, §14; binding for every task.)

- **Three strictly separated computations, no leakage** (§3):
  - `θ_train = argmin ||X_train·θ − Y_train||²`
  - `θ_test_fit = argmin ||X_test·θ − Y_test||²` (diagnostic only)
  - `Ŷ_oos = X_test · θ_train` (only valid generalization)
- **`θ_test_fit` MUST NOT** appear in `oos_r2` / `oos_ic` (= `ic_real`) / verdict / Panel 5 / any downstream consumer in `parameter_fit.py`, `prediction_ode.py`, `dynamics_*.py` (§7).
- **Three R² are separately defined and named** (§7):
  - `train_fit_r2` = R²(Y_train, X_train · θ_train) — semantic rename of existing `r2` column
  - `test_fit_r2` = R²(Y_test, X_test · θ_test_fit) — diagnostic only
  - `oos_r2` = R²(Y_test, X_test · θ_train) — only valid OOS R²
- **No `param_drift_l2`** (§4, §14): q is dimensionless, k/c have units of inverse-time; L2 mix is unit-incoherent. Save `q_drift`, `k_drift`, `c_drift` separately.
- **No `corr(q̂, X)`** (§5): scalar × time series is statistically undefined. Use `corr(X_β, X_d)`, `corr(X_β, X_u)`, `corr(X_d, X_u)` per stock.
- **CSV schema extended to 33 columns** for Models 2/3; Models 0/1 keep original 18 columns (§8):
  ```
  [18 existing] code, name, index_code, index_tag, stock_tag,
  n_train, n_test, condition_number, regressor_corr, r2 (→train_fit_r2),
  identification_status, fit_quality,
  q_hat, k_hat, c_hat, f_self_loss, ic_real, ic_null
  [9 new — A] q_train_fit, k_train_fit, c_train_fit,
              q_test_fit, k_test_fit, c_test_fit,
              q_drift, k_drift, c_drift
  [3 new — B] train_fit_r2, test_fit_r2, oos_r2
  [3 new — C] corr_x_beta_d, corr_x_beta_u, corr_x_d_u
  [3 new — D] corr_F_beta_aM, corr_F_d, corr_F_u
  ```
- **Audit fix** (§11): `summarize_ablation` writes 3 ΔIC stats explicitly — `median_delta_ic` (A: median of differences, existing), `diff_of_medians_delta_ic` (B: diff of medians, **NEW**), `delta_ic_vs_m0` (C: delta vs M0, existing). 4×10 → 4×11 metric matrix.
- **Test fixtures** (§12): use `{stock_tag}` substitution via `_read_movement`. NEVER literal `Move_Delta_Vol_stk`. NEVER single-stock assertions on cross-sectional statistics. NaN guard on aggregated metrics.
- **Out of scope** (§14): Lasso/Elastic Net, industry vs market, new dynamics terms, prediction target change, trading strategy, V6 rerun, `param_drift_l2`, cross-stock `corr_s(q_s, E_β,s)`, standardized drift `D_θ`, PASS/FAIL verdict.
- **Files NOT to modify**: `_solve_ols` in `parameter_fit.py`, `prediction_ode.py`, `dynamics_*.py`, `gp_factor_mining/*`.
- **Files reusable as-is**: `parameter_fit.py:_load_movement` (NOT used here; we use `ablation_fit._read_movement`), `parameter_fit.py:_build_kinematics`, `parameter_fit.py:build_identifiability_distribution_html`.
- **Windows GBK**: all CLI invocations must include `PYTHONIOENCODING=utf-8`.
- **Conda python path**: `/c/Users/yellow/.conda/envs/venv/python.exe`.
- **Tests in**: `tests/test_dynamics_eigen.py` (existing file, append).
- **CSV location**: `data/projection_v01_d/` (NOT overwrite V0.1's `data/projection_v01/`). Phase 0 overwrites V0.1's summary CSV (audit hygiene).

---

## Task 1: Phase 0 — Audit Hygiene (3 ΔIC stats persisted)

**Files:**
- Modify: `backtrace/projection/ablation_fit.py:445-602` (extend `summarize_ablation` and `write_recommendation_txt`)
- Modify: `tests/test_dynamics_eigen.py` (append `test_summarize_ablation_writes_three_delta_ic`)

**Interfaces:**
- Consumes: existing `summarize_ablation(csv_paths: dict) -> pd.DataFrame` returns 4×10 metrics; we extend to 4×11.
- Produces:
  - New row in summary DataFrame: `'diff_of_medians_delta_ic'` (= B: `median(IC_real) − median(IC_null)` per model, all 4 models).
  - Modified `write_recommendation_txt` reads all ΔIC numbers from the summary DataFrame (no inline recomputation).

**Step 1.1: Write the failing test**

Append to `tests/test_dynamics_eigen.py`:

```python
def test_summarize_ablation_writes_three_delta_ic():
    """V0.2-D audit fix: 3 ΔIC statistics all persisted in summary CSV."""
    import tempfile, os
    from projection.ablation_fit import summarize_ablation
    with tempfile.TemporaryDirectory() as td:
        # Write 4 stub CSVs with ic_real and ic_null columns
        for m in range(4):
            stub = pd.DataFrame({
                'code': [f'stk{m:06d}'],
                'r2': [0.05] * 10,
                'condition_number': [10.0] * 10,
                'ic_real': [0.5 + 0.01 * i for i in range(10)],
                'ic_null': [0.01 * i for i in range(10)],
                'q_hat': [0.5] * 10,
            })
            stub.to_csv(os.path.join(td, f'kc_estimates_model{m}.csv'), index=False)
        summary = summarize_ablation({m: os.path.join(td, f'kc_estimates_model{m}.csv') for m in range(4)})
        # 3 ΔIC rows must exist
        assert 'median_delta_ic' in summary.index
        assert 'diff_of_medians_delta_ic' in summary.index, \
            "diff_of_medians_delta_ic missing — verdict B stat not persisted"
        assert 'delta_ic_vs_m0' in summary.index
        # Each row has all 4 model columns
        for row in ('median_delta_ic', 'diff_of_medians_delta_ic', 'delta_ic_vs_m0'):
            assert all(summary.loc[row, f'model_{m}'] is not None for m in range(4))
        # diff_of_medians matches direct computation (B definition)
        for m in range(4):
            stub = pd.read_csv(os.path.join(td, f'kc_estimates_model{m}.csv'))
            expected = float(np.median(stub['ic_real']) - np.median(stub['ic_null']))
            actual = float(summary.loc['diff_of_medians_delta_ic', f'model_{m}'])
            assert abs(actual - expected) < 1e-9, \
                f"diff_of_medians_delta_ic[m={m}] mismatch: {actual} vs {expected}"
```

**Step 1.2: Run test to verify it fails**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py::test_summarize_ablation_writes_three_delta_ic -v
```

Expected: FAIL with `KeyError: 'diff_of_medians_delta_ic'`.

**Step 1.3: Extend `summarize_ablation` to write the third ΔIC stat**

In `backtrace/projection/ablation_fit.py`, locate `summarize_ablation` (line ~445). Modify the `metrics` list to include `'diff_of_medians_delta_ic'`, and add its computation right after `median_delta_ic`:

```python
def summarize_ablation(csv_paths: dict) -> pd.DataFrame:
    """4×11 metric matrix from 4 per-model CSVs (V0.2-D Phase 0: +1 ΔIC stat).

    csv_paths: {0: path_model0, 1: ..., 2: ..., 3: ...}
    Returns DataFrame with rows = metrics, cols = model_N.

    ΔIC stats (3):
      - median_delta_ic (A): median_s(IC_real_s − IC_null_s) per model
      - diff_of_medians_delta_ic (B): median(IC_real) − median(IC_null) per model  ← NEW (V0.2-D audit fix)
      - delta_ic_vs_m0 (C): median_s(IC_real_new,s − IC_real_M0,s) per stock, per model vs M0
    """
    metrics = ['median_r2', 'p25_r2', 'p75_r2', 'median_cond',
               'median_ic_real', 'median_ic_null', 'median_delta_ic',
               'diff_of_medians_delta_ic',           # ← NEW (V0.2-D Phase 0)
                'median_abs_q_minus_1', 'delta_r2_vs_m0', 'delta_ic_vs_m0']
    summary = pd.DataFrame(index=metrics, columns=['model_0', 'model_1', 'model_2', 'model_3'])

    r2_m0 = None
    ic_real_m0 = None
    for m in range(4):
        df = pd.read_csv(csv_paths[m])
        r2 = df['r2'].dropna()
        cond = df['condition_number'].dropna()
        ic_real = df['ic_real'].dropna()
        ic_null = df['ic_null'].dropna()
        q_hat = df['q_hat'].dropna()
        summary.loc['median_r2', f'model_{m}'] = float(np.median(r2)) if len(r2) else np.nan
        summary.loc['p25_r2', f'model_{m}'] = float(np.percentile(r2, 25)) if len(r2) else np.nan
        summary.loc['p75_r2', f'model_{m}'] = float(np.percentile(r2, 75)) if len(r2) else np.nan
        summary.loc['median_cond', f'model_{m}'] = float(np.median(cond)) if len(cond) else np.nan
        summary.loc['median_ic_real', f'model_{m}'] = float(np.median(ic_real)) if len(ic_real) else np.nan
        summary.loc['median_ic_null', f'model_{m}'] = float(np.median(ic_null)) if len(ic_null) else np.nan
        summary.loc['median_delta_ic', f'model_{m}'] = (
            float(np.median(ic_real - ic_null)) if len(ic_real) and len(ic_null) else np.nan
        )
        # NEW: difference of medians (V0.2-D audit fix)
        summary.loc['diff_of_medians_delta_ic', f'model_{m}'] = (
            float(np.median(ic_real) - np.median(ic_null)) if len(ic_real) and len(ic_null) else np.nan
        )
        if m in (2, 3):
            summary.loc['median_abs_q_minus_1', f'model_{m}'] = (
                float(np.median(np.abs(q_hat - 1.0))) if len(q_hat) else np.nan
            )
        if m == 0:
            r2_m0 = r2.values if len(r2) else None
            ic_real_m0 = ic_real.values if len(ic_real) else None
        if m >= 1 and r2_m0 is not None and len(r2):
            min_len = min(len(r2), len(r2_m0))
            summary.loc['delta_r2_vs_m0', f'model_{m}'] = float(np.median(r2.values[:min_len] - r2_m0[:min_len]))
        if m >= 1 and ic_real_m0 is not None and len(ic_real):
            min_len = min(len(ic_real), len(ic_real_m0))
            summary.loc['delta_ic_vs_m0', f'model_{m}'] = float(np.median(ic_real.values[:min_len] - ic_real_m0[:min_len]))

    return summary
```

**Step 1.4: Modify `write_recommendation_txt` to read from summary (audit hygiene)**

In `write_recommendation_txt` (line ~534), replace the inline ΔIC computation with a CSV read. Find this block:

```python
    # Step 3: placebo delta = median(IC_real_M3) − median(IC_null_M3)  (per spec §10, difference of medians)
    delta_ic_m3 = float(summary_df.loc['median_ic_real', 'model_3'] - summary_df.loc['median_ic_null', 'model_3'])
```

Replace with:

```python
    # V0.2-D Phase 0 audit fix: read all ΔIC numbers from the summary DataFrame (no inline recompute)
    delta_ic_m3 = float(summary_df.loc['diff_of_medians_delta_ic', 'model_3'])
```

Also update the diagnostic line in the output to print **all three** ΔIC stats so future readers can cross-check:

```python
        '--- ΔIC (3 stats; see summary CSV row names) ---',
        f'  median_delta_ic (A) M3:          {float(summary_df.loc["median_delta_ic", "model_3"]):+.4f}',
        f'  diff_of_medians_delta_ic (B) M3: {delta_ic_m3:+.4f}    (Step 3 threshold > 0.02)',
        f'  delta_ic_vs_m0 (C) M3:           {float(summary_df.loc["delta_ic_vs_m0", "model_3"]):+.4f}',
```

**Step 1.5: Run test to verify it passes**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py::test_summarize_ablation_writes_three_delta_ic -v
```

Expected: PASS.

**Step 1.6: Run full test suite to verify no regressions**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py -v
```

Expected: 112/112 PASS (1 new test added).

**Step 1.7: Commit**

```bash
git add backtrace/projection/ablation_fit.py tests/test_dynamics_eigen.py
git commit -m "feat(projection): V0.2-D Phase 0 — ΔIC audit hygiene (3 stats persisted)"
```

---

## Task 2: Phase 1 — Parameter Stability (Group A fields)

**Files:**
- Modify: `backtrace/projection/ablation_fit.py` (add `fit_one_split`, extend `CSV_COLUMNS` to 33 cols, update `fit_one_with_placebo` to call it)
- Modify: `tests/test_dynamics_eigen.py` (append 3 tests)

**Interfaces:**
- Produces:
  - `fit_one_split(movement_csv, stock_tag, index_tag, code, name, index_code, model_id) -> dict` returning 33-column row.
  - Extended `CSV_COLUMNS` (33-element list, Models 2/3 only populated; Models 0/1 leave new fields NaN).
  - Modified `write_ablation_csvs` calls `fit_one_split` (not `fit_one_with_placebo`).
  - Modified `fit_one_with_placebo` is replaced by `fit_one_split` (or kept as deprecated alias).

**Step 2.1: Write the failing tests**

Append to `tests/test_dynamics_eigen.py`:

```python
def test_fit_split_returns_train_test_params():
    """Synthetic Model 2 data → q_train_fit ≠ q_test_fit after split."""
    from projection.ablation_fit import fit_one_split
    import tempfile, os
    mv_dir = tempfile.mkdtemp()
    csv_path = os.path.join(mv_dir, "movement_idx_stk000001.csv")
    T = 200
    rng = np.random.default_rng(0)
    beta = 1.2 + 0.001 * np.arange(T)
    delta_v = rng.normal(0, 1, (T, 2))
    delta_u = beta[:, None] * delta_v + rng.normal(0, 0.5, (T, 2))
    pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=T),
        'Move_Delta_Vol_idx': delta_v[:, 0],
        'Move_Delta_Amt_idx': delta_v[:, 1],
        'Move_Delta_Vol_stk': delta_u[:, 0],
        'Move_Delta_Amt_stk': delta_u[:, 1],
        'Move_Proj_Coeff': beta,
    }).to_csv(csv_path, index=False)
    row = fit_one_split(csv_path, 'stk', 'idx', '000001.SZ', 'T', '000001.SH', model_id=2)
    # Group A fields exist
    for k in ('q_train_fit', 'k_train_fit', 'c_train_fit',
              'q_test_fit', 'k_test_fit', 'c_test_fit',
              'q_drift', 'k_drift', 'c_drift'):
        assert k in row, f"missing field {k}"
    # Both fits finite
    assert np.isfinite(row['q_train_fit']) and np.isfinite(row['q_test_fit'])
    # Drift = test − train
    assert abs(row['q_drift'] - (row['q_test_fit'] - row['q_train_fit'])) < 1e-9


def test_param_drift_no_l2_aggregation():
    """V0.2-D §4: param_drift_l2 is FORBIDDEN — only separate drifts exist."""
    from projection import ablation_fit
    src = open(ablation_fit.__file__).read()
    assert 'param_drift_l2' not in src, \
        "V0.2-D §4 forbids param_drift_l2 (q is dimensionless, k/c have units)"


def test_oos_uses_train_params_only():
    """V0.2-D §7: oos_r2 = R²(Y_test, X_test · θ_train) — θ_test must NOT be used."""
    from projection.ablation_fit import fit_one_split
    import tempfile, os
    mv_dir = tempfile.mkdtemp()
    csv_path = os.path.join(mv_dir, "movement_idx_stk000002.csv")
    T = 200
    rng = np.random.default_rng(1)
    beta = 1.2 + 0.001 * np.arange(T)
    delta_v = rng.normal(0, 1, (T, 2))
    delta_u = beta[:, None] * delta_v + rng.normal(0, 0.5, (T, 2))
    pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=T),
        'Move_Delta_Vol_idx': delta_v[:, 0],
        'Move_Delta_Amt_idx': delta_v[:, 1],
        'Move_Delta_Vol_stk': delta_u[:, 0],
        'Move_Delta_Amt_stk': delta_u[:, 1],
        'Move_Proj_Coeff': beta,
    }).to_csv(csv_path, index=False)
    row = fit_one_split(csv_path, 'stk', 'idx', '000002.SZ', 'T', '000001.SH', model_id=2)
    # oos_r2 must use train params only
    delta_u, delta_v, beta_arr = ablation_fit._read_movement(csv_path, 'stk', 'idx')
    u_vec, d_vec, a_u_vec, a_v_vec, bdv_vec = ablation_fit._build_kinematics_ext(delta_u, delta_v, beta_arr)
    X, Y = ablation_fit.BUILDERS[2](u_vec, d_vec, a_u_vec, a_v_vec, beta_arr, bdv_vec)
    mask = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
    valid = np.where(mask)[0]
    n_valid = len(valid)
    n_train = int(np.floor(0.7 * n_valid))
    train_idx = valid[:n_train]
    test_idx = valid[n_train:]
    # Reproduce θ_train
    theta_train = np.array([row['q_train_fit'], row['k_train_fit'], row['c_train_fit']])
    Y_pred_oos = X[test_idx] @ theta_train
    ss_res = np.sum((Y[test_idx] - Y_pred_oos) ** 2)
    ss_tot = np.sum((Y[test_idx] - Y[test_idx].mean()) ** 2)
    expected_oos_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    assert abs(row['oos_r2'] - expected_oos_r2) < 1e-6, \
        f"oos_r2 mismatch: stored={row['oos_r2']:.4f} vs expected={expected_oos_r2:.4f}"
```

**Step 2.2: Run tests to verify they fail**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py::test_fit_split_returns_train_test_params tests/test_dynamics_eigen.py::test_param_drift_no_l2_aggregation tests/test_dynamics_eigen.py::test_oos_uses_train_params_only -v
```

Expected: 3 FAIL (ImportError on `fit_one_split` for the first, KeyError on `param_drift_l2` for the second, ImportError / missing fields for the third).

**Step 2.3: Extend `CSV_COLUMNS` to 33 entries**

In `backtrace/projection/ablation_fit.py`, replace the existing 18-element `CSV_COLUMNS` (line ~132) with:

```python
CSV_COLUMNS = [
    # [18 existing — V0.1]
    'code', 'name', 'index_code', 'index_tag', 'stock_tag',
    'n_train', 'n_test',
    'condition_number', 'regressor_corr', 'r2',  # r2 语义改名 = train_fit_r2
    'identification_status', 'fit_quality',
    'q_hat', 'k_hat', 'c_hat', 'f_self_loss',
    'ic_real', 'ic_null',
    # [9 new — Group A: parameter stability, V0.2-D §4]
    'q_train_fit', 'k_train_fit', 'c_train_fit',
    'q_test_fit',  'k_test_fit',  'c_test_fit',
    'q_drift',     'k_drift',     'c_drift',
    # [3 new — Group B: three R², V0.2-D §7]
    'train_fit_r2', 'test_fit_r2', 'oos_r2',
    # [3 new — Group C: X-X collinearity, V0.2-D §5]
    'corr_x_beta_d', 'corr_x_beta_u', 'corr_x_d_u',
    # [3 new — Group D: residual structure, V0.2-D §6]
    'corr_F_beta_aM', 'corr_F_d', 'corr_F_u',
]
```

(Note: the full set is 36 fields. Re-count: 18 + 9 + 3 + 3 + 3 = 36. Spec §8 says "33 columns" — verify count and update either spec or plan. **Going with 36 to match the explicit list above**, and the spec's count error will be fixed in a plan-amendment commit if needed. Update spec §8 from "33 columns" to "36 columns" in this task's commit message.)

**Step 2.4: Add `fit_one_split` (the new diagnostic orchestrator)**

In `backtrace/projection/ablation_fit.py`, locate the end of `fit_one_with_placebo` (line ~407). Add:

```python
def _oos_r2_from_train_params(X_test: np.ndarray, Y_test: np.ndarray,
                               theta_train: np.ndarray) -> float:
    """V0.2-D §7: oos_r2 = R²(Y_test, X_test · θ_train)."""
    if X_test.shape[0] < 2:
        return np.nan
    Y_pred = X_test @ theta_train
    ss_res = float(np.sum((Y_test - Y_pred) ** 2))
    ss_tot = float(np.sum((Y_test - np.mean(Y_test)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan


def _refit_window(X_window: np.ndarray, Y_window: np.ndarray, model_id: int):
    """Fit OLS on one window; return (theta, f_res, cond, r2)."""
    theta, f_res, n_v, rank, cond, rcorr, r2 = ols_fit(X_window, Y_window)
    return theta, f_res, cond, r2


def fit_one_split(movement_csv: str, stock_tag: str, index_tag: str,
                  code: str, name: str, index_code: str, model_id: int) -> dict:
    """V0.2-D: 36-col diagnostic row — train/test refit + OOS + diagnostics.

    Algorithm (V0.2-D §3 strict layering):
      1. Reconstruct kinematics → X, Y.
      2. mask = isfinite(Y) & all-isfinite(X).
      3. Split valid indices 70/30 → train_idx_abs, test_idx_abs.
      4. Layer 1: OLS on (X_train, Y_train) → θ_train, train_fit_r2.
      5. Layer 2 (diagnostic only): OLS on (X_test, Y_test) → θ_test_fit, test_fit_r2.
      6. Layer 3 (only valid generalization): Ŷ_oos = X_test · θ_train, oos_r2, oos_ic.
      7. Group A fields: q/k/c train/test, drifts.
      8. Group C: X-X correlations on X_train.
      9. Group D: residual F_self = Y_train − X_train · θ_train, correlate with X_train columns.
    """
    delta_u, delta_v, beta = _read_movement(movement_csv, stock_tag, index_tag)
    u_vec, d_vec, a_u_vec, a_v_vec, bdv_vec = _build_kinematics_ext(delta_u, delta_v, beta)
    X, Y = BUILDERS[model_id](u_vec, d_vec, a_u_vec, a_v_vec, beta, bdv_vec)
    mask = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
    valid_indices = np.where(mask)[0]
    n_valid = len(valid_indices)

    # NaN-fill template
    empty = {col: np.nan for col in CSV_COLUMNS}
    empty.update({
        'code': code, 'name': name, 'index_code': index_code,
        'index_tag': index_tag, 'stock_tag': stock_tag,
        'identification_status': 'singular', 'fit_quality': 'uninformative',
        'q_hat': 1.0 if model_id in (0, 1) else np.nan,
    })

    if n_valid < 20:
        empty['n_train'] = n_valid
        empty['n_test'] = 0
        return empty

    train_idx_rel, test_idx_rel = oos_split_indices(n_valid, train_frac=0.7)
    train_idx_abs = valid_indices[train_idx_rel]
    test_idx_abs = valid_indices[test_idx_rel]

    X_train, Y_train = X[train_idx_abs], Y[train_idx_abs]
    X_test, Y_test = X[test_idx_abs], Y[test_idx_abs]

    # Layer 1: train fit (only valid source for OOS prediction)
    theta_train, f_res_train, cond_train, train_fit_r2 = _refit_window(X_train, Y_train, model_id)

    # Layer 2: test refit (diagnostic only — must not leak into oos_r2 / oos_ic)
    theta_test, f_res_test, _, test_fit_r2 = _refit_window(X_test, Y_test, model_id)

    # Layer 3: OOS prediction with train params only
    oos_r2 = _oos_r2_from_train_params(X_test, Y_test, theta_train)
    Y_pred_oos = X_test @ theta_train
    oos_ic = compute_spearman_ic(Y_pred_oos, Y_test)

    # Placebo (reuse existing logic)
    X_train_perm = permute_regressors(X_train, Y_train, seed=PLACEBO_SEED)
    theta_null, *_ = ols_fit(X_train_perm, Y_train)
    Y_pred_null = X_test @ theta_null
    ic_null = compute_spearman_ic(Y_pred_null, Y_test)

    # Group A: parameter stability (separate drifts; NO param_drift_l2)
    if model_id in (0, 1):
        q_train, k_train, c_train = 1.0, float(theta_train[0]), float(theta_train[1])
        q_test_fit, k_test_fit, c_test_fit = 1.0, float(theta_test[0]), float(theta_test[1])
    else:
        q_train, k_train, c_train = (float(theta_train[0]), float(theta_train[1]), float(theta_train[2]))
        q_test_fit, k_test_fit, c_test_fit = (float(theta_test[0]), float(theta_test[1]), float(theta_test[2]))
    q_drift = q_test_fit - q_train
    k_drift = k_test_fit - k_train
    c_drift = c_test_fit - c_train

    # Group C: X-X collinearity on X_train (Task 3 fills in)
    # Group D: residual structure (Task 4 fills in)
    # For now, leave as NaN — Tasks 3 and 4 populate.

    return {
        'code': code, 'name': name, 'index_code': index_code,
        'index_tag': index_tag, 'stock_tag': stock_tag,
        'n_train': X_train.shape[0], 'n_test': X_test.shape[0],
        'condition_number': cond_train,
        'regressor_corr': float(np.nan),  # populated by Task 3 helper if needed
        'r2': train_fit_r2,
        'identification_status': compute_identification_status(
            int(np.linalg.matrix_rank(X_train)), cond_train),
        'fit_quality': compute_fit_quality(train_fit_r2),
        'q_hat': q_train,
        'k_hat': k_train, 'c_hat': c_train,
        'f_self_loss': f_res_train,
        'ic_real': oos_ic, 'ic_null': ic_null,
        # Group A
        'q_train_fit': q_train, 'k_train_fit': k_train, 'c_train_fit': c_train,
        'q_test_fit': q_test_fit, 'k_test_fit': k_test_fit, 'c_test_fit': c_test_fit,
        'q_drift': q_drift, 'k_drift': k_drift, 'c_drift': c_drift,
        # Group B
        'train_fit_r2': train_fit_r2,
        'test_fit_r2': test_fit_r2,
        'oos_r2': oos_r2,
        # Group C placeholders
        'corr_x_beta_d': np.nan, 'corr_x_beta_u': np.nan, 'corr_x_d_u': np.nan,
        # Group D placeholders
        'corr_F_beta_aM': np.nan, 'corr_F_d': np.nan, 'corr_F_u': np.nan,
    }
```

**Step 2.5: Update `write_ablation_csvs` to call `fit_one_split`**

In `backtrace/projection/ablation_fit.py`, locate `write_ablation_csvs` (line ~407). Replace its body so it calls `fit_one_split`:

```python
def write_ablation_csvs(targets: List[Tuple], output_dir: str):
    """V0.2-D: 4 CSVs with full 36-col diagnostic schema (Models 2/3 fully populated; Models 0/1 partial)."""
    os.makedirs(output_dir, exist_ok=True)
    rows_by_model = {m: [] for m in range(4)}
    for code, name, mv_csv, index_tag, stock_tag, index_code in targets:
        for m in range(4):
            row = fit_one_split(mv_csv, stock_tag, index_tag, code, name, index_code, m)
            rows_by_model[m].append(row)
    for m in range(4):
        df = pd.DataFrame(rows_by_model[m], columns=CSV_COLUMNS)
        df.to_csv(os.path.join(output_dir, f'kc_estimates_model{m}.csv'),
                  index=False, encoding='utf-8')
```

**Step 2.6: Run tests to verify they pass**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py::test_fit_split_returns_train_test_params tests/test_dynamics_eigen.py::test_param_drift_no_l2_aggregation tests/test_dynamics_eigen.py::test_oos_uses_train_params_only -v
```

Expected: 3 PASS.

**Step 2.7: Run full suite to verify no regressions**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py -v
```

Expected: 115/115 PASS (3 new tests).

**Step 2.8: Commit**

```bash
git add backtrace/projection/ablation_fit.py tests/test_dynamics_eigen.py
git commit -m "feat(projection): V0.2-D Phase 1 — Group A parameter stability (split + drifts, NO L2)"
```

---

## Task 3: Phase 2 — X-X Collinearity (Group C fields)

**Files:**
- Modify: `backtrace/projection/ablation_fit.py` (add `compute_x_x_correlations`, integrate into `fit_one_split`)
- Modify: `tests/test_dynamics_eigen.py` (append 2 tests)

**Interfaces:**
- Produces:
  - `compute_x_x_correlations(X: np.ndarray) -> (corr_x_beta_d, corr_x_beta_u, corr_x_d_u)` returning 3 floats (NaN if any column has zero variance).

**Step 3.1: Write the failing tests**

Append to `tests/test_dynamics_eigen.py`:

```python
def test_x_x_correlation_per_stock_scalar():
    """V0.2-D §5: X-X correlation is scalar per stock (NOT corr(q, X) which is undefined)."""
    from projection.ablation_fit import compute_x_x_correlations
    rng = np.random.default_rng(0)
    T = 100
    X = rng.normal(0, 1, (T, 3))
    out = compute_x_x_correlations(X)
    assert len(out) == 3, "must return 3 scalars (one per pair)"
    c_beta_d, c_beta_u, c_d_u = out
    # All scalars
    assert all(isinstance(v, float) for v in (c_beta_d, c_beta_u, c_d_u))
    # Reproduce via direct np.corrcoef
    expected_bd = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    expected_bu = float(np.corrcoef(X[:, 0], X[:, 2])[0, 1])
    expected_du = float(np.corrcoef(X[:, 1], X[:, 2])[0, 1])
    assert abs(c_beta_d - expected_bd) < 1e-9
    assert abs(c_beta_u - expected_bu) < 1e-9
    assert abs(c_d_u - expected_du) < 1e-9


def test_x_x_correlation_matches_cond_x_pattern():
    """V0.2-D §5: large |corr_x_beta_d| generally coincides with large condition_number."""
    from projection.ablation_fit import compute_x_x_correlations
    rng = np.random.default_rng(1)
    T = 200
    # Highly collinear X_β with X_d
    X = rng.normal(0, 1, (T, 3))
    X[:, 1] = X[:, 0] + 0.01 * rng.normal(0, 1, T)  # X_d ≈ X_β
    X[:, 2] = rng.normal(0, 1, T)
    c_beta_d, _, _ = compute_x_x_correlations(X)
    assert abs(c_beta_d) > 0.9, f"expected high correlation; got {c_beta_d}"
```

**Step 3.2: Run tests to verify they fail**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py::test_x_x_correlation_per_stock_scalar tests/test_dynamics_eigen.py::test_x_x_correlation_matches_cond_x_pattern -v
```

Expected: 2 FAIL with `ImportError: cannot import name 'compute_x_x_correlations'`.

**Step 3.3: Implement `compute_x_x_correlations`**

In `backtrace/projection/ablation_fit.py`, add (right after `_refit_window`):

```python
def compute_x_x_correlations(X: np.ndarray):
    """V0.2-D §5: pairwise Pearson correlation among 3 design-matrix columns.

    X shape (n, 3): columns are [β·a_M, −d, −u].
    Returns: (corr_x_beta_d, corr_x_beta_u, corr_x_d_u) — 3 floats, NaN if zero variance.

    CRITICAL (V0.2-D §5): this computes corr(X_i, X_j), NOT corr(q̂, X).
    corr(q̂, X) is statistically undefined (scalar × time series).
    """
    if X.shape[0] < 3 or X.shape[1] != 3:
        return (np.nan, np.nan, np.nan)
    stds = X.std(axis=0)
    if np.any(stds < 1e-12):
        return (np.nan, np.nan, np.nan)
    corr = np.corrcoef(X.T)
    return (float(corr[0, 1]), float(corr[0, 2]), float(corr[1, 2]))
```

**Step 3.4: Integrate into `fit_one_split`**

In `backtrace/projection/ablation_fit.py`, locate the section of `fit_one_split` that computes Group C placeholders. Replace:

```python
        # Group C placeholders
        'corr_x_beta_d': np.nan, 'corr_x_beta_u': np.nan, 'corr_x_d_u': np.nan,
```

With:

```python
        # Group C: X-X collinearity on X_train (V0.2-D §5)
        corr_x_beta_d, corr_x_beta_u, corr_x_d_u = compute_x_x_correlations(X_train)
```

And update the return dict:

```python
        # Group C
        'corr_x_beta_d': corr_x_beta_d,
        'corr_x_beta_u': corr_x_beta_u,
        'corr_x_d_u': corr_x_d_u,
```

**Step 3.5: Run tests to verify they pass**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py::test_x_x_correlation_per_stock_scalar tests/test_dynamics_eigen.py::test_x_x_correlation_matches_cond_x_pattern -v
```

Expected: 2 PASS.

**Step 3.6: Run full suite to verify no regressions**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py -v
```

Expected: 117/117 PASS.

**Step 3.7: Commit**

```bash
git add backtrace/projection/ablation_fit.py tests/test_dynamics_eigen.py
git commit -m "feat(projection): V0.2-D Phase 2 — Group C X-X collinearity (corr(X_i, X_j) per stock)"
```

---

## Task 4: Phase 3 — Residual Structure (Group D fields)

**Files:**
- Modify: `backtrace/projection/ablation_fit.py` (add `compute_residual_correlations`, integrate into `fit_one_split`)
- Modify: `tests/test_dynamics_eigen.py` (append 2 tests)

**Interfaces:**
- Produces:
  - `compute_residual_correlations(F_self: np.ndarray, X_train: np.ndarray) -> (corr_F_beta_aM, corr_F_d, corr_F_u)` returning 3 floats.

**Step 4.1: Write the failing tests**

Append to `tests/test_dynamics_eigen.py`:

```python
def test_residual_correlation_white_noise_zero():
    """V0.2-D §6: synthetic Model 2 with Gaussian noise → corr_F_* ≈ 0."""
    from projection.ablation_fit import compute_residual_correlations
    rng = np.random.default_rng(0)
    T = 200
    X_train = rng.normal(0, 1, (T, 3))
    # Fit produces near-zero residuals if model is correctly specified
    theta = np.array([0.5, 0.3, 0.1])
    Y_train = X_train @ theta + rng.normal(0, 0.001, T)
    F_self = Y_train - X_train @ theta
    c_b, c_d, c_u = compute_residual_correlations(F_self, X_train)
    assert all(abs(v) < 0.05 for v in (c_b, c_d, c_u)), \
        f"white-noise residuals must give corr ≈ 0; got ({c_b}, {c_d}, {c_u})"


def test_residual_correlation_missing_term_detects():
    """V0.2-D §6: missing dynamics term leaves systematic residual correlated with X."""
    from projection.ablation_fit import compute_residual_correlations
    rng = np.random.default_rng(2)
    T = 300
    X_train = rng.normal(0, 1, (T, 3))
    theta = np.array([0.5, 0.3, 0.1])
    # Inject a missing-term residual correlated with X[:,1] (= −d column)
    Y_train = X_train @ theta + 0.5 * X_train[:, 1] ** 2 + rng.normal(0, 0.1, T)
    F_self = Y_train - X_train @ theta
    c_b, c_d, c_u = compute_residual_correlations(F_self, X_train)
    assert abs(c_d) > 0.3, f"missing-term residual must correlate with X_d; got {c_d}"
```

**Step 4.2: Run tests to verify they fail**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py::test_residual_correlation_white_noise_zero tests/test_dynamics_eigen.py::test_residual_correlation_missing_term_detects -v
```

Expected: 2 FAIL with `ImportError: cannot import name 'compute_residual_correlations'`.

**Step 4.3: Implement `compute_residual_correlations`**

In `backtrace/projection/ablation_fit.py`, add (right after `compute_x_x_correlations`):

```python
def compute_residual_correlations(F_self: np.ndarray, X_train: np.ndarray):
    """V0.2-D §6: Pearson correlation of residual with each design-matrix column.

    F_self shape (n_train,): residual = Y_train − X_train · θ_train.
    X_train shape (n_train, 3): columns are [β·a_M, −d, −u].

    Returns: (corr_F_beta_aM, corr_F_d, corr_F_u) — 3 floats, NaN if zero variance.

    Diagnostic interpretation:
      - All three near 0 → dynamics correctly specified for in-sample window.
      - corr_F_d systemically > 0.2 → missing term in d-evolution (e.g., d², ḋ).
      - corr_F_u systemically large → u-coupling mis-specified.
    """
    if F_self.shape[0] < 3 or X_train.shape[1] != 3:
        return (np.nan, np.nan, np.nan)
    if F_self.std() < 1e-12:
        return (np.nan, np.nan, np.nan)
    out = []
    for j in range(3):
        col_data = X_train[:, j]
        if col_data.std() < 1e-12:
            out.append(np.nan)
        else:
            out.append(float(np.corrcoef(F_self, col_data)[0, 1]))
    return tuple(out)
```

**Step 4.4: Integrate into `fit_one_split`**

In `backtrace/projection/ablation_fit.py`, locate the Group D placeholder section in `fit_one_split`. Replace:

```python
        # Group D placeholders
        'corr_F_beta_aM': np.nan, 'corr_F_d': np.nan, 'corr_F_u': np.nan,
```

With:

```python
        # Group D: residual correlations (V0.2-D §6)
        F_self = Y_train - X_train @ theta_train
        corr_F_beta_aM, corr_F_d, corr_F_u = compute_residual_correlations(F_self, X_train)
```

And update the return dict:

```python
        # Group D
        'corr_F_beta_aM': corr_F_beta_aM,
        'corr_F_d': corr_F_d,
        'corr_F_u': corr_F_u,
```

**Step 4.5: Run tests to verify they pass**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py::test_residual_correlation_white_noise_zero tests/test_dynamics_eigen.py::test_residual_correlation_missing_term_detects -v
```

Expected: 2 PASS.

**Step 4.6: Run full suite to verify no regressions**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py -v
```

Expected: 119/119 PASS.

**Step 4.7: Commit**

```bash
git add backtrace/projection/ablation_fit.py tests/test_dynamics_eigen.py
git commit -m "feat(projection): V0.2-D Phase 3 — Group D residual structure (corr(F, X) per stock)"
```

---

## Task 5: Phase 4 — Panel 5 + Distribution Reporting

**Files:**
- Modify: `backtrace/projection/ablation_fit.py` (add `build_panel5_html`, `compute_v0_2_d_distributions`, `write_v0_2_d_summary_txt`)
- Modify: `tests/test_dynamics_eigen.py` (append 1 test)

**Interfaces:**
- Produces:
  - `build_panel5_html(model2_csv: str, output_path: str) -> str` — saves scatter HTML.
  - `compute_v0_2_d_distributions(model2_csv: str) -> pd.DataFrame` — D1/D2/D3 distribution tables.
  - `write_v0_2_d_summary_txt(dist_df: pd.DataFrame, output_path: str) -> str` — UTF-8 Chinese diagnostic report.

**Step 5.1: Write the failing test**

Append to `tests/test_dynamics_eigen.py`:

```python
def test_panel5_uses_x_x_corr_not_q_x_corr():
    """V0.2-D §9: Panel 5 x-axis is corr_x_beta_d, NOT corr(q, β·a_M)."""
    from projection.ablation_fit import build_panel5_html
    import tempfile, os
    # Build a stub Model 2 CSV
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, 'kc_estimates_model2.csv')
        rng = np.random.default_rng(0)
        n = 200
        pd.DataFrame({
            'code': [f'stk{i:06d}' for i in range(n)],
            'corr_x_beta_d': rng.normal(0.3, 0.1, n),
            'q_drift': rng.normal(0.1, 0.05, n),
            'ic_real': rng.normal(0, 0.5, n),
        }).to_csv(csv_path, index=False)
        html_path = build_panel5_html(csv_path, os.path.join(td, 'panel5.html'))
        # Read HTML and verify x-axis label
        with open(html_path) as f:
            html = f.read()
        assert 'corr_x_beta_d' in html, "x-axis must be corr_x_beta_d"
        assert 'corr(q' not in html, "x-axis must NOT be the undefined corr(q, β·a_M)"
```

**Step 5.2: Run test to verify it fails**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py::test_panel5_uses_x_x_corr_not_q_x_corr -v
```

Expected: FAIL with `ImportError: cannot import name 'build_panel5_html'`.

**Step 5.3: Implement `build_panel5_html`**

In `backtrace/projection/ablation_fit.py`, add:

```python
def build_panel5_html(model2_csv: str, output_path: str) -> str:
    """V0.2-D §9: scatter of q_drift × corr_x_beta_d, colored by oos_ic.

    Per spec §9 4-quadrant interpretation:
      - upper-right: regime shift + collinearity (highest-risk)
      - upper-left: regime shift dominates
      - lower-right: pure collinearity
      - lower-left: clean
    """
    import plotly.graph_objects as go
    df = pd.read_csv(model2_csv)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df['corr_x_beta_d'], y=df['q_drift'],
        mode='markers',
        marker=dict(
            size=4, opacity=0.6,
            color=df['ic_real'],
            colorscale='RdBu_r', cmin=-1, cmax=1,
            colorbar=dict(title='OOS IC'),
        ),
        text=df['code'],
        hovertemplate='code=%{text}<br>corr_x_beta_d=%{x:.3f}<br>q_drift=%{y:.3f}<br>oos_ic=%{marker.color:.3f}<extra></extra>',
    ))
    fig.update_layout(
        template='plotly_dark',
        title='V0.2-D Panel 5: q_drift × corr_x_beta_d (color = OOS IC, Model 2)',
        xaxis_title='corr_x_beta_d (corr(β·a_M, −d))',
        yaxis_title='q_drift = q_test_fit − q_train_fit',
        height=700,
    )
    fig.write_html(output_path)
    return output_path


def compute_v0_2_d_distributions(model2_csv: str) -> pd.DataFrame:
    """V0.2-D §10: distribution reports for D1 / D2 / D3.

    Returns DataFrame with rows = (gate, statistic) and one column of values.
    Distribution stats: median, p25, p75, P(|x| > threshold) for D1/D2;
                        median, p25, p75, P(x > threshold) for D3.
    """
    df = pd.read_csv(model2_csv)
    rows = []

    def _stats(s: pd.Series, abs_val: bool, threshold: float):
        s_used = s.abs() if abs_val else s
        return {
            'median': float(np.median(s_used.dropna())),
            'p25': float(np.percentile(s_used.dropna(), 25)),
            'p75': float(np.percentile(s_used.dropna(), 75)),
            f'P(>{threshold})': float(np.mean(s_used.dropna() > threshold)),
        }

    for name, col, abs_val, thr in [
        ('D1', 'q_drift', True, 0.3),
        ('D2', 'corr_x_beta_d', True, 0.3),
        ('D3', 'corr_F_d', False, 0.2),
    ]:
        stats = _stats(df[col], abs_val, thr)
        for stat_name, val in stats.items():
            rows.append({'gate': name, 'statistic': stat_name, 'value': val})
    return pd.DataFrame(rows)


def write_v0_2_d_summary_txt(dist_df: pd.DataFrame, output_path: str) -> str:
    """V0.2-D §10: UTF-8 Chinese distribution report (diagnostic only, no PASS/FAIL)."""
    lines = [
        '=' * 60,
        'V0.2-D — OOS Reversal Decomposition (Diagnostic Report)',
        '=' * 60,
        f'Run date:  {pd.Timestamp.now().strftime("%Y-%m-%d")}',
        '',
        'NOTE: This is a diagnostic report. No PASS/FAIL verdicts.',
        'Interpretation routes to V0.2-E or user.',
        '',
    ]
    for gate in ('D1', 'D2', 'D3'):
        gate_df = dist_df[dist_df['gate'] == gate]
        lines.append(f'--- Gate {gate} ---')
        for _, row in gate_df.iterrows():
            lines.append(f'  {row["statistic"]:<8s}: {row["value"]:+.4f}')
        lines.append('')
    lines.append('=' * 60)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return output_path
```

**Step 5.4: Run test to verify it passes**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py::test_panel5_uses_x_x_corr_not_q_x_corr -v
```

Expected: PASS.

**Step 5.5: Run full suite to verify no regressions**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py -v
```

Expected: 120/120 PASS.

**Step 5.6: Commit**

```bash
git add backtrace/projection/ablation_fit.py tests/test_dynamics_eigen.py
git commit -m "feat(projection): V0.2-D Phase 4 — Panel 5 + distribution reporting (no PASS/FAIL)"
```

---

## Task 6: Phase 5 — CLI + Audit Verification

**Files:**
- Create: `backtrace/projection/v0_2_d_decompose.py` (new CLI orchestrator)
- Modify: `tests/test_dynamics_eigen.py` (append 1 CLI smoke test)

**Interfaces:**
- Produces:
  - `v0_2_d_decompose.py --limit N --movement-dir DIR --output-dir DIR` runs full diagnostic and writes `data/projection_v01_d/`.

**Step 6.1: Write the failing CLI smoke test**

Append to `tests/test_dynamics_eigen.py`:

```python
def test_cli_smoke_v0_2_d_full_pipeline():
    """V0.2-D §13: full pipeline runs end-to-end with synthetic movement CSVs."""
    import subprocess, tempfile, os
    with tempfile.TemporaryDirectory() as td:
        mv_dir = os.path.join(td, 'mv')
        os.makedirs(mv_dir)
        # 3 synthetic stocks
        for i in range(3):
            T = 80
            rng = np.random.default_rng(i)
            beta = 1.2 + 0.001 * np.arange(T)
            delta_v = rng.normal(0, 1, (T, 2))
            delta_u = beta[:, None] * delta_v + rng.normal(0, 0.5, (T, 2))
            pd.DataFrame({
                'Date': pd.date_range('2024-01-01', periods=T),
                'Move_Delta_Vol_idx': delta_v[:, 0],
                'Move_Delta_Amt_idx': delta_v[:, 1],
                'Move_Delta_Vol_stk': delta_u[:, 0],
                'Move_Delta_Amt_stk': delta_u[:, 1],
                'Move_Proj_Coeff': beta,
            }).to_csv(os.path.join(mv_dir, f'movement_idx_stk{i:06d}.csv'), index=False)
        out_dir = os.path.join(td, 'out')
        # Run CLI
        result = subprocess.run([
            '/c/Users/yellow/.conda/envs/venv/python.exe',
            'backtrace/projection/v0_2_d_decompose.py',
            '--movement-dir', mv_dir,
            '--output-dir', out_dir,
            '--limit', '3',
        ], capture_output=True, text=True, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        # Verify outputs exist
        for f in ('kc_estimates_model2_diag.csv', 'panel5_drift_vs_collinearity.html',
                   'v0_2_d_distributions.csv', 'v0_2_d_summary.txt'):
            assert os.path.exists(os.path.join(out_dir, f)), f"missing output: {f}"
```

**Step 6.2: Run test to verify it fails**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_smoke_v0_2_d_full_pipeline -v
```

Expected: FAIL (CLI script not found).

**Step 6.3: Create `v0_2_d_decompose.py`**

Create `backtrace/projection/v0_2_d_decompose.py`:

```python
# -*- coding: utf-8 -*-
# v0_2_d_decompose.py — V0.2-D OOS Reversal Decomposition CLI
#
# Spec: docs/superpowers/specs/2026-08-19-dynamics-oos-reversal-decomposition.md
#
# Usage:
#   PYTHONIOENCODING=utf-8 python backtrace/projection/v0_2_d_decompose.py \
#       --movement-dir data/projection --output-dir data/projection_v01_d --limit 0
import sys, os
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
PROJECT_ROOT = os.path.dirname(BACKTRACE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.stdout.reconfigure(encoding='utf-8')

import argparse
from projection.ablation_fit import (
    list_movement_csvs, write_ablation_csvs, summarize_ablation,
    build_panel5_html, compute_v0_2_d_distributions, write_v0_2_d_summary_txt,
    CSV_COLUMNS,
)


def parse_args():
    p = argparse.ArgumentParser(description='V0.2-D — OOS Reversal Decomposition')
    p.add_argument('--movement-dir', default='data/projection',
                   help='Directory containing movement_*.csv')
    p.add_argument('--output-dir', default='data/projection_v01_d',
                   help='Output directory for diagnostic CSV / HTML / TXT')
    p.add_argument('--limit', type=int, default=0,
                   help='Max stocks to process; 0 = all')
    return p.parse_args()


def main():
    args = parse_args()
    targets = list_movement_csvs(args.movement_dir)
    if args.limit > 0:
        targets = targets[:args.limit]
    print(f'输入: {args.movement_dir}/movement_*.csv')
    print(f'目标: {len(targets)} 只 (limit={args.limit})')

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: write 4-model ablation CSVs with full 36-col diagnostic schema
    write_ablation_csvs(targets, args.output_dir)

    # Step 2: regen V0.1 summary (with 3 ΔIC stats from Phase 0 audit fix)
    summary_df = summarize_ablation({m: os.path.join(args.output_dir, f'kc_estimates_model{m}.csv') for m in range(4)})
    summary_df.to_csv(os.path.join(args.output_dir, 'kc_ablation_summary.csv'),
                      encoding='utf-8')

    # Step 3: rename Model 2 CSV for downstream clarity
    src = os.path.join(args.output_dir, 'kc_estimates_model2.csv')
    dst = os.path.join(args.output_dir, 'kc_estimates_model2_diag.csv')
    if os.path.exists(src):
        os.rename(src, dst)

    # Step 4: Panel 5 (Model 2 only)
    panel5_path = build_panel5_html(dst, os.path.join(args.output_dir, 'panel5_drift_vs_collinearity.html'))

    # Step 5: distribution reports
    dist_df = compute_v0_2_d_distributions(dst)
    dist_df.to_csv(os.path.join(args.output_dir, 'v0_2_d_distributions.csv'),
                    index=False, encoding='utf-8')
    summary_txt = write_v0_2_d_summary_txt(dist_df, os.path.join(args.output_dir, 'v0_2_d_summary.txt'))

    print(f'Summary CSV: {args.output_dir}/kc_ablation_summary.csv')
    print(f'Model 2 diag: {dst}')
    print(f'Panel 5:      {panel5_path}')
    print(f'Distributions: {args.output_dir}/v0_2_d_distributions.csv')
    print(f'Summary TXT:   {summary_txt}')


if __name__ == '__main__':
    main()
```

**Step 6.4: Run test to verify it passes**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_smoke_v0_2_d_full_pipeline -v
```

Expected: PASS.

**Step 6.5: Run full suite to verify no regressions**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py -v
```

Expected: 121/121 PASS.

**Step 6.6: Full-market smoke (limited to 100 stocks for fast validation)**

Run:
```bash
PYTHONIOENCODING=utf-8 python backtrace/projection/v0_2_d_decompose.py --limit 100
```

Expected: exit 0; CSV/HTML/TXT produced in `data/projection_v01_d/`.

**Step 6.7: Commit**

```bash
git add backtrace/projection/v0_2_d_decompose.py tests/test_dynamics_eigen.py
git commit -m "feat(projection): V0.2-D Phase 5 — CLI orchestrator + audit verification smoke"
```

---

## Task 7: Whole-branch final review

**Files:** None (review-only)

**Step 7.1: Dispatch final code reviewer (opus)**

Run:
```bash
git log --oneline 75a7b2b..HEAD  # shows V0.2-D commits
git diff 75a7b2b..HEAD --stat    # file-level summary
git diff 75a7b2b..HEAD > /tmp/v0_2_d_full.diff  # full diff for reviewer
```

Dispatch reviewer per SDD skill rules (most-capable model = opus):
- Review spec §1-§17 coverage
- Verify §3 strict layering (θ_test_fit must NOT appear in oos_r2 / oos_ic / verdict / Panel 5)
- Verify §4 no `param_drift_l2`
- Verify §5 corr(X_i, X_j) NOT corr(q̂, X)
- Verify §11 3 ΔIC stats all in CSV
- Verify §14 out-of-scope items absent

**Step 7.2: Apply any reviewer findings (one round maximum)**

If reviewer flags Critical/Important, dispatch one fix round. Park Minor findings as V0.2-D known-issues (deferred to V0.2-D.2 if needed).

**Step 7.3: Final whole-branch approval**

Once review APPROVED, mark V0.2-D complete. Next phase (V0.2-C: industry vs market) is a separate spec, separate plan.

---

## Task 8: Full-market V0.2-D run + memory + commit

**Files:** None (run-only + memory write)

**Step 8.1: Run full-market V0.2-D diagnostic**

Run in background (expected ~10-15 minutes for 5211 stocks):
```bash
PYTHONIOENCODING=utf-8 python backtrace/projection/v0_2_d_decompose.py --limit 0
```

Output to `data/projection_v01_d/`.

**Step 8.2: Verify outputs**

```bash
ls -la data/projection_v01_d/
cat data/projection_v01_d/v0_2_d_summary.txt
head -5 data/projection_v01_d/v0_2_d_distributions.csv
```

Confirm:
- `kc_estimates_model{0,1,2,3}.csv` (Model 2 renamed to `kc_estimates_model2_diag.csv`)
- `kc_ablation_summary.csv` has 11 rows (was 10; +1 ΔIC stat)
- `kc_ablation_recommendation.txt` is regenerated, all ΔIC numbers read from CSV
- `panel5_drift_vs_collinearity.html` is created
- `v0_2_d_distributions.csv` has D1/D2/D3 distributions
- `v0_2_d_summary.txt` is UTF-8 Chinese diagnostic report

**Step 8.3: Write memory entry**

Create `~/.claude/projects/c--Users-yellow-mcp-qtTdx/memory/projection-v02-d-oos-reversal-decomposition.md` with:
- 3 corrections applied (no corr(q̂,X), no L2, no θ_test in OOS)
- 36-col CSV schema
- 4-group diagnostic (A/B/C/D)
- Panel 5 design (q_drift × corr_x_beta_d, color=OOS IC)
- Decision gates as distribution reports (no PASS/FAIL)
- Audit fix (3 ΔIC stats persisted)
- 9 commits, 121 tests PASS

Update `MEMORY.md` to add the pointer.

**Step 8.4: Final commit**

```bash
git add data/projection_v01/kc_ablation_summary.csv data/projection_v01/kc_ablation_recommendation.txt
git commit -m "chore(projection): V0.2-D Phase 0 audit-fix outputs (3 ΔIC stats in summary CSV)"
```

(Note: per .gitignore, the diagnostic outputs in `data/projection_v01_d/` are not committed. The Phase 0 audit fix outputs in `data/projection_v01/` may be committed — verify with the user.)

---

## Self-Review

### 1. Spec coverage

| spec section | task |
|---|---|
| §1 Context | (no impl needed; background) |
| §2 Research question | Task 5 (distribution reporting for D1/D2/D3) |
| §3 Strict layering | Task 2 (`_oos_r2_from_train_params` uses only `θ_train`) |
| §4 Parameter stability | Task 2 (9 new fields) |
| §5 X-X collinearity | Task 3 (3 fields, NOT corr(q̂,X)) |
| §6 Residual structure | Task 4 (3 fields) |
| §7 OOS boundary | Task 2 (reviewer must grep for `θ_test_fit` references in OOS / verdict / Panel 5) |
| §8 CSV schema (36 cols) | Task 2 |
| §9 Panel 5 | Task 5 |
| §10 Decision gates (distribution only) | Task 5 |
| §11 Audit fix | Task 1 |
| §12 Tests | Tasks 2-6 (9 tests total) |
| §13 CLI / outputs | Task 6 |
| §14 Out of scope | All tasks (reviewer must enforce) |
| §15 Phases | Tasks 1-6 (P0 → P5) |

### 2. Placeholder scan

- ✅ No "TBD", "TODO", "implement later" in plan body
- ✅ Each step has full code blocks
- ✅ "Forbidden" patterns are explicit (not "etc.", not "similar to")

### 3. Type consistency

| field | declared in | used in |
|---|---|---|
| `fit_one_split` | Task 2.4 | Tasks 5, 6 |
| `compute_x_x_correlations` | Task 3.3 | Task 4 (different function, similar shape) |
| `compute_residual_correlations` | Task 4.3 | (only Task 4) |
| `build_panel5_html` | Task 5.3 | Task 6 |
| `compute_v0_2_d_distributions` | Task 5.3 | Task 6 |
| `write_v0_2_d_summary_txt` | Task 5.3 | Task 6 |
| `CSV_COLUMNS` (36 elements) | Task 2.3 | Tasks 5, 6 |
| `oos_r2`, `q_drift`, etc. | Task 2.4 | Task 5 (Panel 5 uses `corr_x_beta_d` + `q_drift`) |

### 4. Risks (from spec §16)

- **Test fixture column-naming typo**: Task 2.1 Step test uses `Move_Delta_Vol_stk` literal — this is **allowed** in test code as long as `_read_movement` substitutes `{stock_tag}`. Reviewer must enforce substitution. ✅ (tests pass `'stk'` as `stock_tag`)
- **`oos_r2` SS_tot near zero**: Task 2.4 `_oos_r2_from_train_params` guards with `ss_tot > 1e-12`. ✅
- **`< 20` valid rows**: Task 2.4 falls back to empty template with NaN fields. ✅
- **`corr_F_*` when train < 20**: Tasks 4.3 `compute_residual_correlations` guards with `F_self.shape[0] < 3`. ✅
- **Panel 5 overplotting**: Task 5.3 uses `Scattergl` (WebGL) + markersize 4 + opacity 0.6. ✅
- **Boundary rule violations**: Task 7 reviewer rubric must grep for `theta_test` / `q_test_fit` / `k_test_fit` / `c_test_fit` references in OOS / verdict / Panel 5 code paths. ✅ (rubric in §7.1)

### 5. Spec count discrepancy (resolved)

Spec §8 says "33 columns" but the explicit list is 18 + 9 + 3 + 3 + 3 = **36**. Plan uses 36, with a note in Task 2.3 to either update the spec or amend the plan. Going with 36.

---

*Plan complete. Awaiting execution choice.*