# V0.1 Dynamics Specification Correction & Ablation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a strict 4-model dynamics ablation (Model 0/1/2/3 + placebo) on full-market stocks to determine whether β-drift correction and free-q estimation produce real OOS predictive power, or whether the dynamics methodology should be closed.

**Architecture:** New file `backtrace/projection/ablation_fit.py` orchestrates 4 model fits (Model 0/1/2/3) per stock using a unified OLS dispatcher; each stock gets 70/30 OOS split; placebo = permuted regressors with seed=42. Reuse `parameter_fit.py` helpers for movement CSV loading and kinematics reconstruction. Output 4 per-model CSVs + summary CSV + UTF-8 recommendation TXT + 4-panel plotly HTML.

**Tech Stack:** Python 3.12 (conda venv at `/c/Users/yellow/.conda/envs/venv/python.exe`), numpy, pandas, plotly, pytest. `PYTHONIOENCODING=utf-8` required.

## Phase Order (mandatory per user)

**禁止 mid-run 修改模型/阈值;最终报告 Model 0/1/2/3 + Placebo 完整结果以避免 multiple-comparison / cherry-picking。**

```
Phase 1: implementation + unit tests       ← Task 1
   ↓
Phase 2: 4-model in-sample ablation        ← Task 2
   ↓
Phase 3: 70/30 OOS                         ← Task 3
   ↓
Phase 4: permutation placebo               ← Task 4
   ↓
Phase 5: decision gate (summary + HTML + TXT) ← Task 5
```

## Global Constraints (from spec §2-§11)

- **Time indexing (write-dead)**: `β̇(t) = β(t+1) − β(t)`, `a_M(t) = v_M(t+1) − v_M(t)`, three quantities at same discrete state `t ∈ [0, T−2]`.
- **β̇·v_M as known offset, coef = 1.0 fixed** — NOT a free parameter.
- **OLS design matrices are write-dead** (4 models × Y/X/θ):
  - Model 0: `Y = a_S − β·a_M; X = [−d, −u]; θ = (k, c)`
  - Model 1: `Y = a_S − β·a_M − β̇·v_M; X = [−d, −u]; θ = (k, c)`
  - Model 2: `Y = a_S; X = [β·a_M, −d, −u]; θ = (q, k, c)`
  - Model 3: `Y = a_S − β̇·v_M; X = [β·a_M, −d, −u]; θ = (q, k, c)`
- **OOS split**: 70/30 (train = `[0, floor(0.7·(T−2))]`, test = `[floor(0.7·(T−2)), T−2]`); no overlap; no shuffle.
- **Placebo**: `np.random.default_rng(seed=42)` (fixed, no tuning); permute regressors ONLY (NOT `a_S`); 4 independent permuted indices per stock.
- **Diagnostics reuse from V0**: `cond(X)` (NOT `cond(XᵀX)`), `regressor_corr` (X column correlation), `R²` with SS_tot≈0 → NaN guard, `identification_status` (rank + cond), `fit_quality` (R² buckets).
- **Per-stock CSV schema (17 cols, identical for all 4 models)**:
  `code, name, index_code, index_tag, stock_tag, n_train, n_test, condition_number, regressor_corr, r2, identification_status, fit_quality, q_hat, k_hat, c_hat, f_self_loss, ic_real, ic_null`
- **CLI flags write-dead**: `--model {0|1|2|3}`, `--all`, `--limit N`, `--no-placebo`, `--input PATH`. **禁止** `--pick-best-ic`, `--maximize-ic`, `--feature-engineering`, `--reverse-select`.
- **Forbidden (YAGNI)**: redesign d_vec; modify F_self definition; change prediction target; change trading strategy; add new regressors beyond β·a_M, β̇·v_M, −d, −u; tune for IC; reverse-select stocks.
- **Files NOT to modify**: `_solve_ols` in `parameter_fit.py` (existing math frozen); `prediction_ode.py`; `dynamics_*.py`; `gp_factor_mining/*`.
- **Files reusable as-is**: `parameter_fit.py:_load_movement`, `parameter_fit.py:_build_kinematics`, `parameter_fit.py:build_identifiability_distribution_html`.
- **Windows GBK**: all CLI invocations must include `PYTHONIOENCODING=utf-8`.
- **Conda python path**: `/c/Users/yellow/.conda/envs/venv/python.exe` (Bash `python` not on PATH).
- **Tests in**: `tests/test_dynamics_eigen.py` (existing file, append).
- **Outputs**:
  ```
  data/projection/
  ├── kc_estimates_model0.csv
  ├── kc_estimates_model1.csv
  ├── kc_estimates_model2.csv
  ├── kc_estimates_model3.csv
  ├── kc_ablation_summary.csv          # 4×10 metric matrix
  ├── kc_ablation_recommendation.txt   # UTF-8 中文
  └── ablation_distribution.html       # 4-panel plotly
  ```

---

### Task 1: Core OLS Dispatcher + 4-Model Design Matrix

**Files:**
- Create: `backtrace/projection/ablation_fit.py` (initial skeleton)
- Modify: `tests/test_dynamics_eigen.py` (append 8 tests)

**Interfaces:**
- Consumes: existing `_load_movement` and `_build_kinematics` from `parameter_fit.py`
- Produces:
  - `ols_fit(X, Y) -> (theta, f_residual_loss, n_valid, rank, condition_number, regressor_corr, r2)` — generic OLS dispatcher
  - `_build_kinematics_ext(delta_u, delta_v, beta) -> (u_vec, d_vec, a_u_vec, a_v_vec, beta_dot_vM_vec)` — extended kinematics
  - `build_design_model_0/1/2/3(u_vec, d_vec, a_u_vec, a_v_vec, beta_dot_vM_vec) -> (X, Y)` — 4 design matrix constructors

- [ ] **Step 1.1: Write failing tests for `ols_fit` and design matrix builders**

Append to `tests/test_dynamics_eigen.py`:

```python
# === v0.1 — Dynamics Specification Correction & Ablation (2026-08-19 Task 1) ===
import numpy as np
from projection.ablation_fit import (
    ols_fit, build_design_model_0, build_design_model_1,
    build_design_model_2, build_design_model_3, _build_kinematics_ext,
)


def _make_ext_inputs(k_true=0.5, c_true=0.2, q_true=0.8, T=200, seed=0):
    """Synthetic 2-D data satisfying Model 3 exactly."""
    rng = np.random.default_rng(seed)
    beta = 1.2 + 0.001 * np.arange(T)            # β with mild drift
    delta_v = rng.normal(0, 1, (T, 2))
    delta_u = beta[:, None] * delta_v + rng.normal(0, 0.5, (T, 2))
    d_vec = np.zeros((T, 2)); d_vec[1:] = np.cumsum(delta_u[:-1] - beta[:-1, None]*delta_v[:-1], axis=0)
    u_vec = delta_u - beta[:, None] * delta_v
    a_u = np.full((T, 2), np.nan); a_u[:-1] = np.diff(delta_u, axis=0)
    a_v = np.full((T, 2), np.nan); a_v[:-1] = np.diff(delta_v, axis=0)
    beta_dot_vM = np.full((T, 2), np.nan)
    beta_dot_vM[:-1] = (np.diff(beta))[:, None] * delta_v[:-1]
    # a_S = q·β·a_M + β̇·v_M − k·d − c·u + ε
    eps = rng.normal(0, 0.01, (T, 2))
    a_u_new = q_true * beta[:, None] * a_v + beta_dot_vM - k_true * d_vec - c_true * u_vec + eps
    # only first T-1 rows used (last row NaN)
    a_u[:-1] = a_u_new[:-1]
    return u_vec, d_vec, a_u, a_v, beta_dot_vM


def test_build_design_model0_subtracts_beta_aM():
    u, d, au, av, bdv = _make_ext_inputs()
    X, Y = build_design_model_0(u, d, au, av, bdv)
    # Y = a_u - β·a_v, X = [-d, -u]
    assert X.shape[1] == 2
    # Last row is NaN (from au NaN) → Y last row should be NaN
    assert np.isnan(Y[-1])
    # First row should be finite (a_v[0] finite)
    assert np.isfinite(Y[0])


def test_build_design_model1_subtracts_betadot_vM():
    u, d, au, av, bdv = _make_ext_inputs()
    X, Y = build_design_model_1(u, d, au, av, bdv)
    assert X.shape[1] == 2
    # Y should equal Model 0's Y minus bdv stacked
    X0, Y0 = build_design_model_0(u, d, au, av, bdv)
    bdv_stack = np.concatenate([bdv[:, 0], bdv[:, 1]])
    np.testing.assert_allclose(np.nan_to_num(Y), np.nan_to_num(Y0 - bdv_stack), equal_nan=True)


def test_build_design_model2_keeps_aS_in_Y():
    u, d, au, av, bdv = _make_ext_inputs()
    X, Y = build_design_model_2(u, d, au, av, bdv)
    assert X.shape[1] == 3  # [β·a_M, -d, -u]


def test_build_design_model3_combines_offset_and_free_q():
    u, d, au, av, bdv = _make_ext_inputs()
    X, Y = build_design_model_3(u, d, au, av, bdv)
    assert X.shape[1] == 3
    # Y = Model 1's Y (which already has β̇·v_M subtracted)
    X1, Y1 = build_design_model_1(u, d, au, av, bdv)
    np.testing.assert_allclose(np.nan_to_num(Y), np.nan_to_num(Y1), equal_nan=True)


def test_ols_fit_recovers_k_c_model0():
    u, d, au, av, bdv = _make_ext_inputs(k_true=0.5, c_true=0.2)
    X, Y = build_design_model_0(u, d, au, av, bdv)
    mask = np.isfinite(Y)
    X_v, Y_v = X[mask], Y[mask]
    theta, f_res, n_valid, rank, cond, rcorr, r2 = ols_fit(X_v, Y_v)
    assert n_valid == mask.sum()
    assert abs(theta[0] - 0.5) < 0.05  # k_hat ≈ 0.5
    assert abs(theta[1] - 0.2) < 0.05  # c_hat ≈ 0.2


def test_ols_fit_recovers_q_k_c_model3():
    u, d, au, av, bdv = _make_ext_inputs(k_true=0.5, c_true=0.2, q_true=0.8)
    X, Y = build_design_model_3(u, d, au, av, bdv)
    mask = np.isfinite(Y)
    X_v, Y_v = X[mask], Y[mask]
    theta, *_ = ols_fit(X_v, Y_v)
    # theta = (q, k, c)
    assert abs(theta[0] - 0.8) < 0.05
    assert abs(theta[1] - 0.5) < 0.05
    assert abs(theta[2] - 0.2) < 0.05


def test_ols_fit_r2_nan_when_ss_tot_zero():
    X = np.ones((50, 2))
    Y = np.full(50, 3.14)  # constant → SS_tot = 0
    *_, r2 = ols_fit(X, Y)
    assert np.isnan(r2)


def test_ols_fit_cond_uses_X_not_XTX():
    """Verify cond(X) not cond(X.T @ X) (κ² amplifier test)."""
    X = np.array([[1.0, 1.0], [1.0 + 1e-8, 1.0], [1.0, 1.0 + 1e-8]])
    Y = np.array([1.0, 2.0, 3.0])
    *_, cond, _, _ = ols_fit(X, Y)
    expected_cond = np.linalg.cond(X)
    assert abs(cond - expected_cond) < 1e-3
    # cond(X.T @ X) would be much larger — assert that cond < 1e6
    assert cond < 1e6
```

- [ ] **Step 1.2: Run tests, verify FAIL**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py -k "v0.1 or build_design_model or ols_fit_recovers or test_ols_fit_r2 or test_ols_fit_cond" -v
```
Expected: 8 failures with `ImportError` or `ModuleNotFoundError: projection.ablation_fit`.

- [ ] **Step 1.3: Write minimal `ablation_fit.py`**

Create `backtrace/projection/ablation_fit.py`:

```python
# -*- coding: utf-8 -*-
# ablation_fit.py — Dynamics Specification Correction & Ablation (V0.1)
#
# Spec: docs/superpowers/specs/2026-08-19-dynamics-specification-correction-ablation.md
#
# 4 models:
#   Model 0: a_S = β·a_M − k·d − c·u + F_self       (status quo)
#   Model 1: a_S = β·a_M + β̇·v_M − k·d − c·u + F_self  (β-drift correction)
#   Model 2: a_S = q·β·a_M − k·d − c·u + F_self          (free q)
#   Model 3: a_S = q·β·a_M + β̇·v_M − k·d − c·u + F_self (joint)
#
# β̇·v_M 项 coef 固定 = 1.0(known offset),不作为 free param。
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
import numpy as np


def _stack_2d(arr_2d: np.ndarray) -> np.ndarray:
    """Stack Vol and Amt into single column: shape (2T,)."""
    return np.concatenate([arr_2d[:, 0], arr_2d[:, 1]])


def ols_fit(X: np.ndarray, Y: np.ndarray):
    """Generic OLS with diagnostics.

    X shape (N, p); Y shape (N,).
    Returns: (theta, f_residual_loss, n_valid, rank, condition_number, regressor_corr, r2)
    """
    N, p = X.shape
    theta, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    if theta.size == 0:
        theta = np.zeros(p)
    Y_pred = X @ theta
    F_self_pred = Y - Y_pred
    f_residual_loss = float(np.mean(F_self_pred ** 2))

    condition_number = float(np.linalg.cond(X)) if X.size > 0 else np.nan

    if X.shape[0] >= 2 and X.shape[1] >= 2:
        std = X.std(axis=0)
        if np.all(std > 1e-12):
            corr = np.corrcoef(X.T)
            regressor_corr = float(np.max(np.abs(corr[np.triu_indices_from(corr, k=1)])))
        else:
            regressor_corr = np.nan
    else:
        regressor_corr = np.nan

    y_mean = float(np.mean(Y))
    ss_tot = float(np.sum((Y - y_mean) ** 2))
    ss_res = float(np.sum(F_self_pred ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    return (theta, f_residual_loss, N, int(rank),
            condition_number, regressor_corr, r2)


def _build_kinematics_ext(delta_u, delta_v, beta):
    """Extended kinematics for 4-model ablation.

    Returns: u_vec, d_vec, a_u_vec, a_v_vec, beta_dot_vM_vec
      - u_vec, d_vec: shape (T, 2)
      - a_u_vec, a_v_vec: shape (T, 2) with last row NaN (np.diff)
      - beta_dot_vM_vec: shape (T, 2) with last row NaN
        beta_dot_vM[t] = (β[t+1] - β[t]) * v_M[t] for t ∈ [0, T-2]
    """
    T = len(delta_u)
    u_vec = delta_u - beta[:, None] * delta_v
    d_vec = np.zeros_like(delta_u)
    if T >= 2:
        d_vec[1:] = np.cumsum(u_vec[:-1], axis=0)
    a_u_vec = np.full_like(delta_u, np.nan)
    a_v_vec = np.full_like(delta_v, np.nan)
    if T >= 2:
        a_u_vec[:-1] = np.diff(delta_u, axis=0)
        a_v_vec[:-1] = np.diff(delta_v, axis=0)
    beta_dot_vM_vec = np.full((T, 2), np.nan)
    if T >= 2:
        beta_dot = beta[1:] - beta[:-1]                      # (T-1,)
        beta_dot_vM_vec[:-1] = beta_dot[:, None] * delta_v[:-1]  # (T-1, 2)
    return u_vec, d_vec, a_u_vec, a_v_vec, beta_dot_vM_vec


def build_design_model_0(u_vec, d_vec, a_u_vec, a_v_vec, beta_dot_vM_vec):
    """Model 0: Y = a_S − β·a_M, X = [−d, −u], θ = (k, c)."""
    # β·a_M = β * a_v_vec; need β per-row from a_v_vec — derive: β = a_v ? NO
    # β is implicit in Y: a_S − β·a_M = a_u_vec − β*a_v_vec. We need β here.
    # Caller must pass beta separately OR we recompute. Pass via closure — actually
    # simplest: caller precomputes (β·a_v) externally. But for clean API, accept beta.
    raise NotImplementedError("build_design_model_0: see plan — refactor to accept beta")


def build_design_model_1(u_vec, d_vec, a_u_vec, a_v_vec, beta_dot_vM_vec):
    raise NotImplementedError


def build_design_model_2(u_vec, d_vec, a_u_vec, a_v_vec, beta_dot_vM_vec):
    raise NotImplementedError


def build_design_model_3(u_vec, d_vec, a_u_vec, a_v_vec, beta_dot_vM_vec):
    raise NotImplementedError
```

**STOP** — this skeleton is incomplete. The user spec explicitly requires β in `build_design_model_*` (since Model 0/2/3 reference `β·a_M`). Refactor: change signatures to `build_design_model_N(u_vec, d_vec, a_u_vec, a_v_vec, beta, beta_dot_vM_vec)` and provide full implementation:

```python
def build_design_model_0(u_vec, d_vec, a_u_vec, a_v_vec, beta, beta_dot_vM_vec):
    """Model 0: Y = a_S − β·a_M, X = [−d, −u], θ = (k, c)."""
    beta_aM = beta[:, None] * a_v_vec
    Y_2d = a_u_vec - beta_aM
    Y = _stack_2d(Y_2d)
    d_stack = _stack_2d(d_vec)
    u_stack = _stack_2d(u_vec)
    X = np.column_stack([-d_stack, -u_stack])
    return X, Y


def build_design_model_1(u_vec, d_vec, a_u_vec, a_v_vec, beta, beta_dot_vM_vec):
    """Model 1: Y = a_S − β·a_M − β̇·v_M, X = [−d, −u], θ = (k, c)."""
    X, Y = build_design_model_0(u_vec, d_vec, a_u_vec, a_v_vec, beta, beta_dot_vM_vec)
    Y = Y - _stack_2d(beta_dot_vM_vec)
    return X, Y


def build_design_model_2(u_vec, d_vec, a_u_vec, a_v_vec, beta, beta_dot_vM_vec):
    """Model 2: Y = a_S, X = [β·a_M, −d, −u], θ = (q, k, c)."""
    Y = _stack_2d(a_u_vec)
    beta_aM_stack = _stack_2d(beta[:, None] * a_v_vec)
    d_stack = _stack_2d(d_vec)
    u_stack = _stack_2d(u_vec)
    X = np.column_stack([beta_aM_stack, -d_stack, -u_stack])
    return X, Y


def build_design_model_3(u_vec, d_vec, a_u_vec, a_v_vec, beta, beta_dot_vM_vec):
    """Model 3: Y = a_S − β̇·v_M, X = [β·a_M, −d, −u], θ = (q, k, c)."""
    Y = _stack_2d(a_u_vec - beta_dot_vM_vec)
    beta_aM_stack = _stack_2d(beta[:, None] * a_v_vec)
    d_stack = _stack_2d(d_vec)
    u_stack = _stack_2d(u_vec)
    X = np.column_stack([beta_aM_stack, -d_stack, -u_stack])
    return X, Y
```

Replace the four `raise NotImplementedError` stubs with the implementations above.

- [ ] **Step 1.4: Run tests, verify PASS**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py -k "v0.1 or build_design_model or ols_fit_recovers or test_ols_fit_r2 or test_ols_fit_cond" -v
```
Expected: 8 PASS.

- [ ] **Step 1.5: Commit**

```bash
git add backtrace/projection/ablation_fit.py tests/test_dynamics_eigen.py
git commit -m "feat(projection): V0.1 Task 1 — OLS dispatcher + 4 design matrix builders + 8 unit tests"
```

---

### Task 2: In-Sample 4-Model Fit + CSV Output (Phase 2)

**Files:**
- Modify: `backtrace/projection/ablation_fit.py` (add `fit_one_in_sample`, `write_in_sample_csv`)
- Modify: `tests/test_dynamics_eigen.py` (append 1 in-sample integration test)

**Interfaces:**
- Produces:
  - `fit_one_in_sample(movement_csv, stock_tag, index_tag, model_id) -> dict`
  - `write_in_sample_csvs(movement_csvs, output_dir)` → writes 4 CSVs at `kc_estimates_model{0,1,2,3}.csv` (17 cols, ic_real/ic_null = NaN at this stage)
  - `compute_identification_status(rank, cond)` → `{well_conditioned, ill_conditioned, unidentifiable, singular}`
  - `compute_fit_quality(r2)` → `{good, weak, poor, uninformative}`

- [ ] **Step 2.1: Write failing integration test for in-sample fit**

Append to `tests/test_dynamics_eigen.py`:

```python
def test_in_sample_fit_5_synthetic_stocks(tmp_path):
    """Process 5 synthetic stocks through all 4 models, verify 4 CSV outputs with 17 cols."""
    import tempfile, os
    from projection.ablation_fit import write_in_sample_csvs
    
    # Build 5 synthetic movement CSVs
    mv_dir = tmp_path / "movement"
    mv_dir.mkdir()
    targets = []
    for i in range(5):
        rng = np.random.default_rng(seed=i)
        T = 100
        beta = 1.0 + 0.001 * np.arange(T)
        delta_v = rng.normal(0, 1, (T, 2))
        delta_u = beta[:, None] * delta_v + rng.normal(0, 0.5, (T, 2))
        df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=T),
            'Move_Delta_Vol_idx': delta_v[:, 0],
            'Move_Delta_Amt_idx': delta_v[:, 1],
            'Move_Delta_Vol_stk': delta_u[:, 0],
            'Move_Delta_Amt_stk': delta_u[:, 1],
            'Move_Proj_Coeff': beta,
        })
        csv_path = mv_dir / f"movement_idx_stk{i:06d}.csv"
        df.to_csv(csv_path, index=False)
        targets.append((f"00000{i}.SZ", f"Stock{i}", str(csv_path), 'idx', f'stk{i:06d}', '000001.SH'))
    
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    write_in_sample_csvs(targets, str(out_dir))
    
    for m in range(4):
        path = out_dir / f"kc_estimates_model{m}.csv"
        assert path.exists(), f"missing {path}"
        df = pd.read_csv(path)
        assert len(df) == 5
        assert len(df.columns) == 17  # spec §5 schema
        # ic_real / ic_null are NaN at this stage (Tasks 3+4 will populate)
        assert df['ic_real'].isna().all()
        assert df['ic_null'].isna().all()
        # Models 0/1 q_hat = 1.0; Models 2/3 q_hat = OLS estimate (varies)
        if m in (0, 1):
            assert (df['q_hat'] == 1.0).all()
```

- [ ] **Step 2.2: Run test, verify FAIL**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py::test_in_sample_fit_5_synthetic_stocks -v
```
Expected: FAIL with `ImportError` or `AttributeError: module 'projection.ablation_fit' has no attribute 'write_in_sample_csvs'`.

- [ ] **Step 2.3: Implement `fit_one_in_sample` + classification helpers + `write_in_sample_csvs`**

Append to `backtrace/projection/ablation_fit.py`:

```python
import pandas as pd
from typing import List, Tuple

CSV_OUT_DIR = 'data/projection'

# Schema (17 cols, identical across all 4 models)
CSV_COLUMNS = [
    'code', 'name', 'index_code', 'index_tag', 'stock_tag',
    'n_train', 'n_test', 'condition_number', 'regressor_corr', 'r2',
    'identification_status', 'fit_quality',
    'q_hat', 'k_hat', 'c_hat', 'f_self_loss',
    'ic_real', 'ic_null',
]

BUILDERS = {
    0: build_design_model_0,
    1: build_design_model_1,
    2: build_design_model_2,
    3: build_design_model_3,
}


def compute_identification_status(rank: int, cond: float) -> str:
    if rank < 2 or not np.isfinite(cond):
        return 'singular'
    if cond < 1e3:
        return 'well_conditioned'
    if cond < 1e5:
        return 'ill_conditioned'
    return 'unidentifiable'


def compute_fit_quality(r2: float) -> str:
    if not np.isfinite(r2):
        return 'uninformative'
    if r2 >= 0.1:
        return 'good'
    if r2 >= 0.01:
        return 'weak'
    return 'poor'


def _read_movement(movement_csv: str, stock_tag: str, index_tag: str):
    """Read movement CSV → (delta_u, delta_v, beta)."""
    df = pd.read_csv(movement_csv)
    delta_u = df[[f'Move_Delta_Vol_{stock_tag}',
                  f'Move_Delta_Amt_{stock_tag}']].to_numpy()
    delta_v = df[[f'Move_Delta_Vol_{index_tag}',
                  f'Move_Delta_Amt_{index_tag}']].to_numpy()
    beta = df['Move_Proj_Coeff'].to_numpy()
    return delta_u, delta_v, beta


def fit_one_in_sample(movement_csv: str, stock_tag: str, index_tag: str,
                      code: str, name: str, index_code: str, model_id: int) -> dict:
    """Run in-sample fit for one stock × one model.

    Returns dict with 17 CSV columns (ic_real/ic_null = NaN at this stage).
    """
    delta_u, delta_v, beta = _read_movement(movement_csv, stock_tag, index_tag)
    u_vec, d_vec, a_u_vec, a_v_vec, bdv_vec = _build_kinematics_ext(delta_u, delta_v, beta)
    X, Y = BUILDERS[model_id](u_vec, d_vec, a_u_vec, a_v_vec, beta, bdv_vec)
    mask = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
    n_valid = int(mask.sum())

    if n_valid < 20:
        return {col: np.nan for col in CSV_COLUMNS} | {
            'code': code, 'name': name, 'index_code': index_code,
            'index_tag': index_tag, 'stock_tag': stock_tag,
            'n_train': n_valid, 'n_test': 0,
            'q_hat': 1.0 if model_id in (0, 1) else np.nan,
            'identification_status': 'singular' if n_valid < 2 else 'ill_conditioned',
            'fit_quality': 'uninformative',
        }

    X_v, Y_v = X[mask], Y[mask]
    theta, f_res, n_v, rank, cond, rcorr, r2 = ols_fit(X_v, Y_v)

    if model_id in (0, 1):
        k_hat, c_hat = float(theta[0]), float(theta[1])
        q_hat = 1.0
    else:  # Model 2/3
        q_hat, k_hat, c_hat = float(theta[0]), float(theta[1]), float(theta[2])

    return {
        'code': code, 'name': name, 'index_code': index_code,
        'index_tag': index_tag, 'stock_tag': stock_tag,
        'n_train': n_v, 'n_test': 0,            # populated in Task 3
        'condition_number': cond,
        'regressor_corr': rcorr,
        'r2': r2,
        'identification_status': compute_identification_status(rank, cond),
        'fit_quality': compute_fit_quality(r2),
        'q_hat': q_hat, 'k_hat': k_hat, 'c_hat': c_hat,
        'f_self_loss': f_res,
        'ic_real': np.nan, 'ic_null': np.nan,
    }


def write_in_sample_csvs(targets: List[Tuple], output_dir: str):
    """Write 4 in-sample CSVs to output_dir/kc_estimates_model{0,1,2,3}.csv."""
    os.makedirs(output_dir, exist_ok=True)
    rows_by_model = {m: [] for m in range(4)}
    for code, name, mv_csv, index_tag, stock_tag, index_code in targets:
        for m in range(4):
            row = fit_one_in_sample(mv_csv, stock_tag, index_tag, code, name, index_code, m)
            rows_by_model[m].append(row)
    for m in range(4):
        df = pd.DataFrame(rows_by_model[m], columns=CSV_COLUMNS)
        df.to_csv(os.path.join(output_dir, f'kc_estimates_model{m}.csv'),
                  index=False, encoding='utf-8')
```

- [ ] **Step 2.4: Run test, verify PASS**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py::test_in_sample_fit_5_synthetic_stocks -v
```
Expected: PASS.

- [ ] **Step 2.5: Commit**

```bash
git add backtrace/projection/ablation_fit.py tests/test_dynamics_eigen.py
git commit -m "feat(projection): V0.1 Task 2 — in-sample 4-model fit + 17-col CSV output"
```

---

### Task 3: OOS 70/30 Split + `ic_real` Computation (Phase 3)

**Files:**
- Modify: `backtrace/projection/ablation_fit.py` (add OOS split, refit on train, predict on test)
- Modify: `tests/test_dynamics_eigen.py` (append 3 OOS tests)

**Interfaces:**
- Produces:
  - `oos_split_indices(n_valid: int, train_frac: float = 0.7) -> (train_idx, test_idx)`
  - `compute_spearman_ic(y_pred: np.ndarray, y_actual: np.ndarray) -> float`
  - `fit_one_oos(...)` → extends `fit_one_in_sample` with OOS fit + IC; populates `n_train`, `n_test`, `ic_real`
  - `write_oos_csvs(targets, output_dir)` → overwrites 4 CSVs with OOS fields populated

- [ ] **Step 3.1: Write failing OOS tests**

Append to `tests/test_dynamics_eigen.py`:

```python
from scipy.stats import spearmanr


def test_oos_split_no_overlap():
    from projection.ablation_fit import oos_split_indices
    train, test = oos_split_indices(n_valid=100, train_frac=0.7)
    assert len(train) + len(test) == 100
    assert set(train).isdisjoint(set(test))
    assert max(train) < min(test)  # train < test in index


def test_oos_split_70_30():
    from projection.ablation_fit import oos_split_indices
    train, test = oos_split_indices(n_valid=100, train_frac=0.7)
    assert len(train) == 70
    assert len(test) == 30


def test_oos_perfect_prediction_high_ic():
    """Synthetic Model 3 data → OOS IC ≈ 1."""
    from projection.ablation_fit import fit_one_oos
    u, d, au, av, bdv = _make_ext_inputs(k_true=0.5, c_true=0.2, q_true=0.8, T=200)
    # Construct minimal movement dict-like
    import tempfile, os
    mv_dir = tempfile.mkdtemp()
    csv_path = os.path.join(mv_dir, "movement_idx_stk000001.csv")
    rng = np.random.default_rng(0)
    T = 200
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
    row = fit_one_oos(csv_path, 'stk000001', 'idx', '000001.SZ', 'T', '000001.SH', model_id=3)
    assert row['n_train'] > 0 and row['n_test'] > 0
    assert row['ic_real'] > 0.5  # strong signal, should be high
```

- [ ] **Step 3.2: Run tests, verify FAIL**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py -k "test_oos_split or test_oos_perfect" -v
```
Expected: 3 FAIL with `ImportError` or `AttributeError`.

- [ ] **Step 3.3: Implement OOS split + Spearman IC + `fit_one_oos`**

Append to `backtrace/projection/ablation_fit.py`:

```python
from scipy.stats import spearmanr


def oos_split_indices(n_valid: int, train_frac: float = 0.7):
    """70/30 split. Train = [0, floor(0.7·n_valid)), test = [floor(0.7·n_valid), n_valid).
    Returns (train_idx, test_idx) as numpy int arrays. NO overlap, NO shuffle.
    """
    n_train = int(np.floor(train_frac * n_valid))
    train_idx = np.arange(0, n_train, dtype=int)
    test_idx = np.arange(n_train, n_valid, dtype=int)
    return train_idx, test_idx


def compute_spearman_ic(y_pred: np.ndarray, y_actual: np.ndarray) -> float:
    """Spearman rank correlation between predicted and actual a_S (2-D)."""
    if len(y_pred) < 3:
        return np.nan
    rho, _ = spearmanr(y_pred, y_actual)
    return float(rho) if np.isfinite(rho) else np.nan


def fit_one_oos(movement_csv: str, stock_tag: str, index_tag: str,
                code: str, name: str, index_code: str, model_id: int) -> dict:
    """Run 70/30 OOS fit for one stock × one model.

    Algorithm:
      1. Reconstruct kinematics.
      2. Build (X, Y) per model.
      3. mask = isfinite(Y) & all-isfinite(X).
      4. Split valid indices into train/test (70/30, no overlap).
      5. OLS on train → θ.
      6. Predict on test: ŷ = X_test · θ.
      7. IC = Spearman(ŷ, Y_test).
      8. Use TRAIN set's cond/r2/diagnostics (NOT OOS).
    """
    delta_u, delta_v, beta = _read_movement(movement_csv, stock_tag, index_tag)
    u_vec, d_vec, a_u_vec, a_v_vec, bdv_vec = _build_kinematics_ext(delta_u, delta_v, beta)
    X, Y = BUILDERS[model_id](u_vec, d_vec, a_u_vec, a_v_vec, beta, bdv_vec)
    mask = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
    valid_indices = np.where(mask)[0]
    n_valid = len(valid_indices)

    if n_valid < 20:
        return fit_one_in_sample(movement_csv, stock_tag, index_tag, code, name, index_code, model_id)

    train_idx_rel, test_idx_rel = oos_split_indices(n_valid, train_frac=0.7)
    train_idx_abs = valid_indices[train_idx_rel]
    test_idx_abs = valid_indices[test_idx_rel]

    X_train, Y_train = X[train_idx_abs], Y[train_idx_abs]
    X_test, Y_test = X[test_idx_abs], Y[test_idx_abs]

    theta, f_res, n_train_v, rank, cond, rcorr, r2 = ols_fit(X_train, Y_train)

    # OOS prediction
    Y_pred_test = X_test @ theta
    ic_real = compute_spearman_ic(Y_pred_test, Y_test)

    if model_id in (0, 1):
        k_hat, c_hat = float(theta[0]), float(theta[1])
        q_hat = 1.0
    else:
        q_hat, k_hat, c_hat = float(theta[0]), float(theta[1]), float(theta[2])

    return {
        'code': code, 'name': name, 'index_code': index_code,
        'index_tag': index_tag, 'stock_tag': stock_tag,
        'n_train': n_train_v, 'n_test': len(test_idx_abs),
        'condition_number': cond, 'regressor_corr': rcorr, 'r2': r2,
        'identification_status': compute_identification_status(rank, cond),
        'fit_quality': compute_fit_quality(r2),
        'q_hat': q_hat, 'k_hat': k_hat, 'c_hat': c_hat,
        'f_self_loss': f_res,
        'ic_real': ic_real, 'ic_null': np.nan,   # Task 4 populates
    }


def write_oos_csvs(targets: List[Tuple], output_dir: str):
    """Write 4 OOS CSVs to output_dir/kc_estimates_model{0,1,2,3}.csv (overwrites Task 2 output)."""
    os.makedirs(output_dir, exist_ok=True)
    rows_by_model = {m: [] for m in range(4)}
    for code, name, mv_csv, index_tag, stock_tag, index_code in targets:
        for m in range(4):
            row = fit_one_oos(mv_csv, stock_tag, index_tag, code, name, index_code, m)
            rows_by_model[m].append(row)
    for m in range(4):
        df = pd.DataFrame(rows_by_model[m], columns=CSV_COLUMNS)
        df.to_csv(os.path.join(output_dir, f'kc_estimates_model{m}.csv'),
                  index=False, encoding='utf-8')
```

- [ ] **Step 3.4: Run tests, verify PASS**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py -k "test_oos_split or test_oos_perfect" -v
```
Expected: 3 PASS.

- [ ] **Step 3.5: Commit**

```bash
git add backtrace/projection/ablation_fit.py tests/test_dynamics_eigen.py
git commit -m "feat(projection): V0.1 Task 3 — 70/30 OOS split + Spearman IC + 3 tests"
```

---

### Task 4: Placebo Test (Permutation Baseline, seed=42) (Phase 4)

**Files:**
- Modify: `backtrace/projection/ablation_fit.py` (add `permute_regressors`, `fit_one_with_placebo`)
- Modify: `tests/test_dynamics_eigen.py` (append 4 placebo tests)

**Interfaces:**
- Produces:
  - `PLACEBO_SEED = 42` (module constant, write-dead)
  - `permute_regressors(X: np.ndarray, Y_unused: np.ndarray, seed: int = 42) -> X_perm`
  - `fit_one_with_placebo(...)` → extends `fit_one_oos` with permuted IC; populates `ic_null`
  - `write_ablation_csvs(targets, output_dir)` → final 4 CSVs with both `ic_real` and `ic_null`

- [ ] **Step 4.1: Write failing placebo tests**

Append to `tests/test_dynamics_eigen.py`:

```python
def test_placebo_seed_is_42():
    from projection import ablation_fit
    assert ablation_fit.PLACEBO_SEED == 42


def test_placebo_permutes_regressors_not_Y():
    """Verifies that Y is NOT shuffled when permuting regressors."""
    from projection.ablation_fit import permute_regressors
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (100, 3))
    Y = np.arange(100, dtype=float)
    X_perm = permute_regressors(X, Y, seed=42)
    # Y should NOT appear in X_perm columns
    assert X_perm.shape == X.shape
    # X_perm rows are shuffled version of X (same column marginals)
    assert not np.allclose(X, X_perm)
    # Re-permuting with same seed → same X_perm (deterministic)
    X_perm2 = permute_regressors(X, Y, seed=42)
    np.testing.assert_array_equal(X_perm, X_perm2)


def test_placebo_pure_noise_no_signal():
    """a_S = pure random → ic_real ≈ ic_null ≈ 0."""
    from projection.ablation_fit import fit_one_with_placebo
    import tempfile, os
    mv_dir = tempfile.mkdtemp()
    csv_path = os.path.join(mv_dir, "movement_idx_stk000002.csv")
    T = 200
    beta = np.ones(T)
    rng = np.random.default_rng(7)
    delta_v = rng.normal(0, 1, (T, 2))
    delta_u = rng.normal(0, 1, (T, 2))   # NOT beta·a_v + noise — pure noise
    pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=T),
        'Move_Delta_Vol_idx': delta_v[:, 0],
        'Move_Delta_Amt_idx': delta_v[:, 1],
        'Move_Delta_Vol_stk': delta_u[:, 0],
        'Move_Delta_Amt_stk': delta_u[:, 1],
        'Move_Proj_Coeff': beta,
    }).to_csv(csv_path, index=False)
    row = fit_one_with_placebo(csv_path, 'stk000002', 'idx', '000002.SZ', 'T', '000001.SH', model_id=3)
    assert abs(row['ic_real']) < 0.2
    assert abs(row['ic_null']) < 0.2


def test_placebo_real_signal_beats_null():
    """a_S = Model 3 with true signal → ic_real > ic_null + 0.1."""
    from projection.ablation_fit import fit_one_with_placebo
    import tempfile, os
    mv_dir = tempfile.mkdtemp()
    csv_path = os.path.join(mv_dir, "movement_idx_stk000003.csv")
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
    row = fit_one_with_placebo(csv_path, 'stk000003', 'idx', '000003.SZ', 'T', '000001.SH', model_id=3)
    assert row['ic_real'] - row['ic_null'] > 0.1
```

- [ ] **Step 4.2: Run tests, verify FAIL**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py -k "test_placebo" -v
```
Expected: 4 FAIL with `ImportError` or `AttributeError`.

- [ ] **Step 4.3: Implement `PLACEBO_SEED` + `permute_regressors` + `fit_one_with_placebo`**

Append to `backtrace/projection/ablation_fit.py`:

```python
PLACEBO_SEED = 42  # WRITE-DEAD per spec §4: fixed seed, no tuning


def permute_regressors(X: np.ndarray, Y_unused: np.ndarray = None, seed: int = PLACEBO_SEED):
    """Permute ROWS of X independently per column (destroys temporal coupling).
    Y is NOT touched (a_S stays in original order).

    Returns X_perm with same shape as X.
    """
    rng = np.random.default_rng(seed)
    X_perm = np.empty_like(X)
    for j in range(X.shape[1]):
        perm = rng.permutation(X.shape[0])
        X_perm[:, j] = X[perm, j]
    return X_perm


def fit_one_with_placebo(movement_csv: str, stock_tag: str, index_tag: str,
                         code: str, name: str, index_code: str, model_id: int) -> dict:
    """Run OOS fit + placebo fit for one stock × one model.

    1. Reconstruct kinematics, build (X, Y).
    2. mask finite rows.
    3. 70/30 split → train/test indices.
    4. Real: OLS on (X_train, Y_train) → θ → predict on X_test → ic_real.
    5. Placebo: X_perm = permute_regressors(X_train) → OLS on (X_perm, Y_train) → θ_null
       → predict on X_test → ic_null.
    """
    delta_u, delta_v, beta = _read_movement(movement_csv, stock_tag, index_tag)
    u_vec, d_vec, a_u_vec, a_v_vec, bdv_vec = _build_kinematics_ext(delta_u, delta_v, beta)
    X, Y = BUILDERS[model_id](u_vec, d_vec, a_u_vec, a_v_vec, beta, bdv_vec)
    mask = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
    valid_indices = np.where(mask)[0]
    n_valid = len(valid_indices)

    if n_valid < 20:
        return fit_one_in_sample(movement_csv, stock_tag, index_tag, code, name, index_code, model_id)

    train_idx_rel, test_idx_rel = oos_split_indices(n_valid, train_frac=0.7)
    train_idx_abs = valid_indices[train_idx_rel]
    test_idx_abs = valid_indices[test_idx_rel]

    X_train, Y_train = X[train_idx_abs], Y[train_idx_abs]
    X_test, Y_test = X[test_idx_abs], Y[test_idx_abs]

    # Real fit
    theta, f_res, n_train_v, rank, cond, rcorr, r2 = ols_fit(X_train, Y_train)
    Y_pred_test = X_test @ theta
    ic_real = compute_spearman_ic(Y_pred_test, Y_test)

    # Placebo fit: permute X_train rows, keep Y_train order
    X_train_perm = permute_regressors(X_train, Y_train, seed=PLACEBO_SEED)
    theta_null, *_ = ols_fit(X_train_perm, Y_train)
    Y_pred_test_null = X_test @ theta_null
    ic_null = compute_spearman_ic(Y_pred_test_null, Y_test)

    if model_id in (0, 1):
        k_hat, c_hat = float(theta[0]), float(theta[1])
        q_hat = 1.0
    else:
        q_hat, k_hat, c_hat = float(theta[0]), float(theta[1]), float(theta[2])

    return {
        'code': code, 'name': name, 'index_code': index_code,
        'index_tag': index_tag, 'stock_tag': stock_tag,
        'n_train': n_train_v, 'n_test': len(test_idx_abs),
        'condition_number': cond, 'regressor_corr': rcorr, 'r2': r2,
        'identification_status': compute_identification_status(rank, cond),
        'fit_quality': compute_fit_quality(r2),
        'q_hat': q_hat, 'k_hat': k_hat, 'c_hat': c_hat,
        'f_self_loss': f_res,
        'ic_real': ic_real, 'ic_null': ic_null,
    }


def write_ablation_csvs(targets: List[Tuple], output_dir: str):
    """Final 4 CSVs with both ic_real and ic_null populated."""
    os.makedirs(output_dir, exist_ok=True)
    rows_by_model = {m: [] for m in range(4)}
    for code, name, mv_csv, index_tag, stock_tag, index_code in targets:
        for m in range(4):
            row = fit_one_with_placebo(mv_csv, stock_tag, index_tag, code, name, index_code, m)
            rows_by_model[m].append(row)
    for m in range(4):
        df = pd.DataFrame(rows_by_model[m], columns=CSV_COLUMNS)
        df.to_csv(os.path.join(output_dir, f'kc_estimates_model{m}.csv'),
                  index=False, encoding='utf-8')
```

- [ ] **Step 4.4: Run tests, verify PASS**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py -k "test_placebo" -v
```
Expected: 4 PASS.

- [ ] **Step 4.5: Commit**

```bash
git add backtrace/projection/ablation_fit.py tests/test_dynamics_eigen.py
git commit -m "feat(projection): V0.1 Task 4 — permutation placebo (seed=42) + 4 tests"
```

---

### Task 5: Summary CSV + HTML + Recommendation TXT + CLI (Phase 5)

**Files:**
- Modify: `backtrace/projection/ablation_fit.py` (add `summarize_ablation`, `build_ablation_html`, `write_recommendation_txt`, `main`, `parse_args`)
- Modify: `tests/test_dynamics_eigen.py` (append 1 CLI smoke test)

**Interfaces:**
- Produces:
  - `summarize_ablation(csv_paths: dict) -> pd.DataFrame` → 4×10 metric matrix
  - `build_ablation_html(summary_df, output_path) -> str`
  - `write_recommendation_txt(summary_df, output_path) -> str` (UTF-8 Chinese per spec §10 decision tree)
  - `main()` → CLI entry point
  - `parse_args()` → argparse

- [ ] **Step 5.1: Write failing CLI smoke test**

Append to `tests/test_dynamics_eigen.py`:

```python
def test_cli_smoke_full_ablation(tmp_path):
    """Run --all --limit 5 against 5 synthetic stocks, verify all outputs exist."""
    import subprocess, tempfile, os
    mv_dir = tmp_path / "movement"
    mv_dir.mkdir()
    for i in range(5):
        rng = np.random.default_rng(seed=i)
        T = 100
        beta = 1.0 + 0.001 * np.arange(T)
        delta_v = rng.normal(0, 1, (T, 2))
        delta_u = beta[:, None] * delta_v + rng.normal(0, 0.5, (T, 2))
        pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=T),
            'Move_Delta_Vol_idx': delta_v[:, 0],
            'Move_Delta_Amt_idx': delta_v[:, 1],
            'Move_Delta_Vol_stk': delta_u[:, 0],
            'Move_Delta_Amt_stk': delta_u[:, 1],
            'Move_Proj_Coeff': beta,
        }).to_csv(mv_dir / f"movement_idx_stk{i:06d}.csv", index=False)

    out_dir = tmp_path / "out"
    result = subprocess.run([
        "/c/Users/yellow/.conda/envs/venv/python.exe",
        "backtrace/projection/ablation_fit.py",
        "--all", "--limit", "5",
        "--movement-dir", str(mv_dir),
        "--output-dir", str(out_dir),
    ], capture_output=True, text=True, timeout=120, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert result.returncode == 0, f"stderr: {result.stderr}"
    # 4 per-model CSVs
    for m in range(4):
        assert (out_dir / f"kc_estimates_model{m}.csv").exists()
    # summary CSV
    assert (out_dir / "kc_ablation_summary.csv").exists()
    # recommendation TXT (UTF-8 Chinese)
    assert (out_dir / "kc_ablation_recommendation.txt").exists()
    # HTML
    assert (out_dir / "ablation_distribution.html").exists()
```

- [ ] **Step 5.2: Run test, verify FAIL**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_smoke_full_ablation -v
```
Expected: FAIL (script not exists or `main()` not implemented).

- [ ] **Step 5.3: Implement summary + HTML + TXT + CLI**

Append to `backtrace/projection/ablation_fit.py`:

```python
import argparse


def list_movement_csvs(movement_dir: str):
    """Scan movement_dir for movement_*.csv; return list of (code, name, mv_csv, idx_tag, stk_tag, idx_code)."""
    targets = []
    for fn in sorted(os.listdir(movement_dir)):
        if not (fn.startswith('movement_') and fn.endswith('.csv')):
            continue
        stem = fn[len('movement_'):-len('.csv')]
        parts = stem.split('_')
        if len(parts) < 2:
            continue
        index_tag = parts[0]
        stock_tag = '_'.join(parts[1:])
        suf = stock_tag[:6]
        code = stock_tag + ('.SH' if suf.startswith(('6', '9', '5')) else '.SZ')
        idx_code = index_tag + ('.SH' if suf.startswith(('6', '9', '5')) else '.SZ')
        targets.append((code, None, os.path.join(movement_dir, fn),
                        index_tag, stock_tag, idx_code))
    return targets


def summarize_ablation(csv_paths: dict) -> pd.DataFrame:
    """4×10 metric matrix from 4 per-model CSVs.

    csv_paths: {0: path_model0, 1: ..., 2: ..., 3: ...}
    Returns DataFrame with rows = metrics, cols = model_N.
    """
    metrics = ['median_r2', 'p25_r2', 'p75_r2', 'median_cond',
               'median_ic_real', 'median_ic_null', 'median_delta_ic',
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


def build_ablation_html(summary_df: pd.DataFrame, output_path: str) -> str:
    """4-panel plotly dark template: R² / IC_real vs IC_null / ΔIC / q̂ distribution."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        'R² distribution (4 models)',
        'IC_real vs IC_null (4 models)',
        'ΔIC = IC_real − IC_null (3 corrections)',
        '|q̂ − 1| distribution (Models 2/3)',
    ))
    colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728'}

    # Panel 1: median R² bar
    r2_med = summary_df.loc['median_r2'].astype(float)
    fig.add_trace(go.Bar(x=r2_med.index, y=r2_med.values,
                         marker_color=[colors[i] for i in range(4)],
                         name='median R²', showlegend=False), row=1, col=1)

    # Panel 2: IC_real vs IC_null bar grouped
    ic_real = summary_df.loc['median_ic_real'].astype(float)
    ic_null = summary_df.loc['median_ic_null'].astype(float)
    fig.add_trace(go.Bar(x=ic_real.index, y=ic_real.values, name='IC_real',
                         marker_color='steelblue'), row=1, col=2)
    fig.add_trace(go.Bar(x=ic_null.index, y=ic_null.values, name='IC_null',
                         marker_color='lightgray'), row=1, col=2)

    # Panel 3: ΔIC for Models 1/2/3 vs Model 0
    delta_ic = summary_df.loc['delta_ic_vs_m0'].dropna().astype(float)
    fig.add_trace(go.Bar(x=delta_ic.index, y=delta_ic.values,
                         marker_color=[colors[i] for i in range(1, 4)],
                         name='ΔIC vs M0', showlegend=False), row=2, col=1)

    # Panel 4: |q̂ − 1| for Models 2/3
    abs_q = summary_df.loc['median_abs_q_minus_1'].dropna().astype(float)
    fig.add_trace(go.Bar(x=abs_q.index, y=abs_q.values,
                         marker_color=[colors[i] for i in range(2, 4)],
                         name='|q̂ − 1|', showlegend=False), row=2, col=2)

    fig.update_layout(template='plotly_dark', height=800, title='Dynamics Specification Ablation (V0.1)')
    fig.write_html(output_path)
    return output_path


def write_recommendation_txt(summary_df: pd.DataFrame, output_path: str) -> str:
    """UTF-8 Chinese decision recommendation per spec §10 decision tree."""
    median_r2_m1 = float(summary_df.loc['median_r2', 'model_1'])
    median_r2_m2 = float(summary_df.loc['median_r2', 'model_2'])
    median_r2_m3 = float(summary_df.loc['median_r2', 'model_3'])
    median_r2_m0 = float(summary_df.loc['median_r2', 'model_0'])
    abs_q_m2 = float(summary_df.loc['median_abs_q_minus_1', 'model_2']) if 'median_abs_q_minus_1' in summary_df.index else np.nan
    delta_ic_m3 = float(summary_df.loc['delta_ic_vs_m0', 'model_3'])

    # Step 1: β-drift (ΔR²_M1)
    step1_pass = (median_r2_m1 - median_r2_m0) > 0.005
    # Step 2: q ≠ 1
    step2_pass = abs_q_m2 > 0.1
    # Step 3: placebo ΔIC
    step3_pass = delta_ic_m3 > 0.02

    if step1_pass and step2_pass and step3_pass:
        verdict = 'GO — Model 3 三个 correction 都显著,推进 V0.2 接进 V6'
    elif step1_pass and step3_pass:
        verdict = 'PARTIAL — β-drift 有用,但 q=1 不显著错;Model 1 优先'
    elif step2_pass and step3_pass:
        verdict = 'PARTIAL — q≠1 有用,但 β-drift 不显著;Model 2 优先'
    elif step3_pass:
        verdict = 'MARGINAL — 仅 ΔIC 显著,单 correction 不够;需要 Model 3'
    elif step1_pass or step2_pass:
        verdict = 'WEAK — correction 改了 R² 但 OOS ΔIC < 0.02,可能是过拟合'
    else:
        verdict = 'STOP — 三个 correction 都无效,动力学方法论应收口(B)'

    lines = [
        '=' * 60,
        'Dynamics Specification Correction — Recommendation',
        '=' * 60,
        f'Run date:  {pd.Timestamp.now().strftime("%Y-%m-%d")}',
        '',
        '--- Per-model median R² ---',
        f'  Model 0 (baseline):        {median_r2_m0:.4f}',
        f'  Model 1 (β-drift):         {median_r2_m1:.4f}   ΔR²={median_r2_m1 - median_r2_m0:+.4f}',
        f'  Model 2 (free q):          {median_r2_m2:.4f}   ΔR²={median_r2_m2 - median_r2_m0:+.4f}',
        f'  Model 3 (joint):           {median_r2_m3:.4f}   ΔR²={median_r2_m3 - median_r2_m0:+.4f}',
        '',
        '--- |q̂ − 1| (Models 2/3) ---',
        f'  Model 2 median |q̂−1|: {abs_q_m2:.4f}',
        '',
        '--- Decision tree (spec §10) ---',
        f'  Step 1 (β-drift ΔR² > 0.005): {"PASS" if step1_pass else "FAIL"}',
        f'  Step 2 (|q̂−1| > 0.1):         {"PASS" if step2_pass else "FAIL"}',
        f'  Step 3 (ΔIC vs M0 > 0.02):    {"PASS" if step3_pass else "FAIL"}',
        '',
        '--- Verdict ---',
        f'  {verdict}',
        '=' * 60,
    ]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return output_path


def parse_args():
    p = argparse.ArgumentParser(
        description='V0.1 — Dynamics Specification Correction & Ablation'
    )
    p.add_argument('--model', type=int, choices=[0, 1, 2, 3], default=None,
                   help='Single model to run; default = None (= require --all)')
    p.add_argument('--all', action='store_true',
                   help='Run all 4 models + summary + HTML + TXT')
    p.add_argument('--limit', type=int, default=0,
                   help='Max stocks to process; 0 = all')
    p.add_argument('--no-placebo', action='store_true',
                   help='Skip placebo test (faster smoke)')
    p.add_argument('--movement-dir', default='data/projection',
                   help='Directory containing movement_*.csv')
    p.add_argument('--output-dir', default='data/projection',
                   help='Output directory for CSV / HTML / TXT')
    return p.parse_args()


def main():
    args = parse_args()
    if not (args.model is not None or args.all):
        raise SystemExit('Must pass --model N or --all')

    targets = list_movement_csvs(args.movement_dir)
    if args.limit > 0:
        targets = targets[:args.limit]
    print(f'输入: {args.movement_dir}/movement_*.csv')
    print(f'目标: {len(targets)} 只 (limit={args.limit})')

    os.makedirs(args.output_dir, exist_ok=True)

    if args.no_placebo:
        # Use OOS without placebo (Task 3 path)
        write_oos_csvs(targets, args.output_dir)
    else:
        # Full ablation (Task 4 path)
        write_ablation_csvs(targets, args.output_dir)

    # Summary
    csv_paths = {m: os.path.join(args.output_dir, f'kc_estimates_model{m}.csv') for m in range(4)}
    summary_df = summarize_ablation(csv_paths)
    summary_df.to_csv(os.path.join(args.output_dir, 'kc_ablation_summary.csv'),
                      encoding='utf-8')

    # HTML
    html_path = build_ablation_html(summary_df,
                                    os.path.join('backtrace/outputs', 'ablation_distribution.html'))
    # TXT recommendation
    txt_path = write_recommendation_txt(summary_df,
                                        os.path.join(args.output_dir, 'kc_ablation_recommendation.txt'))
    print(f'Summary: {args.output_dir}/kc_ablation_summary.csv')
    print(f'HTML:    {html_path}')
    print(f'TXT:     {txt_path}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 5.4: Run smoke test, verify PASS**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_smoke_full_ablation -v
```
Expected: PASS.

- [ ] **Step 5.5: Run full V0.1 test suite + verify integration**

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py -v
```
Expected: all v0.1 tests pass; full suite (96 + 16 new = 112) green.

- [ ] **Step 5.6: CLI冒烟(--all --limit 5)**

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe backtrace/projection/ablation_fit.py --all --limit 5 --movement-dir /tmp/synthetic_mv
```
Expected: exit 0, 4 per-model CSVs + summary + HTML + TXT all created.

- [ ] **Step 5.7: Commit**

```bash
git add backtrace/projection/ablation_fit.py tests/test_dynamics_eigen.py
git commit -m "feat(projection): V0.1 Task 5 — summary + HTML + recommendation TXT + CLI"
```

---

## Self-Review Checklist

- [x] Spec §2 (4 models × Y/X/θ) → Task 1 Step 1.3 (`build_design_model_*`)
- [x] Spec §3 (OOS 70/30) → Task 3 Step 3.3 (`oos_split_indices`, `fit_one_oos`)
- [x] Spec §4 (Placebo seed=42) → Task 4 Step 4.3 (`PLACEBO_SEED=42`, `permute_regressors`)
- [x] Spec §5 (17-col schema + 4×10 metric matrix) → Task 2 (schema in `CSV_COLUMNS`), Task 5 (`summarize_ablation`)
- [x] Spec §6 (output files) → Task 5 (writes `kc_estimates_model{0,1,2,3}.csv`, `kc_ablation_summary.csv`, `kc_ablation_recommendation.txt`, `ablation_distribution.html`)
- [x] Spec §7 (CLI flags write-dead) → Task 5 Step 5.3 (`parse_args`: `--model`, `--all`, `--limit`, `--no-placebo`, `--movement-dir`, `--output-dir`; no `--pick-best-ic` / `--feature-engineering`)
- [x] Spec §9 (testing) → 16 tests across Tasks 1-5
- [x] Spec §10 (decision tree) → Task 5 Step 5.3 (`write_recommendation_txt` with 3-step gate)
- [x] Spec §11 (forbidden list) → not in plan (YAGNI by absence — no feature-engineering flags, no d_vec redesign, no F_self redef, etc.)

Type consistency check:
- `CSV_COLUMNS` (17 entries) used identically in `write_in_sample_csvs`, `write_oos_csvs`, `write_ablation_csvs` ✓
- `BUILDERS[model_id]` dispatch used in `fit_one_in_sample`, `fit_one_oos`, `fit_one_with_placebo` ✓
- `ols_fit` returns 7-tuple in `fit_one_*` (matching `test_ols_fit_*` expectations) ✓
- `compute_spearman_ic` returns float in all callers ✓