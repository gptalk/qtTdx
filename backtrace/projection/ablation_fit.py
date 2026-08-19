# -*- coding: utf-8 -*-
# ablation_fit.py — Dynamics Specification Correction & Ablation (V0.1)
#
# Spec: docs/superpowers/specs/2026-08-19-dynamics-specification-correction-ablation.md
#
# 4 models:
#   Model 0: a_S = β·a_M − k·d − c·u + F_self       (status quo)
#   Model 1: a_S = β·a_M + β�·v_M − k·d − c·u + F_self  (β-drift correction)
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
    """Model 3: Y = a_S − β�·v_M, X = [β·a_M, −d, −u], θ = (q, k, c)."""
    Y = _stack_2d(a_u_vec - beta_dot_vM_vec)
    beta_aM_stack = _stack_2d(beta[:, None] * a_v_vec)
    d_stack = _stack_2d(d_vec)
    u_stack = _stack_2d(u_vec)
    X = np.column_stack([beta_aM_stack, -d_stack, -u_stack])
    return X, Y


# === V0.1 Task 2: In-Sample 4-Model Fit + CSV Output ===
import pandas as pd
from typing import List, Tuple

CSV_OUT_DIR = 'data/projection'

# Schema (V0.2-D Phase 1: 36 cols, identical across all 4 models)
#   18 existing (V0.1) + 9 Group A + 3 Group B + 3 Group C + 3 Group D
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


# === V0.1 Task 3: OOS 70/30 Split + Spearman IC ===
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


# === V0.1 Task 4: Permutation Placebo (seed=42) ===
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


# === V0.2-D Phase 1: Train/Test split + parameter stability (Group A/B) ===

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


def compute_x_x_correlations(X: np.ndarray):
    """V0.2-D §5: pairwise Pearson correlation among 3 design-matrix columns.

    Arguments:
      X: design matrix, shape (n, 3); columns are [beta*a_M, -d, -u].

    Returns:
      (corr_x_beta_d, corr_x_beta_u, corr_x_d_u) — 3 floats.
      (nan, nan, nan) if n < 3, if X is not 3-column, or if any column is
      (near-)constant (std < 1e-12), where Pearson correlation is undefined.

    CRITICAL (V0.2-D §5): this computes corr(X_i, X_j), NOT corr(q_hat, X).
    corr(q_hat, X) is statistically undefined (scalar x time series).
    """
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] != 3:
        return (np.nan, np.nan, np.nan)
    stds = X.std(axis=0)
    if np.any(stds < 1e-12):
        return (np.nan, np.nan, np.nan)
    corr = np.corrcoef(X.T)
    return (float(corr[0, 1]), float(corr[0, 2]), float(corr[1, 2]))


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

    # Group A: parameter stability (separate drifts; no L2 aggregation, V0.2-D §4)
    if model_id in (0, 1):
        q_train, k_train, c_train = 1.0, float(theta_train[0]), float(theta_train[1])
        q_test_fit, k_test_fit, c_test_fit = 1.0, float(theta_test[0]), float(theta_test[1])
    else:
        q_train, k_train, c_train = (float(theta_train[0]), float(theta_train[1]), float(theta_train[2]))
        q_test_fit, k_test_fit, c_test_fit = (float(theta_test[0]), float(theta_test[1]), float(theta_test[2]))
    q_drift = q_test_fit - q_train
    k_drift = k_test_fit - k_train
    c_drift = c_test_fit - c_train

    # Group C: X-X collinearity on X_train (V0.2-D §5)
    corr_x_beta_d, corr_x_beta_u, corr_x_d_u = compute_x_x_correlations(X_train)

    # Group D: residual structure (Task 4 fills in)
    # For now, leave as NaN — Task 4 populates.

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
        # Group C
        'corr_x_beta_d': corr_x_beta_d,
        'corr_x_beta_u': corr_x_beta_u,
        'corr_x_d_u': corr_x_d_u,
        # Group D placeholders
        'corr_F_beta_aM': np.nan, 'corr_F_d': np.nan, 'corr_F_u': np.nan,
    }


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


# === V0.1 Task 5: Summary + HTML + Recommendation TXT + CLI ===
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


def build_ablation_html(summary_df: pd.DataFrame, output_path: str) -> str:
    """4-panel plotly dark template: R² / IC_real vs IC_null / ΔIC / q̂ distribution."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        'R² distribution (4 models)',
        'IC_real vs IC_null (4 models)',
        'Placebo ΔIC = IC_real − IC_null (4 models)',
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

    # Panel 3: Placebo ΔIC = median(IC_real − IC_null) per-model (4 models)
    delta_ic = summary_df.loc['median_delta_ic'].astype(float)
    fig.add_trace(go.Bar(x=delta_ic.index, y=delta_ic.values,
                         marker_color=[colors[i] for i in range(4)],
                         name='Placebo ΔIC', showlegend=False), row=2, col=1)

    # Panel 4: |q̂ − 1| for Models 2/3
    abs_q = summary_df.loc['median_abs_q_minus_1'].dropna().astype(float)
    fig.add_trace(go.Bar(x=abs_q.index, y=abs_q.values,
                         marker_color=[colors[i] for i in range(2, 4)],
                         name='|q̂ − 1|', showlegend=False), row=2, col=2)

    fig.update_layout(template='plotly_dark', height=800, title='Dynamics Specification Ablation (V0.1)')
    fig.write_html(output_path)
    return output_path


def write_recommendation_txt(summary_df: pd.DataFrame, output_path: str) -> str:
    """UTF-8 Chinese decision recommendation per spec §10 decision tree.

    Per spec §10 写死:
      - Step 1: ΔR²_M1 = median_s(R²_M1,s − R²_M0,s) > 0.005  (per-stock median delta)
      - Step 2: median_s |q̂_M2,s − 1| > 0.1                    (per-stock median |q̂ − 1|)
      - Step 3: median(IC_real_M3) − median(IC_null_M3) > 0.02  (difference of medians)

    Mid-run 阈值 / verdict 禁止调整。
    """
    median_r2_m1 = float(summary_df.loc['median_r2', 'model_1'])
    median_r2_m2 = float(summary_df.loc['median_r2', 'model_2'])
    median_r2_m3 = float(summary_df.loc['median_r2', 'model_3'])
    median_r2_m0 = float(summary_df.loc['median_r2', 'model_0'])
    abs_q_m2 = float(summary_df.loc['median_abs_q_minus_1', 'model_2']) if 'median_abs_q_minus_1' in summary_df.index else np.nan
    # V0.2-D Phase 0 audit fix: read all ΔIC numbers from the summary DataFrame (no inline recompute)
    delta_ic_m3 = float(summary_df.loc['diff_of_medians_delta_ic', 'model_3'])
    # Per-stock median ΔR² (写死, 不是 difference of medians)
    delta_r2_m1 = float(summary_df.loc['delta_r2_vs_m0', 'model_1'])

    # Step 1: β-drift ΔR² = median(R²_new − R²_old) per-stock > 0.005
    step1_pass = delta_r2_m1 > 0.005
    # Step 2: q ≠ 1 (median of per-stock |q̂ − 1|)
    step2_pass = abs_q_m2 > 0.1
    # Step 3: placebo ΔIC (difference of medians)
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
        f'  Model 1 (β-drift):         {median_r2_m1:.4f}',
        f'  Model 2 (free q):          {median_r2_m2:.4f}',
        f'  Model 3 (joint):           {median_r2_m3:.4f}',
        '',
        '--- ΔR² (per-stock median delta) vs Model 0 ---',
        f'  ΔR²_M1: {delta_r2_m1:+.4f}     (Step 1 threshold > 0.005)',
        '',
        '--- |q̂ − 1| (per-stock median) ---',
        f'  Model 2: {abs_q_m2:.4f}        (Step 2 threshold > 0.1)',
        '',
        '--- ΔIC (3 stats; see summary CSV row names) ---',
        f'  median_delta_ic (A) M3:          {float(summary_df.loc["median_delta_ic", "model_3"]):+.4f}',
        f'  diff_of_medians_delta_ic (B) M3: {delta_ic_m3:+.4f}    (Step 3 threshold > 0.02)',
        f'  delta_ic_vs_m0 (C) M3:           {float(summary_df.loc["delta_ic_vs_m0", "model_3"]):+.4f}',
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
                                    os.path.join(args.output_dir, 'ablation_distribution.html'))
    # TXT recommendation
    txt_path = write_recommendation_txt(summary_df,
                                        os.path.join(args.output_dir, 'kc_ablation_recommendation.txt'))
    print(f'Summary: {args.output_dir}/kc_ablation_summary.csv')
    print(f'HTML:    {html_path}')
    print(f'TXT:     {txt_path}')


if __name__ == '__main__':
    main()
