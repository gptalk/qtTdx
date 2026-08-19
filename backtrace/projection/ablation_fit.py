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
