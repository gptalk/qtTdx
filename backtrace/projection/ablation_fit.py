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
