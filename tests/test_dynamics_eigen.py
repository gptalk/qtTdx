# -*- coding: utf-8 -*-
"""Tests for backtrace/dynamics/analyze_eigenvalues (2026-08-17 v4.1 + v4.2).

覆盖:
  - 11 类稳定分类(全部分支)
  - v4.1 关键边界修正:c=2+k/2 在 k>4 时 **不是** critical_period2;c=k 在 k>4 时 **不是** critical_real_unit
  - v4.2 wedge distance 字段(distance_lower / upper / to_wedge)
  - Gram-Schmidt 能量恒等式(via simulate_trajectory 的 energy_error 字段)
"""
import numpy as np
import pytest
import sys, os

BACKTRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backtrace')
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
from dynamics import analyze_eigenvalues, simulate_trajectory, build_simulation_df


# === 11 分类全分支 ===

def test_stable_oscillatory():
    """k=0.5, c=0.6 → D<0 在楔形内(衰减振荡)"""
    r = analyze_eigenvalues(k=0.5, c=0.6)
    assert r['classification'] == 'stable_oscillatory'
    assert r['stability'] == 'schur_stable'
    assert r['schur_stable'] is True
    assert r['in_wedge'] is True
    assert r['spectral_radius'] < 1.0


def test_stable_overdamped():
    """k=0.1, c=0.8 → D>0 在楔形内(过阻尼)"""
    r = analyze_eigenvalues(k=0.1, c=0.8)
    assert r['classification'] == 'stable_overdamped'
    assert r['stability'] == 'schur_stable'
    assert r['spectral_radius'] < 1.0


def test_stable_critical_damping():
    """k=1, c=2 → c=2k, D=0, 重根"""
    r = analyze_eigenvalues(k=1, c=2)
    assert r['classification'] == 'stable_critical_damping'
    assert r['stability'] == 'schur_stable'


def test_monotonic_divergent():
    """k=1, c=3 → c>2+k/2, 单调发散"""
    r = analyze_eigenvalues(k=1, c=3)
    assert r['classification'] == 'monotonic_divergent'
    assert r['stability'] == 'unstable'
    assert r['spectral_radius'] > 1.0


def test_oscillatory_divergent():
    """k=1, c=0.05 → c<k, D<0, 振荡发散"""
    r = analyze_eigenvalues(k=1, c=0.05)
    assert r['classification'] == 'oscillatory_divergent'
    assert r['stability'] == 'unstable'
    assert r['spectral_radius'] > 1.0


def test_anti_restoring():
    """k<0 → 反回复(趋势强化)"""
    r = analyze_eigenvalues(k=-0.1, c=1.0)
    assert r['classification'] == 'anti_restoring'
    assert r['stability'] == 'unstable'


def test_jordan_drift():
    """k=0, c=0 → Jordan 块, 多项式漂移"""
    r = analyze_eigenvalues(k=0, c=0)
    assert r['classification'] == 'jordan_drift'
    assert r['stability'] == 'critical'


def test_marginal_const():
    """k=0, c>0 → λ₁=1, λ₂=1-c, 边界常数模"""
    r = analyze_eigenvalues(k=0, c=0.5)
    assert r['classification'] == 'marginal_const'
    assert r['stability'] == 'critical'


# === v4.1 边界 bug 回归 ===

def test_critical_period2_when_k_less_4():
    """c=2+k/2, k<4 → ρ=1, λ₁=-1, critical_period2 (正确)"""
    r = analyze_eigenvalues(k=2, c=3.0)
    assert r['classification'] == 'critical_period2'
    assert r['stability'] == 'critical'
    assert abs(r['spectral_radius'] - 1.0) < 1e-7


def test_critical_periodic_when_k_less_4():
    """c=k, k<4 → 复根, ρ=1, critical_periodic (正确)"""
    r = analyze_eigenvalues(k=2, c=2)
    assert r['classification'] == 'critical_periodic'
    assert r['stability'] == 'critical'
    assert abs(r['spectral_radius'] - 1.0) < 1e-7


def test_critical_real_unit_at_special_point():
    """k=4, c=4 → 楔形交点, λ=-1 双根"""
    r = analyze_eigenvalues(k=4, c=4)
    # c=k AND c=2+k/2:边界交点 → ρ=1, λ=-1
    assert abs(r['spectral_radius'] - 1.0) < 1e-7
    # ρ=1 + λ=-1 → critical_period2
    assert r['classification'] == 'critical_period2'
    assert r['stability'] == 'critical'


def test_boundary_fix_c_eq_2pk_half_k_greater_4():
    """v4.1 修正:c=2+k/2, k>4 → λ₁=-1, |λ₂|>1, **不应** critical_period2,而应 unstable

    之前版本错误归为 critical_period2,实际是单调发散。
    """
    r = analyze_eigenvalues(k=6, c=5.0)
    assert r['classification'] != 'critical_period2', \
        'v4.1 边界修正:c=2+k/2 with k>4 必须 NOT 是 critical_period2'
    assert r['stability'] == 'unstable'
    assert r['classification'] == 'monotonic_divergent'
    assert r['spectral_radius'] > 1.0


def test_boundary_fix_c_eq_k_k_greater_4():
    """v4.1 修正:c=k, k>4 → 实根, ρ>1, **不应** critical_real_unit,而应 unstable

    之前版本错误归为 critical_real_unit,实际是发散。
    """
    r = analyze_eigenvalues(k=6, c=6)
    assert r['classification'] != 'critical_real_unit', \
        'v4.1 边界修正:c=k with k>4 必须 NOT 是 critical_real_unit'
    assert r['stability'] == 'unstable'
    assert r['classification'] == 'monotonic_divergent'


# === v4.2 wedge distance 字段 ===

def test_wedge_distance_inside():
    """楔形内:distance_to_wedge > 0"""
    r = analyze_eigenvalues(k=0.5, c=1.0)
    assert r['in_wedge'] is True
    assert r['distance_lower_boundary'] == pytest.approx(0.5)
    assert r['distance_upper_boundary'] == pytest.approx(1.25)
    assert r['distance_to_wedge'] == pytest.approx(0.5)  # min(0.5, 0.5, 1.25)
    assert r['distance_to_wedge'] > 0


def test_wedge_distance_below_lower_boundary():
    """c<k (楔形外下方):distance_lower < 0"""
    r = analyze_eigenvalues(k=0.5, c=0.3)
    assert r['in_wedge'] is False
    assert r['distance_lower_boundary'] < 0
    assert r['distance_to_wedge'] < 0
    assert r['distance_upper_boundary'] > 0


def test_wedge_distance_above_upper_boundary():
    """c>2+k/2 (楔形外上方):distance_upper < 0"""
    r = analyze_eigenvalues(k=0.5, c=3.0)
    assert r['in_wedge'] is False
    assert r['distance_upper_boundary'] < 0
    assert r['distance_to_wedge'] < 0
    assert r['distance_lower_boundary'] > 0


def test_wedge_distance_k_negative():
    """k<0:distance_to_wedge 取决于 c-k 与 2+k/2-c 中的较小者"""
    r = analyze_eigenvalues(k=-0.1, c=1.0)
    # distance_to_wedge = min(k=-0.1, c-k=1.1, 2+k/2-c=0.95) = -0.1
    assert r['distance_to_wedge'] == pytest.approx(-0.1)
    assert r['in_wedge'] is False


# === 数值一致性 ===

def test_eigenvalues_sum_is_trace():
    """λ₁ + λ₂ = trace(A) = 2 - c"""
    r = analyze_eigenvalues(k=0.5, c=0.6)
    lam_sum = r['eigenvalues'][0] + r['eigenvalues'][1]
    assert abs(lam_sum.real - r['trace']) < 1e-10
    assert abs(lam_sum.imag) < 1e-10


def test_eigenvalues_product_is_determinant():
    """λ₁ · λ₂ = det(A) = 1 - c + k"""
    r = analyze_eigenvalues(k=0.5, c=0.6)
    lam_prod = r['eigenvalues'][0] * r['eigenvalues'][1]
    assert abs(lam_prod.real - r['determinant']) < 1e-10
    assert abs(lam_prod.imag) < 1e-10


def test_eigenvalues_complex_conjugate():
    """D<0 时 λ₁, λ₂ 互为共轭"""
    r = analyze_eigenvalues(k=0.5, c=0.6)  # D = 0.36 - 2 = -1.64 < 0
    assert r['mode'] == 'complex_conjugate'
    assert abs(r['eigenvalues'][0] - np.conj(r['eigenvalues'][1])) < 1e-10


# === simulate_trajectory 集成 + Gram-Schmidt 能量恒等式 ===

def test_simulate_trajectory_energy_identity():
    """Gram-Schmidt 严格正交 → E_total = E_market + E_self(数值误差 ~ 1e-15)"""
    v_S = np.array([1.0, 2.0])
    v_M_seq = np.tile(np.array([0.5, 1.5]), (6, 1)) + 0.01 * np.arange(6).reshape(-1, 1)
    beta_seq = np.array([1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
    d_init = np.array([0.3, 0.4])
    sim = simulate_trajectory(
        v_S_init=v_S, d_init=d_init,
        v_M_seq=v_M_seq, beta_seq=beta_seq,
        F_self_seq=np.zeros((5, 2)),
        k=0.5, c=0.6,
    )
    err = sim['energy_error']
    assert np.nanmax(np.abs(err)) < 1e-10, \
        f'Gram-Schmidt 不严格正交,energy_error max={np.nanmax(np.abs(err)):.2e}'


def test_simulate_trajectory_v4_fields_present():
    """simulate_trajectory 返回 dict 必须包含全部 v4 + v4.2 字段"""
    sim = simulate_trajectory(
        v_S_init=np.array([1.0, 2.0]),
        d_init=np.array([0.3, 0.4]),
        v_M_seq=np.tile(np.array([0.5, 1.5]), (6, 1)),
        beta_seq=np.array([1.2, 1.3, 1.4, 1.5, 1.6, 1.7]),
        F_self_seq=np.zeros((5, 2)),
        k=0.5, c=0.6,
    )
    for key in [
        'eigenvalues', 'spectral_radius', 'dynamic_class', 'dynamic_stability',
        'schur_stable', 'in_wedge',
        'distance_lower_boundary', 'distance_upper_boundary', 'distance_to_wedge',
        'energy_error',
    ]:
        assert key in sim, f'Missing key: {key}'


def test_build_simulation_df_v4_columns():
    """build_simulation_df CSV 必须包含全部 v4 + v4.2 列"""
    sim = simulate_trajectory(
        v_S_init=np.array([1.0, 2.0]),
        d_init=np.array([0.3, 0.4]),
        v_M_seq=np.tile(np.array([0.5, 1.5]), (6, 1)),
        beta_seq=np.array([1.2, 1.3, 1.4, 1.5, 1.6, 1.7]),
        F_self_seq=np.zeros((5, 2)),
        k=0.5, c=0.6,
    )
    df = build_simulation_df(sim, dates=None, index_tag='399001', stock_tag='002475')
    for col in [
        'Sim_Lambda1_Real', 'Sim_Lambda1_Imag', 'Sim_Lambda2_Real', 'Sim_Lambda2_Imag',
        'Sim_SpectralRadius', 'Sim_DynamicClass',
        'Sim_DistanceLowerBoundary', 'Sim_DistanceUpperBoundary', 'Sim_DistanceToWedge',
        'Sim_EnergyError',
    ]:
        assert col in df.columns, f'Missing column: {col}'