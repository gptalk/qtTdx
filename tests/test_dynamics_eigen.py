# -*- coding: utf-8 -*-
"""Tests for backtrace/dynamics/analyze_eigenvalues (2026-08-17 v4.1 + v4.2).

覆盖:
  - 11 类稳定分类(全部分支)
  - v4.1 关键边界修正:c=2+k/2 在 k>4 时 **不是** critical_period2;c=k 在 k>4 时 **不是** critical_real_unit
  - v4.2 wedge distance 字段(distance_lower / upper / to_wedge)
  - Gram-Schmidt 能量恒等式(via simulate_trajectory 的 energy_error 字段)
  - v4.3:行业聚合 (ρ 中位数 + 阈值降级 + 降序排)
  - v4.3:交易所拆分 (n_stocks + p25/p75)
  - v4.3:HTML 2x4 + 文本汇总 + 2 个聚合 CSV
"""
import numpy as np
import pandas as pd
import pytest
import sys, os

BACKTRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backtrace')
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
from dynamics import analyze_eigenvalues, simulate_trajectory, build_simulation_df
from dynamics import dynamics_eigen_analysis as EA


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


# === v4.3:行业聚合 ρ 中位数 ===

def test_industry_aggregation_rho_median():
    """构造 100 只票 / 3 个行业的 dummy DataFrame,验证 groupby median + 阈值降级 + 降序排。

    阈值降级:min_stocks=20 / fallback_min=10,3 个行业都 n>=20,但 aggregate_by_industry
    要求 len(agg) >= 5 才锁定阈值,只有 3 个行业时退到 fallback_min=10。
    """
    rng = np.random.default_rng(42)
    rows = []
    # industry_A 50 只 ρ∈[0.5, 1.0]
    for i in range(50):
        rho = rng.uniform(0.5, 1.0)
        rows.append({'code': f'A{i:03d}', 'industry_l1': '881001.SH',
                     'spectral_radius': rho, 'k_hat': 0.1, 'c_hat': 1.0,
                     'schur_stable': rho < 1.0, 'in_wedge': True,
                     'distance_to_wedge': 0.1})
    # industry_B 30 只 ρ∈[1.0, 2.0]
    for i in range(30):
        rho = rng.uniform(1.0, 2.0)
        rows.append({'code': f'B{i:03d}', 'industry_l1': '881002.SH',
                     'spectral_radius': rho, 'k_hat': 0.0, 'c_hat': 2.0,
                     'schur_stable': False, 'in_wedge': False,
                     'distance_to_wedge': -0.5})
    # industry_C 20 只 ρ∈[2.0, 5.0]
    for i in range(20):
        rho = rng.uniform(2.0, 5.0)
        rows.append({'code': f'C{i:03d}', 'industry_l1': '881003.SH',
                     'spectral_radius': rho, 'k_hat': -0.1, 'c_hat': 5.0,
                     'schur_stable': False, 'in_wedge': False,
                     'distance_to_wedge': -2.0})
    df = pd.DataFrame(rows)

    agg, threshold = EA.aggregate_by_industry(df, min_stocks=20, fallback_min=10)
    # 3 个行业(< len>=5 阈值),实际退到 fallback_min=10
    assert threshold == 10
    assert len(agg) == 3
    # A ρ 中位数 ~0.75, B ~1.5, C ~3.5
    rho_med_by_industry = dict(zip(agg['industry_l1'], agg['rho_median']))
    assert 0.6 < rho_med_by_industry['881001.SH'] < 0.9
    assert 1.2 < rho_med_by_industry['881002.SH'] < 1.8
    assert 2.8 < rho_med_by_industry['881003.SH'] < 4.2
    # n_stocks 正确
    n_by_industry = dict(zip(agg['industry_l1'], agg['n_stocks']))
    assert n_by_industry['881001.SH'] == 50
    assert n_by_industry['881002.SH'] == 30
    assert n_by_industry['881003.SH'] == 20
    # 降序排
    assert agg['rho_median'].is_monotonic_decreasing


# === v4.3:交易所拆分 + 误差棒 ===

def test_exchange_split_correctness():
    """SH/SZ 各 50 只,验证 n_stocks=50/50,p25/p75 正确,median 落在 [p25, p75]。"""
    rng = np.random.default_rng(7)
    rows = []
    for i in range(50):
        rows.append({'code': f'SH{i:03d}', 'exchange': 'SH',
                     'spectral_radius': rng.uniform(0.8, 1.5),
                     'k_hat': 0.0, 'c_hat': 1.2, 'schur_stable': False,
                     'in_wedge': False, 'distance_to_wedge': -0.1})
    for i in range(50):
        rows.append({'code': f'SZ{i:03d}', 'exchange': 'SZ',
                     'spectral_radius': rng.uniform(1.0, 2.0),
                     'k_hat': 0.0, 'c_hat': 1.5, 'schur_stable': False,
                     'in_wedge': False, 'distance_to_wedge': -0.2})
    df = pd.DataFrame(rows)

    agg = EA.aggregate_by_exchange(df)
    assert set(agg['exchange']) == {'SH', 'SZ'}
    n_by_ex = dict(zip(agg['exchange'], agg['n_stocks']))
    assert n_by_ex['SH'] == 50
    assert n_by_ex['SZ'] == 50
    # SH ρ 中位数 ~1.15, SZ ~1.5
    rho_by_ex = dict(zip(agg['exchange'], agg['rho_median']))
    assert 1.0 < rho_by_ex['SH'] < 1.3
    assert 1.3 < rho_by_ex['SZ'] < 1.7
    # p25 <= median <= p75
    for _, r in agg.iterrows():
        assert r['rho_p25'] <= r['rho_median'] <= r['rho_p75']


# === v4.3:HTML 2x4 + 文本输出(端到端) ===

def test_html_2x4_layout_and_text_summary(tmp_path, monkeypatch):
    """构造 dummy kc_estimates + dummy stock_basic + dummy sw2/members,
    跑 main(临时改路径),验证 8 子图 HTML + 文本汇总 + 2 个聚合 CSV。"""
    rng = np.random.default_rng(123)

    # --- 1. 构造 dummy kc_estimates(50 只,30 SH + 20 SZ)— 注意:load_kc_estimates 要求 'status' 列
    rows = []
    for i in range(50):
        k = rng.uniform(-0.5, 0.5)
        c = rng.uniform(0.5, 3.0)
        eig = EA.analyze_eigenvalues(k, c)
        rows.append({
            'code': f'{i:06d}.SH' if i < 30 else f'{i:06d}.SZ',
            'name': f'Test{i}',
            'index_tag': '000',
            'stock_tag': f'{i:06d}',
            'status': 'ok',  # ← load_kc_estimates 要求此列(status_filter='ok')
            'k_hat': k,
            'c_hat': c,
            'lam1_real': float(eig['eigenvalues'][0].real),
            'lam1_imag': float(eig['eigenvalues'][0].imag),
            'lam2_real': float(eig['eigenvalues'][1].real),
            'lam2_imag': float(eig['eigenvalues'][1].imag),
            'spectral_radius': eig['spectral_radius'],
            'classification': eig['classification'],
            'stability': eig['stability'],
            'schur_stable': eig['schur_stable'],
            'in_wedge': eig['in_wedge'],
            'distance_lower_boundary': eig['distance_lower_boundary'],
            'distance_upper_boundary': eig['distance_upper_boundary'],
            'distance_to_wedge': eig['distance_to_wedge'],
        })
    df = pd.DataFrame(rows)
    csv_in = tmp_path / 'kc_estimates.csv'
    df.to_csv(csv_in, index=False)

    # --- 2. 构造 dummy stock_basic.csv(code, market, name, status)
    sb_rows = [{'code': rows[i]['code'], 'market': 'SH' if i < 30 else 'SZ',
                'name': f'Test{i}', 'status': 'active'} for i in range(50)]
    sb_path = tmp_path / 'stock_basic.csv'
    pd.DataFrame(sb_rows).to_csv(sb_path, index=False)

    # --- 3. 构造 dummy sw2/members.csv(sector_code, sector_name, member_code)
    sw2_rows = []
    for i in range(30):
        sw2_rows.append({'sector_code': '881001.SH', 'sector_name': 'A组',
                         'member_code': rows[i]['code']})
    for i in range(30, 50):
        sw2_rows.append({'sector_code': '881002.SH', 'sector_name': 'B组',
                         'member_code': rows[i]['code']})
    sw2_path = tmp_path / 'sw2_members.csv'
    pd.DataFrame(sw2_rows).to_csv(sw2_path, index=False)

    # --- 4. 输出路径
    html_out = tmp_path / 'dynsys_eigen.html'
    txt_out = tmp_path / 'dynsys_eigen_summary.txt'

    # --- 5. monkeypatch + 跑 main
    monkeypatch.setattr(EA, 'CSV_OUT_DIR', str(tmp_path))
    monkeypatch.setattr(EA, 'AGG_INDUSTRY_CSV', str(tmp_path / 'v43_eigen_top_industries.csv'))
    monkeypatch.setattr(EA, 'AGG_EXCHANGE_CSV', str(tmp_path / 'v43_eigen_by_exchange.csv'))
    monkeypatch.setattr(EA, 'DEFAULT_TXT_OUTPUT', str(txt_out))

    sys.argv = ['dynamics_eigen_analysis',
                '--input', str(csv_in),
                '--output', str(html_out),
                '--stock-basic', str(sb_path),
                '--sw2-members', str(sw2_path),
                '--limit', '0']
    EA.main()

    # --- 6. 验收
    assert html_out.exists() and html_out.stat().st_size > 50_000
    assert txt_out.exists()
    txt_content = txt_out.read_text(encoding='utf-8')
    assert '=== v4.3 全市场' in txt_content
    assert '--- 11 类分布 ---' in txt_content
    assert '--- 行业 ρ 中位数 top10' in txt_content
    assert '--- 交易所 ---' in txt_content
    assert (tmp_path / 'v43_eigen_top_industries.csv').exists()
    assert (tmp_path / 'v43_eigen_by_exchange.csv').exists()
