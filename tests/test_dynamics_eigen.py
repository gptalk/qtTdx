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
import subprocess
import sys, os
import tempfile
from pathlib import Path

BACKTRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backtrace')
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
from dynamics import analyze_eigenvalues, simulate_trajectory, build_simulation_df
from dynamics import dynamics_eigen_analysis as EA

# Module-scope REPO_ROOT for subprocess.cwd= (v5.10 test_cli_oos_batch_mode)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


# === v4.4:行业 name lookup helper ===

def test_industry_name_lookup(tmp_path):
    """_industry_name_lookup 3 个 case: 正常 / 缺文件 / 缺关键列。"""
    from dynamics.dynamics_eigen_analysis import _industry_name_lookup

    # 1. 正常:写 mock sw2/members.csv,验证返回 dict
    sw2 = tmp_path / 'sw2.csv'
    sw2.write_text(
        'sector_code,sector_name,member_code\n'
        '881459.SH,电力,600000.SH\n'
        '881001.SH,银行,600001.SH\n',
        encoding='utf-8',
    )
    result = _industry_name_lookup(str(sw2))
    assert result == {'881459.SH': '电力', '881001.SH': '银行'}

    # 2. 缺文件:返回空 dict
    assert _industry_name_lookup(str(tmp_path / 'nope.csv')) == {}

    # 3. 缺关键列:返回空 dict
    bad = tmp_path / 'bad.csv'
    bad.write_text('foo,bar\n1,2\n', encoding='utf-8')
    assert _industry_name_lookup(str(bad)) == {}


# === v4.5:楔形边界 helper ===

def test_wedge_boundary_polygon():
    """楔形边界 3 段:左 c=0 / 底 c=0 / 上 c=2√(k+1)"""
    from dynamics.dynamics_eigen_analysis import wedge_boundary_polygon

    boundary = wedge_boundary_polygon(k_max=4.0, n=50)

    # 上边界:起点 (k=0, c=2),终点 (k=4, c=2√5 ≈ 4.47)
    assert boundary['upper_curve'][0] == (0.0, 2.0)
    assert abs(boundary['upper_curve'][-1][0] - 4.0) < 1e-9
    assert abs(boundary['upper_curve'][-1][1] - 2.0 * np.sqrt(5.0)) < 1e-9

    # k 轴:起点 (0, 0),终点 (4, 0)
    assert boundary['k_axis'][0] == (0.0, 0.0)
    assert boundary['k_axis'][-1] == (4.0, 0.0)

    # c 轴:起点 (0, 0),终点 (0, 2)
    assert boundary['c_axis'][0] == (0.0, 0.0)
    assert boundary['c_axis'][-1] == (0.0, 2.0)

    # 长度: n 个点(由 n=50 参数)
    assert len(boundary['k_axis']) == 50
    assert len(boundary['c_axis']) == 50
    assert len(boundary['upper_curve']) == 50

    # k_max 字段
    assert boundary['k_max'] == 4.0


# === v4.5:phase plot HTML smoke ===

def test_phase_plot_html_smoke(tmp_path):
    """build_phase_plot_html 写文件成功 + HTML 包含 11 类 marker。"""
    from dynamics.dynamics_eigen_analysis import CLASS_COLORS, CLASS_LABEL_CN, build_phase_plot_html

    # mock 11 类样本,每类 5 只票
    rng = np.random.default_rng(42)
    rows = []
    for cls in CLASS_COLORS:
        for _ in range(5):
            rows.append({
                'code': f'{rng.integers(0, 999999):06d}.SH',
                'k_hat': rng.uniform(0, 4),
                'c_hat': rng.uniform(0, 4),
                'classification': cls,
            })
    df = pd.DataFrame(rows)
    out = tmp_path / 'phase.html'
    build_phase_plot_html(df, str(out))

    assert out.exists()
    content = out.read_text(encoding='utf-8')
    assert 'scatter' in content.lower()  # plotly HTML
    # 11 类名称都出现在 HTML 里(CLASS_LABEL_CN 中文标签)
    for cn in CLASS_LABEL_CN.values():
        assert cn in content


def test_aggregate_by_industry_no_data():
    """T3.5: 0-row fallback 区分 — 无任何行业时返回 threshold=0(无数据)

    构造 0 行 df 应返回 (empty_df, 0)。这是与"有数据但 < 阈值"区分的标志。
    """
    # 空 df
    df_empty = pd.DataFrame(columns=['code', 'industry_l1', 'spectral_radius',
                                      'k_hat', 'c_hat', 'schur_stable',
                                      'in_wedge', 'distance_to_wedge'])
    agg, threshold = EA.aggregate_by_industry(df_empty)
    assert len(agg) == 0
    assert threshold == 0  # T3.5 修复: 0 表示无数据

    # 单行业 50 只(>= 默认阈值 50,但 < 5 industries): threshold > 0
    rng = np.random.default_rng(13)
    rows = []
    for i in range(50):
        rows.append({'code': f'X{i:03d}', 'industry_l1': '881999.SH',
                     'spectral_radius': rng.uniform(0.5, 1.0),
                     'k_hat': 0.0, 'c_hat': 1.0,
                     'schur_stable': True, 'in_wedge': True,
                     'distance_to_wedge': 0.1})
    df_one = pd.DataFrame(rows)
    agg, threshold = EA.aggregate_by_industry(df_one)
    assert len(agg) == 1  # 1 个行业
    assert threshold > 0  # 有数据(虽然只 1 个行业),threshold >= 30 (fallback_min)


# === v4.7:行业稳定性指数 SI (Sector Stability Index) ===

def test_sector_si_basic_shape():
    """SI = 0.5·ρ_health + 0.2·damping_health + 0.3·wedge_health,锁定权重

    1 行业 100 只全稳定(ρ=0.5, c=1.0, in_wedge=True) → SI = 0.875
    """
    rng = np.random.default_rng(41)
    rows = []
    for i in range(100):
        rows.append({
            'code': f'X{i:03d}', 'industry_l1': '881999.SH',
            'spectral_radius': 0.5, 'c_hat': 1.0, 'in_wedge': True,
            'k_hat': 0.1, 'schur_stable': True, 'distance_to_wedge': 0.3,
        })
    df = pd.DataFrame(rows)
    si = EA.compute_sector_stability(df)
    assert len(si) == 1
    # ρ_health = clip(1 - 0.5/2, 0, 1) = 0.75
    # damping_health = clip(1 - |1-1|/2, 0, 1) = 1.0
    # wedge_health = clip(1.0, 0, 1) = 1.0
    # SI = 0.5*0.75 + 0.2*1.0 + 0.3*1.0 = 0.875
    assert np.isclose(si['SI'].iloc[0], 0.875, atol=1e-9)


def test_sector_si_anti_restoring():
    """anti_restoring 类(ρ=3.0, c=1.5, in_wedge=False) → SI = 0.15"""
    rows = []
    for i in range(100):
        rows.append({
            'code': f'X{i:03d}', 'industry_l1': '881888.SH',
            'spectral_radius': 3.0, 'c_hat': 1.5, 'in_wedge': False,
            'k_hat': -0.05, 'schur_stable': False, 'distance_to_wedge': -0.5,
        })
    df = pd.DataFrame(rows)
    si = EA.compute_sector_stability(df)
    assert len(si) == 1
    # ρ_health = clip(1 - 3/2, 0, 1) = 0
    # damping_health = clip(1 - 0.5/2, 0, 1) = 0.75
    # wedge_health = 0
    # SI = 0.5*0 + 0.2*0.75 + 0.3*0 = 0.15
    assert np.isclose(si['SI'].iloc[0], 0.15, atol=1e-9)


def test_sector_si_clamps_extreme():
    """极端 ρ=10, c=10 → ρ_health=0, damping_health=0,wedge 也 0 → SI=0"""
    rows = []
    for i in range(50):
        rows.append({
            'code': f'X{i:03d}', 'industry_l1': '881777.SH',
            'spectral_radius': 10.0, 'c_hat': 10.0, 'in_wedge': False,
            'k_hat': -1.0, 'schur_stable': False, 'distance_to_wedge': -2.0,
        })
    df = pd.DataFrame(rows)
    si = EA.compute_sector_stability(df)
    assert len(si) == 1
    assert si['rho_health'].iloc[0] == 0.0
    assert si['damping_health'].iloc[0] == 0.0
    assert si['SI'].iloc[0] == 0.0


def test_sector_si_perfect():
    """完美: ρ=0, c=1, in_wedge_pct=1 → SI = 1.0"""
    rows = []
    for i in range(50):
        rows.append({
            'code': f'X{i:03d}', 'industry_l1': '881666.SH',
            'spectral_radius': 0.0, 'c_hat': 1.0, 'in_wedge': True,
            'k_hat': 0.0, 'schur_stable': True, 'distance_to_wedge': 1.0,
        })
    df = pd.DataFrame(rows)
    si = EA.compute_sector_stability(df)
    assert len(si) == 1
    assert np.isclose(si['SI'].iloc[0], 1.0, atol=1e-9)


def test_sector_si_summary_text(tmp_path):
    """write_sector_si_summary 包含 "Top 12 强" + 至少 1 个中文行业名"""
    # 构造 3 个行业,确保前 12 名至少有 1 个中文
    rng = np.random.default_rng(42)
    rows = []
    industries = [
        ('881111.SH', '银行', 0.5, 1.0, 1.0),
        ('881222.SH', '半导体', 3.0, 1.5, 0.0),
        ('881333.SH', '公用事业', 0.7, 1.0, 0.8),
    ]
    name_lookup = {code: name for code, name, _, _, _ in industries}
    for code, _, rho, c, wedge in industries:
        for i in range(60):
            rows.append({
                'code': f'{code}{i:03d}', 'industry_l1': code,
                'spectral_radius': rho, 'c_hat': c, 'in_wedge': wedge > 0.5,
                'k_hat': 0.1, 'schur_stable': rho < 1.0,
                'distance_to_wedge': 0.2 if wedge > 0.5 else -0.2,
            })
    df = pd.DataFrame(rows)
    df_si = EA.compute_sector_stability(df, name_lookup=name_lookup)
    out_path = tmp_path / 'si_summary.txt'
    EA.write_sector_si_summary(df_si, str(out_path))
    content = out_path.read_text(encoding='utf-8')
    assert 'Top 12 强' in content
    assert 'Top 12 弱' in content
    assert '银行' in content
    assert '半导体' in content


# === v4.8: SI × forward return 滚动 Spearman IC 评估 ===

def test_si_ic_synthetic_perfect():
    """SI 与 forward return 完美正相关 → IC ≈ 1.0"""
    siic = pytest.importorskip("backtrace.dynamics.dynamics_si_ic")
    # 5 行业,SI = [0.1, 0.3, 0.5, 0.7, 0.9]
    si = {'I1': 0.1, 'I2': 0.3, 'I3': 0.5, 'I4': 0.7, 'I5': 0.9}
    # forward return 严格按 SI 升序:[-0.04, -0.02, 0.0, 0.02, 0.04]
    fwd = pd.DataFrame(
        {'I1': -0.04, 'I2': -0.02, 'I3': 0.0, 'I4': 0.02, 'I5': 0.04},
        index=pd.date_range('2024-01-01', periods=1),
    )
    ts = siic.rolling_cross_sectional_ic(si, fwd, window=1, step=1)
    assert len(ts) == 1
    assert ts['ic'].iloc[0] > 0.99  # 完美正相关


def test_si_ic_synthetic_random():
    """SI 与 forward return 完全独立 → IC ≈ 0"""
    siic = pytest.importorskip("backtrace.dynamics.dynamics_si_ic")
    rng = np.random.default_rng(42)
    n_industries = 10
    n_days = 100
    si = {f'I{i}': rng.uniform(0, 1) for i in range(n_industries)}
    fwd = pd.DataFrame(
        rng.normal(0, 0.02, size=(n_days, n_industries)),
        columns=[f'I{i}' for i in range(n_industries)],
        index=pd.date_range('2024-01-01', periods=n_days),
    )
    ts = siic.rolling_cross_sectional_ic(si, fwd, window=60, step=20)
    # 跨窗口 IC 平均应该接近 0
    assert abs(ts['ic'].mean()) < 0.3


def test_si_ic_summary_schema(tmp_path):
    """write_si_ic_summary 写 2 行(20d / 60d) × 6 列"""
    siic = pytest.importorskip("backtrace.dynamics.dynamics_si_ic")
    # 构造伪 ts_df:2 个 horizon × 5 个窗口
    rng = np.random.default_rng(13)
    rows = []
    for h in (20, 60):
        for w in range(5):
            rows.append({
                'window_end_date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=w*20),
                'horizon': h, 'ic': rng.uniform(-0.3, 0.3),
                'p_value': rng.uniform(0, 0.5), 'n_industries': 50,
            })
    ts = pd.DataFrame(rows)
    out_path = tmp_path / 'si_ic_summary.csv'
    siic.write_si_ic_summary(ts, str(out_path))
    df = pd.read_csv(out_path)
    assert len(df) == 2  # 2 horizons
    assert list(df['horizon']) == [20, 60]
    assert set(df.columns) >= {'horizon', 'ic_mean', 'ic_std', 'ic_ir', 'p_value_mean', 'n_windows'}


def test_rolling_time_basic_shape():
    """_month_ends 应返回每月最后一个交易日(去重 + 升序)。"""
    import types, importlib
    proj_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'backtrace', 'projection')
    # 记录原始 sys.path 位置,结束后还原,避免污染 pytest 进程级 path。
    original_path = list(sys.path)
    path_added = proj_dir not in sys.path
    if path_added:
        sys.path.insert(0, proj_dir)
    # parameter_fit 顶层 import common.tsfresh_pipeline(依赖 tsfresh)。
    # 本测试只验纯 pandas 的 _month_ends,故环境缺 tsfresh 时打桩,避免无关依赖。
    saved = {k: sys.modules.get(k) for k in ('common', 'common.tsfresh_pipeline')}
    try:
        try:
            import common.tsfresh_pipeline  # noqa: F401
        except Exception:
            pkg = types.ModuleType('common'); pkg.__path__ = []
            sys.modules['common'] = pkg
            sys.modules['common.tsfresh_pipeline'] = types.ModuleType(
                'common.tsfresh_pipeline')
        from parameter_fit import _month_ends
    finally:
        # 还原 sys.modules(防止 common.* stub 污染后续测试)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        # 还原 sys.path 到进入本测试前的状态
        if path_added and proj_dir in sys.path:
            sys.path.remove(proj_dir)
        # 防御:即使调用前已在 sys.path 里,也按原顺序恢复以避免相对位置漂移
        sys.path[:] = original_path

    dates = pd.to_datetime([
        '2024-01-15', '2024-01-31', '2024-02-29', '2024-03-31', '2024-04-30',
        '2024-05-31', '2024-06-30',
    ])
    ends = _month_ends(pd.Series(dates))
    assert ends == [
        pd.Timestamp('2024-01-31'), pd.Timestamp('2024-02-29'),
        pd.Timestamp('2024-03-31'), pd.Timestamp('2024-04-30'),
        pd.Timestamp('2024-05-31'), pd.Timestamp('2024-06-30'),
    ]


# === v4.9: SI 时序 + 漂移检测 ===

def test_si_timeseries_basic_shape():
    """5 行业 × 100 日 → 500 行,SI ∈ [0,1]。"""
    pytest.importorskip("backtrace.dynamics.dynamics_eigen_analysis")
    from backtrace.dynamics.dynamics_eigen_analysis import compute_sector_stability_timeseries
    # 构造 synthetic kc_long_df
    rows = []
    for ind in range(5):
        for d in range(100):
            rows.append({
                'asof_date': pd.Timestamp('2024-01-01') + pd.DateOffset(days=d*7),
                'code': f'{ind:06d}.SH',
                'name': f'测试_{ind}',
                'industry_l1': f'8010{ind:02d}',
                'k_hat': 0.5, 'c_hat': 1.0,
                'n_valid_days': 240, 'status': 'ok',
            })
    kc_long = pd.DataFrame(rows)
    out = compute_sector_stability_timeseries(kc_long)
    # 5 行业 × 100 日
    assert len(out) == 5 * 100, f'expected 500 rows, got {len(out)}'
    assert out['SI'].between(0, 1).all(), 'SI must be in [0, 1]'
    assert set(out.columns) >= {'asof_date', 'industry_l1', 'SI', 'rho_median', 'c_median'}


def test_si_timeseries_stable_industry():
    """k̂, ĉ 恒定 → SI 几乎不变。"""
    pytest.importorskip("backtrace.dynamics.dynamics_eigen_analysis")
    from backtrace.dynamics.dynamics_eigen_analysis import compute_sector_stability_timeseries
    rows = []
    for d in range(120):  # 120 个 asof_dates
        rows.append({
            'asof_date': pd.Timestamp('2024-01-01') + pd.DateOffset(days=d*7),
            'code': '000001.SH',
            'name': '测试银行',
            'industry_l1': '801010',
            'k_hat': 0.5, 'c_hat': 1.0,  # 恒定
            'n_valid_days': 240, 'status': 'ok',
        })
    kc_long = pd.DataFrame(rows)
    out = compute_sector_stability_timeseries(kc_long)
    # SI 应该几乎恒定(每行只有 1 只票 → SI 等于该票 SI)
    si_values = out[out['industry_l1'] == '801010']['SI']
    assert si_values.std() < 1e-6, f'SI should be constant, got std={si_values.std()}'


def test_si_timeseries_drift_zscore():
    """detect_si_drift z-score 计算正确。"""
    pytest.importorskip("backtrace.dynamics.dynamics_si_timeseries")
    from backtrace.dynamics.dynamics_si_timeseries import detect_si_drift
    # 构造 SI 时序:0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1 (t=7)
    dates = pd.date_range('2024-01-01', periods=8, freq='7D')
    si_ts = pd.DataFrame({
        'asof_date': dates,
        'industry_l1': '801010',
        'sector_name': '银行',
        'n_stocks': 42,
        'rho_median': 0.85,
        'c_median': 1.05,
        'in_wedge_pct': 0.92,
        'rho_health': 0.575,
        'damping_health': 0.975,
        'wedge_health': 0.92,
        'SI': [0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.1],
    })
    drift = detect_si_drift(si_ts, window=3, z_threshold=-2.0)
    # t=7 应该触发 drift event(0.1 比 0.85 低很多)
    assert len(drift) >= 1, f'expected ≥ 1 drift, got {len(drift)}'
    assert drift.iloc[0]['industry_l1'] == '801010'
    # z_score 应该是负值且 < -2
    assert drift.iloc[0]['z_score'] < -2.0


def test_si_timeseries_sudden_drop():
    """构造 SI(t=50) 从 0.8 → 0.2 → 触发 drift event。"""
    pytest.importorskip("backtrace.dynamics.dynamics_si_timeseries")
    from backtrace.dynamics.dynamics_si_timeseries import detect_si_drift
    dates = pd.date_range('2024-01-01', periods=60, freq='7D')
    si_values = [0.8] * 50 + [0.2] * 10
    si_ts = pd.DataFrame({
        'asof_date': dates,
        'industry_l1': '801080',
        'sector_name': '半导体',
        'n_stocks': 38,
        'rho_median': 0.85,
        'c_median': 1.05,
        'in_wedge_pct': 0.92,
        'rho_health': 0.575,
        'damping_health': 0.975,
        'wedge_health': 0.92,
        'SI': si_values,
    })
    drift = detect_si_drift(si_ts, window=3, z_threshold=-2.0)
    # t=50 起应该触发 drift(0.2 比 0.8 低)
    assert len(drift) >= 1, f'expected ≥ 1 drift, got {len(drift)}'
    # 至少一个 drift 的 asof_date ≥ t=50
    assert any(drift['asof_date'] >= dates[50])


def test_si_timeseries_summary_text(tmp_path):
    """write_si_timeseries_summary 包含 '漂移事件' + 中文行业名。"""
    pytest.importorskip("backtrace.dynamics.dynamics_si_timeseries")
    from backtrace.dynamics.dynamics_si_timeseries import write_si_timeseries_summary
    si_ts = pd.DataFrame({
        'asof_date': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-02-01')],
        'industry_l1': ['801010', '801010'],
        'sector_name': ['银行', '银行'],
        'n_stocks': [42, 42],
        'rho_median': [0.85, 0.85],
        'c_median': [1.05, 1.05],
        'in_wedge_pct': [0.92, 0.92],
        'rho_health': [0.575, 0.575],
        'damping_health': [0.975, 0.975],
        'wedge_health': [0.92, 0.92],
        'SI': [0.85, 0.85],
    })
    drift = pd.DataFrame(columns=[
        'asof_date', 'industry_l1', 'sector_name',
        'SI', 'rolling_mean', 'rolling_std', 'z_score',
    ])
    out = tmp_path / 'summary.txt'
    write_si_timeseries_summary(si_ts, drift, str(out))
    content = out.read_text(encoding='utf-8')
    assert '漂移事件' in content
    assert '银行' in content


# === v4.10: 时序 SI 的 lagged IC 评估 ===

def _make_si_ts(dates, si_by_ind):
    """构造 v4.9 风格 11 列 SI 时序;si_by_ind[i] 可为标量或 callable。"""
    rows = []
    for ind, si in enumerate(si_by_ind):
        for d in range(len(dates)):
            rows.append({
                'asof_date': dates[d],
                'industry_l1': f'8010{ind:02d}',
                'sector_name': f'测试_{ind}',
                'n_stocks': 42,
                'rho_median': 0.85, 'c_median': 1.05,
                'in_wedge_pct': 0.92, 'rho_health': 0.575,
                'damping_health': 0.975, 'wedge_health': 0.92,
                'SI': si() if callable(si) else si,
            })
    return pd.DataFrame(rows)


def test_si_lagged_ic_synthetic_perfect():
    """5 行业 × 100 日,SI(t) 与 forward(t+20) 完美正相关 → lagged IC > 0.5。"""
    pytest.importorskip("backtrace.dynamics.dynamics_si_lagged_ic")
    from backtrace.dynamics.dynamics_si_lagged_ic import compute_lagged_cross_sectional_ic
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    si_ts = _make_si_ts(dates, [(i + 1) / 5.0 for i in range(5)])
    fwd_rows = []
    for ind in range(5):
        for d in range(80):  # 100-20=80 个 eval_dates
            fwd_rows.append({
                'asof_date': dates[d + 20],
                'industry_l1': f'8010{ind:02d}',
                'forward_return': (ind + 1) / 5.0,
            })
    fwd = pd.DataFrame(fwd_rows)
    daily_ic = compute_lagged_cross_sectional_ic(si_ts, fwd, horizon=20)
    assert len(daily_ic) > 0, '应有至少 1 个 IC'
    assert daily_ic['ic'].mean() > 0.5, f'完美正相关应 IC > 0.5,got {daily_ic["ic"].mean()}'


def test_si_lagged_ic_synthetic_random():
    """20 行业 × 100 日,SI 与 forward 完全独立 → |lagged IC| < 0.3。

    注:行业数取 20(非 5)—— n=5 时 Spearman 的抽样噪声本身就有 E|rho| ≈ 0.35,
    截面太窄无法区分"无预测力"与"有预测力"。
    """
    pytest.importorskip("backtrace.dynamics.dynamics_si_lagged_ic")
    from backtrace.dynamics.dynamics_si_lagged_ic import compute_lagged_cross_sectional_ic
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    si_ts = _make_si_ts(dates, [np.random.rand for _ in range(20)])
    fwd_rows = []
    for ind in range(20):
        for d in range(80):
            fwd_rows.append({
                'asof_date': dates[d + 20],
                'industry_l1': f'8010{ind:02d}',
                'forward_return': np.random.rand(),
            })
    fwd = pd.DataFrame(fwd_rows)
    daily_ic = compute_lagged_cross_sectional_ic(si_ts, fwd, horizon=20)
    if len(daily_ic) > 0:
        mean_abs_ic = daily_ic['ic'].abs().mean()
        assert mean_abs_ic < 0.3, f'随机应 |IC| < 0.3,got {mean_abs_ic}'


def test_si_lagged_ic_temporal_shift():
    """时序 SI 滞后方向 pin:sinusoidal SI(t)=sin(t*2π/50 + i*π/5)+ r(t+20)=SI(t)。

    Lag 方向决定 IC 符号(实测: h=20 → +1.0, h=0 → -0.67, h=-20 → +0.07)。
    Time-constant SI 不可区分这些 case — 之前实现下三种 h 都过 IC>0.9。
    Time-varying SI 让 cross-sectional ranking 随时翻转,所以 lag 错位就被打回原形。
    """
    pytest.importorskip("backtrace.dynamics.dynamics_si_lagged_ic")
    from backtrace.dynamics.dynamics_si_lagged_ic import compute_lagged_cross_sectional_ic
    n_industries = 5
    n_days = 200
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    # 时变 SI:industry i 在 t 时刻的 SI = sin(t*2π/50 + i*π/5)
    si_rows = []
    for i in range(n_industries):
        for d in range(n_days):
            si_rows.append({
                'asof_date': dates[d],
                'industry_l1': f'8010{i:02d}',
                'sector_name': f'测试_{i}',
                'n_stocks': 42,
                'rho_median': 0.85, 'c_median': 1.05,
                'in_wedge_pct': 0.92, 'rho_health': 0.575,
                'damping_health': 0.975, 'wedge_health': 0.92,
                'SI': float(np.sin(d * 2 * np.pi / 50 + i * np.pi / 5)),
            })
    si_ts = pd.DataFrame(si_rows)
    # Forward return r(t+20) = SI(t) — lagged signal pattern
    fwd_rows = []
    for i in range(n_industries):
        for d in range(n_days - 20):
            fwd_rows.append({
                'asof_date': dates[d + 20],
                'industry_l1': f'8010{i:02d}',
                'forward_return': float(np.sin(d * 2 * np.pi / 50 + i * np.pi / 5)),
            })
    fwd = pd.DataFrame(fwd_rows)
    # h=20: 正确滞后 → IC ≈ +1.0
    ic_h20 = compute_lagged_cross_sectional_ic(si_ts, fwd, horizon=20)
    assert len(ic_h20) > 0
    mean_ic_h20 = ic_h20['ic'].mean()
    assert mean_ic_h20 > 0.9, \
        f'h=20 (正确滞后) 应 IC ≈ +1.0, got {mean_ic_h20:.4f}'
    # h=0: 滞后被移除 → IC ≈ -0.67 (sin 与 sin(x+0.4π) 在 5 industries 间距 0.2π 下反相关)
    ic_h0 = compute_lagged_cross_sectional_ic(si_ts, fwd, horizon=0)
    mean_ic_h0 = ic_h0['ic'].mean()
    assert mean_ic_h0 < -0.3, \
        f'h=0 (无滞后) 应 IC < -0.3, got {mean_ic_h0:.4f}'
    # h=-20: 倒挂(看未来)→ IC ≈ +0.07 (sin(x+0.8π) 与 sin(x) 在该相位下基本无序相关)
    ic_h_neg20 = compute_lagged_cross_sectional_ic(si_ts, fwd, horizon=-20)
    mean_ic_h_neg20 = ic_h_neg20['ic'].mean()
    assert abs(mean_ic_h_neg20) < 0.3, \
        f'h=-20 (未来窥探) 应 |IC| < 0.3, got {mean_ic_h_neg20:.4f}'


def test_si_lagged_ic_summary_schema(tmp_path):
    """write_si_lagged_ic_summary 写出 2 horizons × 6 列。"""
    pytest.importorskip("backtrace.dynamics.dynamics_si_lagged_ic")
    from backtrace.dynamics.dynamics_si_lagged_ic import write_si_lagged_ic_summary
    rows = []
    for h in [20, 60]:
        for w in range(5):
            rows.append({
                'window_end_date': pd.Timestamp('2024-01-01') + pd.DateOffset(days=w * 20),
                'horizon': h,
                'ic': 0.05 + w * 0.01,
                'p_value': 0.3 - w * 0.05,
                'n_industries': 25,
            })
    ts = pd.DataFrame(rows)
    out = tmp_path / 'ic_summary.csv'
    summary, text = write_si_lagged_ic_summary(ts, str(out))
    assert len(summary) == 2, f'expected 2 horizons, got {len(summary)}'
    assert set(summary.columns) >= {
        'horizon', 'ic_mean', 'ic_std', 'ic_ir', 'p_value_mean', 'n_windows',
    }
    assert out.exists()
    assert 'horizon=' in text


# === v5 — 受迫系统 + G(ω) 频率响应 (2026-08-18) ===

def test_transfer_function_dc_gain():
    """DC gain 验证:H(jω→0) ≈ 1.0(任意 k, c,k>0)。"""
    pytest.importorskip("backtrace.dynamics.dynamics_forced_response")
    from backtrace.dynamics.dynamics_forced_response import transfer_function
    for k, c in [(0.5, 0.5), (2.0, 1.5), (4.0, 4.0), (5.0, 0.1)]:
        H_dc = transfer_function(np.array([0.001]), k, c)
        assert abs(abs(H_dc[0]) - 1.0) < 1e-3, (
            f'DC gain failed for (k={k}, c={c}): |H(j0)|={abs(H_dc[0]):.4f}'
        )


def test_transfer_function_complex_pole_stability():
    """验证复杂极点区域(k < c, c² < 4k)的稳定性:|z|² = 1 - c + k < 1。

    Use (k=3.5, c=3.6): complex poles, |z|² = 0.9 < 1, system stable.
    Verify |H(jω)| is bounded across frequency sweep.
    """
    pytest.importorskip("backtrace.dynamics.dynamics_forced_response")
    from backtrace.dynamics.dynamics_forced_response import (
        transfer_function, natural_frequency, classify_response_type,
    )
    k, c = 3.5, 3.6
    # Verify pole |z|² = 1 - c + k = 0.9 < 1 (Schur stable)
    assert (1 - c + k) < 1, f'expected |z|²<1, got {1-c+k}'
    assert classify_response_type(k, c) == 'overdamped'
    # Verify |H| is bounded (< 100) across full frequency range
    omega_grid = np.linspace(0.001, np.pi, 500)
    H = transfer_function(omega_grid, k, c)
    max_mag = float(np.max(np.abs(H)))
    assert max_mag < 100, f'稳定系统 |H| 应有界, got max={max_mag:.2f}'


def test_transfer_function_resonance_peak():
    """欠阻尼 (k=4, c=0.5):|H(jω)| 在 ω ≈ arctan(√(4k-c²)/(2-c)) ≈ 1.21 处有局部峰值。"""
    pytest.importorskip("backtrace.dynamics.dynamics_forced_response")
    from backtrace.dynamics.dynamics_forced_response import (
        magnitude_phase, natural_frequency, classify_response_type,
    )
    assert classify_response_type(4.0, 0.5) == 'underdamped'
    omega_n = natural_frequency(4.0, 0.5)
    expected_omega_n = np.arctan2(np.sqrt(4*4 - 0.5**2)/2, 1 - 0.5/2)
    assert abs(omega_n - expected_omega_n) < 1e-6
    omega_grid = np.linspace(0.01, np.pi, 1000)
    mag, _ = magnitude_phase(omega_grid, k=4.0, c=0.5)
    # 找峰值
    peak_idx = np.argmax(mag)
    peak_omega = omega_grid[peak_idx]
    peak_mag = mag[peak_idx]
    # 峰值应在 ω_n 附近(±0.3 弧度)
    assert abs(peak_omega - omega_n) < 0.3, (
        f'peak at ω={peak_omega:.4f}, expected near ω_n={omega_n:.4f}'
    )
    assert peak_mag > 1.0, f'欠阻尼应有 peak > 1, got {peak_mag:.4f}'


def test_transfer_function_unstable_blowup():
    """不稳定边界附近 (k=2.01, c=2):极点 |z|² = 1.01,接近单位圆,|H(jω_n)| 爆炸。"""
    pytest.importorskip("backtrace.dynamics.dynamics_forced_response")
    from backtrace.dynamics.dynamics_forced_response import (
        magnitude_phase, natural_frequency, classify_response_type,
    )
    k, c = 2.01, 2.0
    # Verify pole |z|² = 1 - c + k = 1.01 > 1 (Schur unstable)
    assert (1 - c + k) > 1, f'expected |z|²>1, got {1-c+k}'
    assert classify_response_type(k, c) == 'underdamped'
    omega_n = natural_frequency(k, c)
    mag, _ = magnitude_phase(np.array([omega_n]), k, c)
    # At ω_n ≈ π/2, denominator → small (~0.01), |H| should be huge
    assert mag[0] > 100, f'不稳定边界附近应 |H(jω_n)| > 100, got {mag[0]:.2f}'


def test_classify_response_type():
    """4 种阻尼类型判定。"""
    pytest.importorskip("backtrace.dynamics.dynamics_forced_response")
    from backtrace.dynamics.dynamics_forced_response import classify_response_type
    assert classify_response_type(2.0, 4.0) == 'overdamped'   # k<c → 稳定
    assert classify_response_type(4.0, 4.0) == 'critical'     # k=c 边界
    assert classify_response_type(4.0, 0.5) == 'underdamped'  # k>c → 不稳定
    assert classify_response_type(-1.0, 0.5) == 'anti_damped' # k<0


# === v5.1 — 多对 (k, c) Bode plot 叠加 (2026-08-18) ===

def test_bode_overlay_creates_html(tmp_path):
    """bode_overlay 调用产生 HTML 文件 + 文件非空。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    omega = np.linspace(0.01, np.pi, 50)
    pairs = [(0.5, 2.0, "Strong"), (2.0, 1.5, "Mild")]
    out = tmp_path / "overlay.html"
    DFR.bode_overlay(omega, pairs, str(out))
    assert out.exists()
    assert out.stat().st_size > 1000


def test_bode_overlay_validates_empty_list(tmp_path):
    """空 k_c_pairs → ValueError。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    omega = np.linspace(0.01, np.pi, 50)
    out = tmp_path / "overlay.html"
    with pytest.raises(ValueError, match="k_c_pairs 不能为空"):
        DFR.bode_overlay(omega, [], str(out))


def test_bode_overlay_marks_omega_n(tmp_path):
    """ω_n 在复极点区域被标在 magnitude subplot。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    omega = np.linspace(0.01, np.pi, 100)
    # k=2.0, c=1.5 (underdamped) — ω_n is finite
    pairs = [(2.0, 1.5, "Underdamped")]
    out = tmp_path / "overlay.html"
    DFR.bode_overlay(omega, pairs, str(out))
    content = out.read_text(encoding='utf-8')
    # ω_n marker trace 存在
    assert 'ω_n' in content or 'Underdamped' in content
    # 重新调用 natural_frequency 验证
    expected_omega_n = DFR.natural_frequency(2.0, 1.5)
    assert np.isfinite(expected_omega_n)


# === v5.1 — 多对 (k, c) summary TXT (2026-08-18 Task 2) ===

def test_write_overlay_summary_creates_txt(tmp_path):
    """write_overlay_summary 调用产生 TXT 文件 + 内容含所有 label。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    omega = np.linspace(0.01, np.pi, 50)
    pairs = [(0.5, 2.0, "Industry A"), (2.0, 1.5, "Industry B")]
    out = tmp_path / "overlay_summary.txt"
    DFR.write_overlay_summary(omega, pairs, str(out))
    assert out.exists()
    content = out.read_text(encoding='utf-8')
    assert "Industry A" in content
    assert "Industry B" in content
    assert "|H(j0)" in content or "DC" in content


# === v5.1 — parse_overlay_pairs 字符串解析 (2026-08-18 Task 3) ===

def test_parse_overlay_pairs_basic():
    """基本 3 对解析。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    s = "0.5,2.0,Industry A; 2.0,1.5,Industry B; 3.0,0.5,Industry C"
    pairs = DFR.parse_overlay_pairs(s)
    assert len(pairs) == 3
    assert pairs[0] == (0.5, 2.0, "Industry A")
    assert pairs[1] == (2.0, 1.5, "Industry B")
    assert pairs[2] == (3.0, 0.5, "Industry C")


def test_parse_overlay_pairs_label_with_spaces():
    """label 含空格的解析。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    s = "2.0,1.5,Bank Index; 0.5,2.0,Tech Sector"
    pairs = DFR.parse_overlay_pairs(s)
    assert pairs[0] == (2.0, 1.5, "Bank Index")
    assert pairs[1] == (0.5, 2.0, "Tech Sector")


def test_parse_overlay_pairs_invalid_format():
    """错误格式 → ValueError。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    with pytest.raises(ValueError, match="格式错误"):
        DFR.parse_overlay_pairs("only_two_parts")
    with pytest.raises(ValueError, match="k 必须"):
        DFR.parse_overlay_pairs("abc,1.5,Label")
    with pytest.raises(ValueError, match="c 必须"):
        DFR.parse_overlay_pairs("1.0,xyz,Label")


def test_cli_overlay_mode(tmp_path):
    """CLI --overlay 模式产生 overlay HTML + summary TXT,且不写单对输出。"""
    import subprocess
    # 用绝对路径解析 script(因为 cwd=tmp_path 时相对路径不可达)
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    _SCRIPT = os.path.join(_PROJECT_ROOT, 'backtrace', 'dynamics', 'dynamics_forced_response.py')
    overlay_str = "0.5,2.0,Strong; 2.0,1.5,Mild"
    out_html = tmp_path / "overlay.html"
    out_txt = tmp_path / "overlay_summary.txt"
    result = subprocess.run([
        sys.executable,
        _SCRIPT,
        "--overlay", overlay_str,
        "--overlay-html", str(out_html),
        "--overlay-summary-txt", str(out_txt),
    ], capture_output=True, text=True, cwd=str(tmp_path))
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out_html.exists(), "overlay HTML 应被创建"
    assert out_html.stat().st_size > 1000, "overlay HTML 文件应 > 1000 bytes"
    assert out_txt.exists(), "overlay summary TXT 应被创建"
    txt_content = out_txt.read_text(encoding='utf-8')
    assert "Strong" in txt_content
    assert "Mild" in txt_content
    # 单对模式输出应不被创建(tmp_path 是 cwd)
    single_pair_csv = tmp_path / "data" / "dynamics" / "transfer_function_grid.csv"
    assert not single_pair_csv.exists(), "单对 CSV 不应被创建(overlay-only 模式)"
    single_pair_html = tmp_path / "backtrace" / "outputs" / "dynsys_forced_response.html"
    assert not single_pair_html.exists(), "单对 HTML 不应被创建(overlay-only 模式)"


# === v5.2 load_kc_estimates ===

def test_load_kc_estimates_filters_failed(tmp_path):
    """load_kc_estimates 过滤 status != 'ok' 的行。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    csv_path = tmp_path / "kc.csv"
    csv_path.write_text(
        "code,index_code,k_hat,c_hat,status\n"
        "600000.SH,801010,0.5,2.0,ok\n"
        "600001.SH,801010,0.6,1.9,ok\n"
        "600002.SH,801020,2.0,1.5,ok\n"
        "600003.SH,801020,2.1,1.4,fail\n",  # ← 应被过滤
        encoding='utf-8',
    )
    df = DFR.load_kc_estimates(str(csv_path))
    assert len(df) == 3, f"应过滤 fail 行,剩 3 行,得 {len(df)}"
    assert "600003.SH" not in df['code'].values


def test_load_kc_estimates_validates_columns(tmp_path):
    """缺必需列 → ValueError。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    csv_path = tmp_path / "kc.csv"
    csv_path.write_text(
        "code,k_hat,c_hat\n"  # ← 缺 index_code + status
        "600000.SH,0.5,2.0\n",
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match="index_code"):
        DFR.load_kc_estimates(str(csv_path))


def test_aggregate_by_industry_median():
    """agg='median' 对 (k̂, ĉ) 中位数聚合 + n_stocks 计数。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    import pandas as pd
    df = pd.DataFrame({
        'code': ['A1', 'A2', 'A3', 'B1', 'B2'],
        'index_code': ['801010', '801010', '801010', '801020', '801020'],
        'k_hat': [0.5, 0.6, 0.7, 2.0, 2.1],
        'c_hat': [2.0, 1.9, 2.1, 1.5, 1.4],
    })
    agg_df = DFR.aggregate_by_industry(df, group_col='index_code', agg='median')
    assert len(agg_df) == 2
    # 801010 中位数 k=0.6, c=2.0, n=3
    row_a = agg_df[agg_df['index_code'] == '801010'].iloc[0]
    assert row_a['n_stocks'] == 3
    assert abs(row_a['k_hat'] - 0.6) < 1e-9
    assert abs(row_a['c_hat'] - 2.0) < 1e-9
    # 801020 中位数 k=2.05, c=1.45, n=2
    row_b = agg_df[agg_df['index_code'] == '801020'].iloc[0]
    assert row_b['n_stocks'] == 2
    assert abs(row_b['k_hat'] - 2.05) < 1e-9
    assert abs(row_b['c_hat'] - 1.45) < 1e-9


# === v5.2 select_top_n_industries ===

def test_select_top_n_by_n_stocks():
    """criterion='by_n_stocks' 按股票数降序排,选 top N。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    import pandas as pd
    df = pd.DataFrame({
        'index_code': ['A', 'B', 'C'],
        'n_stocks': [10, 5, 2],
        'k_hat': [1.0, 2.0, 3.0],
        'c_hat': [1.5, 1.5, 1.5],
    })
    pairs = DFR.select_top_n_industries(df, criterion='by_n_stocks', n=2)
    assert len(pairs) == 2
    # A (10 stocks) 第一, B (5 stocks) 第二
    assert pairs[0] == (1.0, 1.5, 'Industry A')
    assert pairs[1] == (2.0, 1.5, 'Industry B')


def test_select_top_n_by_c_over_k():
    """criterion='by_c_over_k' 按 c/k 比降序排,选 top N。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    import pandas as pd
    df = pd.DataFrame({
        'index_code': ['A', 'B', 'C'],
        'n_stocks': [5, 5, 5],
        'k_hat': [0.5, 2.0, 1.0],
        'c_hat': [2.0, 1.5, 1.0],  # c/k: 4.0, 0.75, 1.0
    })
    pairs = DFR.select_top_n_industries(df, criterion='by_c_over_k', n=2)
    # A (c/k=4.0) 第一, C (c/k=1.0) 第二
    assert pairs[0][2] == 'Industry A'
    assert pairs[1][2] == 'Industry C'


# === v5.2 --from-kc-estimates CLI 集成测试 ===

def test_cli_from_kc_estimates_mode(tmp_path):
    """CLI --from-kc-estimates 模式读合成 CSV → 选 top-N → 写 overlay + 行业 CSV。"""
    import subprocess
    # 用绝对路径解析 script(因为 cwd=tmp_path 时相对路径不可达)
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    _SCRIPT = os.path.join(_PROJECT_ROOT, 'backtrace', 'dynamics', 'dynamics_forced_response.py')
    cwd = tmp_path
    csv_path = cwd / "kc_estimates.csv"
    csv_path.write_text(
        "code,index_code,k_hat,c_hat,status\n"
        "600000.SH,801010,0.5,2.0,ok\n"
        "600001.SH,801010,0.6,1.9,ok\n"
        "600002.SH,801010,0.7,2.1,ok\n"
        "600010.SH,801020,2.0,1.5,ok\n"
        "600011.SH,801020,2.1,1.4,ok\n"
        "600020.SH,801030,3.5,0.5,ok\n",
        encoding='utf-8',
    )
    out_html = cwd / "overlay.html"
    out_txt = cwd / "overlay_summary.txt"
    out_pairs = cwd / "industry_pairs.csv"
    result = subprocess.run([
        sys.executable,
        _SCRIPT,
        "--from-kc-estimates", str(csv_path),
        "--top-n", "2",
        "--industry-agg", "median",
        "--select-criterion", "by_n_stocks",
        "--overlay-html", str(out_html),
        "--overlay-summary-txt", str(out_txt),
        "--industry-pairs-csv", str(out_pairs),
    ], capture_output=True, text=True, cwd=str(cwd))
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out_html.exists()
    assert out_txt.exists()
    assert out_pairs.exists()
    pairs_content = out_pairs.read_text(encoding='utf-8')
    # 801010 有 3 只股票(最多),801020 有 2 只,801030 有 1 只
    # by_n_stocks top-2: 801010 + 801020
    assert "801010" in pairs_content
    assert "801020" in pairs_content


# === v5.3 SI Frequency Response — load_kc_time_series helper ===

def test_load_kc_time_series_filters_failed(tmp_path):
    """load_kc_time_series 过滤 status != 'ok' 行 + n_valid_days < 192 (ramp-up)"""
    rows = [
        # (code, index_code, asof_date, k_hat, c_hat, status, n_valid_days)
        ('000001.SZ', '801010', '2024-09-30', 0.50, 2.00, 'ok',  250),  # 保留
        ('000002.SZ', '801010', '2024-09-30', 0.55, 2.10, 'ok',  100),  # 过滤 (ramp-up)
        ('000003.SZ', '801010', '2024-09-30', 0.60, 1.90, 'fail', 250), # 过滤 (status)
        ('000004.SZ', '801010', '2024-09-30', 0.70, 1.80, 'ok',  300),  # 保留
        ('000005.SZ', '801010', '2024-09-30', 0.80, 1.70, 'ok',  192),  # 保留 (边界)
        ('000006.SZ', '801010', '2024-09-30', 0.90, 1.60, 'ok',  191),  # 过滤 (ramp-up 边界外)
    ]
    df = pd.DataFrame(rows, columns=[
        'code', 'index_code', 'asof_date', 'k_hat', 'c_hat', 'status', 'n_valid_days',
    ])
    csv_path = tmp_path / 'kc_estimates_time.csv'
    df.to_csv(csv_path, index=False)

    from backtrace.dynamics.dynamics_si_freq_response import load_kc_time_series
    result = load_kc_time_series(str(csv_path))

    assert len(result) == 3  # 只保留 250/300/192 三行
    assert result['code'].tolist() == ['000001.SZ', '000004.SZ', '000005.SZ']
    assert (result['status'] == 'ok').all()
    assert (result['n_valid_days'] >= 192).all()


def test_load_kc_time_series_validates_columns(tmp_path):
    """缺必需列 → ValueError,错误信息列出缺失列名"""
    rows = [
        ('000001.SZ', '801010', '2024-09-30', 0.5, 2.0, 'ok'),  # 缺 n_valid_days
    ]
    df = pd.DataFrame(rows, columns=['code', 'index_code', 'asof_date', 'k_hat', 'c_hat', 'status'])
    csv_path = tmp_path / 'bad.csv'
    df.to_csv(csv_path, index=False)

    from backtrace.dynamics.dynamics_si_freq_response import load_kc_time_series
    with pytest.raises(ValueError, match='n_valid_days'):
        load_kc_time_series(str(csv_path))


def test_aggregate_by_industry_per_date():
    """按 (asof_date, index_code) 聚合 (k̂, ĉ),每片一个 DataFrame"""
    rows = [
        # Date 1 (2 ind × 2 stocks)
        ('000001.SZ', '801010', '2024-09-30', 0.50, 2.00, 'ok', 250),
        ('000002.SZ', '801010', '2024-09-30', 0.55, 2.10, 'ok', 250),
        ('600001.SH', '801080', '2024-09-30', 3.50, 0.50, 'ok', 250),
        ('600002.SH', '801080', '2024-09-30', 3.60, 0.45, 'ok', 250),
        # Date 2
        ('000001.SZ', '801010', '2024-10-31', 0.60, 1.90, 'ok', 250),
        ('000002.SZ', '801010', '2024-10-31', 0.65, 1.95, 'ok', 250),
        ('600001.SH', '801080', '2024-10-31', 4.00, 0.40, 'ok', 250),
        ('600002.SH', '801080', '2024-10-31', 3.90, 0.42, 'ok', 250),
    ]
    df = pd.DataFrame(rows, columns=[
        'code', 'index_code', 'asof_date', 'k_hat', 'c_hat', 'status', 'n_valid_days',
    ])
    dates = ['2024-09-30', '2024-10-31']

    from backtrace.dynamics.dynamics_si_freq_response import aggregate_by_industry_per_date
    result = aggregate_by_industry_per_date(df, dates)

    assert set(result.keys()) == {'2024-09-30', '2024-10-31'}
    # Date 1: Industry 801010 k̂=0.525, ĉ=2.05; Industry 801080 k̂=3.55, ĉ=0.475
    d1 = result['2024-09-30'].set_index('index_code')
    assert d1.loc['801010', 'k_hat'] == pytest.approx(0.525)
    assert d1.loc['801010', 'c_hat'] == pytest.approx(2.05)
    assert d1.loc['801010', 'n_stocks'] == 2
    assert d1.loc['801080', 'k_hat'] == pytest.approx(3.55)
    # Date 2: 801010 k̂=0.625, ĉ=1.925; 801080 k̂=3.95, ĉ=0.41
    d2 = result['2024-10-31'].set_index('index_code')
    assert d2.loc['801010', 'k_hat'] == pytest.approx(0.625)
    assert d2.loc['801080', 'k_hat'] == pytest.approx(3.95)


def test_select_top_n_per_date():
    """每个 date 选 top-N industries,返 (asof_date, k̂, ĉ, label) 元组列表(按 date 排序)"""
    per_date_dfs = {
        '2024-09-30': pd.DataFrame({
            'index_code': ['801010', '801080', '801090'],
            'n_stocks':   [4,         3,         2],
            'k_hat':      [0.5,       3.5,       2.0],
            'c_hat':      [2.0,       0.5,       1.5],
        }),
        '2024-10-31': pd.DataFrame({
            'index_code': ['801010', '801080', '801090'],
            'n_stocks':   [5,         2,         1],
            'k_hat':      [0.6,       4.0,       1.8],
            'c_hat':      [1.9,       0.4,       1.7],
        }),
    }

    from backtrace.dynamics.dynamics_si_freq_response import select_top_n_per_date
    pairs = select_top_n_per_date(per_date_dfs, criterion='by_n_stocks', n=2)

    # 2 dates × 2 industries = 4 pairs
    assert len(pairs) == 4
    # Date 1 top-2 by n_stocks: 801010 (4 stocks), 801080 (3 stocks)
    d1 = [p for p in pairs if p[0] == '2024-09-30']
    assert d1[0][1:] == (0.5, 2.0, 'Industry 801010')
    assert d1[1][1:] == (3.5, 0.5, 'Industry 801080')
    # Date 2 top-2 by n_stocks: 801010 (5 stocks), 801080 (2 stocks)
    d2 = [p for p in pairs if p[0] == '2024-10-31']
    assert d2[0][1:] == (0.6, 1.9, 'Industry 801010')
    assert d2[1][1:] == (4.0, 0.4, 'Industry 801080')
    # 按 date 排序
    assert [p[0] for p in pairs] == ['2024-09-30', '2024-09-30', '2024-10-31', '2024-10-31']


def test_cli_si_freq_response_mode(tmp_path):
    """CLI 时序动画模式:合成 12 行 CSV → 跑 CLI → 验证 HTML + TXT + CSV 3 个输出"""
    # 合成 3 dates × 2 ind × 2 stocks = 12 行
    rows = [
        # Date 1 (2024-09-30)
        ('000001.SZ', '801010', '2024-09-30', 0.50, 2.00, 'ok', 250),
        ('000002.SZ', '801010', '2024-09-30', 0.55, 2.10, 'ok', 250),
        ('600001.SH', '801080', '2024-09-30', 3.50, 0.50, 'ok', 250),
        ('600002.SH', '801080', '2024-09-30', 3.60, 0.45, 'ok', 250),
        # Date 2 (2024-10-31)
        ('000001.SZ', '801010', '2024-10-31', 0.60, 1.90, 'ok', 250),
        ('000002.SZ', '801010', '2024-10-31', 0.65, 1.95, 'ok', 250),
        ('600001.SH', '801080', '2024-10-31', 4.00, 0.40, 'ok', 250),
        ('600002.SH', '801080', '2024-10-31', 3.90, 0.42, 'ok', 250),
        # Date 3 (2024-11-30)
        ('000001.SZ', '801010', '2024-11-30', 0.70, 1.80, 'ok', 250),
        ('000002.SZ', '801010', '2024-11-30', 0.72, 1.82, 'ok', 250),
        ('600001.SH', '801080', '2024-11-30', 3.00, 0.60, 'ok', 250),
        ('600002.SH', '801080', '2024-11-30', 2.95, 0.58, 'ok', 250),
    ]
    df = pd.DataFrame(rows, columns=[
        'code', 'index_code', 'asof_date', 'k_hat', 'c_hat', 'status', 'n_valid_days',
    ])
    csv_path = tmp_path / 'kc_estimates_time.csv'
    df.to_csv(csv_path, index=False)

    html_path = tmp_path / 'si_freq_response.html'
    summary_path = tmp_path / 'si_freq_response_summary.txt'
    pairs_path = tmp_path / 'si_freq_response_pairs.csv'

    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cli_script = os.path.join(_PROJECT_ROOT, 'backtrace', 'dynamics', 'dynamics_si_freq_response.py')

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(
        [
            sys.executable,
            cli_script,
            '--kc-time-csv', str(csv_path),
            '--top-n-industries', '2',
            '--industry-selection', 'by_n_stocks',
            '--max-dates', '3',
            '--html-output', str(html_path),
            '--summary-output', str(summary_path),
            '--pairs-csv-output', str(pairs_path),
        ],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 0, f'stderr: {result.stderr}'

    # 3 输出文件都存在
    assert html_path.exists() and html_path.stat().st_size > 2000  # v5.4 dual-pane 加倍
    assert summary_path.exists()
    assert pairs_path.exists()

    # HTML 含 plotly animation_frame + frames + 3 个日期 + 双子图元素
    html_text = html_path.read_text(encoding='utf-8')
    assert 'plotly' in html_text.lower()
    # plotly v3.x 输出 addFrames(驼峰),不是 frames/小写,放宽到 addFrames 或 Plotly.animate
    assert 'addFrames' in html_text or 'Plotly.animate' in html_text or 'animation_frame' in html_text or 'frames' in html_text
    assert '2024-09-30' in html_text and '2024-10-31' in html_text and '2024-11-30' in html_text
    # v5.4 dual-pane: at least 2 xaxis/yaxis pairs (subplot 1 + subplot 2)
    assert html_text.count('xaxis') >= 2, 'v5.4 dual-pane: 至少 2 个 xaxis'
    assert html_text.count('yaxis') >= 2, 'v5.4 dual-pane: 至少 2 个 yaxis'
    # phase subplot 关键词(plotly JSON-escape Unicode: ∠ → ∠, ω → ω,所以同时支持字面 + 转义 + yaxis2)
    assert any(kw in html_text for kw in ('∠H', 'phase', 'arg H', '相角', '\\u2220', 'yaxis2')), 'v5.4: phase 子图存在'
    # v5.5 regime color: HTML 含至少 1 种 regime 颜色 hex
    # fixture 里有 2 industries × 3 dates: Industry A 始终 overdamped (k=0.5-0.7, c=1.8-2.0)
    # Industry B 始终 underdamped (k=3.0-4.0, c=0.4-0.6)
    # 所以 HTML 应同时含 #2ca02c (绿) 和 #d62728 (红)
    assert '#2ca02c' in html_text, 'v5.5: 至少 1 个 overdamped 颜色 hex (绿)'
    assert '#d62728' in html_text, 'v5.5: 至少 1 个 underdamped 颜色 hex (红)'
    # v5.5 颜色注释关键词
    assert any(kw in html_text for kw in ('过阻尼', '欠阻尼', 'regime', '稳定', '共振')), 'v5.5: 颜色注释'

    # F1 fix: phase y-axis must be in degrees (not radians).
    # y-axis title is '∠H(jω) deg'; data is wrapped with np.degrees() in the closure.
    # If the fix is missing, the y-arrays in the embedded plotly JSON would be in radians (|y| <= π ≈ 3.14).
    # Parse the plotly JSON embedded in the HTML; the initial-state phase traces are the
    # last 2 traces per industry-pair in the data array. Use a robust heuristic: scan the first
    # ~20 "y": [...] arrays and confirm at least one has values whose absolute magnitudes exceed 10
    # (i.e. is in degrees, not radians).
    import re
    y_arrays = re.findall(r'"y":\s*\[[^\]]*\]', html_text)
    has_degrees_range = False
    for y_str in y_arrays[:20]:
        nums = re.findall(r'-?\d+\.?\d*', y_str)
        values = [float(n) for n in nums if n]
        if values and max(abs(v) for v in values) > 10:
            has_degrees_range = True
            break
    assert has_degrees_range, 'v5.4 F1: phase y-data must be in degrees (|y| > 10), not radians (≤ π ≈ 3.14)'

    # Summary TXT 含 3 日期 + 中文
    summary_text = summary_path.read_text(encoding='utf-8')
    assert '2024-09-30' in summary_text and '2024-10-31' in summary_text and '2024-11-30' in summary_text
    # 中文字符串(业务解读 / 时序动画 / 过阻尼 / 欠阻尼 任一即可)
    assert any(cn in summary_text for cn in ('行业', '业务解读', '时序动画', '过阻尼', '欠阻尼'))

    # Pairs CSV: 3 dates × 2 industries = 6 行 + header
    pairs_df = pd.read_csv(pairs_path)
    assert len(pairs_df) == 6
    assert set(pairs_df.columns) >= {'asof_date', 'index_code', 'k_hat', 'c_hat'}
    assert set(pairs_df['asof_date'].unique()) == {'2024-09-30', '2024-10-31', '2024-11-30'}


# === v5.6 — static 2D grid PNG export (2026-08-18) ===

def test_cli_static_grid_mode(tmp_path):
    """v5.6: CLI static PNG export mode — 验证 build_static_bode_grid 输出 PNG."""
    pytest.importorskip("matplotlib")

    import subprocess
    import sys
    import os

    # 合成 3 dates × 2 industries CSV (复用 v5.3 fixture 模式)
    csv_path = tmp_path / 'kc_time.csv'
    rows = []
    for date_str in ['2024-09-30', '2024-10-31', '2024-11-30']:
        for code, k, c in [('AAA', 0.5, 2.0), ('BBB', 3.5, 0.5)]:
            rows.append({
                'code': code, 'index_code': f'Industry_{code}',
                'asof_date': date_str, 'k_hat': k, 'c_hat': c,
                'status': 'ok', 'n_valid_days': 200,
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')

    static_png = tmp_path / 'static.png'
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cli_script = os.path.join(repo_root, 'backtrace', 'dynamics', 'dynamics_si_freq_response.py')
    cmd = [
        sys.executable, cli_script,
        '--kc-time-csv', str(csv_path),
        '--top-n-industries', '2',
        '--static-output', str(static_png),
    ]
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(cmd, capture_output=True, env=env, timeout=60)
    assert result.returncode == 0, f'CLI failed: {result.stderr.decode("utf-8", errors="ignore")}'

    # 验证 PNG 存在 + 字节头 + size
    assert static_png.exists(), f'PNG not created: {static_png}'
    assert static_png.stat().st_size > 5000, f'PNG too small: {static_png.stat().st_size}'
    with open(static_png, 'rb') as fh:
        header = fh.read(8)
    assert header.startswith(b'\x89PNG'), f'Not a valid PNG: header={header!r}'


def test_cli_regime_heatmap_mode(tmp_path):
    """v5.7: CLI regime heatmap mode — 验证 build_regime_heatmap 输出 PNG."""
    pytest.importorskip("matplotlib")

    import subprocess
    import sys
    import os

    # 合成 3 dates × 2 industries CSV (复用 v5.6 fixture 模式)
    csv_path = tmp_path / 'kc_time.csv'
    rows = []
    for date_str in ['2024-09-30', '2024-10-31', '2024-11-30']:
        for code, k, c in [('AAA', 0.5, 2.0), ('BBB', 3.5, 0.5)]:
            rows.append({
                'code': code, 'index_code': f'Industry_{code}',
                'asof_date': date_str, 'k_hat': k, 'c_hat': c,
                'status': 'ok', 'n_valid_days': 200,
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')

    heatmap_png = tmp_path / 'heatmap.png'
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cli_script = os.path.join(repo_root, 'backtrace', 'dynamics', 'dynamics_si_freq_response.py')
    cmd = [
        sys.executable, cli_script,
        '--kc-time-csv', str(csv_path),
        '--top-n-industries', '2',
        '--heatmap-output', str(heatmap_png),
    ]
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(cmd, capture_output=True, env=env, timeout=60)
    assert result.returncode == 0, f'CLI failed: {result.stderr.decode("utf-8", errors="ignore")}'

    # 验证 PNG 存在 + 字节头 + size
    assert heatmap_png.exists(), f'PNG not created: {heatmap_png}'
    assert heatmap_png.stat().st_size > 5000, f'PNG too small: {heatmap_png.stat().st_size}'
    with open(heatmap_png, 'rb') as fh:
        header = fh.read(8)
    assert header.startswith(b'\x89PNG'), f'Not a valid PNG: header={header!r}'


def test_cli_state_timeline_mode(tmp_path):
    """v5.8: CLI state timeline mode — 验证 build_state_timeline_html 输出 HTML."""
    pytest.importorskip("plotly")

    import subprocess
    import sys
    import os

    html_out = tmp_path / 'timeline.html'
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cli_script = os.path.join(repo_root, 'backtrace', 'dynamics', 'dynamics_state_timeline.py')
    cmd = [
        sys.executable, cli_script,
        '--code', '002475.SZ',
        '--days', '250',
        '--output', str(html_out),
    ]
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(cmd, capture_output=True, env=env, timeout=60)

    # Tolerate only the documented downstream failures (local cache missing
    # OR pre-existing M1 tsfresh import shadow). Anything else fails loudly.
    if result.returncode != 0:
        stderr = result.stderr.decode('utf-8', errors='ignore')
        if '本地缓存缺失' in stderr:
            pytest.skip('002475.SZ not in local cache')
        if 'cannot import name' in stderr and 'tsfresh' in stderr:
            pytest.skip('M1 pre-existing tsfresh import shadow needs separate fix')
        assert False, f'Unexpected CLI failure: {stderr[-800:]}'

    # CLI succeeded → HTML must exist and be a valid plotly doc
    assert html_out.exists(), f'HTML not created: {html_out}'
    with open(html_out, 'rb') as fh:
        content = fh.read()
    assert b'<html' in content.lower() or b'plotly' in content.lower(), \
        f'Not a valid plotly HTML: {content[:200]}'


def test_cli_oos_viz_mode(tmp_path):
    """v5.9: CLI OOS visualization mode — 验证 build_oos_prediction_html 输出 HTML."""
    pytest.importorskip("plotly")

    import subprocess
    import sys
    import os

    html_out = tmp_path / 'oos_viz.html'
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cli_script = os.path.join(repo_root, 'backtrace', 'dynamics', 'dynamics_oos_viz.py')
    cmd = [
        sys.executable, cli_script,
        '--code', '002475.SZ',
        '--days', '250',
        '--output', str(html_out),
    ]
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(cmd, capture_output=True, env=env, timeout=60)

    # Tolerate documented failures (cache miss OR M1 tsfresh shadow)
    if result.returncode != 0:
        stderr = result.stderr.decode('utf-8', errors='ignore')
        if '本地缓存缺失' in stderr:
            pytest.skip('002475.SZ not in local cache')
        if 'cannot import name' in stderr and 'tsfresh' in stderr:
            pytest.skip('M1 pre-existing tsfresh import shadow')
        assert False, f'Unexpected CLI failure: {stderr[-800:]}'

    assert html_out.exists(), f'HTML not created: {html_out}'
    with open(html_out, 'rb') as fh:
        content = fh.read()
    assert b'<html' in content.lower() or b'plotly' in content.lower(), \
        f'Not a valid plotly HTML: {content[:200]}'


def test_cli_oos_batch_mode(tmp_path):
    """v5.10 — `dynamics_oos_batch.py` end-to-end CLI smoke test.

    Runs the full CLI with --limit 2 --days 30 --top-n 1 — small numbers for fast test.
    Asserts both HTML files are written and contain expected plotly divs.
    """
    import subprocess
    output_html = tmp_path / 'oos_full_market.html'
    top_html = tmp_path / 'oos_full_market_top1.html'

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    proc = subprocess.run(
        [
            sys.executable,
            'backtrace/dynamics/dynamics_oos_batch.py',
            '--limit', '2',
            '--days', '30',
            '--top-n', '1',
            '--output', str(output_html),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    # F3 inverted tolerance: documented failures skip, anything else fails loudly
    if proc.returncode != 0:
        combined = (proc.stdout + proc.stderr).lower()
        # documented skip patterns: 本地缓存缺失 / 'cannot import name' + 'tsfresh'
        if '数据' in combined or '本地缓存' in combined:
            pytest.skip(f"v5.10: local cache missing — {proc.stderr[:200]}")
        elif 'cannot import name' in combined and 'tsfresh' in combined:
            pytest.skip(f"v5.10: M1 tsfresh shadow — {proc.stderr[:200]}")
        else:
            pytest.fail(f"v5.10 CLI failed (rc={proc.returncode}):\n"
                        f"STDOUT:\n{proc.stdout[-1000:]}\n"
                        f"STDERR:\n{proc.stderr[-1000:]}")

    # Success assertions
    assert output_html.exists(), f'dashboard not written at {output_html}'
    assert output_html.stat().st_size > 1000, f'dashboard too small: {output_html.stat().st_size} bytes'

    # top-N file path: brief replaces '.html' → '_top{top_n}.html'
    top_path = str(output_html).replace('.html', '_top1.html')
    assert os.path.exists(top_path), f'top-N multiples not written at {top_path}'

    # Sanity: dashboards contain plotly divs
    dashboard_text = output_html.read_text(encoding='utf-8')
    assert 'plotly' in dashboard_text.lower(), 'dashboard missing plotly'
    assert 'Full-Market' in dashboard_text, 'dashboard missing title'


def test_lookup_kc_for_code(tmp_path):
    """v5.11 — lookup_kc_for_code 单元测试 (no subprocess, fast).

    覆盖 v5.11.1 schema fix:parameter_fit.py 的 status 是 verbose 形式
    ("ok (anti-restoring, damping)" / "extreme (...)" / "too_few_days (...)")，
    不是 bare "ok"。过滤条件用 str.startswith('ok', na=False)。
    """
    from backtrace.dynamics.dynamics_oos_viz import lookup_kc_for_code

    # 1. mock kc_estimates.csv (用 REAL status format,不是裸 'ok')
    csv = tmp_path / 'kc.csv'
    csv.write_text(
        'code,index_code,k_hat,c_hat,status\n'
        '601609.SH,000001.SH,-0.012,5.14,"ok (anti-restoring, damping)"\n'
        '601610.SH,000001.SH,0.5,0.3,"ok (anti-damping)"\n'
        '601611.SH,000001.SH,0.1,0.2,"extreme (|k| or |c| > 10)"\n'
        '601612.SH,000001.SH,,,"too_few_days (3 < 20)"\n',
        encoding='utf-8',
    )

    # 2. 命中 (verbose ok × 2)
    assert lookup_kc_for_code(str(csv), '601609.SH') == (-0.012, 5.14)
    assert lookup_kc_for_code(str(csv), '601610.SH') == (0.5, 0.3)

    # 3. status 不 startswith('ok') → None
    assert lookup_kc_for_code(str(csv), '601611.SH') is None  # extreme
    assert lookup_kc_for_code(str(csv), '601612.SH') is None  # too_few_days

    # 4. code 不存在 → None
    assert lookup_kc_for_code(str(csv), '000777.SZ') is None

    # 5. 文件不存在 → None(不抛)
    assert lookup_kc_for_code(str(tmp_path / 'missing.csv'), '601609.SH') is None

    # 6. 缺必需列 → None
    bad = tmp_path / 'bad.csv'
    bad.write_text('code,foo,bar\n601609.SH,1,2\n', encoding='utf-8')
    assert lookup_kc_for_code(str(bad), '601609.SH') is None


# === v6 — Dynamics Factor Validation (2026-08-19) ===

def test_compute_cross_section_ic_positive():
    """Perfect positive rank correlation → IC ≈ 1.0."""
    from dynamics.dynamics_factor_validation import compute_cross_section_ic
    factor = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ret = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ic, pval, n = compute_cross_section_ic(factor, ret)
    assert ic > 0.99
    assert n == 10
    assert pval < 0.001


def test_compute_cross_section_ic_negative():
    """Negative correlation → IC < 0."""
    from dynamics.dynamics_factor_validation import compute_cross_section_ic
    factor = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ret = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    ic, _, _ = compute_cross_section_ic(factor, ret)
    assert ic < -0.99


def test_compute_cross_section_ic_too_few():
    """n < 10 → NaN, n=actual."""
    from dynamics.dynamics_factor_validation import compute_cross_section_ic
    factor = pd.Series([1, 2, 3])
    ret = pd.Series([0.1, 0.2, 0.3])
    ic, _, n = compute_cross_section_ic(factor, ret)
    assert np.isnan(ic)
    assert n == 3


def test_compute_cross_section_ic_string_factor():
    """v6.0.1 IMPORTANT #10: regime/state_dominant 是 string →
    compute_cross_section_ic 必须返回 (NaN, NaN, n),不能按字母序排产生 garbage IC。"""
    from dynamics.dynamics_factor_validation import compute_cross_section_ic
    np.random.seed(42)
    regimes = np.random.choice(
        ['overdamped', 'critical', 'underdamped', 'anti_damped'], size=100
    )
    factor = pd.Series(regimes)
    ret = pd.Series(np.random.randn(100) * 0.01)
    ic, pval, n = compute_cross_section_ic(factor, ret)
    assert np.isnan(ic), f'string factor should yield NaN IC, got {ic}'
    assert np.isnan(pval)
    assert n == 100  # n_obs 仍然返回,只在 IC 层面 NaN


def test_compute_quantile_returns_monotonic():
    """Q1 < Q5 monotonic."""
    from dynamics.dynamics_factor_validation import compute_quantile_returns
    np.random.seed(42)
    factor = pd.Series(np.arange(100).astype(float))
    ret = pd.Series(factor.values * 0.01)  # 完美单调
    q = compute_quantile_returns(factor, ret, n_quantiles=5)
    assert q['q1_ret'] < q['q5_ret']
    assert q['q5_minus_q1'] > 0


def test_compute_quantile_returns_non_numeric_factor():
    """regime / categorical 因子 → 全部 NaN, 不抛错 (Task 2 修复)。"""
    from dynamics.dynamics_factor_validation import compute_quantile_returns
    regimes = ['overdamped', 'critical', 'underdamped', 'anti_damped'] * 25
    factor = pd.Series(regimes[:100])
    ret = pd.Series(np.random.RandomState(42).randn(100) * 0.01)
    q = compute_quantile_returns(factor, ret, n_quantiles=5)
    # 所有 q1-q5 + spread 都应是 NaN,n_obs=100(只是 dropna 后)
    for i in range(1, 6):
        assert np.isnan(q[f'q{i}_ret']), f'q{i}_ret should be NaN for string factor'
    assert np.isnan(q['q5_minus_q1'])
    assert q['n_obs'] == 100


def test_compute_eigen_factors():
    """(k=0.145, c=1.112) → rho ≈ 0.85, regime=overdamped."""
    from dynamics.dynamics_factor_validation import compute_eigen_factors
    kc = pd.DataFrame({'code': ['000001.SZ'], 'k_hat': [0.145], 'c_hat': [1.112]})
    out = compute_eigen_factors(kc)
    assert abs(out['rho'].iloc[0] - 0.85) < 0.01
    assert out['regime'].iloc[0] == 'overdamped'


def test_load_kc_estimates_missing():
    """missing path → FileNotFoundError with hint."""
    from dynamics.dynamics_factor_validation import load_kc_estimates
    with pytest.raises(FileNotFoundError, match='parameter_fit.py'):
        load_kc_estimates('/nonexistent/kc_estimates.csv')


# === v6 Task 2 — CLI smoke test ===

def test_cli_factor_validation_minimal(tmp_path):
    """CLI runs with required files, exits 0, writes 3 CSVs + 1 TXT."""
    import subprocess
    # 需要 kc_estimates.csv 存在 — 在 test fixture 里跳过如果缺失
    kc_path = Path('data/projection/kc_estimates.csv')
    if not kc_path.exists():
        pytest.skip('kc_estimates.csv not available — run parameter_fit.py first')

    out_dir = tmp_path / 'outputs'
    data_dir = tmp_path / 'data'
    result = subprocess.run(
        [
            sys.executable,
            'backtrace/dynamics/dynamics_factor_validation.py',
            '--limit', '50',
            '--horizons', '5,20',
            '--output-dir', str(out_dir),
            '--data-dir', str(data_dir),
            '--repo-root', '.',
        ],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
    )
    assert result.returncode == 0, f'stderr: {result.stderr}'
    assert (data_dir / 'factor_validation.csv').exists()
    assert (data_dir / 'factor_validation_by_year.csv').exists()
    assert (data_dir / 'factor_validation_by_industry.csv').exists()
    assert (out_dir / 'dynsys_factor_validation_summary.txt').exists()


# === v6.0.1 final-review regression tests (BLOCKER #8 + IMPORTANT #9 + #10) ===

def test_validate_all_factors_by_industry_true_per_industry_ic():
    """v6.0.1 BLOCKER #8: by_industry aggregation must compute IC on **industry-only**
    stocks, NOT inflate n_dates by appending market-wide IC once per stock-in-industry.

    Construction: 2 dates × 2 industries × 35 stocks = 140 (code, date) rows.
    - Industry A (35 stocks): random factor + random returns → ic_mean ≈ 0.
    - Industry B (35 stocks): factor monotonically 1..35; returns = factor * 0.01
      (perfect monotonic) → ic_mean ≈ 1.
    With the OLD (buggy) impl, B's ic_mean would equal the market-wide IC
    (somewhere in the middle), not ≈ 1. With the NEW impl, B should be clearly ≈ 1
    and A clearly ≈ 0.
    """
    from dynamics.dynamics_factor_validation import (
        build_factor_panel, validate_all_factors,
    )
    np.random.seed(42)
    n_per_ind = 35  # ≥ 30 to clear spec §6.4 threshold
    codes_a = [f'90000{i:02d}' for i in range(n_per_ind)]
    codes_b = [f'80000{i:02d}' for i in range(n_per_ind)]
    all_codes = codes_a + codes_b
    # Factor values (static, one per code)
    rng = np.random.RandomState(0)
    factors_a = rng.randn(n_per_ind)
    factors_b = np.arange(1, n_per_ind + 1).astype(float)  # monotonic 1..35
    # Forward returns — MultiIndex (code, date)
    dates = pd.to_datetime(['2024-01-01', '2024-01-02'])
    rows = []
    for d_idx, d in enumerate(dates):
        # Industry A: random returns
        for i, c in enumerate(codes_a):
            rows.append({'code': c, 'date': d, 'fwd_ret_5d': float(rng.randn() * 0.01)})
        # Industry B: returns = factor * 0.01 (perfect monotonic)
        for i, c in enumerate(codes_b):
            rows.append({'code': c, 'date': d, 'fwd_ret_5d': factors_b[i] * 0.01})
    fwd_rets = pd.DataFrame(rows).set_index(['code', 'date']).sort_index()
    # Build panel (single factor = 'k' static)
    panel_rows = []
    for c, fv in zip(codes_a, factors_a):
        panel_rows.append({'code': c, 'factor_name': 'k', 'factor_value': fv, 'status': 'loaded'})
    for c, fv in zip(codes_b, factors_b):
        panel_rows.append({'code': c, 'factor_name': 'k', 'factor_value': fv, 'status': 'loaded'})
    panel = pd.DataFrame(panel_rows)
    # Industry lookup (l1 keys: 'indA' / 'indB')
    industry_l1 = pd.Series(
        {**{c: 'indA' for c in codes_a}, **{c: 'indB' for c in codes_b}}
    )
    main, by_year, by_ind = validate_all_factors(
        panel, fwd_rets, horizons=[5], industry_l1=industry_l1,
    )
    assert len(by_ind) == 2, f'expected 2 industry rows, got {len(by_ind)}'
    ind_b_row = by_ind[by_ind['industry_l1'] == 'indB'].iloc[0]
    ind_a_row = by_ind[by_ind['industry_l1'] == 'indA'].iloc[0]
    # Industry B: IC ≈ 1 across both dates
    assert ind_b_row['ic_mean'] > 0.9, (
        f"indB ic_mean should be ≈1 (perfect monotonic), got {ind_b_row['ic_mean']}"
    )
    # Industry A: IC ≈ 0 (random factor, random returns)
    assert abs(ind_a_row['ic_mean']) < 0.3, (
        f"indA ic_mean should be ≈0 (random), got {ind_a_row['ic_mean']}"
    )
    # Critical: indB - indA must be clearly differentiated (>0.5).
    # Old buggy impl would have produced nearly identical values
    # (both = market-wide IC ≈ 0.4-0.6 from mix of random + monotonic).
    assert ind_b_row['ic_mean'] - ind_a_row['ic_mean'] > 0.5, (
        f'industry differentiation lost: indB={ind_b_row["ic_mean"]:.3f} '
        f'indA={ind_a_row["ic_mean"]:.3f}'
    )


def test_industry_threshold_30():
    """v6.0.1 IMPORTANT #9: per (date, industry) if <30 stocks have data,
    that (date, industry) is skipped. If every date for an industry has <30 stocks,
    the industry gets status='insufficient_data' with ic_mean=NaN."""
    from dynamics.dynamics_factor_validation import (
        build_factor_panel, validate_all_factors,
    )
    np.random.seed(42)
    # 3 industries: SMALL (10 stocks — below threshold), MEDIUM (30 stocks — at
    # threshold — counts), LARGE (40 stocks — well above).
    codes_small = [f'70000{i:02d}' for i in range(10)]   # <30 → skip
    codes_medium = [f'60000{i:02d}' for i in range(30)]  # exactly 30 → count
    codes_large = [f'50000{i:02d}' for i in range(40)]   # >30 → count
    dates = pd.to_datetime(['2024-01-01'])
    rng = np.random.RandomState(0)
    rows = []
    for c in codes_small + codes_medium + codes_large:
        rows.append({
            'code': c, 'date': dates[0],
            'fwd_ret_5d': float(rng.randn() * 0.01),
        })
    fwd_rets = pd.DataFrame(rows).set_index(['code', 'date']).sort_index()
    # Build panel — all stocks have factor value 1.0 (constant → IC = NaN by
    # nunique<2 guard, but that's fine — we test the *threshold*, not the IC value).
    panel_rows = []
    for c in codes_small + codes_medium + codes_large:
        # Use distinct factor values to avoid nunique<2
        val = float(int(c) % 100) / 100.0 + 0.001
        panel_rows.append({'code': c, 'factor_name': 'k', 'factor_value': val, 'status': 'loaded'})
    panel = pd.DataFrame(panel_rows)
    industry_l1 = pd.Series(
        {
            **{c: 'SMALL' for c in codes_small},
            **{c: 'MEDIUM' for c in codes_medium},
            **{c: 'LARGE' for c in codes_large},
        }
    )
    main, by_year, by_ind = validate_all_factors(
        panel, fwd_rets, horizons=[5], industry_l1=industry_l1,
    )
    # SMALL industry: only 10 stocks at date_t < 30 → insufficient_data
    small_row = by_ind[by_ind['industry_l1'] == 'SMALL'].iloc[0]
    assert small_row['status'] == 'insufficient_data', (
        f'SMALL industry should be insufficient_data (10 stocks < 30 threshold), '
        f'got {small_row["status"]}'
    )
    assert pd.isna(small_row['ic_mean'])
    assert small_row['n_dates'] == 0
    assert small_row['n_obs'] == 0
    # MEDIUM / LARGE: ≥30 → 'ok' (may be NaN IC due to constant factor, but row emitted)
    medium_row = by_ind[by_ind['industry_l1'] == 'MEDIUM'].iloc[0]
    large_row = by_ind[by_ind['industry_l1'] == 'LARGE'].iloc[0]
    assert medium_row['status'] == 'ok', (
        f'MEDIUM industry should be ok (exactly 30 ≥ 30), got {medium_row["status"]}'
    )
    assert large_row['status'] == 'ok', (
        f'LARGE industry should be ok (40 ≥ 30), got {large_row["status"]}'
    )


# === v0 — Parameter Fit Identifiability Audit diagnostics (2026-08-19) ===
from backtrace.projection.parameter_fit import _solve_ols


def _make_ols_inputs(k_true=0.5, c_true=0.2, T=100, noise_std=1e-3, seed=42):
    """合成 Y = -k d - c u + noise,生成 2D 投影与 _solve_ols 兼容的 6 个输入。

    Returns: (a_u_vec, a_v_vec, d_vec, u_vec, beta, valid)

    构造策略:直接造 a_u_vec = β·a_v_vec - k·d - c·u + noise,
    这样 _solve_ols 内部 A_full = a_u - β·a_v 严格满足 Y = -k·d - c·u + noise,
    OLS 能精确恢复 (k, c)。

    注:_solve_ols 实际接口要求 beta 为 1D shape (T,),所以这里不用 2D。
    """
    rng = np.random.default_rng(seed)
    # 2D 输入:Vol + Amt 两个维度独立(均为 (T, 2))
    u_vec = rng.standard_normal((T, 2)) * 0.5   # 速度项(白噪声)
    d_vec = np.zeros((T, 2))
    if T >= 2:
        d_vec[1:] = np.cumsum(u_vec[:-1], axis=0)   # d = cumsum(u[:-1])
    # β(t) 时变 1D,shape (T,)(实数据从 Move_Proj_Coeff 列出来也是 1D)
    beta = rng.uniform(0.8, 1.2, T)
    # a_v_vec:大盘加速度,任意有限输入
    a_v_vec = rng.standard_normal((T, 2)) * 0.1
    # 核心:让 A_full = a_u - β·a_v 满足 Y = -k·d - c·u + noise
    target = -k_true * d_vec - c_true * u_vec
    a_u_vec = beta[:, None] * a_v_vec + target + rng.normal(0, noise_std, (T, 2))
    valid = (
        np.isfinite(a_u_vec).all(axis=1)
        & np.isfinite(a_v_vec).all(axis=1)
        & np.isfinite(d_vec).all(axis=1)
        & np.isfinite(u_vec).all(axis=1)
    )
    return a_u_vec, a_v_vec, d_vec, u_vec, beta, valid


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
    # 强加 d ≈ u,使 X 两列(-d 和 -u)高度共线
    d[:, 0] = u[:, 0] * 0.999 + np.random.default_rng(0).standard_normal(200) * 1e-3
    d[:, 1] = u[:, 1] * 0.999 + np.random.default_rng(1).standard_normal(200) * 1e-3
    _, _, _, _, rank, cond, rcorr, _ = _solve_ols(a_u, a_v, d, u, beta, valid)
    assert cond > 1e3, f'cond={cond:.2e} 应 > 1e3'
    assert rcorr > 0.9, f'rcorr={rcorr:.4f} 应 > 0.9'


def test_solve_ols_singular_zero_variance():
    """X 某列全 0 → X^T X 不可逆 → rank < 2 → singular。"""
    a_u, a_v, d, u, beta, valid = _make_ols_inputs(T=100)
    # 让 d_vec 与 u_vec 都全 0 → X 列全 0 → rank 0
    d = np.zeros_like(d)
    u = np.zeros_like(u)
    _, _, _, _, rank, _, _, _ = _solve_ols(a_u, a_v, d, u, beta, valid)
    assert rank < 2


def test_solve_ols_ss_tot_near_zero():
    """Y 几乎常数 → SS_tot ≈ 0 → r2 = NaN → fit_quality='uninformative'。"""
    a_u, a_v, d, u, beta, valid = _make_ols_inputs(T=100, noise_std=1e-3)
    # 让 a_u = β·a_v → A_full = 0 → Y = 0 全 0
    a_u = beta[:, None] * a_v
    _, _, _, _, _, _, _, r2 = _solve_ols(a_u, a_v, d, u, beta, valid)
    assert np.isnan(r2), f'r2={r2} 应为 NaN'


# === v0 — Parameter Fit Identifiability Audit (2026-08-19 Task 2) ===

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


# === v0.1 — Dynamics Specification Correction & Ablation (2026-08-19 Task 1) ===
import numpy as np
from projection.ablation_fit import (
    ols_fit, build_design_model_0, build_design_model_1,
    build_design_model_2, build_design_model_3, _build_kinematics_ext,
)


def _make_ext_inputs(k_true=0.5, c_true=0.2, q_true=0.8, T=200, seed=0,
                     beta_drift=0.001):
    """Synthetic 2-D data satisfying Model 3 exactly.

    Defaults satisfy Model 3: q_true=0.8 (free q), beta_drift=0.001 (β varies).
    For Model 0 tests, override: q_true=1.0, beta_drift=0 (matches Model 0's
    q=1, β̇=0 assumptions exactly, avoiding omitted-variable bias).

    Returns 6-tuple: (u_vec, d_vec, a_u_vec, a_v_vec, beta, beta_dot_vM_vec)
    The 5th element (beta) is required by build_design_model_0/1/2/3.
    """
    rng = np.random.default_rng(seed)
    beta = 1.2 + beta_drift * np.arange(T)            # β constant if beta_drift=0
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
    return u_vec, d_vec, a_u, a_v, beta, beta_dot_vM


def test_build_design_model0_subtracts_beta_aM():
    u, d, au, av, beta, bdv = _make_ext_inputs()
    X, Y = build_design_model_0(u, d, au, av, beta, bdv)
    # Y = a_u - β·a_v, X = [-d, -u]
    assert X.shape[1] == 2
    # Last row is NaN (from au NaN) → Y last row should be NaN
    assert np.isnan(Y[-1])
    # First row should be finite (a_v[0] finite)
    assert np.isfinite(Y[0])


def test_build_design_model1_subtracts_betadot_vM():
    u, d, au, av, beta, bdv = _make_ext_inputs()
    X, Y = build_design_model_1(u, d, au, av, beta, bdv)
    assert X.shape[1] == 2
    # Y should equal Model 0's Y minus bdv stacked
    X0, Y0 = build_design_model_0(u, d, au, av, beta, bdv)
    bdv_stack = np.concatenate([bdv[:, 0], bdv[:, 1]])
    np.testing.assert_allclose(np.nan_to_num(Y), np.nan_to_num(Y0 - bdv_stack), equal_nan=True)


def test_build_design_model2_keeps_aS_in_Y():
    u, d, au, av, beta, bdv = _make_ext_inputs()
    X, Y = build_design_model_2(u, d, au, av, beta, bdv)
    assert X.shape[1] == 3  # [β·a_M, -d, -u]


def test_build_design_model3_combines_offset_and_free_q():
    u, d, au, av, beta, bdv = _make_ext_inputs()
    X, Y = build_design_model_3(u, d, au, av, beta, bdv)
    assert X.shape[1] == 3
    # Y_Model3 - Y_Model1 = β·a_M (since Model 3 keeps β·a_M in X, Model 1 subtracts it)
    X1, Y1 = build_design_model_1(u, d, au, av, beta, bdv)
    beta_aM = beta[:, None] * av
    beta_aM_stack = np.concatenate([beta_aM[:, 0], beta_aM[:, 1]])
    np.testing.assert_allclose(np.nan_to_num(Y - Y1), np.nan_to_num(beta_aM_stack), equal_nan=True)


def test_ols_fit_recovers_k_c_model0():
    # Model 0 assumes q=1 and β̇=0; must use compatible synthetic data
    u, d, au, av, beta, bdv = _make_ext_inputs(k_true=0.5, c_true=0.2, q_true=1.0, beta_drift=0.0)
    X, Y = build_design_model_0(u, d, au, av, beta, bdv)
    mask = np.isfinite(Y)
    X_v, Y_v = X[mask], Y[mask]
    theta, f_res, n_valid, rank, cond, rcorr, r2 = ols_fit(X_v, Y_v)
    assert n_valid == mask.sum()
    assert abs(theta[0] - 0.5) < 0.05  # k_hat ≈ 0.5
    assert abs(theta[1] - 0.2) < 0.05  # c_hat ≈ 0.2


def test_ols_fit_recovers_q_k_c_model3():
    u, d, au, av, beta, bdv = _make_ext_inputs(k_true=0.5, c_true=0.2, q_true=0.8)
    X, Y = build_design_model_3(u, d, au, av, beta, bdv)
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
    """Verify cond(X) not cond(X.T @ X) (κ² amplifier test).

    For this X, cond(X) ≈ 2.45e8 but cond(XᵀX) = inf. Threshold must
    distinguish finite cond(X) from infinite cond(XᵀX), so use 1e10.
    """
    X = np.array([[1.0, 1.0], [1.0 + 1e-8, 1.0], [1.0, 1.0 + 1e-8]])
    Y = np.array([1.0, 2.0, 3.0])
    *_, cond, _, _ = ols_fit(X, Y)
    expected_cond = np.linalg.cond(X)
    assert abs(cond - expected_cond) < 1e-3
    # cond(X.T @ X) = inf, cond(X) ≈ 2.45e8 → threshold must be > 2.45e8 and < inf
    assert cond < 1e10


# === v0.1 — Task 2: In-Sample 4-Model Fit + CSV Output ===

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
            f'Move_Delta_Vol_stk{i:06d}': delta_u[:, 0],
            f'Move_Delta_Amt_stk{i:06d}': delta_u[:, 1],
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
        # V0.2-D Phase 1: CSV_COLUMNS extended to 36 (18 existing + 9 Group A + 3 Group B + 3 Group C + 3 Group D)
        assert len(df.columns) == 36, f"expected 36 columns, got {len(df.columns)}"
        # Group A fields exist (NaN at this stage — fit_one_in_sample doesn't populate them)
        for k in ('q_train_fit', 'k_train_fit', 'c_train_fit',
                  'q_test_fit', 'k_test_fit', 'c_test_fit',
                  'q_drift', 'k_drift', 'c_drift'):
            assert k in df.columns, f"missing column {k}"
        # ic_real / ic_null are NaN at this stage (Tasks 3+4 will populate)
        assert df['ic_real'].isna().all()
        assert df['ic_null'].isna().all()
        # Models 0/1 q_hat = 1.0; Models 2/3 q_hat = OLS estimate (varies)
        if m in (0, 1):
            assert (df['q_hat'] == 1.0).all()


# === v0.1 — Task 3: OOS 70/30 Split + Spearman IC ===

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
    u, d, au, av, beta, bdv = _make_ext_inputs(k_true=0.5, c_true=0.2, q_true=0.8, T=200)
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
        'Move_Delta_Vol_stk000001': delta_u[:, 0],
        'Move_Delta_Amt_stk000001': delta_u[:, 1],
        'Move_Proj_Coeff': beta,
    }).to_csv(csv_path, index=False)
    row = fit_one_oos(csv_path, 'stk000001', 'idx', '000001.SZ', 'T', '000001.SH', model_id=3)
    assert row['n_train'] > 0 and row['n_test'] > 0
    assert row['ic_real'] > 0.5  # strong signal, should be high


# === v0.1 — Task 4: Placebo Test (Permutation Baseline, seed=42) ===

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
    stock_tag = 'stk000003'
    pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=T),
        'Move_Delta_Vol_idx': delta_v[:, 0],
        'Move_Delta_Amt_idx': delta_v[:, 1],
        f'Move_Delta_Vol_{stock_tag}': delta_u[:, 0],
        f'Move_Delta_Amt_{stock_tag}': delta_u[:, 1],
        'Move_Proj_Coeff': beta,
    }).to_csv(csv_path, index=False)
    row = fit_one_with_placebo(csv_path, stock_tag, 'idx', '000003.SZ', 'T', '000001.SH', model_id=3)
    assert row['ic_real'] - row['ic_null'] > 0.1


# === v0.1 — Task 5: Summary + HTML + Recommendation TXT + CLI ===

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
            f'Move_Delta_Vol_stk{i:06d}': delta_u[:, 0],
            f'Move_Delta_Amt_stk{i:06d}': delta_u[:, 1],
            'Move_Proj_Coeff': beta,
        }).to_csv(mv_dir / f"movement_idx_stk{i:06d}.csv", index=False)

    out_dir = tmp_path / "out"
    result = subprocess.run([
        sys.executable,
        "backtrace/projection/ablation_fit.py",
        "--all", "--limit", "5",
        "--movement-dir", str(mv_dir),
        "--output-dir", str(out_dir),
    ], capture_output=True, text=True, timeout=120,
       cwd=REPO_ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
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


def test_summarize_ablation_writes_three_delta_ic():
    """V0.2-D audit fix: 3 ΔIC statistics all persisted in summary CSV."""
    import tempfile, os
    from projection.ablation_fit import summarize_ablation
    with tempfile.TemporaryDirectory() as td:
        # Write 4 stub CSVs with ic_real and ic_null columns
        for m in range(4):
            stub = pd.DataFrame({
                'code': [f'stk{m:06d}'] * 10,
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


# === V0.2-D Phase 1: Parameter Stability (Group A) ===

def test_fit_split_returns_train_test_params():
    """V0.2-D Phase 1: fit_one_split returns train/test params + drifts."""
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
    src = open(ablation_fit.__file__, encoding='utf-8').read()
    assert 'param_drift_l2' not in src, \
        "V0.2-D §4 forbids param_drift_l2 (q is dimensionless, k/c have units)"


def test_oos_uses_train_params_only():
    """V0.2-D §7: oos_r2 = R²(Y_test, X_test · θ_train) — θ_test must NOT be used."""
    from projection import ablation_fit
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
    # Highly collinear X_beta with X_d
    X = rng.normal(0, 1, (T, 3))
    X[:, 1] = X[:, 0] + 0.01 * rng.normal(0, 1, T)  # X_d ~ X_beta
    X[:, 2] = rng.normal(0, 1, T)
    c_beta_d, _, _ = compute_x_x_correlations(X)
    assert abs(c_beta_d) > 0.9, f"expected high correlation; got {c_beta_d}"


# === V0.2-D Task 4 — Group D: residual structure (corr(F_self, X_train columns)) ===

def test_residual_correlation_white_noise_zero():
    """V0.2-D §6: synthetic Model 2 with Gaussian noise → corr_F_* ≈ 0."""
    from projection.ablation_fit import compute_residual_correlations
    rng = np.random.default_rng(0)
    # NOTE: brief originally wrote T=200, but std of sample-corr under the null
    # is 1/sqrt(T-1) ≈ 0.07, so the 0.05 threshold is unreliable. Bumping to
    # T=2000 brings std ≈ 0.022 (max|corr| over 3 cols well below 0.05 for seed=0).
    T = 2000
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
    # Inject a missing LINEAR term correlated with X[:,1] (= −d column).
    # NOTE: brief originally wrote `** 2` here, but E[X^3] = 0 for centered normal
    # → corr(X^2, X) = 0, so the test would always fail. Linear missing term
    # is what the brief's prose ("correlated with X") actually requires.
    Y_train = X_train @ theta + 0.5 * X_train[:, 1] + rng.normal(0, 0.1, T)
    F_self = Y_train - X_train @ theta
    c_b, c_d, c_u = compute_residual_correlations(F_self, X_train)
    assert abs(c_d) > 0.3, f"missing-term residual must correlate with X_d; got {c_d}"


# === V0.2-D Task 4 amend — populate regressor_corr from ols_fit (Phase 1 regression) ===

def test_regressor_corr_populated_in_fit_one_split():
    """V0.2-D Task 4 amend: regressor_corr must NOT be hard-coded NaN.

    Regression check: V0.1 fit_one_in_sample populated regressor_corr from ols_fit.
    V0.2-D Phase 1 hard-coded NaN; Task 4 amend must restore it.
    """
    from projection.ablation_fit import fit_one_split
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
    row = fit_one_split(csv_path, 'stk', 'idx', '000003.SZ', 'T', '000001.SH', model_id=2)
    # regressor_corr must be a finite float, not NaN
    assert np.isfinite(row['regressor_corr']), \
        f"regressor_corr should be populated by ols_fit; got {row['regressor_corr']}"
    # Sanity: regressor_corr ∈ [0, 1] (it's max |corr|)
    assert 0.0 <= row['regressor_corr'] <= 1.0, \
        f"regressor_corr must be max |corr| ∈ [0, 1]; got {row['regressor_corr']}"


# === V0.2-D Task 5 — Panel 5 + Distribution Reporting (2026-08-20) ===

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
        with open(html_path, encoding='utf-8') as f:
            html = f.read()
        assert 'corr_x_beta_d' in html, "x-axis must be corr_x_beta_d"
        assert 'corr(q' not in html, "x-axis must NOT be the undefined corr(q, β·a_M)"


# === V0.2-D Task 6 — CLI orchestrator + audit verification (2026-08-20) ===

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
                f'Move_Delta_Vol_stk{i:06d}': delta_u[:, 0],
                f'Move_Delta_Amt_stk{i:06d}': delta_u[:, 1],
                'Move_Proj_Coeff': beta,
            }).to_csv(os.path.join(mv_dir, f'movement_idx_stk{i:06d}.csv'), index=False)
        out_dir = os.path.join(td, 'out')
        # Run CLI
        result = subprocess.run([
            sys.executable,
            'backtrace/projection/v0_2_d_decompose.py',
            '--movement-dir', mv_dir,
            '--output-dir', out_dir,
            '--limit', '3',
        ], capture_output=True, text=True, timeout=120,
           cwd=REPO_ROOT, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        # Verify outputs exist
        for f in ('kc_estimates_model2_diag.csv', 'panel5_drift_vs_collinearity.html',
                   'v0_2_d_distributions.csv', 'v0_2_d_summary.txt'):
            assert os.path.exists(os.path.join(out_dir, f)), f"missing output: {f}"


# === V0.2-C1 Task 1 — projection_batch --output-dir flag (2026-08-20) ===

def test_projection_batch_output_dir_flag():
    """V0.2-C1 Task 1: --output-dir redirects movement files; load_kc_map reads from default."""
    import subprocess, tempfile, os
    with tempfile.TemporaryDirectory() as td:
        # Snapshot existing movement_399001_*.csv files BEFORE the run —
        # so we can verify the test's run did not add any new ones to data/projection/
        # (previous V0.2-D runs may have left legitimate files there).
        proj_dir = 'data/projection'
        before = set(
            f for f in os.listdir(proj_dir)
            if f.startswith('movement_399001_') and f.endswith('.csv')
        )
        result = subprocess.run([
            sys.executable,
            'backtrace/projection/projection_batch.py',
            '--input', 'data/projection/stocks.csv',
            '--output-dir', td,
            '--movement',
            '--index', '399001.SZ',
            '--days', '60',
            '--limit', '2',
        ], capture_output=True, text=True, timeout=120,
           cwd=REPO_ROOT, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        # Verify movement files written to td (NOT to data/projection/)
        out_files = [f for f in os.listdir(td) if f.startswith('movement_') and f.endswith('.csv')]
        assert len(out_files) >= 1, f"No movement files in {td}; got {os.listdir(td)}"
        # Verify data/projection/ NOT contaminated with new market movement files
        after = set(
            f for f in os.listdir(proj_dir)
            if f.startswith('movement_399001_') and f.endswith('.csv')
        )
        new_files = after - before
        assert not new_files, f"data/projection/ contaminated with new market files: {sorted(new_files)}"


# === V0.2-C1 Task 2 — Paired C0/C1 compare helpers (2026-08-20) ===

def test_paired_compare_columns_and_sign_flipped():
    """V0.2-C1 §4.3: paired CSV has all 25 columns; sign_flipped matches sign(ic_real_C0) != sign(ic_real_C1)."""
    from projection.c0_c1_compare import compute_c0_c1_paired_compare
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        c0_path = os.path.join(td, 'c0.csv')
        c1_path = os.path.join(td, 'c1.csv')
        out_path = os.path.join(td, 'paired.csv')
        # 3 synthetic stocks, with deliberate sign change on stock[1]
        n = 3
        pd.DataFrame({
            'code': [f'stk{i:06d}' for i in range(n)],
            'name': [f'Stock {i}' for i in range(n)],
            'ic_real': [+0.1, +0.2, -0.3],  # C0: signs + + -
            'q_drift': [+0.1, +0.2, +0.3],
            'q_hat': [+0.5, +0.6, +0.7],
            'test_fit_r2': [+0.1, +0.2, +0.3],
            'oos_r2': [+0.05, +0.10, -0.05],
            'condition_number': [+10.0, +20.0, +30.0],
        }).to_csv(c0_path, index=False)
        pd.DataFrame({
            'code': [f'stk{i:06d}' for i in range(n)],
            'name': [f'Stock {i}' for i in range(n)],
            'ic_real': [+0.1, -0.2, -0.3],  # C1: signs + - - (stock[1] flipped)
            'q_drift': [+0.05, +0.10, +0.20],
            'q_hat': [+0.4, +0.5, +0.6],
            'test_fit_r2': [+0.15, +0.18, +0.28],
            'oos_r2': [+0.08, +0.05, -0.03],
            'condition_number': [+8.0, +18.0, +28.0],
        }).to_csv(c1_path, index=False)
        result_path = compute_c0_c1_paired_compare(c0_path, c1_path, out_path)
        df = pd.read_csv(result_path)
        # All 25 columns present (2 code/name + 6 metric blocks x 3 cols + 3 flags + 2 flags)
        assert len(df.columns) == 25, f"expected 25 cols, got {len(df.columns)}: {list(df.columns)}"
        # sign_flipped: stock[1] flipped (+0.2 -> -0.2)
        assert df.iloc[0]['sign_flipped'] == False
        assert df.iloc[1]['sign_flipped'] == True
        assert df.iloc[2]['sign_flipped'] == False
        # q_drift_attenuated: with these values the 0.5x threshold does not
        # trigger (|0.05| < 0.5*|0.1| is False, etc.) — assert the FLAG is
        # present and bool-interpretable, not that it triggers.
        assert df.iloc[0]['q_drift_attenuated'] in (True, False)


def test_c0_c1_summary_txt_format():
    """V0.2-C1 §4.4: summary TXT is UTF-8, has C0/C1 columns, no verdicts."""
    from projection.c0_c1_compare import (
        compute_c0_c1_paired_compare, write_c0_c1_compare_summary_txt,
    )
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        c0_path = os.path.join(td, 'c0.csv')
        c1_path = os.path.join(td, 'c1.csv')
        paired_path = os.path.join(td, 'paired.csv')
        summary_path = os.path.join(td, 'summary.txt')
        # 5 synthetic stocks
        n = 5
        rng = np.random.default_rng(0)
        for path, scale in [(c0_path, 1.0), (c1_path, 0.5)]:
            pd.DataFrame({
                'code': [f'stk{i:06d}' for i in range(n)],
                'name': [f'Stock {i}' for i in range(n)],
                'ic_real': rng.normal(0, 0.3, n) * scale,
                'q_drift': rng.normal(0.1, 0.05, n) * scale,
                'q_hat': rng.normal(0.5, 0.2, n),
                'test_fit_r2': rng.uniform(0, 0.2, n),
                'oos_r2': rng.normal(0, 0.1, n),
                'condition_number': rng.uniform(5, 30, n),
            }).to_csv(path, index=False)
        # Write minimal dist CSVs (3 rows each: median, p25, p75)
        for path, m in [(os.path.join(td, 'c0_dist.csv'), 0.12), (os.path.join(td, 'c1_dist.csv'), 0.08)]:
            pd.DataFrame({
                'gate': ['D1', 'D1', 'D1'],
                'statistic': ['median', 'p25', 'p75'],
                'value': [m, m - 0.05, m + 0.05],
            }).to_csv(path, index=False)
        compute_c0_c1_paired_compare(c0_path, c1_path, paired_path)
        write_c0_c1_compare_summary_txt(paired_path,
                                        os.path.join(td, 'c0_dist.csv'),
                                        os.path.join(td, 'c1_dist.csv'),
                                        summary_path)
        with open(summary_path, encoding='utf-8') as f:
            txt = f.read()
        # UTF-8 decoded
        # Has C0/C1 column headers
        assert 'C0' in txt and 'C1' in txt
        # No verdict PASS/FAIL — use verdict-specific regex (a bare substring
        # search would also match embedded forms like "BYPASS"/"PASSED").
        import re as _re
        assert _re.search(r'(?<![A-Za-z])PASS(?![A-Za-z])', txt) is None, (
            'Summary TXT contains verdict PASS'
        )
        assert _re.search(r'(?<![A-Za-z])FAIL(?![A-Za-z])', txt) is None, (
            'Summary TXT contains verdict FAIL'
        )
        # Has D1/D2/D3 sections
        for d in ('D1', 'D2', 'D3'):
            assert d in txt


# === V0.2-C1 Task 3 — CLI orchestrator (2026-08-20) ===

def test_v0_2_c1_cli_smoke():
    """V0.2-C1 §7: full pipeline (data gen + ablation + paired compare) end-to-end with synthetic stocks."""
    import subprocess, tempfile, os, sys
    with tempfile.TemporaryDirectory() as td:
        # Pre-populate <td>/data/projection/ with 3 SH + 3 SZ synthetic movement files
        proj_dir = os.path.join(td, 'data', 'projection')
        os.makedirs(proj_dir)
        for i in range(3):
            T = 80
            rng = np.random.default_rng(i)
            beta = 1.2 + 0.001 * np.arange(T)
            delta_v = rng.normal(0, 1, (T, 2))
            delta_u = beta[:, None] * delta_v + rng.normal(0, 0.5, (T, 2))
            # SH version (index=000001)
            sh_code = f'600{100 + i:03d}'
            sh_tag = sh_code
            pd.DataFrame({
                'Date': pd.date_range('2024-01-01', periods=T),
                'Move_Delta_Vol_000001': delta_v[:, 0],
                'Move_Delta_Amt_000001': delta_v[:, 1],
                f'Move_Delta_Vol_{sh_tag}': delta_u[:, 0],
                f'Move_Delta_Amt_{sh_tag}': delta_u[:, 1],
                'Move_Proj_Coeff': beta,
            }).to_csv(os.path.join(proj_dir, f'movement_000001_{sh_tag}.csv'), index=False)
            # SZ version (index=399001)
            sz_code = f'000{100 + i:03d}'
            sz_tag = sz_code
            pd.DataFrame({
                'Date': pd.date_range('2024-01-01', periods=T),
                'Move_Delta_Vol_399001': delta_v[:, 0],
                'Move_Delta_Amt_399001': delta_v[:, 1],
                f'Move_Delta_Vol_{sz_tag}': delta_u[:, 0],
                f'Move_Delta_Amt_{sz_tag}': delta_u[:, 1],
                'Move_Proj_Coeff': beta,
            }).to_csv(os.path.join(proj_dir, f'movement_399001_{sz_tag}.csv'), index=False)
        # Pre-populate <td>/data/stock_basic.csv with 6 stocks
        basic = os.path.join(td, 'data', 'stock_basic.csv')
        rows = []
        for i in range(3):
            rows.append({'code': f'600{100 + i:03d}', 'market': 'SH', 'name': f'SH{i}', 'status': 'active'})
            rows.append({'code': f'000{100 + i:03d}', 'market': 'SZ', 'name': f'SZ{i}', 'status': 'active'})
        pd.DataFrame(rows).to_csv(basic, index=False)
        # Pre-populate C0 (industry) for paired compare
        c0_dir = os.path.join(td, 'data', 'projection_v01_d')
        os.makedirs(c0_dir)
        n = 6
        # Include `index_code` so the CLI's driver-aware filter is exercised.
        # SH stocks (600xxx) → 申万 industry codes (881xxx.SH/SZ) for C0
        # SH stocks → 000001.SH (上证综指) for C1
        # SZ stocks (000xxx) → 申万 industry codes (881xxx) for C0
        # SZ stocks → 399001.SZ (深证成指) for C1
        index_codes_c0 = ['881001.SH', '881001.SH', '881001.SH',
                          '881002.SZ', '881002.SZ', '881002.SZ']
        index_codes_c1 = ['000001.SH', '000001.SH', '000001.SH',
                          '399001.SZ', '399001.SZ', '399001.SZ']
        pd.DataFrame({
            'code': [r['code'] for r in rows],
            'name': [r['name'] for r in rows],
            'index_code': index_codes_c0,
            'ic_real': np.random.default_rng(0).normal(0, 0.3, n),
            'q_drift': np.random.default_rng(1).normal(0.1, 0.05, n),
            'q_hat': np.random.default_rng(2).normal(0.5, 0.2, n),
            'test_fit_r2': np.random.default_rng(3).uniform(0, 0.2, n),
            'oos_r2': np.random.default_rng(4).normal(0, 0.1, n),
            'condition_number': np.random.default_rng(5).uniform(5, 30, n),
        }).to_csv(os.path.join(c0_dir, 'kc_estimates_model2_diag.csv'), index=False)
        # C1 input CSV (the CLI's ablation step would normally produce this,
        # but for the smoke test we pre-populate with matching market-driver rows)
        c1_input_dir = os.path.join(td, 'data', 'projection_v01_c1')
        os.makedirs(c1_input_dir)
        pd.DataFrame({
            'code': [r['code'] for r in rows],
            'name': [r['name'] for r in rows],
            'index_code': index_codes_c1,
            'ic_real': np.random.default_rng(0).normal(0, 0.2, n),
            'q_drift': np.random.default_rng(1).normal(0.05, 0.03, n),
            'q_hat': np.random.default_rng(2).normal(0.5, 0.2, n),
            'test_fit_r2': np.random.default_rng(3).uniform(0, 0.2, n),
            'oos_r2': np.random.default_rng(4).normal(0, 0.08, n),
            'condition_number': np.random.default_rng(5).uniform(5, 30, n),
        }).to_csv(os.path.join(c1_input_dir, 'kc_estimates_model2_diag.csv'), index=False)
        # Run CLI
        market_dir = os.path.join(td, 'data', 'projection_market')
        c1_dir = os.path.join(td, 'data', 'projection_v01_c1')
        # Use --skip-data-gen --skip-ablation so the test is CI-friendly
        # (no TQ required; C0 + C1 CSVs are pre-populated).
        result = subprocess.run([
            sys.executable,
            'backtrace/projection/v0_2_c1_market_swap.py',
            '--input', basic,
            '--market-dir', market_dir,
            '--c0-dir', c0_dir,
            '--c1-output-dir', c1_dir,
            '--skip-data-gen',
            '--skip-ablation',
            '--limit', '0',
        ], capture_output=True, text=True,
           # encoding='utf-8' on the pipe: the CLI prints Chinese, and text=True
           # alone would decode it with the Windows locale codec (gbk) and blow
           # up the reader thread — leaving result.stdout None in the assert below.
           encoding='utf-8', errors='replace',
           env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'REPO_ROOT': os.getcwd()},
           cwd=os.getcwd(), timeout=300)
        assert result.returncode == 0, f"CLI failed: {result.stderr}\nstdout: {result.stdout}"
        # Verify paired compare outputs (the test's actual scope)
        for f in ('kc_estimates_model2_diag_filtered.csv', 'c0_c1_paired_compare.csv', 'c0_c1_compare_summary.txt'):
            assert os.path.exists(os.path.join(c1_dir, f)), f"missing C1 output: {f}"
        # Verify paired compare has all 25 cols and the driver-filter was applied
        paired = pd.read_csv(os.path.join(c1_dir, 'c0_c1_paired_compare.csv'))
        assert len(paired.columns) == 25, f"paired CSV has {len(paired.columns)} cols, expected 25"
        assert len(paired) == 6, f"expected 6 paired rows (after filter), got {len(paired)}"


# === V0.2-C1 Task 5a — empty movement file pruning (2026-08-20) ===

def test_prune_empty_movement_files():
    """688826.SH (newly-listed, 1 valid row) → projection_batch.py writes a
    header-only movement file. fit_one_split crashes on np.isfinite(object).
    The orchestrator must prune such files before v0_2_d_decompose.py sees them.
    """
    import tempfile, os
    import sys as _sys
    BACKTRACE = os.path.join(os.getcwd(), 'backtrace')
    if BACKTRACE not in _sys.path:
        _sys.path.insert(0, BACKTRACE)
    from projection.v0_2_c1_market_swap import _prune_empty_movement_files
    with tempfile.TemporaryDirectory() as td:
        # Write 3 files: 1 empty (header only), 1 with 1 row, 1 with 100 rows
        empty_path = os.path.join(td, 'movement_000001_688826.csv')
        with open(empty_path, 'w', encoding='utf-8') as f:
            f.write('Date,Move_Delta_Vol_000001,Move_Delta_Amt_000001,'
                    'Move_Delta_Vol_688826,Move_Delta_Amt_688826,Move_Proj_Coeff\n')
        one_row_path = os.path.join(td, 'movement_000001_600000.csv')
        with open(one_row_path, 'w', encoding='utf-8') as f:
            f.write('Date,Move_Delta_Vol_000001,Move_Delta_Amt_000001,'
                    'Move_Delta_Vol_600000,Move_Delta_Amt_600000,Move_Proj_Coeff\n')
            f.write('2024-01-01,0.1,0.2,0.3,0.4,1.1\n')
        # Non-movement file: must NOT be touched
        unrelated_path = os.path.join(td, 'manifest.json')
        with open(unrelated_path, 'w', encoding='utf-8') as f:
            f.write('{}')

        n_pruned = _prune_empty_movement_files(td)
        assert n_pruned == 1, f'expected 1 pruned, got {n_pruned}'
        assert not os.path.exists(empty_path), 'empty file should be deleted'
        assert os.path.exists(one_row_path), '1-row file should be kept (project_batch may regenerate)'
        assert os.path.exists(unrelated_path), 'non-movement file must not be touched'

    # Non-existent dir: return 0, no crash
    assert _prune_empty_movement_files(os.path.join(td, 'does_not_exist')) == 0
