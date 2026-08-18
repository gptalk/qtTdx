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
