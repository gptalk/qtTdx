# -*- coding: utf-8 -*-
# dynamics_eigen_analysis.py — 批量 (k̂, ĉ) → 特征值 + 稳定性分类 + HTML 报告(2026-08-17 v4 Plan)
#
# 目标:
#   把 parameter_fit.py 估出的 (k̂, ĉ) 喂给 analyze_eigenvalues,得到每只票的:
#     - 特征值 λ₁, λ₂
#     - 谱半径 ρ(A) = max(|λ|)
#     - 8 类稳定性分类(stable_oscillatory / oscillatory_divergent / ...)
#     - Schur 稳定性(楔形内 vs 楔形外)
#   画图:
#     1. (k, c) 散点 + 楔形稳定区叠加
#     2. ρ 分布直方图(看经验分布是 <1 还是 >1)
#     3. 8 类分类饼图
#     4. summary 表(均值/中位数/占比)
#
# 用法:
#   PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_eigen_analysis.py
#   PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_eigen_analysis.py --limit 500
#   PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_eigen_analysis.py --input data/projection/kc_estimates.csv
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
from collections import Counter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 让 from dynamics import ... 能找到 _dynamics_core
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
from dynamics import analyze_eigenvalues

CSV_OUT_DIR = 'data/dynamics'
AGG_INDUSTRY_CSV = os.path.join(CSV_OUT_DIR, 'v43_eigen_top_industries.csv')
AGG_EXCHANGE_CSV = os.path.join(CSV_OUT_DIR, 'v43_eigen_by_exchange.csv')
DEFAULT_INPUT = 'data/projection/kc_estimates.csv'
DEFAULT_OUTPUT_HTML = 'backtrace/outputs/dynsys_eigen.html'
DEFAULT_STOCK_BASIC = 'data/stock_basic.csv'
DEFAULT_SW2_MEMBERS = 'data/sw2/members.csv'
DEFAULT_TXT_OUTPUT = 'backtrace/outputs/dynsys_eigen_summary.txt'

# 8 类标签(分类配色)
CLASS_COLORS = {
    'stable_oscillatory':       '#2ca02c',   # 绿 — 稳定振荡
    'stable_overdamped':        '#1f77b4',   # 蓝 — 稳定过阻尼
    'stable_critical_damping':  '#9467bd',   # 紫 — 临界阻尼
    'oscillatory_divergent':    '#d62728',   # 红 — 振荡发散(共振本质)
    'monotonic_divergent':      '#ff7f0e',   # 橙 — 单调发散
    'anti_restoring':           '#8c564b',   # 棕 — 反回复(趋势强化)
    'critical_periodic':        '#bcbd22',   # 黄绿 — 周期振荡边界
    'critical_period2':         '#17becf',   # 青 — λ=-1 边界
    'critical_real_unit':       '#e377c2',   # 粉 — 实根单位圆
    'marginal_const':           '#7f7f7f',   # 灰 — k=0 有界常数模
    'jordan_drift':             '#000000',   # 黑 — Jordan 漂移
}
CLASS_LABEL_CN = {
    'stable_oscillatory':       '稳定振荡',
    'stable_overdamped':        '稳定过阻尼',
    'stable_critical_damping':  '临界阻尼稳定',
    'oscillatory_divergent':    '振荡发散',
    'monotonic_divergent':      '单调发散',
    'anti_restoring':           '反回复',
    'critical_periodic':        '临界周期振荡',
    'critical_period2':         '临界 λ=-1',
    'critical_real_unit':       '临界实根单位圆',
    'marginal_const':           '边界常数模',
    'jordan_drift':             'Jordan 漂移',
}


def parse_args():
    p = argparse.ArgumentParser(description='(k̂, ĉ) → 特征值 + 11 类稳定性分类 + HTML 报告 (v4.3)')
    p.add_argument('--input', default=DEFAULT_INPUT,
                   help=f'kc_estimates CSV。默认 {DEFAULT_INPUT}')
    p.add_argument('--output', default=DEFAULT_OUTPUT_HTML,
                   help=f'HTML 输出路径。默认 {DEFAULT_OUTPUT_HTML}')
    p.add_argument('--limit', type=int, default=0, help='最多处理多少只;0=全部')
    p.add_argument('--status-filter', default='ok', help='只分析 status 以此前缀开头的行;默认 "ok"')
    p.add_argument('--stock-basic', default='data/stock_basic.csv',
                   help='stock_basic CSV 路径(反查 exchange);默认 data/stock_basic.csv')
    p.add_argument('--sw2-members', default='data/sw2/members.csv',
                   help='sw2/members CSV 路径(反查 industry_l1/l2);默认 data/sw2/members.csv')
    p.add_argument('--phase-plot', action='store_true', help='画 (k,c) 11 类 phase plot 到独立 HTML(默认 off)')
    return p.parse_args()


def load_kc_estimates(
    path: str, status_filter: str = 'ok', limit: int = 0,
    stock_basic_path: str = 'data/stock_basic.csv',
    sw2_members_path: str = 'data/sw2/members.csv',
) -> pd.DataFrame:
    """读 kc_estimates.csv,反查 stock_basic(exchange)+ sw2/members(industry_l1/l2)。"""
    df = pd.read_csv(path, dtype={'code': str})
    if status_filter:
        df = df[df['status'].astype(str).str.startswith(status_filter)].copy()
    if limit and len(df) > limit:
        df = df.head(limit).copy()
    # merge 前 drop 同名列(避免 _x / _y 后缀污染)
    for col in ['industry_l1', 'industry_l2', 'exchange']:
        if col in df.columns:
            df = df.drop(columns=[col])
    ex_lookup = load_exchange_lookup(stock_basic_path)
    df = df.merge(ex_lookup, on='code', how='left')
    ind_lookup = load_industry_lookup(sw2_members_path)
    df = df.merge(ind_lookup, on='code', how='left')
    for col in ['industry_l1', 'industry_l2', 'exchange']:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('').astype(str)
    return df


def load_exchange_lookup(path: str = 'data/stock_basic.csv') -> pd.DataFrame:
    """读 stock_basic.csv,返回 code → exchange 反查表。

    stock_basic 列: code, market, name, status。`market` 即交易所(SH/SZ/BJ)。
    缺文件 / 缺列 → 返回空表,eigen_analysis 不致命(行业列留空)。
    """
    if not os.path.exists(path):
        print(f'[eigen] ⚠ stock_basic 不存在: {path},exchange 列将留空')
        return pd.DataFrame(columns=['code', 'exchange'])
    df = pd.read_csv(path, dtype={'code': str})
    if 'market' not in df.columns:
        print(f'[eigen] ⚠ stock_basic 缺 market 列: {path},exchange 列将留空')
        return pd.DataFrame(columns=['code', 'exchange'])
    df['exchange'] = df['market'].fillna('').astype(str).str.strip()
    df.loc[df['exchange'].isin(['-', 'nan', 'None']), 'exchange'] = ''
    return df[['code', 'exchange']]


def load_industry_lookup(path: str = 'data/sw2/members.csv') -> pd.DataFrame:
    """读 sw2/members.csv,返回 code → {industry_l1, industry_l2} 反查表。

    sw2/members 列: sector_code, sector_name, member_code。
    - `sector_code`(881xxx.SH) → industry_l1
    - `sector_name`(中文,例如"银行") → industry_l2
    - `member_code` → code(join key)

    注意:同 code 可能属于多个 industry_l1(同一只票同时是"银行"和"金融")。
    默认取首条(`.drop_duplicates('code', keep='first')`)。
    """
    if not os.path.exists(path):
        print(f'[eigen] ⚠ sw2/members 不存在: {path},industry 列将留空')
        return pd.DataFrame(columns=['code', 'industry_l1', 'industry_l2'])
    df = pd.read_csv(path, dtype={'member_code': str})
    if 'member_code' not in df.columns or 'sector_name' not in df.columns:
        print(f'[eigen] ⚠ sw2/members 缺关键列: {path}')
        return pd.DataFrame(columns=['code', 'industry_l1', 'industry_l2'])
    df['industry_l1'] = df['sector_code'].fillna('').astype(str).str.strip() if 'sector_code' in df.columns else ''
    df['industry_l2'] = df['sector_name'].fillna('').astype(str).str.strip()
    df['code'] = df['member_code']
    df = df[['code', 'industry_l1', 'industry_l2']].copy()
    df = df.drop_duplicates('code', keep='first')
    for col in ['industry_l1', 'industry_l2']:
        df.loc[df[col].isin(['-', 'nan', 'None']), col] = ''
    return df


def aggregate_by_industry(
    df: pd.DataFrame, min_stocks: int = 50, fallback_min: int = 30,
) -> tuple[pd.DataFrame, int]:
    """按 industry_l1 聚合 ρ 中位数。

    Returns:
        agg_df: top 10(降序),列: industry_l1, n_stocks, rho_median, rho_p25, rho_p75,
               k_hat_median, c_hat_median, schur_stable_pct, in_wedge_pct, dist_wedge_median
        threshold_used: 实际生效的 n_stocks 阈值(50 或 30;若都 <5 则返回 fallback_min)
    """
    for thr in (min_stocks, fallback_min):
        agg = df.groupby('industry_l1').agg(
            n_stocks=('code', 'count'),
            rho_median=('spectral_radius', 'median'),
            rho_p25=('spectral_radius', lambda s: s.quantile(0.25)),
            rho_p75=('spectral_radius', lambda s: s.quantile(0.75)),
            k_hat_median=('k_hat', 'median'),
            c_hat_median=('c_hat', 'median'),
            schur_stable_pct=('schur_stable', 'mean'),
            in_wedge_pct=('in_wedge', 'mean'),
            dist_wedge_median=('distance_to_wedge', 'median'),
        ).reset_index()
        agg = agg[agg['n_stocks'] >= thr].sort_values('rho_median', ascending=False).head(10)
        if len(agg) >= 5:
            return agg, thr
    return agg, fallback_min   # 都凑不够 5,返回最后尝试的结果


def aggregate_by_exchange(df: pd.DataFrame) -> pd.DataFrame:
    """按 exchange 聚合(SH / SZ / BJ),列同行业聚合(无 n_stocks >= 50 阈值)。"""
    agg = df.groupby('exchange').agg(
        n_stocks=('code', 'count'),
        rho_median=('spectral_radius', 'median'),
        rho_p25=('spectral_radius', lambda s: s.quantile(0.25)),
        rho_p75=('spectral_radius', lambda s: s.quantile(0.75)),
        k_hat_median=('k_hat', 'median'),
        c_hat_median=('c_hat', 'median'),
        schur_stable_pct=('schur_stable', 'mean'),
        in_wedge_pct=('in_wedge', 'mean'),
        dist_wedge_median=('distance_to_wedge', 'median'),
    ).reset_index().sort_values('rho_median')
    return agg


def _industry_name_lookup(sw2_members_path: str = 'data/sw2/members.csv') -> dict:
    """sector_code → sector_name 反查表。

    文件不存在 / 缺关键列 → 返回空 dict(让 caller 走 fallback)。
    """
    if not os.path.exists(sw2_members_path):
        print(f'[eigen] ⚠ sw2_members 不存在: {sw2_members_path},行业 label 走 fallback')
        return {}
    df = pd.read_csv(sw2_members_path, dtype={'sector_code': str})
    if 'sector_code' not in df.columns or 'sector_name' not in df.columns:
        print(f'[eigen] ⚠ sw2_members 缺关键列: {sw2_members_path}')
        return {}
    return df.drop_duplicates('sector_code').set_index('sector_code')['sector_name'].to_dict()


def write_text_summary(
    summary_df: pd.DataFrame,
    cls_count: Counter,
    agg_l1: pd.DataFrame,
    l1_threshold: int,
    agg_ex: pd.DataFrame,
    path: str,
    sw2_members_path: str = 'data/sw2/members.csv',
    name_lookup: dict | None = None,
) -> None:
    """写 dynsys_eigen_summary.txt 纯文本汇总(UTF-8)。

    行业 label 增强:agg_l1.groupby key 是 industry_l1(sector_code),
    这里再读 sw2/members.csv 把 sector_name 拼过来,显示更可读。
    sw2_members_path 默认即仓库内 data/sw2/members.csv。
    name_lookup 若 caller 已构造好(测试隔离 / main() 复用)则用之,否则 fallback 到 helper。
    """
    import datetime as _dt
    N = len(summary_df)
    rho = summary_df['spectral_radius']
    k_hat = summary_df['k_hat']
    c_hat = summary_df['c_hat']
    schur_n = int(summary_df['schur_stable'].sum())
    wedge_n = int(summary_df['in_wedge'].sum())
    rho_gt1_n = int((rho > 1.0 + 1e-8).sum())
    dist = summary_df['distance_to_wedge']
    timestamp = _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 行业 label 增强:industry_l1(sector_code) → industry_l2(sector_name)
    # 优先用 caller 传入的 lookup(避免重复读 csv),否则 fallback 到 helper
    if name_lookup is None:
        name_lookup = _industry_name_lookup(sw2_members_path)
    agg_l1_label = agg_l1.copy()
    agg_l1_label['industry_label'] = agg_l1_label['industry_l1'].map(
        lambda c: f'{name_lookup.get(c, c)}({c})' if c else '(未知)'
    )

    lines = []
    lines.append('=== v4.3 全市场 (k̂, ĉ) 经验分布报告 ===')
    lines.append(f'样本数: N = {N}')
    lines.append('数据来源: data/projection/kc_estimates.csv')
    lines.append(f'报告时间: {timestamp}')
    lines.append('')
    lines.append('--- 全市场 ---')
    lines.append(f'ρ 中位数: {rho.median():.4f} | p25: {rho.quantile(0.25):.4f} | p75: {rho.quantile(0.75):.4f}')
    lines.append(f'k̂ 中位数: {k_hat.median():.4f} | p25: {k_hat.quantile(0.25):.4f} | p75: {k_hat.quantile(0.75):.4f}')
    lines.append(f'ĉ 中位数: {c_hat.median():.4f} | p25: {c_hat.quantile(0.25):.4f} | p75: {c_hat.quantile(0.75):.4f}')
    lines.append(f'Schur 稳定(ρ<1):   {schur_n}/{N} ({schur_n/N*100:.1f}%)')
    lines.append(f'楔形内:            {wedge_n}/{N} ({wedge_n/N*100:.1f}%)')
    lines.append(f'ρ > 1(发散):       {rho_gt1_n}/{N} ({rho_gt1_n/N*100:.1f}%)')
    lines.append(f'distance_to_wedge 中位数: {dist.median():+.4f} (>0 在楔形内)')
    lines.append('')
    lines.append('--- 11 类分布 ---')
    for cls, cnt in sorted(cls_count.items(), key=lambda x: -x[1]):
        lines.append(f'  {cls:<28} {cnt:>5} ({cnt/N*100:>5.1f}%)')
    lines.append('')
    lines.append(f'--- 行业 ρ 中位数 top10 (n_stocks >= {l1_threshold}) ---')
    if len(agg_l1_label) >= 5:
        for _, r in agg_l1_label.iterrows():
            lines.append(
                f'  {r["industry_label"]:<26} n={int(r["n_stocks"]):>4}, '
                f'ρ_med={r["rho_median"]:.3f}, p25={r["rho_p25"]:.3f}, p75={r["rho_p75"]:.3f}, '
                f'k̂_med={r["k_hat_median"]:.3f}, ĉ_med={r["c_hat_median"]:.3f}, '
                f'楔形内%={r["in_wedge_pct"]*100:.1f}%'
            )
    else:
        lines.append(f'  (行业不足 5 个,n_stocks >= {l1_threshold} 仅 {len(agg_l1_label)} 个)')
    lines.append('')
    lines.append('--- 交易所 ---')
    for _, r in agg_ex.iterrows():
        lines.append(
            f'  {r["exchange"]:<5} n={int(r["n_stocks"]):>4}, '
            f'ρ_med={r["rho_median"]:.3f}, p25={r["rho_p25"]:.3f}, p75={r["rho_p75"]:.3f}, '
            f'楔形内%={r["in_wedge_pct"]*100:.1f}%'
        )
    lines.append('')

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'[eigen] ✓ text summary: {path}')


def wedge_boundary_polygon(k_max: float = 4.0, n: int = 100) -> dict:
    """楔形稳定区边界 3 段折线。

    Schur 稳定区定义: c² ≤ 4(k+1) AND c ≥ 0 AND k ≥ 0
    边界曲线:
      - k 轴: c = 0, k ∈ [0, k_max]
      - c 轴: k = 0, c ∈ [0, 2]
      - 上抛物线: c = 2√(k+1), k ∈ [0, k_max]

    Returns:
        dict with keys: 'k_axis', 'c_axis', 'upper_curve', 'k_max'
        每段都是 list[(k, c)] 长度 n。
    """
    k_axis = [(k, 0.0) for k in np.linspace(0, k_max, n)]
    c_axis = [(0.0, c) for c in np.linspace(0, 2.0, n)]
    upper_curve = [(k, 2.0 * np.sqrt(k + 1.0)) for k in np.linspace(0, k_max, n)]
    return {'k_axis': k_axis, 'c_axis': c_axis, 'upper_curve': upper_curve, 'k_max': k_max}


def build_phase_plot_html(summary_df: pd.DataFrame, output_path: str) -> None:
    """画 (k̂, ĉ) 散点 + 11 类颜色 + 楔形稳定区边界 overlay。

    独立 HTML,不动 v4.3 2x4 输出。被 main() 通过 --phase-plot flag 调用。
    """
    import json as _json

    fig = go.Figure()

    # 楔形稳定区填充(浅绿背景)
    k_max = summary_df['k_hat'].quantile(0.99)
    boundary = wedge_boundary_polygon(k_max=k_max)
    fill_k = [k for k, c in boundary['upper_curve']] + [k for k, c in boundary['k_axis']][::-1]
    fill_c = [c for k, c in boundary['upper_curve']] + [c for k, c in boundary['k_axis']][::-1]
    fig.add_trace(go.Scatter(
        x=fill_k, y=fill_c, fill='toself', fillcolor='rgba(44, 160, 44, 0.08)',
        line=dict(color='rgba(0,0,0,0)'), name='楔形稳定区', showlegend=True, hoverinfo='skip',
    ))

    # 楔形边界 3 段虚线
    for label, pts in [('c=0', boundary['k_axis']),
                        ('k=0', boundary['c_axis']),
                        ('c=2√(k+1)', boundary['upper_curve'])]:
        fig.add_trace(go.Scatter(
            x=[k for k, c in pts], y=[c for k, c in pts],
            mode='lines', line=dict(color='black', width=1.5, dash='dash'),
            name=label, showlegend=False, hoverinfo='skip',
        ))

    # 11 类散点(每类 1 trace,图例 1 entry)
    for cls in CLASS_COLORS:
        sub = summary_df[summary_df['classification'] == cls]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub['k_hat'], y=sub['c_hat'],
            mode='markers',
            marker=dict(color=CLASS_COLORS[cls], size=6, opacity=0.7, line=dict(width=0)),
            name=f'{CLASS_LABEL_CN[cls]} ({len(sub)})',
            hovertemplate=f'<b>{cls}</b><br>k̂=%{{x:.4f}}<br>ĉ=%{{y:.4f}}<extra></extra>',
            showlegend=True,
        ))

    fig.update_layout(
        title='全市场 (k̂, ĉ) 11 类稳定性分类 phase plot',
        xaxis_title='k̂ (回复力强度)',
        yaxis_title='ĉ (阻尼系数)',
        width=1100, height=750,
        legend=dict(title='11 类分类', x=1.02, y=1, bgcolor='rgba(255,255,255,0.9)'),
        template='plotly_white',
    )

    # plotly 默认 to_html 会把中文转 \uXXXX,这里手动拼 HTML 保留原始字符
    fig_dict = fig.to_dict()
    data_json = _json.dumps(fig_dict.get('data', []), ensure_ascii=False)
    layout_json = _json.dumps(fig_dict.get('layout', {}), ensure_ascii=False)
    config_json = _json.dumps({'responsive': True}, ensure_ascii=False)
    html = (
        '<!DOCTYPE html>\n<html><head><meta charset="utf-8" />\n'
        '<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: \'local\'};</script>\n'
        '<script charset="utf-8" src="https://cdn.plot.ly/plotly-3.1.0.min.js"></script>\n'
        '</head><body>\n'
        '<div id="phase-plot-div" class="plotly-graph-div" style="height:750px; width:1100px;"></div>\n'
        '<script type="text/javascript">\n'
        'window.PLOTLYENV = window.PLOTLYENV || {};\n'
        'if (document.getElementById("phase-plot-div")) {\n'
        f'    Plotly.newPlot("phase-plot-div", {data_json}, {layout_json}, {config_json});\n'
        '};\n</script>\n</body></html>'
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        print(f'[eigen] ✗ 输入文件不存在: {args.input}')
        print('  请先跑 backtrace/projection/parameter_fit.py 生成 kc_estimates.csv')
        sys.exit(1)

    df = load_kc_estimates(
        args.input, args.status_filter, args.limit,
        stock_basic_path=args.stock_basic,
        sw2_members_path=args.sw2_members,
    )
    if len(df) == 0:
        print(f'[eigen] ✗ 过滤后 0 行(status_filter={args.status_filter!r})')
        sys.exit(1)

    print(f'[eigen] 输入: {args.input} ({len(df)} 行,status 前缀 {args.status_filter!r})')
    print(f'[eigen] 行业来源: {args.sw2_members} | 交易所来源: {args.stock_basic} — v4.3')

    # ---------- 1. 每行算 analyze_eigenvalues ----------
    rows = []
    for _, row in df.iterrows():
        k, c = float(row['k_hat']), float(row['c_hat'])
        eig = analyze_eigenvalues(k, c)
        rows.append({
            'code': row['code'],
            'name': row.get('name', ''),
            'index_tag': row.get('index_tag', ''),
            'stock_tag': row.get('stock_tag', ''),
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
            # v4.2:楔形距离
            'distance_lower_boundary': eig['distance_lower_boundary'],
            'distance_upper_boundary': eig['distance_upper_boundary'],
            'distance_to_wedge': eig['distance_to_wedge'],
            # v4.3:行业 + 交易所(via stock_basic + sw2/members)
            'industry_l1': row.get('industry_l1', ''),
            'industry_l2': row.get('industry_l2', ''),
            'exchange': row.get('exchange', ''),
        })
    summary_df = pd.DataFrame(rows)
    out_csv = os.path.join(CSV_OUT_DIR, 'eigen_summary.csv')
    os.makedirs(CSV_OUT_DIR, exist_ok=True)
    summary_df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f'[eigen] ✓ eigen_summary: {out_csv}({len(summary_df)} 行)')

    # ---------- 2. 分类统计 ----------
    cls_count = Counter(summary_df['classification'])
    total = len(summary_df)
    print()
    print('=== 11 类稳定性分类分布(v4.1:ρ-primary) ===')
    print(f'{"分类":<28} {"标签":<14} {"数量":>5} {"占比":>7}')
    for cls, count in sorted(cls_count.items(), key=lambda x: -x[1]):
        label = CLASS_LABEL_CN.get(cls, cls)
        print(f'{cls:<28} {label:<14} {count:>5} {count / total * 100:>6.1f}%')

    schur_n = summary_df['schur_stable'].sum()
    wedge_n = summary_df['in_wedge'].sum()
    rho_gt1_n = (summary_df['spectral_radius'] > 1.0 + 1e-8).sum()
    wedge_close_n = (np.abs(summary_df['distance_to_wedge']) < 0.1).sum()
    print()
    print(f'Schur 稳定(ρ<1):  {schur_n}/{total}({schur_n / total * 100:.1f}%)')
    print(f'在楔形内:         {wedge_n}/{total}({wedge_n / total * 100:.1f}%)')
    print(f'ρ > 1(发散):      {rho_gt1_n}/{total}({rho_gt1_n / total * 100:.1f}%)')
    print(f'距楔形边界 < 0.1: {wedge_close_n}/{total}({wedge_close_n / total * 100:.1f}%)')
    rho_median = float(summary_df['spectral_radius'].median())
    rho_mean = float(summary_df['spectral_radius'].mean())
    dist_median = float(summary_df['distance_to_wedge'].median())
    dist_mean = float(summary_df['distance_to_wedge'].mean())
    print(f'ρ 中位数: {rho_median:.4f} | 均值: {rho_mean:.4f}')
    print(f'楔形距离 中位数: {dist_median:+.4f} | 均值: {dist_mean:+.4f}(>0 在楔形内,<0 在外)')

    # ---------- 3. 画 HTML(plotly) ----------
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=(
            '(k̂, ĉ) 散点 + 楔形(颜色=分类)',
            'ρ 分布直方图',
            '11 类分类分布',
            '行业 ρ 中位数 top10',                       # ← 新
            '(k̂, ĉ) 散点(颜色=楔形距离)',
            '楔形距离分布',
            'ρ vs 楔形距离',
            '交易所 ρ 中位数(SH vs SZ)',                  # ← 新
        ),
        specs=[[{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'bar'},    {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'scatter'}, {'type': 'bar'}]],
        horizontal_spacing=0.06,                          # ← 0.08 改 0.06(4 列更挤)
        vertical_spacing=0.18,
    )

    # 楔形背景(填充):c ∈ (k, 2 + k/2),k ∈ (0, 4)
    k_grid = np.linspace(0.001, 4.0, 200)
    c_lower = k_grid
    c_upper = 2.0 + k_grid / 2.0
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([k_grid, k_grid[::-1]]),
            y=np.concatenate([c_lower, c_upper[::-1]]),
            fill='toself',
            fillcolor='rgba(44, 160, 44, 0.15)',
            line=dict(color='rgba(44, 160, 44, 0.4)', width=1),
            name='Schur 楔形',
            hoverinfo='skip',
            showlegend=True,
        ),
        row=1, col=1,
    )
    # 楔形上下边界
    fig.add_trace(
        go.Scatter(x=k_grid, y=c_lower, mode='lines',
                   line=dict(color='green', dash='dash', width=1),
                   name='c=k', hoverinfo='skip'),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=k_grid, y=c_upper, mode='lines',
                   line=dict(color='green', dash='dash', width=1),
                   name='c=2+k/2', hoverinfo='skip'),
        row=1, col=1,
    )

    # (k, c) 散点(按分类上色)
    for cls in sorted(set(summary_df['classification'])):
        sub = summary_df[summary_df['classification'] == cls]
        fig.add_trace(
            go.Scatter(
                x=sub['k_hat'], y=sub['c_hat'],
                mode='markers',
                marker=dict(
                    color=CLASS_COLORS.get(cls, '#888888'),
                    size=8,
                    line=dict(color='white', width=0.5),
                ),
                name=f'{cls} ({len(sub)})',
                text=sub['code'] + ' / ' + sub['classification'],
                hovertemplate='<b>%{text}</b><br>k̂=%{x:.4f}<br>ĉ=%{y:.4f}<extra></extra>',
                showlegend=True,
            ),
            row=1, col=1,
        )
    fig.update_xaxes(title_text='k̂ (恢复系数)', row=1, col=1)
    fig.update_yaxes(title_text='ĉ (阻尼系数)', row=1, col=1)

    # ρ 直方图
    fig.add_trace(
        go.Histogram(
            x=summary_df['spectral_radius'],
            nbinsx=40,
            marker=dict(color='steelblue', line=dict(color='white', width=1)),
            name='ρ 分布',
            hovertemplate='ρ 区间: %{x}<br>数量: %{y}<extra></extra>',
        ),
        row=1, col=2,
    )
    # ρ=1 边界线
    fig.add_vline(x=1.0, line_dash='dash', line_color='red', row=1, col=2,
                  annotation_text='ρ=1 临界', annotation_position='top right')
    fig.update_xaxes(title_text='ρ(A) = max(|λ|)', row=1, col=2)
    fig.update_yaxes(title_text='数量', row=1, col=2)

    # 11 类分类柱状图(v4.1:ρ-primary)
    cls_sorted = sorted(cls_count.items(), key=lambda x: -x[1])
    cls_labels = [CLASS_LABEL_CN.get(c, c) for c, _ in cls_sorted]
    cls_counts = [n for _, n in cls_sorted]
    cls_colors = [CLASS_COLORS.get(c, '#888888') for c, _ in cls_sorted]
    fig.add_trace(
        go.Bar(
            x=cls_labels, y=cls_counts,
            marker_color=cls_colors,
            name='11 类分布',
            text=[f'{c / total * 100:.1f}%' for c in cls_counts],
            textposition='outside',
            hovertemplate='%{x}<br>%{y} 只 (%{text})<extra></extra>',
            showlegend=False,
        ),
        row=1, col=3,
    )
    fig.update_xaxes(title_text='分类', row=1, col=3, tickangle=-30)
    fig.update_yaxes(title_text='数量', row=1, col=3)

    # (1,4) 行业 ρ 中位数 top10(误差棒 p25-p75)
    agg_l1, l1_threshold = aggregate_by_industry(summary_df)
    # 行业 label 增强(sector_code → sector_name),(1,4) bar chart + 文本汇总共用
    name_lookup = _industry_name_lookup(args.sw2_members)
    agg_l1_label = agg_l1.copy()
    agg_l1_label['industry_label'] = agg_l1_label['industry_l1'].map(
        lambda c: f'{name_lookup.get(c, c)}({c})' if c else '(未知)'
    )
    if len(agg_l1) >= 5:
        fig.add_trace(
            go.Bar(
                x=agg_l1_label['industry_label'],
                y=agg_l1['rho_median'],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=agg_l1['rho_p75'] - agg_l1['rho_median'],
                    arrayminus=agg_l1['rho_median'] - agg_l1['rho_p25'],
                    color='black',
                    thickness=1.5,
                    width=4,
                ),
                marker_color='steelblue',
                name=f'行业 top10 (n≥{l1_threshold})',
                text=[f"n={n}<br>ρ={r:.2f}" for n, r in zip(agg_l1['n_stocks'], agg_l1['rho_median'])],
                hovertemplate='<b>%{x}</b><br>ρ 中位数: %{y:.3f}<br>%{text}<extra></extra>',
                showlegend=False,
            ),
            row=1, col=4,
        )
    else:
        fig.add_annotation(
            text=f'行业不足(n<{l1_threshold},仅 {len(agg_l1)} 个)',
            xref='x4 domain', yref='y4 domain', x=0.5, y=0.5,
            showarrow=False, font=dict(size=12, color='gray'),
        )
    fig.update_xaxes(title_text=f'行业 (n≥{l1_threshold})', row=1, col=4, tickangle=-30)
    fig.update_yaxes(title_text='ρ 中位数', row=1, col=4)

    # (2,1) — (k̂, ĉ) 散点按楔形距离上色(连续 colormap)
    fig.add_trace(
        go.Scatter(
            x=summary_df['k_hat'], y=summary_df['c_hat'],
            mode='markers',
            marker=dict(
                color=summary_df['distance_to_wedge'],
                colorscale='RdYlGn',
                cmin=-2.0, cmax=2.0,
                size=8,
                colorbar=dict(title='楔形距离', x=1.02, len=0.5, y=0.25, yanchor='middle'),
                line=dict(color='white', width=0.5),
            ),
            text=summary_df['code'] + ' / dist=' + summary_df['distance_to_wedge'].round(3).astype(str),
            hovertemplate='<b>%{text}</b><br>k̂=%{x:.4f}<br>ĉ=%{y:.4f}<extra></extra>',
            showlegend=False,
        ),
        row=2, col=1,
    )
    # 楔形上下边界也加到 (2,1)
    fig.add_trace(
        go.Scatter(x=k_grid, y=c_lower, mode='lines',
                   line=dict(color='black', dash='dash', width=1),
                   name='c=k', hoverinfo='skip', showlegend=False),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=k_grid, y=c_upper, mode='lines',
                   line=dict(color='black', dash='dash', width=1),
                   name='c=2+k/2', hoverinfo='skip', showlegend=False),
        row=2, col=1,
    )
    fig.update_xaxes(title_text='k̂', row=2, col=1)
    fig.update_yaxes(title_text='ĉ', row=2, col=1)

    # (2,2) — 楔形距离分布直方图
    fig.add_trace(
        go.Histogram(
            x=summary_df['distance_to_wedge'],
            nbinsx=40,
            marker=dict(color='steelblue', line=dict(color='white', width=1)),
            name='楔形距离分布',
            hovertemplate='距离区间: %{x}<br>数量: %{y}<extra></extra>',
        ),
        row=2, col=2,
    )
    fig.add_vline(x=0.0, line_dash='dash', line_color='red', row=2, col=2,
                  annotation_text='楔形边界', annotation_position='top right')
    fig.update_xaxes(title_text='distance_to_wedge (>0 楔形内)', row=2, col=2)
    fig.update_yaxes(title_text='数量', row=2, col=2)

    # (2,3) — ρ vs 楔形距离(整体稳定性视角)
    fig.add_trace(
        go.Scatter(
            x=summary_df['distance_to_wedge'], y=summary_df['spectral_radius'],
            mode='markers',
            marker=dict(
                color=summary_df['distance_to_wedge'],
                colorscale='RdYlGn',
                cmin=-2.0, cmax=2.0,
                size=6,
                line=dict(color='white', width=0.3),
            ),
            text=summary_df['code'] + ' / ' + summary_df['classification'],
            hovertemplate='<b>%{text}</b><br>dist=%{x:.3f}<br>ρ=%{y:.3f}<extra></extra>',
            showlegend=False,
        ),
        row=2, col=3,
    )
    fig.add_hline(y=1.0, line_dash='dash', line_color='red', row=2, col=3)
    fig.add_vline(x=0.0, line_dash='dash', line_color='red', row=2, col=3)
    fig.update_xaxes(title_text='distance_to_wedge', row=2, col=3)
    fig.update_yaxes(title_text='ρ(A)', row=2, col=3)

    # (2,4) 交易所 ρ 中位数 SH vs SZ vs BJ
    agg_ex = aggregate_by_exchange(summary_df)
    ex_colors = {'SH': '#1f77b4', 'SZ': '#ff7f0e', 'BJ': '#2ca02c'}
    fig.add_trace(
        go.Bar(
            x=agg_ex['exchange'],
            y=agg_ex['rho_median'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=agg_ex['rho_p75'] - agg_ex['rho_median'],
                arrayminus=agg_ex['rho_median'] - agg_ex['rho_p25'],
                color='black',
                thickness=1.5,
                width=8,
            ),
            marker_color=[ex_colors.get(e, '#888888') for e in agg_ex['exchange']],
            name='交易所 ρ 中位数',
            text=[f"n={n}<br>ρ={r:.2f}" for n, r in zip(agg_ex['n_stocks'], agg_ex['rho_median'])],
            hovertemplate='<b>%{x}</b><br>ρ 中位数: %{y:.3f}<br>%{text}<extra></extra>',
            showlegend=False,
        ),
        row=2, col=4,
    )
    fig.update_xaxes(title_text='交易所', row=2, col=4)
    fig.update_yaxes(title_text='ρ 中位数', row=2, col=4)

    fig.update_layout(
        height=1000,        # ← 950 改 1000
        width=1800,         # ← 1500 改 1800
        title_text=f'动力系统特征值分析 v4.3 ({total} 只,Schur 稳定 {schur_n}/{total},楔形内 {wedge_n}/{total})',  # ← v4.1 改 v4.3
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='v', yanchor='top', y=1.0, xanchor='left', x=1.32),  # ← 1.30 改 1.32(4 列 legend 位置)
    )
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.write_html(args.output)
    print()
    print(f'[eigen] ✓ HTML: {args.output}')

    # ---------- 4. 聚合表 + 文本汇总 ----------
    os.makedirs(CSV_OUT_DIR, exist_ok=True)
    agg_l1.to_csv(AGG_INDUSTRY_CSV, index=False, encoding='utf-8')
    print(f'[eigen] ✓ industry agg: {AGG_INDUSTRY_CSV} ({len(agg_l1)} 行)')
    agg_ex.to_csv(AGG_EXCHANGE_CSV, index=False, encoding='utf-8')
    print(f'[eigen] ✓ exchange agg: {AGG_EXCHANGE_CSV} ({len(agg_ex)} 行)')

    # 文本汇总(便于 grep / CI)
    write_text_summary(
        summary_df, cls_count, agg_l1, l1_threshold, agg_ex, DEFAULT_TXT_OUTPUT,
        sw2_members_path=args.sw2_members, name_lookup=name_lookup,
    )

    # ---------- 5. (可选) (k,c) phase plot ----------
    if args.phase_plot:
        # 用 stem 派生,避免 --output 自定义路径时 replace no-op 覆盖 2x4 HTML
        from pathlib import Path
        output_p = Path(args.output)
        phase_path = str(output_p.with_name(output_p.stem + '_phase' + output_p.suffix))
        build_phase_plot_html(summary_df, phase_path)
        print(f'[eigen] ✓ phase plot: {phase_path}')


if __name__ == '__main__':
    main()
