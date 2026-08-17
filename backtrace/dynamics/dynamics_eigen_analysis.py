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
        rows=2, cols=3,
        subplot_titles=(
            '(k̂, ĉ) 散点 + 楔形(颜色=分类)',
            'ρ 分布直方图',
            '11 类分类分布',
            '(k̂, ĉ) 散点(颜色=楔形距离)',
            '楔形距离分布',
            'ρ vs 楔形距离',
        ),
        specs=[[{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'scatter'}]],
        horizontal_spacing=0.08,
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

    fig.update_layout(
        height=950, width=1500,
        title_text=f'动力系统特征值分析 v4.1 ({total} 只,Schur 稳定 {schur_n}/{total},楔形内 {wedge_n}/{total})',
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='v', yanchor='top', y=1.0, xanchor='left', x=1.30),
    )
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.write_html(args.output)
    print()
    print(f'[eigen] ✓ HTML: {args.output}')


if __name__ == '__main__':
    main()