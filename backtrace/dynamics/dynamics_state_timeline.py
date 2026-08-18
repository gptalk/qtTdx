# -*- coding: utf-8 -*-
"""v5.8: State Timeline + Force Decomposition HTML (plotly).

闭环 _projection_core.py 3 个高级函数 → 业务可读可视化:
- compute_dynamics() → 9 指标
- compute_forces() → 4 力分解
- classify_states() → 7 状态

Top 子图: 7 状态颜色时间线 (1 行/industry)
Bottom 子图: 4 力 stacked area (F_market/F_restore/F_damp/F_self)

业务读法: 哪个行业哪天共振/加速偏离, 哪个力在主导。
"""
import os
import sys
import argparse
import warnings

warnings.filterwarnings('ignore')

BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common import tsfresh_pipeline as P
from backtrace.projection._projection_core import (
    load_pair,
    compute_movement_projection,
    compute_dynamics,
    compute_forces,
    classify_states,
    STATE_COLORS,
    STATE_LABELS,
)


# 7 状态 ordinal mapping (y-axis 坐标)
STATE_Y = {label: i for i, label in enumerate(STATE_LABELS)}


def load_state_force_timeseries(
    stock_code: str,
    days: int,
    pipeline,
    prefer_industry: bool = True,
    lambda_q: float | None = None,
    k_restore: float = 0.0,
    c_damp: float = 0.0,
) -> dict:
    """load_pair → compute_movement_projection → compute_dynamics →
    compute_forces → classify_states 一步到位。

    Returns:
        dict with keys: stock_df, index_df, common_idx, index_code, index_name,
                       mv, dyn, frc, states.
    """
    pair = load_pair(stock_code, days, pipeline, prefer_industry=prefer_industry)
    stock_df = pair['stock_df']
    index_df = pair['index_df']
    common_idx = pair['common_idx']

    mv = compute_movement_projection(stock_df, index_df)
    dyn = compute_dynamics(mv, lambda_q=lambda_q)
    frc = compute_forces(dyn, mv, k_restore=k_restore, c_damp=c_damp)

    # 4 thresholds 默认 (R_low=0.10, R_high=0.50, theta_following=30°, theta_against=90°)
    thresholds = (0.10, 0.50, np.deg2rad(30), np.deg2rad(90))
    states = classify_states(dyn['R'], dyn['theta'], dyn['E_self'], thresholds)

    return {
        'stock_df': stock_df,
        'index_df': index_df,
        'common_idx': common_idx[1:],  # 与 mv/dyn 长度对齐 (丢首行 diff)
        'index_code': pair['index_code'],
        'index_name': pair['index_name'],
        'mv': mv,
        'dyn': dyn,
        'frc': frc,
        'states': states,
    }


def build_state_timeline_html(
    series_per_industry: list,
    output_path: str,
    title: str = 'Industry State Timeline + Force Decomposition',
) -> None:
    """Render N industries' state timeline + 4-force stacked area as 2-row plotly HTML."""
    if not series_per_industry:
        raise ValueError('series_per_industry 为空,无法构建 state timeline')

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.6],
        vertical_spacing=0.08,
        subplot_titles=('State Timeline (7 categories)', 'Force Decomposition (4 forces stacked)'),
    )

    # ---- Top: state timeline ----
    for s in series_per_industry:
        industry_code = s['industry_code']
        common_idx = s['common_idx']
        states = s['states']
        dyn = s['dyn']

        y_vals = [STATE_Y[st] for st in states]
        colors = [STATE_COLORS[st] for st in states]

        fig.add_trace(
            go.Scatter(
                x=common_idx,
                y=y_vals,
                mode='markers+lines',
                marker=dict(size=8, color=colors),
                line=dict(color='lightgray', width=1),
                name=industry_code,
                customdata=np.column_stack([
                    states,
                    dyn['q_t'],
                    dyn['R'],
                    np.degrees(dyn['theta']),
                    dyn['E_self'],
                ]),
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    'Industry: ' + industry_code + '<br>'
                    'State: %{customdata[0]}<br>'
                    'q_t: %{customdata[1]:.3f}<br>'
                    'R: %{customdata[2]:.3f}<br>'
                    'θ: %{customdata[3]:.1f}°<br>'
                    'E_self: %{customdata[4]:.2e}'
                ),
            ),
            row=1, col=1,
        )

    fig.update_yaxes(
        tickmode='array',
        tickvals=list(range(7)),
        ticktext=STATE_LABELS,
        row=1, col=1,
    )

    # ---- Bottom: 4 forces stacked area ----
    force_colors = {
        'F_market':  '#1f77b4',  # 蓝
        'F_restore': '#2ca02c',  # 绿
        'F_damp':    '#ff7f0e',  # 橙
        'F_self':    '#d62728',  # 红
    }
    for s in series_per_industry:
        industry_code = s['industry_code']
        common_idx = s['common_idx']
        frc = s['frc']

        for force_name in ['F_market', 'F_restore', 'F_damp', 'F_self']:
            fig.add_trace(
                go.Scatter(
                    x=common_idx,
                    y=frc[force_name],
                    mode='lines',
                    stackgroup=f'force_{industry_code}',
                    name=f'{force_name} ({industry_code})',
                    line=dict(width=0.5, color=force_colors[force_name]),
                    fillcolor=force_colors[force_name],
                    legendgroup=industry_code,
                    showlegend=(force_name == 'F_market'),
                ),
                row=2, col=1,
            )

    fig.update_layout(
        title=title,
        height=800,
        hovermode='closest',
        legend_tracegroupgap=10,
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')


def parse_args():
    parser = argparse.ArgumentParser(description='v5.8: State Timeline + Force Decomposition HTML')
    parser.add_argument('--code', type=str, required=True, help='Stock code, e.g. 002475.SZ')
    parser.add_argument('--days', type=int, default=250, help='Days lookback (default 250)')
    parser.add_argument('--prefer-industry', action='store_true', default=True,
                        help='Use industry index (default True)')
    parser.add_argument('--no-prefer-industry', dest='prefer_industry', action='store_false')
    parser.add_argument('--lambda-q', type=float, default=None,
                        help='Anchoring strength (None=adaptive)')
    parser.add_argument('--k-restore', type=float, default=0.0, help='Restoration coefficient k')
    parser.add_argument('--c-damp', type=float, default=0.0, help='Damping coefficient c')
    parser.add_argument('--output', type=str,
                        default='backtrace/outputs/dynsys_state_timeline.html',
                        help='Output HTML path')
    return parser.parse_args()


def main():
    args = parse_args()

    series = load_state_force_timeseries(
        stock_code=args.code,
        days=args.days,
        pipeline=P,
        prefer_industry=args.prefer_industry,
        lambda_q=args.lambda_q,
        k_restore=args.k_restore,
        c_damp=args.c_damp,
    )

    series_per_industry = [{
        'industry_code': series['index_code'],
        'common_idx': series['common_idx'],
        'states': series['states'],
        'frc': series['frc'],
        'dyn': series['dyn'],
    }]

    title = f"{series['index_name']} ({series['index_code']}) — State Timeline + Force Decomposition"
    build_state_timeline_html(series_per_industry, args.output, title=title)
    print(f'[v5.8] state timeline 已写入 {args.output}')


if __name__ == '__main__':
    main()
