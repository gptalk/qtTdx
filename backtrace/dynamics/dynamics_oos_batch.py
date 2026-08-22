# -*- coding: utf-8 -*-
# dynamics_oos_batch.py — v5.10 Task 1: scaffold + per-stock metrics + aggregator
#              v5.11 Task 3: --kc-estimates-csv 透传 load_oos_predictions(读真实 k̂, ĉ)
#
# 本模块给 v5.10 「全市场 OOS 分布」分析打地基:
#   1. compute_oos_metrics — 单股调用 load_oos_predictions(v5.9) → 算 hit/rmse/mae/dir_acc
#   2. aggregate_oos_metrics — 批量汇总 → median/quantiles/ranked
#
# 设计要点:
#   - 0 重写 projection / dynamics 数学(全部 import)
#   - 0 新依赖(numpy/pandas 已存在)
#   - plotly 仅用于 build_full_market_oos_html(Task 2,已存在依赖)
#   - REPO_ROOT sys.path 沿用 v5.9.1 修复模式
#
# 已知坑:
#   - load_oos_predictions 不传 lambda_q 时由 compute_dynamics 自适应(None = 自适应)
#   - M1 tsfresh shadow tolerated(详见 dynamics_1step_oos README)
#   - n_oos == 0 时全部返回 NaN(命中"零样本"边界)
import warnings
warnings.filterwarnings('ignore')
import sys, os
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
REPO_ROOT = os.path.dirname(BACKTRACE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import argparse
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtrace.dynamics.dynamics_oos_viz import load_oos_predictions
from dynamics.dynamics_granularity_compare import output_subdir_for_period

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

DEFAULTS = dict(
    days=250,
    limit=0,
    prefer_industry=True,
    top_n=5,
)
DEFAULT_OUTPUT = 'backtrace/outputs/dynsys_oos_full_market.html'


# === Per-stock OOS metrics =================================================
def compute_oos_metrics(
    stock_code: str,
    days: int = 250,
    *,
    prefer_industry: bool = True,
    k: float | None = None,
    c: float | None = None,
    f_self_window: int = 10,
    kc_estimates_path: str | None = None,  # v5.11 NEW:透传给 load_oos_predictions
    period: str = 'daily',
) -> dict:
    """Per-stock OOS prediction quality metrics.

    Returns dict with keys:
        code: str
        n_oos: int
        hit_rate: float
        rmse: float
        mae: float
        direction_accuracy: float
        k_used: float
        c_used: float
    """
    # 1) 跑 v5.9 的对齐预测
    oos = load_oos_predictions(
        stock_code=stock_code,
        days=days,
        prefer_industry=prefer_industry,
        k=k,
        c=c,
        f_self_window=f_self_window,
        kc_estimates_path=kc_estimates_path,  # v5.11 NEW
        period=period,
    )

    common_idx = oos['common_idx']
    a_pred = oos['a_pred']
    a_actual = oos['a_actual']
    state_pred = oos['state_pred']
    state_actual = oos['state_actual']
    k_used = oos['k_used']
    c_used = oos['c_used']

    n_oos = len(common_idx)

    # 2) 零样本兜底(避免 NaN propagation)
    if n_oos == 0:
        return {
            'code': stock_code,
            'n_oos': 0,
            'hit_rate': float('nan'),
            'rmse': float('nan'),
            'mae': float('nan'),
            'direction_accuracy': float('nan'),
            'k_used': float(k_used),
            'c_used': float(c_used),
        }

    # 3) 幅度误差
    a_pred_mag = np.linalg.norm(a_pred, axis=1)
    a_actual_mag = np.linalg.norm(a_actual, axis=1)
    error = a_pred_mag - a_actual_mag

    # 4) rmse / mae(nan 防护,虽然本函数上游不应产生 NaN)
    rmse = float(np.sqrt(np.nanmean(np.square(error))))
    mae = float(np.nanmean(np.abs(error)))

    # 5) 状态命中率
    n_hit = sum(1 for sp, sa in zip(state_pred, state_actual) if sp == sa)
    hit_rate = float(n_hit / n_oos)

    # 6) 方向一致率(按加速度幅度符号)
    n_same_dir = sum(
        1
        for p, a in zip(a_pred_mag, a_actual_mag)
        if np.sign(p) == np.sign(a)
    )
    direction_accuracy = float(n_same_dir / n_oos)

    return {
        'code': stock_code,
        'n_oos': int(n_oos),
        'hit_rate': hit_rate,
        'rmse': rmse,
        'mae': mae,
        'direction_accuracy': direction_accuracy,
        'k_used': float(k_used),
        'c_used': float(c_used),
    }


# === Cross-stock aggregator =================================================
def aggregate_oos_metrics(metrics_list: list[dict]) -> dict:
    """Aggregate per-stock metrics into population summary.

    Returns dict with:
        n_stocks: int
        median_hit_rate, p25_hit_rate, p75_hit_rate: float
        median_rmse, median_mae: float
        median_direction_acc: float
        ranked: list[dict]  # sorted by hit_rate desc
    """
    # 1) 空列表兜底
    if not metrics_list:
        return {
            'n_stocks': 0,
            'median_hit_rate': float('nan'),
            'p25_hit_rate': float('nan'),
            'p75_hit_rate': float('nan'),
            'median_rmse': float('nan'),
            'median_mae': float('nan'),
            'median_direction_acc': float('nan'),
            'ranked': [],
        }

    # 2) DataFrame + 中位数 / 四分位
    df = pd.DataFrame(metrics_list)
    ranked = df.sort_values('hit_rate', ascending=False).to_dict('records')

    return {
        'n_stocks': int(len(df)),
        'median_hit_rate': float(np.nanmedian(df['hit_rate'])),
        'p25_hit_rate': float(np.nanpercentile(df['hit_rate'], 25)),
        'p75_hit_rate': float(np.nanpercentile(df['hit_rate'], 75)),
        'median_rmse': float(np.nanmedian(df['rmse'])),
        'median_mae': float(np.nanmedian(df['mae'])),
        'median_direction_acc': float(np.nanmedian(df['direction_accuracy'])),
        'ranked': ranked,
    }


# === Full-market 2x2 dashboard =============================================
def build_full_market_oos_html(
    metrics_list: list[dict],
    output_path: str,
    title: str = 'Full-Market OOS Prediction Quality Distribution',
) -> None:
    """2x2 plotly dashboard of full-market OOS prediction quality (spec 3.4).

    Panels:
        (1,1) hit-rate histogram + median/p25/p75 vlines
        (1,2) RMSE histogram + median vline
        (2,1) hit-rate vs RMSE scatter (Viridis, hover = code)
        (2,2) hit-rate CDF + median vline
    """
    # 1) 空输入 → 明确报错(而不是画空图)
    if not metrics_list:
        raise ValueError('metrics_list is empty — nothing to plot')

    # 2) DataFrame
    df = pd.DataFrame(metrics_list)
    hit_rates = df['hit_rate'].to_numpy()
    rmses = df['rmse'].to_numpy()

    # 3) 汇总统计
    median_hr = float(np.median(hit_rates))
    p25_hr = float(np.percentile(hit_rates, 25))
    p75_hr = float(np.percentile(hit_rates, 75))
    median_rmse = float(np.median(rmses))

    # 4) 2x2 子图骨架
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Hit-rate distribution (median={median_hr:.3f}, IQR=[{p25_hr:.3f}, {p75_hr:.3f}])',
            f'RMSE distribution (median={median_rmse:.4f})',
            'Hit-rate vs RMSE scatter',
            'Hit-rate CDF',
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # 5) (1,1) hit-rate 直方图
    fig.add_trace(
        go.Histogram(
            x=hit_rates, nbinsx=40,
            marker=dict(color='#3498db', line=dict(color='#2c3e50', width=0.5)),
            name='hit-rate',
        ),
        row=1, col=1,
    )
    for marker_val, label, color in [(median_hr, 'median', '#e74c3c'),
                                     (p25_hr, 'p25', '#95a5a6'),
                                     (p75_hr, 'p75', '#95a5a6')]:
        fig.add_vline(x=marker_val, line_dash='dash', line_color=color,
                      annotation_text=label, row=1, col=1)

    # 6) (1,2) RMSE 直方图
    fig.add_trace(
        go.Histogram(
            x=rmses, nbinsx=40,
            marker=dict(color='#e67e22', line=dict(color='#2c3e50', width=0.5)),
            name='RMSE',
        ),
        row=1, col=2,
    )
    fig.add_vline(x=median_rmse, line_dash='dash', line_color='#e74c3c',
                  annotation_text='median', row=1, col=2)

    # 7) (2,1) hit-rate vs RMSE 散点
    fig.add_trace(
        go.Scatter(
            x=hit_rates, y=rmses,
            mode='markers',
            marker=dict(
                size=6,
                color=hit_rates,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='hit-rate', x=0.45, len=0.4, y=0.2),
                line=dict(color='#2c3e50', width=0.5),
            ),
            text=df['code'].tolist(),
            hovertemplate='<b>%{text}</b><br>hit-rate: %{x:.3f}<br>RMSE: %{y:.4f}<extra></extra>',
            name='stocks',
        ),
        row=2, col=1,
    )

    # 8) (2,2) hit-rate CDF
    sorted_hr = np.sort(hit_rates)
    cdf = np.arange(1, len(sorted_hr) + 1) / len(sorted_hr)
    fig.add_trace(
        go.Scatter(
            x=sorted_hr, y=cdf,
            mode='lines',
            line=dict(color='#2ecc71', width=2),
            name='CDF',
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.2)',
        ),
        row=2, col=2,
    )
    fig.add_vline(x=median_hr, line_dash='dash', line_color='#e74c3c',
                  annotation_text='median', row=2, col=2)

    # 9) 轴标签
    fig.update_xaxes(title_text='hit-rate', row=1, col=1)
    fig.update_xaxes(title_text='RMSE', row=1, col=2)
    fig.update_xaxes(title_text='hit-rate', row=2, col=1)
    fig.update_xaxes(title_text='hit-rate', row=2, col=2)
    fig.update_yaxes(title_text='count', row=1, col=1)
    fig.update_yaxes(title_text='count', row=1, col=2)
    fig.update_yaxes(title_text='RMSE', row=2, col=1)
    fig.update_yaxes(title_text='CDF', row=2, col=2)

    # 10) layout + 落盘
    n_stocks = len(metrics_list)
    fig.update_layout(
        title=f"{title} — N={n_stocks}",
        height=800, showlegend=False,
        template='plotly_white',
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn', full_html=True)
    log.info(f"[v5.10] wrote {output_path} ({n_stocks} stocks, median hit-rate={median_hr:.3f})")


# === Top-5 small multiples ===============================================
def build_top5_small_multiples(
    top5_data: list[dict],   # list of {code, common_idx, a_pred, a_actual, hit_rate, rmse}
    output_path: str,
    title: str = 'Top-5 OOS Prediction Detail',
) -> None:
    """N mini 4-row charts in single figure (spec 3.5).

    N×1 row layout: 1 column, N rows = len(top5_data).
    Each row: predicted (blue #3498db) + actual (orange #e67e22) magnitude lines.
    """
    # 1) Validate
    if not top5_data:
        raise ValueError('top5_data is empty')

    # 2) N-row subplots (1 column, shared_xaxes=False)
    n = len(top5_data)
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.04,
        subplot_titles=[f"{d['code']} (hit={d['hit_rate']:.3f}, RMSE={d['rmse']:.4f})"
                        for d in top5_data],
    )

    # 3) 2 traces per stock (predicted + actual)
    for i, d in enumerate(top5_data, start=1):
        a_pred_mag = np.linalg.norm(d['a_pred'], axis=1)
        a_actual_mag = np.linalg.norm(d['a_actual'], axis=1)
        common_idx = d['common_idx']

        fig.add_trace(
            go.Scatter(
                x=common_idx, y=a_pred_mag,
                mode='lines', line=dict(color='#3498db', width=1),
                name=f"{d['code']} pred" if i == 1 else None,
                legendgroup='series',
                showlegend=(i == 1),
            ),
            row=i, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=common_idx, y=a_actual_mag,
                mode='lines', line=dict(color='#e67e22', width=1),
                name=f"{d['code']} actual" if i == 1 else None,
                legendgroup='series',
                showlegend=(i == 1),
            ),
            row=i, col=1,
        )
        fig.update_yaxes(title_text='|a_S|', row=i, col=1)

    # 4) Layout + output
    fig.update_xaxes(title_text='date', row=n, col=1)
    fig.update_layout(
        title=f"{title} — top {n} by hit-rate",
        height=250 * n, showlegend=True,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn', full_html=True)
    log.info(f"[v5.10] wrote {output_path} (top {n})")


__all__ = [
    'compute_oos_metrics',
    'aggregate_oos_metrics',
    'build_full_market_oos_html',
    'build_top5_small_multiples',
    'DEFAULTS',
    'DEFAULT_OUTPUT',
]


# === CLI: stock-code loader ===============================================
def _load_stock_codes(limit: int) -> list[str]:
    """Load stock codes from data/manifest.json (TQ本地缓存)。

    Falls back to scanning data/stocks/*.csv if manifest missing.
    """
    manifest_path = 'data/manifest.json'
    if os.path.exists(manifest_path):
        import json
        with open(manifest_path, encoding='utf-8') as fh:
            manifest = json.load(fh)
        # manifest has nested `entries` keyed by code; each entry has
        # `kind` ('stocks'|'sectors'|'indices') + `status` ('ok'|'failed').
        entries = manifest.get('entries', manifest) if isinstance(manifest, dict) else {}
        codes = [
            c for c, info in entries.items()
            if isinstance(info, dict)
            and info.get('kind', 'stocks') == 'stocks'
            and info.get('status', 'ok') != 'failed'
            and info.get('rows', 0) > 0
        ]
        codes.sort()
        if limit > 0:
            codes = codes[:limit]
        return codes

    # Fallback: scan directory
    stock_dir = 'data/stocks'
    if not os.path.isdir(stock_dir):
        raise FileNotFoundError(f'No manifest.json or data/stocks/ dir found')
    files = [f for f in os.listdir(stock_dir) if f.endswith('.csv')]
    files.sort()
    codes = [f.replace('.csv', '') for f in files]
    if limit > 0:
        codes = codes[:limit]
    return codes


# === CLI: main() =========================================================
def main():
    p = argparse.ArgumentParser(
        description='v5.10 — Full-market OOS prediction quality distribution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--days', type=int, default=DEFAULTS['days'], help='trading days per stock')
    p.add_argument('--limit', type=int, default=DEFAULTS['limit'],
                   help='0 = all stocks in local cache, else first N')
    p.add_argument('--prefer-industry', dest='prefer_industry',
                   action='store_true', default=DEFAULTS['prefer_industry'])
    p.add_argument('--no-prefer-industry', dest='prefer_industry',
                   action='store_false')
    p.add_argument('--top-n', dest='top_n', type=int, default=DEFAULTS['top_n'],
                   help='number of top stocks to render as small multiples')
    p.add_argument('--codes-file', dest='codes_file', type=str, default=None,
                   help='optional file with one stock code per line')
    p.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                   help='output HTML path')
    p.add_argument('--kc-estimates-csv', dest='kc_estimates_csv', type=str, default=None,
                   help='v5.11: 透传给 compute_oos_metrics → load_oos_predictions')
    p.add_argument('--period', choices=['daily', '15m', '5m', '1m'], default='daily',
                   help='缓存粒度(daily = 默认)')
    args = p.parse_args()

    # 1. Load stock codes
    if args.codes_file and os.path.exists(args.codes_file):
        with open(args.codes_file) as fh:
            codes = [line.strip() for line in fh if line.strip()]
        log.info(f"[v5.10] loaded {len(codes)} codes from {args.codes_file}")
    else:
        codes = _load_stock_codes(args.limit)
        log.info(f"[v5.10] loaded {len(codes)} codes from manifest/cache")

    if not codes:
        raise ValueError('No stock codes found')

    log.info(f"[v5.10] days={args.days} prefer_industry={args.prefer_industry} top_n={args.top_n}")

    # 2. Compute per-stock metrics
    metrics_list = []
    per_stock_lookup = {}  # for top-N small multiples
    for idx, code in enumerate(codes, start=1):
        try:
            m = compute_oos_metrics(
                stock_code=code,
                days=args.days,
                prefer_industry=args.prefer_industry,
                kc_estimates_path=args.kc_estimates_csv,  # v5.11 NEW
                period=args.period,
            )
            if m['n_oos'] > 0:
                metrics_list.append(m)
                per_stock_lookup[code] = m
                log.info(f"[{idx}/{len(codes)}] {code}: hit={m['hit_rate']:.3f}, RMSE={m['rmse']:.4f}, "
                         f"k̂={m['k_used']:.4f}, ĉ={m['c_used']:.4f}")  # v5.11 NEW: ĉ 字段
            else:
                log.warning(f"[{idx}/{len(codes)}] {code}: 0 OOS days, skip")
        except Exception as e:
            log.warning(f"[{idx}/{len(codes)}] {code}: ERROR ({type(e).__name__}: {e}), skip")
            continue

    if not metrics_list:
        raise ValueError('No valid metrics computed — check data/manifest.json')

    # 3. Aggregate
    agg = aggregate_oos_metrics(metrics_list)
    log.info(f"[v5.10] aggregated: N={agg['n_stocks']}, "
             f"median_hit={agg['median_hit_rate']:.3f}, "
             f"median_rmse={agg['median_rmse']:.4f}")

    # 4. Render 2×2 distribution dashboard
    args.output = output_subdir_for_period(args.output, args.period)
    build_full_market_oos_html(
        metrics_list=metrics_list,
        output_path=args.output,
        title=f"Full-Market OOS — {agg['n_stocks']} stocks, {args.days} days",
    )

    # 5. Render top-N small multiples (separate file)
    top_n = min(args.top_n, agg['n_stocks'])
    top_codes = [r['code'] for r in agg['ranked'][:top_n]]
    top_data = []
    for code in top_codes:
        try:
            d = load_oos_predictions(
                stock_code=code, days=args.days,
                prefer_industry=args.prefer_industry,
                period=args.period,
            )
            top_data.append(dict(
                code=code,
                common_idx=d['common_idx'],
                a_pred=d['a_pred'],
                a_actual=d['a_actual'],
                hit_rate=per_stock_lookup[code]['hit_rate'],
                rmse=per_stock_lookup[code]['rmse'],
            ))
        except Exception as e:
            log.warning(f"top-{top_n}: failed to reload {code} for small multiples: {e}")

    # Compute top5_path unconditionally so the final log line never references
    # an unbound name when top_data is empty (e.g. --top-n 0 or all reloads fail).
    top5_path = args.output.replace('.html', '_top{}.html'.format(top_n))
    if top_data:
        build_top5_small_multiples(
            top5_data=top_data,
            output_path=top5_path,
            title=f"Top-{top_n} OOS Detail",
        )
        log.info(f"[v5.10] DONE — wrote {args.output} + top{top_n}-multiples at {top5_path}")
    else:
        log.info(f"[v5.10] DONE — wrote {args.output} (no top{top_n}-multiples; top_data empty)")


if __name__ == '__main__':
    main()
