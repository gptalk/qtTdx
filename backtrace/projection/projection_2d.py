# -*- coding: utf-8 -*-
# 2-D 投影验证 — legacy，将个股 (STOCK_CODE) 的成交量 / 成交额向量投影到基线指数的方向上
# 个股通过 --code / --name / --days 参数化;基线指数默认按个股交易所自动选大盘
# (SZ→深证成指 / SH→上证综指),通过 --index 可显式覆盖为任意行业指数或自定义基线
# 输出:6 个 HTML 到 backtrace/outputs/ + 1 个 CSV 到 data/projection/
# 数学/数据载入/CSV 组装统一在 _projection_core.py;本脚本只负责 plotly 可视化与文件落地
# 用法:已不推荐,主要用作早期可正交性可视化实验;研究请改用 vbt/tsfresh 系列
# 批量版见 projection_batch.py
#
# 参数:
#   --code          str   个股代码(带 .SH / .SZ 后缀)。默认 002475.SZ
#   --name          str   个股中文名(仅用于图例标签)。默认 立讯精密
#   --days          int   回看交易日数。默认 240
#   --index         str   基线指数代码(带 .SH / .SZ 后缀)。默认 None 时按个股交易所自动选大盘。
#                        示例:
#                          881427.SH(申万二级行业-体育)
#                          000001.SH(上证综指,显式指定)
#                          399001.SZ(深证成指,显式指定)
#                        任意能解析的 TQ 代码均可;数据需在 data/sectors/ 或 data/indices/ 缓存中。
#   --two-day-vec   flag  向量扩展为 4-D(今日+前一日 Vol/Amt);首日丢弃。默认 2-D。
#   --movement      flag  运动向量投影模式(正交于 --two-day-vec):把个股 (ΔVol, ΔAmt) 投到
#                        大盘 (ΔVol, ΔAmt) 的运动方向上,首行同样丢弃(因 .diff 无前一日)。
#                        产出前缀 `projmv_*.html` 的 4 个 HTML + 1 个 movement_*.csv,
#                        与 2-D / 4-D 的 `proj2d_*.html` / `proj2d_4d_*.html` 不冲突。
#
# 向量维度说明:
#   默认(2-D 状态):向量 v = (Volume, Amount),每个交易日产出 19 列 CSV;
#     HTML 文件前缀 `proj2d_*.html`(例:proj2d_002475_index.html)
#   --two-day-vec(4-D 状态):v = (Volume_today, Amount_today, Volume_yesterday, Amount_yesterday),
#     产生 27 列 CSV(新增 prev_raw / prev_norm 两组共 8 列);首日无前一日数据被丢弃;
#     HTML 文件前缀切到 `proj2d_4d_*.html`(例:proj2d_4d_002475_index.html)避免覆盖 2-D 结果。
#   --movement(运动向量投影):不算"当前成交状态"投影,而是把 Δv_s 投到 Δv_i 方向上,
#     产出 13 列 movement CSV(ΔV/ΔA/Proj_Coeff/Proj_Delta/Resi_Delta/Magnitude/Dot_After);
#     HTML 落到 `projmv_movement_scatter.html` / `movement_projection_verify.html` /
#              `movement_coeff.html` / `movement_orthogonality.html`(共 4 个)。
#   状态投影与运动投影是两种独立特征,可同时启用(--two-day-vec + --movement)。
#
# CLI:
#   python backtrace/projection/projection_2d.py                                       # 默认 002475.SZ / 立讯精密 / 240 日 / 大盘基线 / 2-D
#   python backtrace/projection/projection_2d.py --code 688318.SH                      # 科创板虹软 → 上证综指
#   python backtrace/projection/projection_2d.py --code 600519.SH --name 贵州茅台 --days 120
#   python backtrace/projection/projection_2d.py --code 300651.SZ --index 881427.SH     # 金陵体育 → 申万二级体育指数
#   python backtrace/projection/projection_2d.py --code 002475.SZ --index 000001.SH    # 立讯精密 → 上证综指(显式)
#   python backtrace/projection/projection_2d.py --code 002475.SZ --two-day-vec        # 4-D:含前一日 Vol/Amt,HTML 落到 proj2d_4d_*.html
#   python backtrace/projection/projection_2d.py --code 600519.SH --two-day-vec --days 120 # 4-D + 120 日回看
#   python backtrace/projection/projection_2d.py --code 002475.SZ --movement           # 运动向量投影(产 projmv_*.html + movement_*.csv)
#   python backtrace/projection/projection_2d.py --code 600519.SH --movement --days 60 # 运动投影 + 60 日回看
#   python backtrace/projection/projection_2d.py --code 002475.SZ --two-day-vec --movement  # 状态 4-D + 运动投影双产出
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 走本地 data/ 缓存 — 不依赖 TQ 客户端;首次跑需先执行 backtrace/data_fetch/fetch_daily.py 拉数
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
from common import tsfresh_pipeline as P
from _projection_core import (
    load_pair,
    compute_vectors,
    compute_projections,
    build_result_df,
    compute_movement_projection,
    build_movement_result_df,
)

# ========================= CLI 参数 =========================
def parse_args():
    p = argparse.ArgumentParser(description='单股 2-D 投影分析(HTML + CSV)')
    p.add_argument('--code', default='002475.SZ', help='个股代码(带 .SH / .SZ 后缀)。默认 002475.SZ')
    p.add_argument(
        '--name', default=None,
        help=(
            '个股中文名(仅用于图例标签)。不传时按 --code 推默认名 '
            '(002475.SZ → 立讯精密);其他代码不传则空,避免误标。'
        ),
    )
    p.add_argument('--days', type=int, default=240, help='回看交易日数。默认 240')
    p.add_argument(
        '--index', default=None,
        help=(
            '基线指数代码(带 .SH / .SZ 后缀)。'
            '默认按个股交易所自动选择大盘(SZ→深证成指 / SH→上证综指);'
            '传 881xxx.SH 可改用申万二级行业指数;'
            '传 000001.SH / 399001.SZ 显式指定大盘。'
            '示例:--index 881427.SH(半导体)'
        ),
    )
    p.add_argument(
        '--two-day-vec', action='store_true',
        help='将向量扩展为 4-D (今日 + 前一日 Vol/Amt);首日丢弃。默认 2-D。',
    )
    p.add_argument(
        '--movement', action='store_true',
        help=(
            '运动向量投影模式:不投影当前成交状态,而是把个股 (ΔVol, ΔAmt) '
            '投影到大盘 (ΔVol, ΔAmt) 的运动方向上。首行丢弃(因 .diff 无前一日)。'
            '产出 movement HTML + CSV,与 --two-day-vec 可叠加。'
        ),
    )
    return p.parse_args()

args = parse_args()
TWO_DAY_VEC = args.two_day_vec
LAG = 1 if TWO_DAY_VEC else 0
STOCK_CODE = args.code
# --name 缺省时:先从 stocks_info 反查(用户跑过 fetch_stock_basic.py 后),
# 再回退到 002475.SZ 旧默认(避免非该代码误标);最后才是空字符串。
if args.name:
    STOCK_NAME = args.name
else:
    from common import stocks_info
    STOCK_NAME = stocks_info.lookup_name(STOCK_CODE) or {'002475.SZ': '立讯精密'}.get(STOCK_CODE, '')
days = args.days
INDEX_OVERRIDE = args.index

# ========================= 输出布局 =========================
OUT_DIR = 'backtrace/outputs' # HTML 报告输出目录(CLAUDE.md 约定)
FILE_PREFIX = 'proj2d_4d_' if TWO_DAY_VEC else 'proj2d_'  # HTML 文件统一前缀,4-D 模式切前缀
MOVEMENT_PREFIX = 'projmv_'  # movement HTML / CSV 统一前缀,与 state 投影分开
CSV_OUT = 'data/projection'   # 分析结果 CSV 输出子目录(与 INDEX/STOCK 标签组合文件名)
# ======================================================

# 由配置派生:六位数字代码(去交易所后缀)用于变量标签 / CSV 列名 / 图例
loaded = load_pair(STOCK_CODE, days, P, index_code=INDEX_OVERRIDE, lag=LAG)
data_stock = loaded['stock_df']
data_index = loaded['index_df']
common_idx = loaded['common_idx']
INDEX_CODE = loaded['index_code']
INDEX_NAME = loaded['index_name']
INDEX_TAG = loaded['index_tag']
STOCK_TAG = loaded['stock_tag']
INDEX_LABEL = f'{INDEX_CODE} ({INDEX_NAME})'
STOCK_LABEL = f'{STOCK_CODE} ({STOCK_NAME})'

baseline_kind = (
    f'显式指定基线 {INDEX_CODE}' if INDEX_OVERRIDE
    else ('大盘指数(按个股交易所)' if INDEX_CODE in ('000001.SH', '399001.SZ') else '行业指数(自动)')
)
print(f"基线选择: {baseline_kind}")
print(f"向量维度: {'4-D (今日+前一日 Vol/Amt)' if TWO_DAY_VEC else '2-D (今日 Vol/Amt)'}")

def out(name):
    """HTML 报告:backtrace/outputs/<FILE_PREFIX><name>"""
    return os.path.join(OUT_DIR, FILE_PREFIX + name).replace('\\', '/')

def out_csv(name):
    """分析 CSV:data/projection/<name>"""
    return os.path.join(CSV_OUT, name).replace('\\', '/')

print(f"从本地 data/ 缓存读取最近{days}日日线... 指数={INDEX_LABEL} 个股={STOCK_LABEL}")

vec_index, vec_stock, vec_index_norm, vec_stock_norm, norm_params = compute_vectors(
    data_stock, data_index, INDEX_TAG, STOCK_TAG, lag=LAG,
)

print(f"共同交易日数量: {len(common_idx)}")
print(f"Volume {INDEX_TAG} 范围: [{vec_index[:,0].min():.2e}, {vec_index[:,0].max():.2e}]")
print(f"Volume {STOCK_TAG} 范围: [{vec_stock[:,0].min():.2e}, {vec_stock[:,0].max():.2e}]")
print(f"Amount {INDEX_TAG} 范围: [{vec_index[:,1].min():.2e}, {vec_index[:,1].max():.2e}]")
print(f"Amount {STOCK_TAG} 范围: [{vec_stock[:,1].min():.2e}, {vec_stock[:,1].max():.2e}]")
print(f"\n归一化后向量范围: [0, 1]")

# 二维投影计算(全部委托给 _projection_core)
proj = compute_projections(vec_stock_norm, vec_index_norm)
projections = proj['projections']
residuals = proj['residuals']
dot_products_after = proj['dot_after']
proj_coefficients = proj['proj_coeffs']
proj_magnitudes = proj['proj_mags']
proj_prices = proj['proj_prices']
resi_prices = proj['resi_prices']

# 运动向量投影(可选,与状态投影并存)
movement_data = None
if args.movement:
    mv = compute_movement_projection(data_stock, data_index)
    movement_data = build_movement_result_df(common_idx[1:], mv, INDEX_TAG, STOCK_TAG)
    # 同时落 CSV(供非 HTML 路径消费)
    mv_csv = os.path.join(CSV_OUT, f'movement_{INDEX_TAG}_{STOCK_TAG}.csv')
    os.makedirs(CSV_OUT, exist_ok=True)
    movement_data.to_csv(mv_csv, index=False, encoding='utf-8')
    print(f"运动投影: 共 {len(movement_data)} 日 (首行丢弃),CSV → {mv_csv}")

# ============== 图形显示 ==============

# 图1: 所有日期的向量对比散点图 (归一化后)
fig1 = go.Figure()

# 指数向量 (归一化)
fig1.add_trace(go.Scatter(
    x=vec_index_norm[:, 0], y=vec_index_norm[:, 1],
    mode='markers', name=INDEX_LABEL,
    marker=dict(color='blue', size=6, opacity=0.7)
))

# 个股向量 (归一化)
fig1.add_trace(go.Scatter(
    x=vec_stock_norm[:, 0], y=vec_stock_norm[:, 1],
    mode='markers', name=STOCK_LABEL,
    marker=dict(color='red', size=6, opacity=0.7)
))

fig1.update_layout(
    title='Volume-Amount 二维空间向量分布 (Min-Max归一化)',
    xaxis_title='Volume (normalized)',
    yaxis_title='Amount (normalized)',
    template='plotly_dark',
    height=600, width=800
)
fig1.write_html(out('vector_scatter.html'))

# 图1b: 个股 stock 当日 - 前一日 (Volume, Amount) 差额的 3-D 折线
# X = 日序号(int,避免日期轴引入时分秒)、Y = Volume_diff、Z = Amount_diff
# lag=0 时 np.diff;lag=1 时 vec_stock 已有 Volume_prev / Amount_prev,直接减
if LAG == 0:
    vol_today = vec_stock[:, 0]
    amt_today = vec_stock[:, 1]
    vol_prev = np.concatenate([[vol_today[0]], vol_today[:-1]])   # pad 首个,保证长度一致
    amt_prev = np.concatenate([[amt_today[0]], amt_today[:-1]])
else:
    vol_today = vec_stock[:, 0]
    amt_today = vec_stock[:, 1]
    vol_prev = vec_stock[:, 2]
    amt_prev = vec_stock[:, 3]
vol_diff = vol_today - vol_prev
amt_diff = amt_today - amt_prev
day_idx = np.arange(len(common_idx))

fig1b = go.Figure()
fig1b.add_trace(go.Scatter3d(
    x=day_idx, y=vol_diff, z=amt_diff,
    mode='lines+markers',
    name=f'{STOCK_TAG} 日差',
    line=dict(color='cyan', width=2),
    marker=dict(
        size=3, color='orange', opacity=0.9,
        line=dict(color='white', width=0.2),
    ),
    hovertemplate=(
        'Day#%{x}<br>'
        'ΔVolume: %{y:.2e}<br>ΔAmount: %{z:.2e}<extra></extra>'
    ),
))
fig1b.update_layout(
    title=f'{STOCK_LABEL} 日间差额 3-D 折线 (ΔAmount / ΔVolume / 日序)',
    scene=dict(
        xaxis_title='日序 (Day #)',
        yaxis_title='ΔVolume (今日 - 前一日)',
        zaxis_title='ΔAmount (今日 - 前一日)',
        aspectmode='manual',
        aspectratio=dict(x=1.5, y=1, z=1),
    ),
    template='plotly_dark',
    height=700, width=900,
)
fig1b.write_html(out('stock_diff_3d.html'))

# 图2: 投影验证 - 选取几个代表性日期(在可用交易日内均匀取样,避免硬编码越界)
sample_indices = sorted(set(np.linspace(0, len(common_idx) - 1, 4, dtype=int).tolist()))

fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f'{str(common_idx[i])[:10]} 投影验证' for i in sample_indices],
    horizontal_spacing=0.15, vertical_spacing=0.15
)

for idx, (si, row, col) in enumerate(zip(sample_indices, [1,1,2,2], [1,2,1,2])):
    u = vec_stock_norm[si]
    v = vec_index_norm[si]
    proj_pt = projections[si]
    residual = residuals[si]

    # 原点到v (指数)
    fig2.add_trace(go.Scatter(
        x=[0, v[0]], y=[0, v[1]],
        mode='lines+markers', name=f'v ({INDEX_TAG})' if idx==0 else None,
        line=dict(color='blue', width=3), marker=dict(size=8)
    ), row=row, col=col)

    # 原点到u (个股)
    fig2.add_trace(go.Scatter(
        x=[0, u[0]], y=[0, u[1]],
        mode='lines+markers', name=f'u ({STOCK_TAG})' if idx==0 else None,
        line=dict(color='red', width=3), marker=dict(size=8)
    ), row=row, col=col)

    # 投影
    fig2.add_trace(go.Scatter(
        x=[0, proj_pt[0]], y=[0, proj_pt[1]],
        mode='lines+markers', name='proj(u->v)' if idx==0 else None,
        line=dict(color='green', width=2, dash='dash'), marker=dict(size=6)
    ), row=row, col=col)

    # 残差 (正交分量)
    fig2.add_trace(go.Scatter(
        x=[proj_pt[0], u[0]], y=[proj_pt[1], u[1]],
        mode='lines', name='residual (正交)' if idx==0 else None,
        line=dict(color='orange', width=2)
    ), row=row, col=col)

    # 标注正交验证
    fig2.add_annotation(
        x=u[0]*0.7, y=u[1]*0.7,
        text=f"u·v={np.dot(u,v):.2e}<br>residual·v={dot_products_after[si]:.2e}",
        showarrow=False, font=dict(size=8), row=row, col=col
    )

fig2.update_layout(
    title=f'{STOCK_CODE} → {INDEX_CODE} 投影分解 (正交验证)',
    template='plotly_dark',
    height=700, width=900,
    showlegend=True
)
fig2.write_html(out('projection_verify.html'))

# 图3: 正交性时序图 (叠加Close收盘价)
close_stock = data_stock['Close'].to_numpy()
close_stock_norm = (close_stock - close_stock.min()) / (close_stock.max() - close_stock.min())

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=list(common_idx), y=dot_products_after,
    mode='lines', name='residual · v',
    line=dict(color='orange')
))
fig3.add_trace(go.Scatter(
    x=list(common_idx), y=[0]*len(common_idx),
    mode='lines', name='y=0 (理想正交)',
    line=dict(color='gray', dash='dash')
))
fig3.add_trace(go.Scatter(
    x=list(common_idx), y=close_stock_norm,
    mode='lines', name=f'{STOCK_TAG} Close收盘价 (归一化到点积范围)',
    line=dict(color='cyan'),
    opacity=0.7
))
fig3.update_layout(
    title='正交性验证: (u - proj) · v 应为 0 (叠加Close收盘价)',
    xaxis_title='日期', yaxis_title='点积值 / 收盘价(归一化)',
    template='plotly_dark', height=400
)
fig3.write_html(out('orthogonality_check.html'))

# 图4: 投影函数图形
# 4a: 投影系数时序图
dot_abs_max = np.abs(proj_coefficients).max()
fig4a = go.Figure()
fig4a.add_trace(go.Scatter(
    x=list(common_idx), y=proj_coefficients,
    mode='lines', name='投影系数',
    line=dict(color='green')
))
fig4a.add_trace(go.Scatter(
    x=list(common_idx), y=close_stock_norm * dot_abs_max,
    mode='lines', name=f'{STOCK_TAG} Close收盘价 (归一化到点积范围)',
    line=dict(color='cyan'),
    opacity=0.7
))
fig4a.update_layout(
    title=f'投影系数时序 ({STOCK_TAG}→{INDEX_TAG})',
    xaxis_title='日期', yaxis_title='系数 (u·v / v·v)',
    template='plotly_dark', height=300
)
fig4a.write_html(out('proj_coefficient.html'))

# 4f: proj_prices 时序图
fig4f = go.Figure()
fig4f.add_trace(go.Scatter(
    x=list(common_idx), y=proj_prices,
    mode='lines', name='proj_prices (Amount/Volume 投影)',
    line=dict(color='purple')
))
fig4f.add_trace(go.Scatter(
    x=list(common_idx), y=close_stock,
    mode='lines', name=f'{STOCK_TAG} Close收盘价',
    line=dict(color='cyan'), opacity=0.7, yaxis='y2'
))
fig4f.update_layout(
    title='proj_prices 时序图 (投影向量的 Amount/Volume 比,叠加Close)',
    xaxis_title='日期',
    yaxis=dict(title='proj_prices'),
    yaxis2=dict(title=f'{STOCK_TAG} Close', overlaying='y', side='right', showgrid=False),
    template='plotly_dark', height=350
)
fig4f.write_html(out('proj_prices.html'))

# 4g: resi_prices 时序图
fig4g = go.Figure()
fig4g.add_trace(go.Scatter(
    x=list(common_idx), y=resi_prices,
    mode='lines', name='resi_prices (Amount/Volume 残差)',
    line=dict(color='red')
))
fig4g.add_trace(go.Scatter(
    x=list(common_idx), y=close_stock,
    mode='lines', name=f'{STOCK_TAG} Close收盘价',
    line=dict(color='cyan'), opacity=0.7, yaxis='y2'
))
fig4g.update_layout(
    title='resi_prices 时序图 (残差向量的 Amount/Volume 比,叠加Close)',
    xaxis_title='日期',
    yaxis=dict(title='resi_prices'),
    yaxis2=dict(title=f'{STOCK_TAG} Close', overlaying='y', side='right', showgrid=False),
    template='plotly_dark', height=350
)
fig4g.write_html(out('resi_prices.html'))

# 图 M: 运动向量投影(仅 --movement 启用时绘制)
if args.movement:
    mv_idx = list(common_idx[1:])           # common_idx 丢首行,与 diff 对齐
    mv_stock = movement_data[f'ΔV_{STOCK_TAG}'].to_numpy()
    mv_amt = movement_data[f'ΔA_{STOCK_TAG}'].to_numpy()
    mv_iv = movement_data[f'ΔV_{INDEX_TAG}'].to_numpy()
    mv_ia = movement_data[f'ΔA_{INDEX_TAG}'].to_numpy()
    mv_proj_v = movement_data['Proj_Delta_Vol'].to_numpy()
    mv_proj_a = movement_data['Proj_Delta_Amt'].to_numpy()
    mv_res_v = movement_data['Resi_Delta_Vol'].to_numpy()
    mv_res_a = movement_data['Resi_Delta_Amt'].to_numpy()
    mv_coeff = movement_data['Proj_Coeff'].to_numpy()
    mv_dot_after = movement_data['Dot_After_Proj'].to_numpy()

    def mv_out(name):
        return os.path.join(OUT_DIR, MOVEMENT_PREFIX + name).replace('\\', '/')

    # M1: 个股/大盘 运动向量散点 (ΔV, ΔA) — 两行独立 Y 轴避免量纲差异
    #     (大盘 Amount Δ ≈ 1e7, 个股 Amount Δ ≈ 1e4,共享轴会让个股被压扁到 y≈0)
    figm1 = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Δ{INDEX_LABEL} (大盘基线)',
            f'Δ{STOCK_LABEL} (个股)',
        ),
        vertical_spacing=0.15,
        shared_xaxes=False,
    )
    figm1.add_trace(go.Scatter(
        x=mv_iv, y=mv_ia, mode='markers', name=f'Δ{INDEX_LABEL}',
        marker=dict(color='blue', size=6, opacity=0.7),
        legendgroup='idx',
    ), row=1, col=1)
    figm1.add_trace(go.Scatter(
        x=mv_stock, y=mv_amt, mode='markers', name=f'Δ{STOCK_LABEL}',
        marker=dict(color='red', size=6, opacity=0.7),
        legendgroup='stk',
    ), row=2, col=1)
    figm1.update_xaxes(title_text='ΔVolume', row=1, col=1)
    figm1.update_xaxes(title_text='ΔVolume', row=2, col=1)
    figm1.update_yaxes(title_text='ΔAmount', row=1, col=1)
    figm1.update_yaxes(title_text='ΔAmount', row=2, col=1)
    figm1.update_layout(
        title='运动向量散点 (ΔAmount / ΔVolume,原始量纲;上下分轴避免量纲压缩)',
        template='plotly_dark', height=800, width=800,
    )
    figm1.write_html(mv_out('movement_scatter.html'))

    # M2: 投影分解验证 (2×2 子图,每个交易日: v (index) / u (stock) / proj / residual)
    mv_sample_indices = sorted(set(
        np.linspace(0, len(mv_idx) - 1, min(4, len(mv_idx)), dtype=int).tolist()
    ))
    figm2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'{str(mv_idx[i])[:10]} 运动投影' for i in mv_sample_indices],
        horizontal_spacing=0.15, vertical_spacing=0.15,
    )
    for idx, mi in enumerate(mv_sample_indices):
        row = idx // 2 + 1
        col = idx % 2 + 1
        v_vec = np.array([mv_iv[mi], mv_ia[mi]])
        u_vec = np.array([mv_stock[mi], mv_amt[mi]])
        proj_vec = np.array([mv_proj_v[mi], mv_proj_a[mi]])
        res_vec = u_vec - proj_vec
        # v (index 运动)
        figm2.add_trace(go.Scatter(
            x=[0, v_vec[0]], y=[0, v_vec[1]],
            mode='lines+markers', name=f'i ({INDEX_TAG})' if idx == 0 else None,
            line=dict(color='blue', width=3), marker=dict(size=8),
        ), row=row, col=col)
        # u (stock 运动)
        figm2.add_trace(go.Scatter(
            x=[0, u_vec[0]], y=[0, u_vec[1]],
            mode='lines+markers', name=f's ({STOCK_TAG})' if idx == 0 else None,
            line=dict(color='red', width=3), marker=dict(size=8),
        ), row=row, col=col)
        # 投影
        figm2.add_trace(go.Scatter(
            x=[0, proj_vec[0]], y=[0, proj_vec[1]],
            mode='lines+markers', name='proj(s→i)' if idx == 0 else None,
            line=dict(color='green', width=2, dash='dash'), marker=dict(size=6),
        ), row=row, col=col)
        # 残差
        figm2.add_trace(go.Scatter(
            x=[proj_vec[0], u_vec[0]], y=[proj_vec[1], u_vec[1]],
            mode='lines', name='residual' if idx == 0 else None,
            line=dict(color='orange', width=2),
        ), row=row, col=col)
        # 正交性注释
        figm2.add_annotation(
            x=u_vec[0] * 0.7, y=u_vec[1] * 0.7,
            text=f'β={mv_coeff[mi]:.3f}<br>res·i={mv_dot_after[mi]:.2e}',
            showarrow=False, font=dict(size=8), row=row, col=col,
        )
    figm2.update_layout(
        title=f'{STOCK_LABEL} → {INDEX_LABEL} 运动投影分解',
        template='plotly_dark', height=700, width=900, showlegend=True,
    )
    figm2.write_html(mv_out('movement_projection_verify.html'))

    # M3: 投影系数 β 时序图
    figm3 = go.Figure()
    figm3.add_trace(go.Scatter(
        x=mv_idx, y=mv_coeff, mode='lines', name='β (运动映射系数)',
        line=dict(color='green'),
    ))
    figm3.add_trace(go.Scatter(
        x=mv_idx, y=[0] * len(mv_idx), mode='lines', name='β=0',
        line=dict(color='gray', dash='dash'),
    ))
    figm3.update_layout(
        title=f'运动投影系数 β 时序 ({STOCK_TAG} → {INDEX_TAG})',
        xaxis_title='日期', yaxis_title='β = (Δu·Δv) / (Δv·Δv)',
        template='plotly_dark', height=350,
    )
    figm3.write_html(mv_out('movement_coeff.html'))

    # M4: 正交性时序图 (residual · v 应为 0)
    figm4 = go.Figure()
    figm4.add_trace(go.Scatter(
        x=mv_idx, y=mv_dot_after, mode='lines', name='residual · Δv',
        line=dict(color='orange'),
    ))
    figm4.add_trace(go.Scatter(
        x=mv_idx, y=[0] * len(mv_idx), mode='lines', name='y=0 (理想正交)',
        line=dict(color='gray', dash='dash'),
    ))
    figm4.update_layout(
        title='运动投影正交性验证: (Δu - proj) · Δv 应为 0',
        xaxis_title='日期', yaxis_title='点积值',
        template='plotly_dark', height=350,
    )
    figm4.write_html(mv_out('movement_orthogonality.html'))

    # M5: proj_prices 时序图 — 运动投影的边际成交价(β·ΔA / β·ΔV = ΔA_i / ΔV_i)
    # 与状态投影 4f/4g 同主题但语义不同:这里是大盘边际成交均价,描述市场状态
    # 切换时的成交均价,与个股 Volume 方向相同;残差是个股偏离大盘方向的程度。
    assert movement_data is not None   # 整段在 if args.movement: 内,必非 None
    mv_proj_prices = movement_data['Proj_Price'].to_numpy()
    mv_resi_prices = movement_data['Resi_Price'].to_numpy()
    figm5 = go.Figure()
    figm5.add_trace(go.Scatter(
        x=mv_idx, y=mv_proj_prices, mode='lines',
        name='proj_price (大盘边际成交均价 ΔA_i/ΔV_i)',
        line=dict(color='cyan'),
    ))
    figm5.add_trace(go.Scatter(
        x=mv_idx, y=[0] * len(mv_idx), mode='lines', name='y=0',
        line=dict(color='gray', dash='dash'),
    ))
    figm5.update_layout(
        title=f'运动 proj_prices 时序 ({STOCK_TAG} → {INDEX_TAG})',
        xaxis_title='日期', yaxis_title='proj_price = β·ΔA / β·ΔV',
        template='plotly_dark', height=350,
    )
    figm5.write_html(mv_out('movement_proj_prices.html'))

    # M6: resi_prices 时序图 — 残差向量的 Amount/Volume 比(已限幅到 ±3)。
    # 识别个股是否在大盘方向之外额外放量(>0)/缩量(<0)。
    figm6 = go.Figure()
    figm6.add_trace(go.Scatter(
        x=mv_idx, y=mv_resi_prices, mode='lines',
        name='resi_price (个股残差边际价)',
        line=dict(color='magenta'),
    ))
    figm6.add_trace(go.Scatter(
        x=mv_idx, y=[0] * len(mv_idx), mode='lines', name='y=0',
        line=dict(color='gray', dash='dash'),
    ))
    figm6.update_layout(
        title=f'运动 resi_prices 时序 ({STOCK_TAG} → {INDEX_TAG})',
        xaxis_title='日期', yaxis_title='resi_price = residual_ΔA / residual_ΔV (限幅 ±3)',
        template='plotly_dark', height=350,
    )
    figm6.write_html(mv_out('movement_resi_prices.html'))

    print("\n运动向量投影 HTML 已生成:")
    print(f"  M1. {mv_out('movement_scatter.html')}            - ΔV/ΔA 运动向量散点")
    print(f"  M2. {mv_out('movement_projection_verify.html')} - 运动投影分解验证")
    print(f"  M3. {mv_out('movement_coeff.html')}              - β 系数时序")
    print(f"  M4. {mv_out('movement_orthogonality.html')}      - 运动正交性验证")
    print(f"  M5. {mv_out('movement_proj_prices.html')}        - 运动 proj_prices 时序")
    print(f"  M6. {mv_out('movement_resi_prices.html')}        - 运动 resi_prices 时序")

print("\n图形已生成:")
print(f"  1. {out('vector_scatter.html')}      - Volume-Amount向量散点图")
print(f"  1b. {out('stock_diff_3d.html')}      - 个股日间差额 3-D 折线 (ΔVolume/ΔAmount/日序)")
print(f"  2. {out('projection_verify.html')}  - 投影分解验证图")
print(f"  3. {out('orthogonality_check.html')} - 正交性时序检验图")
print(f"  4. {out('proj_coefficient.html')}    - 投影系数时序图")
print(f"  5. {out('proj_prices.html')}        - proj_prices 时序图")
print(f"  6. {out('resi_prices.html')}        - resi_prices 时序图")

# 保存CSV(组装 19 列 DataFrame)
result_df = build_result_df(
    common_idx, vec_index, vec_stock, vec_index_norm, vec_stock_norm,
    projections, residuals, dot_products_after,
    proj_coefficients, proj_magnitudes, proj_prices, resi_prices,
    norm_params, INDEX_TAG, STOCK_TAG, lag=LAG,
)
csv_path = out_csv(f'projection_{INDEX_TAG}_{STOCK_TAG}.csv')
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
result_df.to_csv(csv_path, index=False, encoding='utf-8')
print(f"\n数据已保存到 {csv_path}")