# -*- coding: utf-8 -*-
# 2-D 投影验证 — legacy，将个股 (STOCK_CODE) 的成交量 / 成交额向量投影到大盘指数 (INDEX_CODE) 的方向上
# 指数与个股均在下方「配置」区参数化，改两行即可换标的
# 输出:6 个 HTML + 1 个 CSV 到 backtrace/ 根目录（vector_scatter / projection_verify / orthogonality_check / 等）
# 用法:已不推荐,主要用作早期可正交性可视化实验;研究请改用 vbt/tsfresh 系列
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 走本地 data/ 缓存 — 不依赖 TQ 客户端;首次跑需先执行 backtrace/data_fetch/fetch_daily.py 拉数
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
from common import tsfresh_pipeline as P

# ========================= 配置 =========================
INDEX_CODE = '399001.SZ'      # 大盘指数 — 深证成指(399001.SZ) / 上证指数(000001.SH) / 沪深300(000300.SH)
INDEX_NAME = '深证成指'
STOCK_CODE = '002475.SZ'      # 个股
STOCK_NAME = '立讯精密'
days = 240
OUT_DIR = 'backtrace'         # HTML 报告输出目录
CSV_OUT = 'data/projection'   # 分析结果 CSV 输出子目录(与 INDEX/STOCK 标签组合文件名)
# ======================================================

# 由配置派生:六位数字代码(去交易所后缀)用于变量标签 / CSV 列名 / 图例
INDEX_TAG = INDEX_CODE.split('.')[0]
STOCK_TAG = STOCK_CODE.split('.')[0]
INDEX_LABEL = f'{INDEX_CODE} ({INDEX_NAME})'
STOCK_LABEL = f'{STOCK_CODE} ({STOCK_NAME})'

def out(name):
    """HTML 报告:backtrace/<name>"""
    return os.path.join(OUT_DIR, name).replace('\\', '/')

def out_csv(name):
    """分析 CSV:data/projection/<name>"""
    return os.path.join(CSV_OUT, name).replace('\\', '/')

print(f"从本地 data/ 缓存读取最近{days}日日线... 指数={INDEX_LABEL} 个股={STOCK_LABEL}")

# use_tq=False 强制走 data_store.load_daily;数据缺失时返回 None(报清楚)
data_index_full = P.load_ohlcva(INDEX_CODE, use_tq=False, verbose=True)
data_stock_full = P.load_ohlcva(STOCK_CODE, use_tq=False, verbose=True)
if data_index_full is None:
    raise RuntimeError(
        f"本地缓存缺失 {INDEX_CODE}。请先跑:\n"
        f"  PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py"
    )
if data_stock_full is None:
    raise RuntimeError(
        f"本地缓存缺失 {STOCK_CODE}。请先跑:\n"
        f"  PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py"
    )

# 用最近 `days` 行(本地缓存保留 500 交易日,够用)
data_index = data_index_full[['Volume', 'Amount', 'Close']].tail(days).dropna()
data_stock = data_stock_full[['Volume', 'Amount', 'Close']].tail(days).dropna()

common_idx = data_index.index.intersection(data_stock.index)
data_index = data_index.loc[common_idx]
data_stock = data_stock.loc[common_idx]

print(f"共同交易日数量: {len(common_idx)}")

vec_index = data_index[['Volume', 'Amount']].values
vec_stock = data_stock[['Volume', 'Amount']].values

# 分别对 Volume 和 Amount 进行归一化 (Min-Max到[0,1])
vol_min_index, vol_max_index = vec_index[:, 0].min(), vec_index[:, 0].max()
vol_min_stock, vol_max_stock = vec_stock[:, 0].min(), vec_stock[:, 0].max()
amt_min_index, amt_max_index = vec_index[:, 1].min(), vec_index[:, 1].max()
amt_min_stock, amt_max_stock = vec_stock[:, 1].min(), vec_stock[:, 1].max()

vec_index_norm = np.column_stack([
    (vec_index[:, 0] - vol_min_index) / (vol_max_index - vol_min_index),
    (vec_index[:, 1] - amt_min_index) / (amt_max_index - amt_min_index)
])
vec_stock_norm = np.column_stack([
    (vec_stock[:, 0] - vol_min_stock) / (vol_max_stock - vol_min_stock),
    (vec_stock[:, 1] - amt_min_stock) / (amt_max_stock - amt_min_stock)
])

print(f"Volume {INDEX_TAG} 范围: [{vol_min_index:.2e}, {vol_max_index:.2e}]")
print(f"Volume {STOCK_TAG} 范围: [{vol_min_stock:.2e}, {vol_max_stock:.2e}]")
print(f"Amount {INDEX_TAG} 范围: [{amt_min_index:.2e}, {amt_max_index:.2e}]")
print(f"Amount {STOCK_TAG} 范围: [{amt_min_stock:.2e}, {amt_max_stock:.2e}]")
print(f"\n归一化后向量范围: [0, 1]")

# ============== 二维投影计算 ==============
def project_u_onto_v(u, v):
    v_norm_sq = np.dot(v, v)
    if v_norm_sq == 0:
        return np.zeros_like(u)
    coeff = np.dot(u, v) / v_norm_sq
    return coeff * v

projections = []
residuals = []
dot_products_after = []
proj_coefficients = []  # 投影系数
proj_magnitudes = []   # 投影向量模长
proj_prices = []  # 投影向量对应的价格 (个股的Close)
resi_prices = []  # 残差向量对应的价格 (个股的Close)

for i in range(len(common_idx)):
    u = vec_stock_norm[i]  # 归一化后的个股
    v = vec_index_norm[i]  # 归一化后的指数
    proj = project_u_onto_v(u, v)
    residual = u - proj
    projections.append(proj)
    residuals.append(residual)
    dot_products_after.append(np.dot(residual, v))
    proj_coefficients.append(np.dot(u, v) / np.dot(v, v))
    proj_magnitudes.append(np.linalg.norm(proj))
    proj_prices.append(proj[1]/proj[0] if proj[0] != 0 else np.sign(proj[1]))  # 投影向量的价格比 (Amount/Volume)
    resi_prices.append(residual[1]/residual[0] if residual[0] != 0 else residual[1]/abs(residual[1]))  # 残差向量的价格比 (Amount/Volume)
    #resi_prices中大于100的数字被认为是异常值，可能是由于Volume接近0导致的价格比异常大。设置为最大值
    if abs(resi_prices[-1]) > 3:
        resi_prices[-1] = np.sign(resi_prices[-1]) * np.max(np.abs(resi_prices[:-2]))



projections = np.array(projections)
residuals = np.array(residuals)
dot_products_after = np.array(dot_products_after)
proj_coefficients = np.array(proj_coefficients)
proj_magnitudes = np.array(proj_magnitudes)
proj_prices = np.array(proj_prices)
resi_prices = np.array(resi_prices)

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
    proj = projections[si]
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
        x=[0, proj[0]], y=[0, proj[1]],
        mode='lines+markers', name='proj(u->v)' if idx==0 else None,
        line=dict(color='green', width=2, dash='dash'), marker=dict(size=6)
    ), row=row, col=col)

    # 残差 (正交分量)
    fig2.add_trace(go.Scatter(
        x=[proj[0], u[0]], y=[proj[1], u[1]],
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
dot_abs_max = np.abs(dot_products_after).max()

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
    # x=list(common_idx), y=close_stock_norm * dot_abs_max,
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
    # x=list(common_idx), y=close_stock_norm,
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

print("\n图形已生成:")
print(f"  1. {out('vector_scatter.html')}      - Volume-Amount向量散点图")
print(f"  2. {out('projection_verify.html')}  - 投影分解验证图")
print(f"  3. {out('orthogonality_check.html')} - 正交性时序检验图")
print(f"  4. {out('proj_coefficient.html')}    - 投影系数时序图")
print(f"  5. {out('proj_prices.html')}        - proj_prices 时序图")
print(f"  6. {out('resi_prices.html')}        - resi_prices 时序图")

# 保存CSV
norm_params = (
    f"vol_{INDEX_TAG}:[{vol_min_index:.2e},{vol_max_index:.2e}] "
    f"amt_{INDEX_TAG}:[{amt_min_index:.2e},{amt_max_index:.2e}] "
    f"vol_{STOCK_TAG}:[{vol_min_stock:.2e},{vol_max_stock:.2e}] "
    f"amt_{STOCK_TAG}:[{amt_min_stock:.2e},{amt_max_stock:.2e}]"
)
result_df = pd.DataFrame({
    'Date': common_idx,
    f'Vol_{INDEX_TAG}_raw': vec_index[:, 0],
    f'Amt_{INDEX_TAG}_raw': vec_index[:, 1],
    f'Vol_{STOCK_TAG}_raw': vec_stock[:, 0],
    f'Amt_{STOCK_TAG}_raw': vec_stock[:, 1],
    f'Vol_{INDEX_TAG}_norm': vec_index_norm[:, 0],
    f'Amt_{INDEX_TAG}_norm': vec_index_norm[:, 1],
    f'Vol_{STOCK_TAG}_norm': vec_stock_norm[:, 0],
    f'Amt_{STOCK_TAG}_norm': vec_stock_norm[:, 1],
    'Proj_Vol': projections[:, 0],
    'Proj_Amt': projections[:, 1],
    'Residual_Vol': residuals[:, 0],
    'Residual_Amt': residuals[:, 1],
    'Proj_Coeff': proj_coefficients,
    'Proj_Magnitude': proj_magnitudes,
    'Proj_Price': proj_prices,
    'Resi_Price': resi_prices,
    'Dot_After_Proj': dot_products_after,
    'Norm_Params': [norm_params] * len(common_idx)
})
csv_path = out_csv(f'projection_{INDEX_TAG}_{STOCK_TAG}.csv')
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
result_df.to_csv(csv_path, index=False, encoding='utf-8')
print(f"\n数据已保存到 {csv_path}")
