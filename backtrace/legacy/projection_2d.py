# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqcenter import tq
from datetime import datetime, timedelta

tq.initialize(__file__)

# ========================= 配置 =========================
stock_list = ['000001.SH', '002475.SZ']
days = 240
# ======================================================

end_time = datetime.now().strftime("%Y%m%d")
start_time = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")

print(f"获取最近{days}日日线数据...")

df = tq.get_market_data(
    field_list=['Volume', 'Amount', 'Close'],
    stock_list=stock_list,
    start_time=start_time,
    end_time=end_time,
    dividend_type='front',
    period='1d',
    fill_data=True
)

vol_df = tq.price_df(df, 'Volume', column_names=stock_list)
amount_df = tq.price_df(df, 'Amount', column_names=stock_list)

close_df = tq.price_df(df, 'Close', column_names=stock_list)
data_000001 = pd.DataFrame({'Volume': vol_df['000001.SH'], 'Amount': amount_df['000001.SH'], 'Close': close_df['000001.SH']}).dropna()
data_002475 = pd.DataFrame({'Volume': vol_df['002475.SZ'], 'Amount': amount_df['002475.SZ'], 'Close': close_df['002475.SZ']}).dropna()

common_idx = data_000001.index.intersection(data_002475.index)
data_000001 = data_000001.loc[common_idx]
data_002475 = data_002475.loc[common_idx]

print(f"共同交易日数量: {len(common_idx)}")

vec_000001 = data_000001[['Volume', 'Amount']].values
vec_002475 = data_002475[['Volume', 'Amount']].values

# 分别对 Volume 和 Amount 进行归一化 (Min-Max到[0,1])
vol_min_000001, vol_max_000001 = vec_000001[:, 0].min(), vec_000001[:, 0].max()
vol_min_002475, vol_max_002475 = vec_002475[:, 0].min(), vec_002475[:, 0].max()
amt_min_000001, amt_max_000001 = vec_000001[:, 1].min(), vec_000001[:, 1].max()
amt_min_002475, amt_max_002475 = vec_002475[:, 1].min(), vec_002475[:, 1].max()

vec_000001_norm = np.column_stack([
    (vec_000001[:, 0] - vol_min_000001) / (vol_max_000001 - vol_min_000001),
    (vec_000001[:, 1] - amt_min_000001) / (amt_max_000001 - amt_min_000001)
])
vec_002475_norm = np.column_stack([
    (vec_002475[:, 0] - vol_min_002475) / (vol_max_002475 - vol_min_002475),
    (vec_002475[:, 1] - amt_min_002475) / (amt_max_002475 - amt_min_002475)
])

print(f"Volume 000001 范围: [{vol_min_000001:.2e}, {vol_max_000001:.2e}]")
print(f"Volume 002475 范围: [{vol_min_002475:.2e}, {vol_max_002475:.2e}]")
print(f"Amount 000001 范围: [{amt_min_000001:.2e}, {amt_max_000001:.2e}]")
print(f"Amount 002475 范围: [{amt_min_002475:.2e}, {amt_max_002475:.2e}]")
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

for i in range(len(common_idx)):
    u = vec_002475_norm[i]  # 归一化后的002475
    v = vec_000001_norm[i]  # 归一化后的000001
    proj = project_u_onto_v(u, v)
    residual = u - proj
    projections.append(proj)
    residuals.append(residual)
    dot_products_after.append(np.dot(residual, v))
    proj_coefficients.append(np.dot(u, v) / np.dot(v, v))
    proj_magnitudes.append(np.linalg.norm(proj))

projections = np.array(projections)
residuals = np.array(residuals)
dot_products_after = np.array(dot_products_after)
proj_coefficients = np.array(proj_coefficients)
proj_magnitudes = np.array(proj_magnitudes)

# ============== 图形显示 ==============

# 图1: 所有日期的向量对比散点图 (归一化后)
fig1 = go.Figure()

# 000001向量 (归一化)
fig1.add_trace(go.Scatter(
    x=vec_000001_norm[:, 0], y=vec_000001_norm[:, 1],
    mode='markers', name='000001.SH (上证指数)',
    marker=dict(color='blue', size=6, opacity=0.7)
))

# 002475向量 (归一化)
fig1.add_trace(go.Scatter(
    x=vec_002475_norm[:, 0], y=vec_002475_norm[:, 1],
    mode='markers', name='002475.SZ (立讯精密)',
    marker=dict(color='red', size=6, opacity=0.7)
))

fig1.update_layout(
    title='Volume-Amount 二维空间向量分布 (Min-Max归一化)',
    xaxis_title='Volume (normalized)',
    yaxis_title='Amount (normalized)',
    template='plotly_dark',
    height=600, width=800
)
fig1.write_html('backtrace/vector_scatter.html')

# 图2: 投影验证 - 选取几个代表性日期
fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f'{str(common_idx[i])[:10]} 投影验证' for i in [0, 50, 100, 174]],
    horizontal_spacing=0.15, vertical_spacing=0.15
)

sample_indices = [0, 50, 100, 174]
for idx, (si, row, col) in enumerate(zip(sample_indices, [1,1,2,2], [1,2,1,2])):
    u = vec_002475_norm[si]
    v = vec_000001_norm[si]
    proj = projections[si]
    residual = residuals[si]

    # 原点到v (000001)
    fig2.add_trace(go.Scatter(
        x=[0, v[0]], y=[0, v[1]],
        mode='lines+markers', name='v (000001)' if idx==0 else None,
        line=dict(color='blue', width=3), marker=dict(size=8)
    ), row=row, col=col)

    # 原点到u (002475)
    fig2.add_trace(go.Scatter(
        x=[0, u[0]], y=[0, u[1]],
        mode='lines+markers', name='u (002475)' if idx==0 else None,
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
    title='002475.SZ → 000001.SH 投影分解 (正交验证)',
    template='plotly_dark',
    height=700, width=900,
    showlegend=True
)
fig2.write_html('backtrace/projection_verify.html')

# 图3: 正交性时序图
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
fig3.update_layout(
    title='正交性验证: (u - proj) · v 应为 0',
    xaxis_title='日期', yaxis_title='点积值',
    template='plotly_dark', height=400
)
fig3.write_html('backtrace/orthogonality_check.html')

# 图4: 投影函数图形
# 4a: 投影系数时序图
fig4a = go.Figure()
fig4a.add_trace(go.Scatter(
    x=list(common_idx), y=proj_coefficients,
    mode='lines', name='投影系数',
    line=dict(color='green')
))
fig4a.update_layout(
    title='投影系数时序 (002475→000001)',
    xaxis_title='日期', yaxis_title='系数 (u·v / v·v)',
    template='plotly_dark', height=300
)
fig4a.write_html('backtrace/proj_coefficient.html')

# 4b: 投影向量模长时序图 (叠加Close收盘价)
close_002475 = data_002475['Close'].values
close_002475_norm = (close_002475 - close_002475.min()) / (close_002475.max() - close_002475.min())
proj_magnitudes_scaled = proj_magnitudes * (close_002475.max() / proj_magnitudes.max())
residual_magnitudes = np.linalg.norm(residuals, axis=1)

fig4b = go.Figure()
fig4b.add_trace(go.Scatter(
    x=list(common_idx), y=proj_magnitudes,
    mode='lines', name='投影向量模长',
    line=dict(color='purple')
))
fig4b.add_trace(go.Scatter(
    x=list(common_idx), y=close_002475_norm * proj_magnitudes.max(),
    mode='lines', name='Close收盘价 (归一化到模长范围)',
    line=dict(color='orange'),
    opacity=0.7
))
fig4b.update_layout(
    title='投影向量模长时序 (叠加Close收盘价)',
    xaxis_title='日期', yaxis_title='模长 / 收盘价(归一化)',
    template='plotly_dark', height=300
)
fig4b.write_html('backtrace/proj_magnitude.html')

# 4d: 投影垂直向量模长时序图 (叠加Close收盘价)
fig4d = go.Figure()
fig4d.add_trace(go.Scatter(
    x=list(common_idx), y=residual_magnitudes,
    mode='lines', name='垂直分量模长 (residual)',
    line=dict(color='red')
))
fig4d.add_trace(go.Scatter(
    x=list(common_idx), y=close_002475_norm * residual_magnitudes.max(),
    mode='lines', name='Close收盘价 (归一化到模长范围)',
    line=dict(color='orange'),
    opacity=0.7
))
fig4d.update_layout(
    title='投影垂直分量模长时序 (叠加Close收盘价)',
    xaxis_title='日期', yaxis_title='模长 / 收盘价(归一化)',
    template='plotly_dark', height=300
)
fig4d.write_html('backtrace/residual_magnitude.html')

# 4c: 投影函数3D曲面 (u在单位圆上变化，v固定为(1,0))
theta = np.linspace(0, 2*np.pi, 100)
u_unit_circle = np.column_stack([np.cos(theta), np.sin(theta)])
v_fixed = np.array([1.0, 0.0])  # 固定v方向

proj_on_fixed_v = np.array([project_u_onto_v(u, v_fixed) for u in u_unit_circle])
proj_mags = np.linalg.norm(proj_on_fixed_v, axis=1)

fig4c = go.Figure()
fig4c.add_trace(go.Scatter(
    x=np.cos(theta), y=np.sin(theta),
    mode='lines', name='u (单位圆)',
    line=dict(color='red', width=2)
))
fig4c.add_trace(go.Scatter(
    x=proj_on_fixed_v[:, 0], y=proj_on_fixed_v[:, 1],
    mode='lines', name='proj(u->v)',
    line=dict(color='green', width=2)
))
# v方向
fig4c.add_trace(go.Scatter(
    x=[0, v_fixed[0]], y=[0, v_fixed[1]],
    mode='lines+markers', name='v 方向',
    line=dict(color='blue', width=3), marker=dict(size=10)
))
fig4c.update_layout(
    title='投影函数: u在单位圆上变化, v=(1,0)固定',
    xaxis_title='x', yaxis_title='y',
    template='plotly_dark', height=600, width=700,
    xaxis=dict(range=[-1.5, 1.5]), yaxis=dict(range=[-1.5, 1.5])
)
fig4c.write_html('backtrace/proj_function.html')

print("\n图形已生成:")
print("  1. backtrace/vector_scatter.html      - Volume-Amount向量散点图")
print("  2. backtrace/projection_verify.html  - 投影分解验证图")
print("  3. backtrace/orthogonality_check.html - 正交性时序检验图")
print("  4. backtrace/proj_coefficient.html    - 投影系数时序图")
print("  5. backtrace/proj_magnitude.html      - 投影向量模长时序图")
print("  6. backtrace/proj_function.html      - 投影函数可视化")
print("  7. backtrace/residual_magnitude.html  - 投影垂直分量模长时序图")

# 保存CSV
result_df = pd.DataFrame({
    'Date': common_idx,
    'Vol_000001_raw': vec_000001[:, 0],
    'Amt_000001_raw': vec_000001[:, 1],
    'Vol_002475_raw': vec_002475[:, 0],
    'Amt_002475_raw': vec_002475[:, 1],
    'Vol_000001_norm': vec_000001_norm[:, 0],
    'Amt_000001_norm': vec_000001_norm[:, 1],
    'Vol_002475_norm': vec_002475_norm[:, 0],
    'Amt_002475_norm': vec_002475_norm[:, 1],
    'Proj_Vol': projections[:, 0],
    'Proj_Amt': projections[:, 1],
    'Residual_Vol': residuals[:, 0],
    'Residual_Amt': residuals[:, 1],
    'Dot_After_Proj': dot_products_after,
    'Norm_Params': [f"vol_000001:[{vol_min_000001:.2e},{vol_max_000001:.2e}] amt_000001:[{amt_min_000001:.2e},{amt_max_000001:.2e}] vol_002475:[{vol_min_002475:.2e},{vol_max_002475:.2e}] amt_002475:[{amt_min_002475:.2e},{amt_max_002475:.2e}]"] * len(common_idx)
})
result_df.to_csv('backtrace/projection_result.csv', index=False, encoding='utf-8')
print("\n数据已保存到 backtrace/projection_result.csv")

tq.close()