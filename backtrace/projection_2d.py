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
    field_list=['Volume', 'Amount'],
    stock_list=stock_list,
    start_time=start_time,
    end_time=end_time,
    dividend_type='front',
    period='1d',
    fill_data=True
)

vol_df = tq.price_df(df, 'Volume', column_names=stock_list)
amount_df = tq.price_df(df, 'Amount', column_names=stock_list)

data_000001 = pd.DataFrame({'Volume': vol_df['000001.SH'], 'Amount': amount_df['000001.SH']}).dropna()
data_002475 = pd.DataFrame({'Volume': vol_df['002475.SZ'], 'Amount': amount_df['002475.SZ']}).dropna()

common_idx = data_000001.index.intersection(data_002475.index)
data_000001 = data_000001.loc[common_idx]
data_002475 = data_002475.loc[common_idx]

print(f"共同交易日数量: {len(common_idx)}")

vec_000001 = data_000001[['Volume', 'Amount']].values
vec_002475 = data_002475[['Volume', 'Amount']].values

# 归一化: 将每个向量 (Volume, Amount) 归一化为单位向量
vec_000001_norm = vec_000001 / np.linalg.norm(vec_000001, axis=1, keepdims=True)
vec_002475_norm = vec_002475 / np.linalg.norm(vec_002475, axis=1, keepdims=True)

print(f"归一化后 000001.SH 向量长度验证: {np.linalg.norm(vec_000001_norm, axis=1)[:3]}")
print(f"归一化后 002475.SZ 向量长度验证: {np.linalg.norm(vec_002475_norm, axis=1)[:3]}")

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

for i in range(len(common_idx)):
    u = vec_002475_norm[i]  # 归一化后的002475
    v = vec_000001_norm[i]  # 归一化后的000001
    proj = project_u_onto_v(u, v)
    residual = u - proj
    projections.append(proj)
    residuals.append(residual)
    dot_products_after.append(np.dot(residual, v))

projections = np.array(projections)
residuals = np.array(residuals)
dot_products_after = np.array(dot_products_after)

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
    title='Volume-Amount 二维空间向量分布 (归一化后)',
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

print("\n图形已生成:")
print("  1. backtrace/vector_scatter.html      - Volume-Amount向量散点图")
print("  2. backtrace/projection_verify.html  - 投影分解验证图")
print("  3. backtrace/orthogonality_check.html - 正交性时序检验图")

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
    'Dot_After_Proj': dot_products_after
})
result_df.to_csv('backtrace/projection_result.csv', index=False, encoding='utf-8')
print("\n数据已保存到 backtrace/projection_result.csv")

tq.close()