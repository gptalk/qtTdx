# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqcenter import tq
from datetime import datetime, timedelta

tq.initialize(__file__)

# ========================= 配置 =========================
stock_list = ['000001.SH', '002475.SZ']  # 上证指数、002475
days = 240
# ======================================================

end_time = datetime.now().strftime("%Y%m%d")
start_time = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")

print(f"获取最近{days}日日线数据...")
print(f"时间范围: {start_time} - {end_time}")

# 获取数据
df = tq.get_market_data(
    field_list=['Open', 'High', 'Low', 'Close', 'Volume', 'Amount'],
    stock_list=stock_list,
    start_time=start_time,
    end_time=end_time,
    dividend_type='front',
    period='1d',
    fill_data=True
)

open_df = tq.price_df(df, 'Open', column_names=stock_list)
high_df = tq.price_df(df, 'High', column_names=stock_list)
low_df = tq.price_df(df, 'Low', column_names=stock_list)
close_df = tq.price_df(df, 'Close', column_names=stock_list)
vol_df = tq.price_df(df, 'Volume', column_names=stock_list)
amount_df = tq.price_df(df, 'Amount', column_names=stock_list)

for code in stock_list:
    data = pd.DataFrame({
        'Open': open_df[code],
        'High': high_df[code],
        'Low': low_df[code],
        'Close': close_df[code],
        'Volume': vol_df[code],
        'Amount': amount_df[code]
    })
    data = data.dropna().tail(240)

    # 保存CSV
    csv_path = f'backtrace/{code.replace(".", "_")}_daily.csv'
    data.to_csv(csv_path, encoding='utf-8')
    print(f"{code} 数据已保存到 {csv_path}，共{len(data)}条记录")

    # === 绘制K线图 ===
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{code} K线图 (最近240日)', '成交量')
    )

    # K线
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='K线',
            increasing_line_color='#ef5350',
            decreasing_line_color='#26a69a'
        ),
        row=1, col=1
    )

    # 均线 (5, 10, 20)
    for window, color in [(5, 'orange'), (10, 'blue'), (20, 'purple')]:
        ma = data['Close'].rolling(window=window).mean()
        fig.add_trace(
            go.Scatter(x=data.index, y=ma, mode='lines', name=f'MA{window}',
                       line=dict(color=color, width=1)),
            row=1, col=1
        )

    # 成交量
    colors = ['#ef5350' if data['Close'].iloc[i] >= data['Open'].iloc[i] else '#26a69a'
              for i in range(len(data))]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='成交量', marker_color=colors),
        row=2, col=1
    )

    fig.update_layout(
        height=800,
        width=1200,
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )

    html_path = f'backtrace/{code.replace(".", "_")}_kline.html'
    fig.write_html(html_path)
    print(f"K线图已保存到 {html_path}")

print("\n完成！请在浏览器中打开生成的HTML文件查看交互式K线图。")
tq.close()