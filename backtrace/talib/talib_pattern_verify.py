# TALib 形态验证 - 检查形态后第2天涨跌概率 (直接计算，不使用vbt portfolio)
import sys
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')
import numpy as np
import pandas as pd
import talib
from tqcenter import tq
from datetime import datetime

tq.initialize(__file__)

pd.set_option('future.no_silent_downcasting', True)

# ========================= 配置 =========================
stock_code = '000001.SH'  # 上证指数代表A股
start_time = '20250101'
end_time = datetime.now().strftime("%Y%m%d")
# ======================================================

# 获取数据
df_real = tq.get_market_data(
    field_list=['Open', 'High', 'Low', 'Close'],
    stock_list=[stock_code],
    start_time=start_time,
    end_time=end_time,
    dividend_type='front',
    period='1d',
    fill_data=True
)

close = tq.price_df(df_real, 'Close', column_names=[stock_code])[stock_code]
openp = tq.price_df(df_real, 'Open', column_names=[stock_code])[stock_code]
high = tq.price_df(df_real, 'High', column_names=[stock_code])[stock_code]
low = tq.price_df(df_real, 'Low', column_names=[stock_code])[stock_code]

print(f"{'='*80}")
print(f"TALib形态验证 - {stock_code} ({start_time} 至 {end_time})")
print(f"验证形态后第2天 (T+1) 涨跌概率")
print(f"{'='*80}\n")

# TALib 形态识别函数列表
pattern_funcs = {
    'CDL2CROWS': talib.CDL2CROWS,
    'CDL3BLACKCROWS': talib.CDL3BLACKCROWS,
    'CDL3INSIDE': talib.CDL3INSIDE,
    'CDL3LINESTRIKE': talib.CDL3LINESTRIKE,
    'CDL3OUTSIDE': talib.CDL3OUTSIDE,
    'CDL3STARSINSOUTH': talib.CDL3STARSINSOUTH,
    'CDL3WHITESOLDIERS': talib.CDL3WHITESOLDIERS,
    'CDLABANDONEDBABY': talib.CDLABANDONEDBABY,
    'CDLADVANCEBLOCK': talib.CDLADVANCEBLOCK,
    'CDLBELTHOLD': talib.CDLBELTHOLD,
    'CDLBREAKAWAY': talib.CDLBREAKAWAY,
    'CDLCLOSINGMARUBOZU': talib.CDLCLOSINGMARUBOZU,
    'CDLCONCEALBABYSWALL': talib.CDLCONCEALBABYSWALL,
    'CDLCOUNTERATTACK': talib.CDLCOUNTERATTACK,
    'CDLDARKCLOUDCOVER': talib.CDLDARKCLOUDCOVER,
    'CDLDOJI': talib.CDLDOJI,
    'CDLDOJISTAR': talib.CDLDOJISTAR,
    'CDLDRAGONFLYDOJI': talib.CDLDRAGONFLYDOJI,
    'CDLENGULFING': talib.CDLENGULFING,
    'CDLEVENINGDOJISTAR': talib.CDLEVENINGDOJISTAR,
    'CDLEVENINGSTAR': talib.CDLEVENINGSTAR,
    'CDLGAPSIDESIDEWHITE': talib.CDLGAPSIDESIDEWHITE,
    'CDLGRAVESTONEDOJI': talib.CDLGRAVESTONEDOJI,
    'CDLHAMMER': talib.CDLHAMMER,
    'CDLHANGINGMAN': talib.CDLHANGINGMAN,
    'CDLHARAMI': talib.CDLHARAMI,
    'CDLHARAMICROSS': talib.CDLHARAMICROSS,
    'CDLHIGHWAVE': talib.CDLHIGHWAVE,
    'CDLHIKKAKE': talib.CDLHIKKAKE,
    'CDLHIKKAKEMOD': talib.CDLHIKKAKEMOD,
    'CDLHOMINGPIGEON': talib.CDLHOMINGPIGEON,
    'CDLIDENTICAL3CROWS': talib.CDLIDENTICAL3CROWS,
    'CDLINNECK': talib.CDLINNECK,
    'CDLINVERTEDHAMMER': talib.CDLINVERTEDHAMMER,
    'CDLKICKING': talib.CDLKICKING,
    'CDLKICKINGBYLENGTH': talib.CDLKICKINGBYLENGTH,
    'CDLLADDERBOTTOM': talib.CDLLADDERBOTTOM,
    'CDLLONGLEGGEDDOJI': talib.CDLLONGLEGGEDDOJI,
    'CDLLONGLINE': talib.CDLLONGLINE,
    'CDLMARUBOZU': talib.CDLMARUBOZU,
    'CDLMATCHINGLOW': talib.CDLMATCHINGLOW,
    'CDLMATHOLD': talib.CDLMATHOLD,
    'CDLMORNINGDOJISTAR': talib.CDLMORNINGDOJISTAR,
    'CDLMORNINGSTAR': talib.CDLMORNINGSTAR,
    'CDLONNECK': talib.CDLONNECK,
    'CDLPIERCING': talib.CDLPIERCING,
    'CDLRICKSHAWMAN': talib.CDLRICKSHAWMAN,
    'CDLRISEFALL3METHODS': talib.CDLRISEFALL3METHODS,
    'CDLSEPARATINGLINES': talib.CDLSEPARATINGLINES,
    'CDLSHOOTINGSTAR': talib.CDLSHOOTINGSTAR,
    'CDLSHORTLINE': talib.CDLSHORTLINE,
    'CDLSPINNINGTOP': talib.CDLSPINNINGTOP,
    'CDLSTALLEDPATTERN': talib.CDLSTALLEDPATTERN,
    'CDLSTICKSANDWICH': talib.CDLSTICKSANDWICH,
    'CDLTAKURI': talib.CDLTAKURI,
    'CDLTASUKIGAP': talib.CDLTASUKIGAP,
    'CDLTHRUSTING': talib.CDLTHRUSTING,
    'CDLTRISTAR': talib.CDLTRISTAR,
    'CDLUNIQUE3RIVER': talib.CDLUNIQUE3RIVER,
    'CDLUPSIDEGAP2CROWS': talib.CDLUPSIDEGAP2CROWS,
    'CDLXSIDEGAP3METHODS': talib.CDLXSIDEGAP3METHODS,
}

# 计算每日涨跌
daily_change = close.pct_change()  # T+1相对于T的涨跌

results = []

for name, func in pattern_funcs.items():
    try:
        signal = func(openp, high, low, close)

        # 形态在T日形成，T+1日信号确认，T+2日检查涨跌
        # bullish信号(T日) -> T+1日买入 -> T+2日检查
        bullish_mask = (signal == 100).values

        if bullish_mask.sum() == 0:
            continue

        # 找到形态出现的位置 (T日)
        pattern_positions = np.where(bullish_mask)[0]

        # 需要T+2存在才能验证
        valid_positions = pattern_positions[pattern_positions + 2 < len(close)]

        if len(valid_positions) == 0:
            continue

        # T+2日的涨跌 (相对于T+1日收盘价)
        #实际上：形态在T日形成，T+1日买入，T+2日收盘时检查
        # 涨跌 = close[T+2] / close[T+1] - 1

        wins = 0
        total = len(valid_positions)

        for pos in valid_positions:
            # T+1日收盘价 (买入价)
            buy_price = close.iloc[pos + 1]
            # T+2日收盘价 (卖出价)
            sell_price = close.iloc[pos + 2]

            if buy_price <= 0 or sell_price <= 0:
                continue

            # T+2相比T+1上涨算赢
            if sell_price > buy_price:
                wins += 1

        win_rate = wins / total * 100 if total > 0 else 0

        results.append({
            'Pattern': name,
            'Bullish_Signals': bullish_mask.sum(),
            'Valid_Trades': total,
            'Win_Count': wins,
            'Lose_Count': total - wins,
            'Win_Rate(%)': round(win_rate, 2),
            'Avg_Change(%)': round((close.iloc[valid_positions + 2].values / close.iloc[valid_positions + 1].values -1).mean() * 100, 2),
        })

    except Exception as e:
        print(f"Error in {name}: {e}")

if results:
    results_df = pd.DataFrame(results).sort_values('Win_Rate(%)', ascending=False)
    print(f"共 {len(results_df)} 种形态有信号\n")
    print(results_df.to_string(index=False))
    results_df.to_csv('backtrace/talib_pattern_verify.csv', index=False)
    print(f"\n结果已保存到 backtrace/talib_pattern_verify.csv")
else:
    print("未发现任何形态信号")

tq.close()