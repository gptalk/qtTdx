# TALib 形态识别回测 - 全市场验证 (用上证指数代表A股)
# 输出:backtrace/talib_pattern_verify_<code>.csv(每形态的 hit / total / win_rate)
# 用法:`python talib/pattern_backtest.py` → 验证 TALib 形态是否真有预测力(单只票上)
import sys
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')
import numpy as np
import pandas as pd
import vectorbt as vbt
import talib
from tqcenter import tq
from datetime import datetime

tq.initialize(__file__)

pd.set_option('future.no_silent_downcasting', True)

# ========================= 配置 =========================
# 使用上证指数代表A股整体市场
stock_code = '000001.SH'
start_time = '20250101'
end_time = datetime.now().strftime("%Y%m%d")
# 形态确认后持有1天，看第2天涨跌概率
hold_days = 1
init_cash = 100000
fees = 0.001
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
print(f"持有天数: {hold_days}天 | 验证形态后第2天涨跌概率")
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


def build_exits_vectorized(entries, hold_days, close_len):
    """向量化构建exit信号"""
    entry_positions = np.where(entries.values)[0]
    if len(entry_positions) == 0:
        return pd.Series(False, index=close.index)

    # 确保exit位置有效且在entry之后
    exit_positions = entry_positions + hold_days
    #过滤掉超出范围的位置
    valid_mask = exit_positions < close_len
    exit_positions = exit_positions[valid_mask]

    if len(exit_positions) == 0:
        return pd.Series(False, index=close.index)

    exits = pd.Series(False, index=close.index)
    exits.iloc[np.unique(exit_positions)] = True
    return exits


results = []

for name, func in pattern_funcs.items():
    try:
        signal = func(openp, high, low, close)
        bullish_count = int((signal == 100).sum())
        bearish_count = int((signal == -100).sum())

        if bullish_count == 0:
            continue

        # 形态确认后次日买入
        entries = (signal == 100).shift(1).fillna(False).infer_objects(copy=False)

        if entries.sum() == 0:
            continue

        # 向量化构建exit信号
        exits = build_exits_vectorized(entries, hold_days, len(close))

        # VectorBT回测
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            price=openp,
            init_cash=init_cash,
            fees=fees,
            freq='D',
            size_granularity=100,
            upon_long_conflict='exit'
        )

        trades = portfolio.trades.records_readable
        stats = portfolio.stats()

        if len(trades) > 0:
            winning = trades[trades['PnL'] > 0]
            losing = trades[trades['PnL'] <= 0]

            win_rate = len(winning) / len(trades) * 100

            # Profit Factor
            total_win = winning['PnL'].sum() if len(winning) > 0 else 0
            total_loss = abs(losing['PnL'].sum()) if len(losing) > 0 else 0
            profit_factor = total_win / total_loss if total_loss > 0 else np.inf

            sharpe = stats.get('Sharpe Ratio', 0)
            sharpe = round(sharpe, 2) if not pd.isna(sharpe) else 0

            results.append({
                'Pattern': name,
                'Bullish': bullish_count,
                'Bearish': bearish_count,
                'Trades': len(trades),
                'Win Rate(%)': round(win_rate, 2),
                'Total Win': round(total_win, 2),
                'Total Loss': round(total_loss, 2),
                'Profit Factor': round(profit_factor, 2) if profit_factor != np.inf else 'inf',
                'Sharpe': sharpe,
                'Total Return(%)': round(stats.get('Total Return [%]', 0), 2),
            })
    except Exception as e:
        print(f"Error in {name}: {e}")

if results:
    results_df = pd.DataFrame(results).sort_values('Win Rate(%)', ascending=False)
    print(f"共 {len(results_df)} 种形态有信号\n")
    print(results_df.to_string(index=False))
    results_df.to_csv(f'backtrace/talib_pattern_verify_000001_SH.csv', index=False)
    print(f"\n结果已保存到 backtrace/talib_pattern_verify_000001_SH.csv")
else:
    print("未发现任何形态信号")

tq.close()