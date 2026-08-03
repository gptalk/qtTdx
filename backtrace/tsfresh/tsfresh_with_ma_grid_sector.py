# tsfresh_with_ma 跨通达信88 板块验证
# 对每只股票跑 basic vs with_ma 两套方案,看 with_ma 是否稳定胜出
# 输出:tsfresh_with_ma_grid_<code>_<start>_<end>.csv
# 用法:板块层面跑一遍,验证 with_ma 方案在多数票上是否真的更稳(不是过拟合单只票)
import warnings
warnings.filterwarnings('ignore')

import sys, os, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import numpy as np
import pandas as pd
import vectorbt as vbt
from tqcenter import tq

from common import tsfresh_config as C
from common import tsfresh_pipeline as P
from common import jhzq_fees as F

tq.initialize(__file__)

# ============== 配置 ==============
SECTOR_NAME   = '通达信88'
TARGET_START  = '20250101'
TARGET_END    = '20251231'
INIT_CASH     = 100_000
MAX_POS_PCT   = 0.95

INIT_TRAIN_SIZE = 200
STEP = 50
ENTRY_TSF = 0.60
EXIT_TSF  = 0.50

CHANNELS_BASIC = ['Open', 'High', 'Low', 'Close', 'Volume']
CHANNELS_MA    = CHANNELS_BASIC + ['ma5', 'ma10', 'ma20', 'rel_ma5']
# ===================================


def tsfresh_walkforward_proba(ohlcv_df, channels, label):
    """单只股票跑 tsfresh + walk-forward,返回 proba Series"""
    df_fill = ohlcv_df[channels].copy().fillna(0.0)
    long_df = P.to_long_format(df_fill, channels=channels, id_value=ohlcv_df.name or 'X')
    X_all = P.extract_window_features(long_df, use_kind=True, verbose=False)
    y_all, X_all = P.make_labels(X_all, ohlcv_df['Close'].values, verbose=False)

    X_sel = P.select_relevant(X_all, y_all, verbose=False)
    if X_sel.shape[1] == 0:
        X_sel = X_all   # demo 兜底

    date_index = pd.DatetimeIndex(ohlcv_df.index)
    proba_records = []
    scaler_w = clf_w = None
    for pos, idx in enumerate(X_sel.index):
        end_t = idx[1]
        if end_t >= len(date_index):
            continue
        if pos < INIT_TRAIN_SIZE:
            proba_records.append((date_index[end_t], np.nan))
            continue
        if pos == INIT_TRAIN_SIZE or (pos - INIT_TRAIN_SIZE) % STEP == 0:
            scaler_w, clf_w = P.fit_logreg(X_sel.iloc[:pos], y_all.iloc[:pos], verbose=False)
        p = float(clf_w.predict_proba(scaler_w.transform(X_sel.iloc[[pos]].values))[0, 1])
        proba_records.append((date_index[end_t], p))

    proba = pd.Series([v for _, v in proba_records],
                      index=pd.DatetimeIndex([d for d, _ in proba_records]),
                      name='proba').sort_index()
    return proba[~proba.index.duplicated(keep='last')].dropna()


def run_backtest(ohlcv_df, proba, stock_code):
    """单只股票跑 vbt + jhzq_fees,返回汇总 dict"""
    init_open = float(ohlcv_df['Open'].iloc[0])
    if init_open <= 0:
        return None
    shares = int(np.floor(INIT_CASH * MAX_POS_PCT / init_open / 100) * 100)
    if shares < 100:
        return None

    aligned = proba.reindex(ohlcv_df.index)
    entries = (aligned > ENTRY_TSF).shift(1).fillna(False).astype(bool)
    exits   = (aligned < EXIT_TSF).shift(1).fillna(False).astype(bool)

    pf = vbt.Portfolio.from_signals(
        close=ohlcv_df['Close'], entries=entries, exits=exits,
        price=ohlcv_df['Open'], init_cash=INIT_CASH,
        fees=0, freq='D', size=shares, size_type='amount',
        size_granularity=100, upon_long_conflict='exit',
    )
    trades = pf.trades.records_readable
    if len(trades) == 0:
        return {'stock': stock_code, 'trades': 0, 'net_pnl': 0.0, 'net_ret': 0.0,
                'win_rate': 0.0}

    summary = F.summary_after_fees(trades, stock_code)
    summary['stock'] = stock_code
    pnl_col = next(c for c in trades.columns if 'PnL' in c and '扣' not in c)
    wins = (trades[pnl_col] > 0).sum()
    summary['win_rate'] = wins / len(trades)
    summary['net_ret'] = summary['net_pnl'] / INIT_CASH
    return summary


# ---------- 1. 拉全板块 ----------
print("=" * 70)
print(f"[{SECTOR_NAME}] 拉板块全部成员...")
stock_data = P.load_sector(sector_name=SECTOR_NAME, verbose=True)
print("=" * 70)
if not stock_data:
    print("❌ 板块数据为空"); tq.close(); raise SystemExit(1)

# ---------- 2. 对每只股票跑 2 方案 ----------
results = []
total = len(stock_data)
t_start = time.time()

for i, (code, raw) in enumerate(stock_data.items(), 1):
    df = raw.loc[TARGET_START:TARGET_END].copy()
    if len(df) < 100:
        print(f"  [{i:2d}/{total}] {code} 样本不足 ({len(df)}),跳过")
        continue

    # 算 MA 通道
    for w in [5, 10, 20]:
        df[f'ma{w}'] = vbt.MA.run(df['Close'], window=w).ma.ffill()
    df['rel_ma5'] = (df['Close'] - df['ma5']) / df['ma5'].replace(0, np.nan)
    df.name = code

    print(f"  [{i:2d}/{total}] {code} ({len(df)} 交易日) ...")
    rec = {'stock': code, 'n_days': len(df)}

    # basic
    try:
        proba_basic = tsfresh_walkforward_proba(df, CHANNELS_BASIC, 'basic')
        r_basic = run_backtest(df, proba_basic, code)
        rec['basic_net_ret'] = r_basic['net_ret'] if r_basic else None
        rec['basic_trades']  = r_basic['trades']  if r_basic else 0
        rec['basic_winrate'] = r_basic['win_rate'] if r_basic else None
    except Exception as e:
        print(f"     basic 失败:{e}")
        rec['basic_net_ret'] = None

    # with_ma
    try:
        proba_ma = tsfresh_walkforward_proba(df, CHANNELS_MA, 'with_ma')
        r_ma = run_backtest(df, proba_ma, code)
        rec['with_ma_net_ret'] = r_ma['net_ret'] if r_ma else None
        rec['with_ma_trades']  = r_ma['trades']  if r_ma else 0
        rec['with_ma_winrate'] = r_ma['win_rate'] if r_ma else None
    except Exception as e:
        print(f"     with_ma 失败:{e}")
        rec['with_ma_net_ret'] = None

    rec['with_ma_wins'] = (
        (rec.get('with_ma_net_ret') or -1) > (rec.get('basic_net_ret') or -1)
        if rec.get('with_ma_net_ret') is not None and rec.get('basic_net_ret') is not None
        else None
    )
    results.append(rec)

    elapsed = time.time() - t_start
    eta = elapsed / i * (total - i)
    print(f"     basic={rec.get('basic_net_ret', 'NA')}, with_ma={rec.get('with_ma_net_ret', 'NA')},"
          f" 胜={rec.get('with_ma_wins', 'NA')} | 累计 {elapsed:.0f}s,ETA {eta:.0f}s")

# ---------- 3. 汇总 ----------
df_res = pd.DataFrame(results)
df_res['alpha_diff_pp'] = (df_res['with_ma_net_ret'] - df_res['basic_net_ret']) * 100

print("\n" + "=" * 90)
print("=== 88 板块 basic vs with_ma 汇总 ===")
print("=" * 90)
both_valid = df_res.dropna(subset=['basic_net_ret', 'with_ma_net_ret'])
print(f"有效股票数:{len(both_valid)} / {len(df_res)}")
print(f"with_ma 胜出(>basic):{(both_valid['with_ma_wins'] == True).sum()} 只"
      f"  ({(both_valid['with_ma_wins'] == True).mean():.1%})")
print(f"with_ma 跑赢平均提升:{both_valid['alpha_diff_pp'].mean():+.2f} pp(中位数 {both_valid['alpha_diff_pp'].median():+.2f} pp)")
print(f"\n平均净收益:basic={both_valid['basic_net_ret'].mean():.2%}  with_ma={both_valid['with_ma_net_ret'].mean():.2%}")
print(f"中位净收益:basic={both_valid['basic_net_ret'].median():.2%}  with_ma={both_valid['with_ma_net_ret'].median():.2%}")

# Top 10 with_ma 表现最好
top = both_valid.sort_values('with_ma_net_ret', ascending=False).head(10)
print("\n=== Top 10 with_ma 净收益最高 ===")
print(top[['stock', 'basic_net_ret', 'with_ma_net_ret', 'alpha_diff_pp']].to_string(index=False))

# 保存
out_csv = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/outputs",
    f'tsfresh_with_ma_sector_{SECTOR_NAME}_{TARGET_START}_{TARGET_END}.csv',
)
df_res.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到 {out_csv}")

tq.close()