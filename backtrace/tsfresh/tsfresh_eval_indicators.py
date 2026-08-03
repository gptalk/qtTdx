# tsfresh 指标评测:Phase 1 - IC(信息系数)快速评估
# 对通达信88 每只股票,每个 vbt 指标计算"指标值 vs 未来 5 日收益"的 Pearson 相关
# 输出:每个指标的板块 IC 中位数 + 胜率 + Top 5 表现最好股票
# 输出文件:tsfresh_indicator_ic_<sector>_<start>_<end>.csv + _summary.csv
# 用法:`python tsfresh/eval_indicators.py` → 11 个指标横向打分,挑有效的往下做
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import numpy as np
import pandas as pd
import vectorbt as vbt
from tqcenter import tq

from common import tsfresh_config as C
from common import tsfresh_pipeline as P

tq.initialize(__file__)

# ============== 配置 ==============
SECTOR_NAME   = '通达信88'
TARGET_START  = '20250101'
TARGET_END    = '20251231'
HORIZON       = 5
# ===================================

# ---------- 1. 定义 11 个指标通道(全部 pandas 手算,绕开 vbt 内部 EMA bug) ----------
def _wilder_smooth(series, period):
    """Wilder 平滑(RSI / ATR 用)"""
    return series.ewm(alpha=1/period, adjust=False).mean()


def build_indicators(df):
    """返回 {name: pd.Series} 字典"""
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    out = {}

    # 趋势类
    for w in [5, 10, 20]:
        out[f'ma{w}'] = c.rolling(w).mean().ffill()

    # RSI 14 (Wilder 平滑)
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = _wilder_smooth(gain, 14)
    avg_loss = _wilder_smooth(loss, 14)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out['rsi14'] = (100 - 100 / (1 + rs)).fillna(50)

    # ATR 14 (Wilder)
    prev_close = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)
    out['atr14'] = _wilder_smooth(tr, 14).ffill()

    # BBANDS %b
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    bb_range = (upper - lower).replace(0, np.nan)
    out['bband_pctb'] = ((c - lower) / bb_range).fillna(0.5)

    # MACD (EMA12 - EMA26)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out['macd'] = (ema12 - ema26).ffill()

    # STOCH %K (14 日)
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    stoch_range = (high14 - low14).replace(0, np.nan)
    out['stoch_k'] = ((c - low14) / stoch_range * 100).fillna(50)

    # OBV
    sign = np.sign(c.diff()).fillna(0)
    out['obv'] = (sign * v).cumsum().ffill()

    # MFI 14
    tp = (h + l + c) / 3
    mf = tp * v
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    neg_mf_safe = neg_mf.replace(0, np.nan)
    mfr = pos_mf / neg_mf_safe
    out['mfi14'] = (100 - 100 / (1 + mfr)).fillna(50)

    # ADX 简化代理(用 ATR / Close * 100,作为趋势强度近似)
    out['adx14'] = (out['atr14'] / c.replace(0, np.nan) * 100).fillna(0)

    # 资金流类(成交额相关,大资金动向)
    if 'Amount' in df.columns:
        out['amount'] = pd.to_numeric(df['Amount'], errors='coerce').ffill()
    else:
        out['amount'] = (c * v).ffill()        # 兜底 = Close × Volume
    v_safe = v.replace(0, np.nan)
    out['vwap'] = (out['amount'] / v_safe).ffill()   # 近似 VWAP

    return out


INDICATOR_NAMES = ['ma5', 'ma10', 'ma20', 'rsi14', 'atr14', 'bband_pctb',
                   'macd', 'stoch_k', 'obv', 'mfi14', 'adx14',
                   'amount', 'vwap']


# ---------- 2. 单只股票 IC 评估 ----------
def ic_per_stock(df, indicators):
    """对单只股票,返回 {indicator_name: ic_value}"""
    df_demo = df.loc[TARGET_START:TARGET_END].copy()
    if len(df_demo) < 50:
        return None

    # 未来 HORIZON 日收益率
    fwd_ret = df_demo['Close'].pct_change(HORIZON).shift(-HORIZON)

    ic_dict = {}
    for name in INDICATOR_NAMES:
        if name not in indicators:
            continue
        ind = indicators[name].reindex(df_demo.index)
        valid = ind.notna() & fwd_ret.notna()
        if valid.sum() < 30:
            ic_dict[name] = np.nan
            continue
        ic = np.corrcoef(ind[valid].values, fwd_ret[valid].values)[0, 1]
        ic_dict[name] = ic
    return ic_dict


# ---------- 3. 拉板块 + 跑全 IC ----------
print("=" * 70)
print(f"[{SECTOR_NAME}] 拉板块全部成员(带 Amount)...")
stock_data = P.load_sector(sector_name=SECTOR_NAME, verbose=True)
print("=" * 70)

results = []
total = len(stock_data)
for i, (code, raw) in enumerate(stock_data.items(), 1):
    try:
        indicators = build_indicators(raw)
        ic_dict = ic_per_stock(raw, indicators)
        if ic_dict is None:
            continue
        rec = {'stock': code}
        rec.update(ic_dict)
        results.append(rec)
        if i % 10 == 0:
            print(f"  [{i:2d}/{total}] 已处理")
    except Exception as e:
        print(f"  [{i:2d}/{total}] {code} 失败:{e}")

df_ic = pd.DataFrame(results)
print(f"\n有效股票:{len(df_ic)} / {total}")

# ---------- 4. 聚合每个指标的板块 IC ----------
print("\n" + "=" * 70)
print("=== 11 个指标板块 IC 评估(2025-01-01 ~ 2025-12-31, HORIZON=5 日) ===")
print("=" * 70)

agg_rows = []
for name in INDICATOR_NAMES:
    if name not in df_ic.columns:
        continue
    ic_series = df_ic[name].dropna()
    if len(ic_series) == 0:
        continue
    pos = (ic_series > 0).sum()
    strong_pos = (ic_series > 0.05).sum()      # |IC| > 5% 算较强
    strong_neg = (ic_series < -0.05).sum()
    agg_rows.append({
        'indicator':     name,
        'n':             len(ic_series),
        'ic_median':     ic_series.median(),
        'ic_mean':       ic_series.mean(),
        'ic_std':        ic_series.std(),
        'ic_pos_count':  int(pos),
        'ic_pos_rate':   pos / len(ic_series),
        'ic_strong_pos': int(strong_pos),
        'ic_strong_neg': int(strong_neg),
    })

agg_df = pd.DataFrame(agg_rows).sort_values('ic_median', ascending=False).reset_index(drop=True)

print(f"\n{'指标':<14} {'样本':>5} {'IC中位数':>10} {'IC均值':>10} {'IC标准差':>10} "
      f"{'IC>0':>6} {'胜率':>8} {'|IC|>5%':>10}")
print("-" * 90)
for _, r in agg_df.iterrows():
    print(f"{r['indicator']:<14} {int(r['n']):>5} "
          f"{r['ic_median']:>+10.4f} {r['ic_mean']:>+10.4f} {r['ic_std']:>10.4f} "
          f"{int(r['ic_pos_count']):>6} {r['ic_pos_rate']:>8.1%} "
          f"+{int(r['ic_strong_pos'])}/-{int(r['ic_strong_neg']):<3}")

# Top 3 指标 Top 5 表现最好股票
print("\n=== Top 3 指标 + Top 5 表现最好股票 ===")
for name in agg_df.head(3)['indicator']:
    top5 = df_ic[['stock', name]].dropna().sort_values(name, ascending=False).head(5)
    print(f"\n[{name}]")
    print(top5.to_string(index=False))

# 保存
out_csv = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/outputs",
    f'tsfresh_indicator_ic_{SECTOR_NAME}_{TARGET_START}_{TARGET_END}.csv',
)
df_ic.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n每只股票 IC 明细已保存到 {out_csv}")

agg_path = out_csv.replace('.csv', '_summary.csv')
agg_df.to_csv(agg_path, index=False, encoding='utf-8-sig')
print(f"板块聚合已保存到 {agg_path}")

tq.close()