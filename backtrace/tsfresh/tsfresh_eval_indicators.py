# -*- coding: utf-8 -*-
# tsfresh 指标评测:Phase 1 - IC(信息系数)快速评估
# 对 SW2 行业并集 每只股票,每个 vbt 指标计算"指标值 vs 未来 5 日收益"的 Spearman rank IC
# 输出:每个指标的板块 IC 中位数 / IC_IR / 胜率 / Top 5 表现最好股票 / Top 3 IC 分布
# 输出文件:tsfresh_indicator_ic_<sector>_<start>_<end>.csv + _summary.csv
# 用法:`PYTHONIOENCODING=utf-8 python backtrace/tsfresh/tsfresh_eval_indicators.py`
#
# 数据源:本地 data/ 缓存(由 fetch_daily.py 落盘,不依赖 TQ 客户端)。
#   原版用 '通达信88' 板块,本地缓存只有 128 申万二级行业 + 它们的成分股并集 ~5000 只;
#   这里用 SW2 行业成分股并集(data/sw2/union.csv)作为「全市场近似」 universe。
#   之前的 2-只 fallback(LOCAL_FALLBACK_CODES)对 IC 统计毫无意义,这里直接绕开。
#
# 实现要点:
# - 先在**连续时间轴**上算 fwd_ret,再对涨跌停日的指标值置 NaN(不删行);
#   删行会让 pct_change(H) 跨过缺失日期,实际跨度 > H 自然日,污染 IC
# - Spearman rank IC(对极端值鲁棒)+ p < 0.05 显著性过滤(不显著不采信)
# - 涨跌停过滤(|daily_ret| >= 9.5%)避免停牌/复牌/涨停样本污染 IC
# - OBV 改为 20 日 rolling z-score,跨股票 IC 才有可比性
# - VWAP 改为价格相对 VWAP 的偏离百分比(vwap_dev),跨股票量纲一致
# - 聚合加 IC_IR(mean / std),> 0.5 算稳健(注:Phase 1 是截面 IC,严格说 IR 应在时序上算)
import warnings
warnings.filterwarnings('ignore')

import sys, os
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import data_store
from common import tsfresh_config as C

# ============== 配置 ==============
SECTOR_NAME   = 'SW2_并集'   # 128 申万二级行业成分股并集(原 '通达信88' 的本地近似)
TARGET_START  = '20250101'
TARGET_END    = datetime.now().strftime('%Y%m%d')   # 动态取「当天」;df.loc 切片自然截到 data/ 最新日
HORIZON       = 5
SW2_UNION_CSV = os.path.join(C.DATA_DIR, 'sw2', 'union.csv')
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

    # OBV — 原始是 cumsum,量纲随时间增长,跨股票 IC 不可比
    # 改为 20 日 rolling z-score,标准化后才进 IC 评估
    sign = np.sign(c.diff()).fillna(0)
    obv_raw = (sign * v).cumsum()
    obv_mu = obv_raw.rolling(20).mean()
    obv_sd = obv_raw.rolling(20).std()
    out['obv'] = ((obv_raw - obv_mu) / (obv_sd + 1e-9)).ffill()

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
    vwap_raw = (out['amount'] / v_safe).ffill()      # 近似 VWAP(价格量级,跨股票不可比)
    # 用 vwap_dev(价格相对 VWAP 的偏离百分比)而不是裸 VWAP:
    # 5 元股 vs 500 元股的裸 VWAP 量纲差 100 倍;vwap_dev 反映「当前价相对
    # 当日均价偏离多少」,跨股票量纲一致,IC 才有意义
    out['vwap_dev'] = ((c - vwap_raw) / vwap_raw.replace(0, np.nan)).ffill()

    return out


INDICATOR_NAMES = ['ma5', 'ma10', 'ma20', 'rsi14', 'atr14', 'bband_pctb',
                   'macd', 'stoch_k', 'obv', 'mfi14', 'adx14',
                   'amount', 'vwap_dev']


# ---------- 2. 单只股票 IC 评估 ----------
def ic_per_stock(df, indicators):
    """对单只股票,返回 {indicator_name: ic_value}。

    实现要点:
    - 先在**连续时间轴**上算 fwd_ret,再对涨跌停日的指标值置 NaN(不删行)
      — 若先删行再算 pct_change(H),被删行的位置会让 H 实际跨度 > H 自然日,污染 IC
    - 涨跌停过滤:只把指标值置 NaN(不影响收益率计算),保留连续时间轴
    - Spearman rank IC(对极端值鲁棒),p < 0.05 才采信
    - Pearson 在涨跌停/复牌/停牌样本下易被一个异常点拉偏
    """
    df_demo = df.loc[TARGET_START:TARGET_END].copy()
    if len(df_demo) < 50:
        return None

    # 1) 先在完整连续时间轴上算前瞻收益率 — 关键时序:fwd_ret 用 df_demo 完整索引
    fwd_ret = df_demo['Close'].pct_change(HORIZON).shift(-HORIZON)

    # 2) 涨跌停过滤:对单日 |return| >= 9.5% 的行,只把指标值置 NaN(不删行)
    daily_ret = df_demo['Close'].pct_change()
    limit_mask = daily_ret.abs() >= 0.095

    ic_dict = {}
    for name in INDICATOR_NAMES:
        if name not in indicators:
            continue
        # reindex 到 df_demo 索引,并对涨跌停日把指标值置 NaN(链式 where 不修改原 indicators)
        ind = indicators[name].reindex(df_demo.index).where(~limit_mask)
        valid = ind.notna() & fwd_ret.notna()
        if valid.sum() < 30:
            ic_dict[name] = np.nan
            continue
        ic, pval = spearmanr(ind[valid].values, fwd_ret[valid].values)
        # 不显著的 IC 不采信(填 NaN,在聚合时 dropna 排除)
        ic_dict[name] = ic if pval < 0.05 else np.nan
    return ic_dict


# ---------- 3. 拉板块 + 跑全 IC ----------
print("=" * 70)
print(f"[{SECTOR_NAME}] 从本地缓存拉全市场近似(带 Amount)...")
if not os.path.exists(SW2_UNION_CSV):
    raise FileNotFoundError(
        f"未找到 {SW2_UNION_CSV} — 请先跑 `PYTHONIOENCODING=utf-8 python "
        f"backtrace/data_fetch/fetch_daily.py` 生成本地缓存"
    )
union = pd.read_csv(SW2_UNION_CSV, dtype={'code': str})
print(f"  union.csv 共 {len(union)} 只")

stock_data = {}
miss = 0
for code in union['code']:
    df = data_store.load_daily(code)
    if df is None or len(df) < 50:
        miss += 1
        continue
    stock_data[code] = df
print(f"  本地命中 {len(stock_data)} 只 (缺失或不足 {miss} 只)")
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
        if i % 100 == 0 or i == total:
            print(f"  [{i:4d}/{total}] 已处理")
    except Exception as e:
        print(f"  [{i:2d}/{total}] {code} 失败:{e}")

df_ic = pd.DataFrame(results)
print(f"\n有效股票:{len(df_ic)} / {total}")

# ---------- 4. 聚合每个指标的板块 IC ----------
print("\n" + "=" * 70)
print(f"=== 11 个指标板块 IC 评估({TARGET_START} ~ {TARGET_END}, HORIZON=5 日) ===")
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
    # IC_IR = mean / std,衡量 IC 时序稳定性(Phase 1 是截面 IC,严格说跨时间不独立,
    # 但仍是衡量「是否大多数股票都给出一致方向」的便捷指标;IC_IR > 0.5 算稳健)
    ic_ir = ic_series.mean() / (ic_series.std() + 1e-9)
    agg_rows.append({
        'indicator':     name,
        'n':             len(ic_series),
        'ic_median':     ic_series.median(),
        'ic_mean':       ic_series.mean(),
        'ic_std':        ic_series.std(),
        'ic_ir':         ic_ir,
        'ic_pos_count':  int(pos),
        'ic_pos_rate':   pos / len(ic_series),
        'ic_strong_pos': int(strong_pos),
        'ic_strong_neg': int(strong_neg),
    })

agg_df = pd.DataFrame(agg_rows).sort_values('ic_median', ascending=False).reset_index(drop=True)

print(f"\n{'指标':<14} {'样本':>5} {'IC中位数':>10} {'IC均值':>10} {'IC标准差':>10} "
      f"{'IC_IR':>8} {'IC>0':>6} {'胜率':>8} {'|IC|>5%':>10}")
print("-" * 96)
for _, r in agg_df.iterrows():
    print(f"{r['indicator']:<14} {int(r['n']):>5} "
          f"{r['ic_median']:>+10.4f} {r['ic_mean']:>+10.4f} {r['ic_std']:>10.4f} "
          f"{r['ic_ir']:>+8.3f} "
          f"{int(r['ic_pos_count']):>6} {r['ic_pos_rate']:>8.1%} "
          f"+{int(r['ic_strong_pos'])}/-{int(r['ic_strong_neg']):<3}")

# Top 3 指标 Top 5 表现最好股票
print("\n=== Top 3 指标 + Top 5 表现最好股票 ===")
for name in agg_df.head(3)['indicator']:
    top5 = df_ic[['stock', name]].dropna().sort_values(name, ascending=False).head(5)
    print(f"\n[{name}]")
    print(top5.to_string(index=False))

# Top 3 指标 IC 分布(便于一眼看出「是否多数 IC 同号 + 强弱分布」)
print("\n=== Top 3 指标 IC 分布 ===")
for name in agg_df.head(3)['indicator']:
    ic_vals = df_ic[name].dropna()
    if len(ic_vals) == 0:
        continue
    neg = int((ic_vals < 0).sum())
    near0 = int((ic_vals.abs() < 0.05).sum())
    pos = int((ic_vals > 0).sum())
    q25, q50, q75 = ic_vals.quantile([0.25, 0.50, 0.75])
    print(f"\n[{name}] n={len(ic_vals)}")
    print(f"  负IC: {neg}  |IC|<0.05: {near0}  正IC: {pos}")
    print(f"  Q25={q25:+.3f}  Q50={q50:+.3f}  Q75={q75:+.3f}")

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