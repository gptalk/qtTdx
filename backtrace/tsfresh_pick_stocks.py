# 用训练好的模型对多只股票打分 → 输出选股 CSV
# 数据源:优先 TQ(通达信88) → 自动回退到本地 CSV
import warnings
warnings.filterwarnings('ignore')

import sys
import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute, roll_time_series

# ========================= 配置 =========================
SECTOR_NAME = '通达信88'                    # TQ 起来后,模型会对该板块全部成员打分
LOCAL_FALLBACK_CODES = ['000001.SH', '002475.SZ']  # TQ 起不来时,本地 CSV 兜底列表
TRAIN_CODE  = '002475.SZ'                  # 用哪只股票的数据重训 scaler+clf
WINDOW      = 30
HORIZON     = 5
LOOKBACK_YEARS = 5
USE_TQ      = True                        # False 强制只用本地 CSV
# ========================================================


# ---------- 数据获取(优先 TQ,失败回退本地 CSV) ----------
def fetch_stock_data(lookback_years=5, use_tq=True):
    """
    优先 TQ 拉 SECTOR_NAME 板块日线;失败回退到本地 CSV。
    返回 (stock_data: {code: DataFrame}, codes: list)
    """
    stock_data = {}

    if use_tq:
        try:
            sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')
            from tqcenter import tq
            tq.initialize(__file__)

            sector_codes = tq.get_stock_list_in_sector(SECTOR_NAME) or []
            if not sector_codes:
                raise RuntimeError(f"板块 {SECTOR_NAME} 拉不到成分股")

            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=lookback_years * 365 + 30)).strftime("%Y%m%d")
            df = tq.get_market_data(
                field_list=['Open', 'High', 'Low', 'Close', 'Volume'],
                stock_list=sector_codes,
                start_time=start, end_time=end,
                dividend_type='front', period='1d', fill_data=True,
            )
            for c in sector_codes:
                if c in df['Close'].columns:
                    stock_data[c] = pd.DataFrame({
                        'Open':   pd.to_numeric(df['Open'][c],   errors='coerce'),
                        'High':   pd.to_numeric(df['High'][c],   errors='coerce'),
                        'Low':    pd.to_numeric(df['Low'][c],    errors='coerce'),
                        'Close':  pd.to_numeric(df['Close'][c],  errors='coerce'),
                        'Volume': pd.to_numeric(df['Volume'][c], errors='coerce'),
                    }).sort_index()
            tq.close()
            if stock_data:
                print(f"[TQ] 板块 {SECTOR_NAME} 拉到 {len(stock_data)} 只股票日线 "
                      f"({start}~{end}),回退路径不触发")
                return stock_data, list(stock_data.keys())
            print("[TQ] 接口返回空,回退到本地 CSV")
        except Exception as e:
            print(f"[TQ] 拉取失败({type(e).__name__}: {e}),回退到本地 CSV")

    # 本地 CSV 回退
    base = os.path.dirname(os.path.abspath(__file__))
    for c in LOCAL_FALLBACK_CODES:
        p = os.path.join(base, f'{c.replace(".", "_")}_daily.csv')
        if not os.path.exists(p):
            print(f"[CSV] 缺少 {p},跳过 {c}")
            continue
        stock_data[c] = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    print(f"[CSV] 本地回退拿到 {len(stock_data)} 只股票")
    return stock_data, list(stock_data.keys())


# ---------- 加载训练集 + 重训模型 ----------
print("=" * 70)
stock_data, codes = fetch_stock_data(lookback_years=LOOKBACK_YEARS, use_tq=USE_TQ)
print("=" * 70)

# 加载训练用特征名
selected_csv = f'backtrace/tsfresh_selected_{TRAIN_CODE.replace(".", "_")}.csv'
selected = pd.read_csv(selected_csv, index_col=0)
feat_cols = selected.index.astype(str).tolist()
print(f"载入训练用显著特征 {len(feat_cols)} 个\n")

# 准备训练数据(滚动窗口)
if TRAIN_CODE not in stock_data:
    print(f"❌ 训练集 {TRAIN_CODE} 不在 stock_data 中,无法训练")
    sys.exit(1)

train_raw = stock_data[TRAIN_CODE].reset_index(drop=True)
train_raw['id']   = TRAIN_CODE
train_raw['time'] = train_raw.index
train_ts = train_raw[['id', 'time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
    train_ts[c] = pd.to_numeric(train_ts[c], errors='coerce')

print("重新提训练集滑动窗口特征...")
rolled_train = roll_time_series(
    train_ts, column_id='id', column_sort='time',
    max_timeshift=WINDOW - 1, min_timeshift=WINDOW - 1,
    n_jobs=0, disable_progressbar=True,
)
X_train_all = extract_features(rolled_train, column_id='id', column_sort='time',
                               n_jobs=0, disable_progressbar=False)
impute(X_train_all)
X_train = X_train_all[feat_cols].copy()

close_arr = pd.to_numeric(train_raw['Close'], errors='coerce').values
y_train = pd.Series(
    {idx: (1 if close_arr[idx[1] + HORIZON] > close_arr[idx[1]] else 0)
         if idx[1] + HORIZON < len(close_arr) else np.nan
     for idx in X_train.index}
).dropna().astype(int)
X_train = X_train.loc[y_train.index]

scaler = StandardScaler().fit(X_train.values)
clf = LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced', random_state=42).fit(
    scaler.transform(X_train.values), y_train.values
)
print(f"模型在 {len(X_train)} 个训练样本上重新训练完成\n")
print("=" * 70)


# ---------- 遍历每只股票 → 提最新窗口 → 模型打分 ----------
results = []
for code, raw in stock_data.items():
    if len(raw) < WINDOW + HORIZON:
        print(f"  ⚠️  {code} 样本不足 {WINDOW+HORIZON},跳过")
        continue

    win = raw.iloc[-WINDOW:].copy().reset_index(drop=True)
    win['id']   = code
    win['time'] = win.index
    ts = win[['id', 'time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        ts[c] = pd.to_numeric(ts[c], errors='coerce')

    print(f"提取 {code} 最新 {WINDOW} 日窗口特征...")
    X_win_all = extract_features(ts, column_id='id', column_sort='time',
                                 n_jobs=0, disable_progressbar=False)
    impute(X_win_all)

    # 兜底:训练集里有、当前窗口因 tsfresh 差异缺失的列填 0
    for c in feat_cols:
        if c not in X_win_all.columns:
            X_win_all[c] = 0
    X_win = X_win_all[feat_cols].iloc[[-1]]

    p = clf.predict_proba(scaler.transform(X_win.values))[0, 1]

    cur_close = float(raw['Close'].iloc[-1])

    # 真实未来 5 日收益:最新窗口结束日就是 raw.index[-1],无未来数据 → 留空
    real_fwd_ret = None
    real_fwd_dir = None

    # 已发生回看(过去 5 日):用 iloc[-1-HORIZON] 取最新窗口开始前 5 天
    if len(raw) > HORIZON:
        past5_close = float(raw['Close'].iloc[-1 - HORIZON])
        past5_ret   = (cur_close / past5_close - 1) * 100
        past5_dir   = 'up' if past5_ret > 0 else 'down'
    else:
        past5_ret, past5_dir = None, None

    results.append({
        'stock':         code,
        'win_end_date':  raw.index[-1].strftime('%Y-%m-%d'),
        'cur_close':     round(cur_close, 2),
        'up_proba':      round(p, 4),
        'past_5d_ret':   round(past5_ret, 2) if past5_ret is not None else None,
        'past_5d_dir':   past5_dir,
        'real_fwd_ret':  real_fwd_ret,
        'real_fwd_dir':  real_fwd_dir,
    })


# ---------- 输出 ----------
out_csv = f'backtrace/tsfresh_pick_stocks_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
df = pd.DataFrame(results).sort_values('up_proba', ascending=False).reset_index(drop=True)
print(f"\n" + "=" * 70)
print("=== 选股打分结果(按 up_proba 降序)===")
print("=" * 70)
print(df.to_string(index=False))

df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n已保存到 {out_csv}")

# 命中率回看:基于 past_5d_dir(未来收益尚不可知)
if df['past_5d_dir'].notna().all():
    hit = (df['up_proba'] > 0.5).astype(int).values == \
          (df['past_5d_dir'] == 'up').astype(int).values
    print(f"\n  注:真实未来收益尚未发生,以下用 past_5d_dir 近似评估(窗口内已发生部分):")
    print(f"  模型预测命中率(proba>0.5 -> past_5d_dir=up):"
          f"{hit.sum()}/{len(hit)} = {hit.mean():.1%}")
    print(f"  [WARN] 严格意义上,选股打分要等 HORIZON 日后才能验证。")
