# tsfresh 多通道联合特征提取:个股 002475 + 大盘 000001 + 相对强弱
# 标签:未来 5 日个股相对大盘是否跑赢
# 数据源:本地 CSV,失败自动 TQ 回退
import warnings
warnings.filterwarnings('ignore')

import sys
import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute, roll_time_series

# ========================= 配置 =========================
STOCK_CODE  = '002475.SZ'
INDEX_CODE  = '000001.SH'           # 上证指数
WINDOW      = 30
HORIZON     = 5
LOOKBACK_YEARS = 5
USE_TQ      = True
out_csv     = f'backtrace/tsfresh_multichannel_{STOCK_CODE.replace(".", "_")}_vs_{INDEX_CODE.replace(".", "_")}.csv'
# ========================================================


# ---------- 数据加载:TQ 优先 → 失败回退本地 CSV ----------
def load_pair(stock_code, index_code, lookback_years=5, use_tq=True):
    """返回 (stock_df, index_df),两者按交易日 inner join,DatetimeIndex"""
    import traceback

    def _try_local(code):
        base = os.path.dirname(os.path.abspath(__file__))
        p = os.path.join(base, f'{code.replace(".", "_")}_daily.csv')
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
            print(f"[CSV] {code}  本地 {len(df)} 行  {df.index[0].date()} → {df.index[-1].date()}")
            return df
        return None

    def _init_path():
        # tq.initialize 要求传脚本路径,__file__ 在 -c 模式下可能没定义
        if '__file__' in globals() and __file__:
            return os.path.abspath(__file__)
        return os.path.abspath(sys.argv[0]) if sys.argv else os.getcwd()

    stock_df = index_df = None

    # ---- TQ 优先 ----
    if use_tq:
        try:
            sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')
            from tqcenter import tq
            tq.initialize(_init_path())
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=lookback_years * 365 + 30)).strftime("%Y%m%d")
            for code in [stock_code, index_code]:
                df_real = tq.get_market_data(
                    field_list=['Open', 'High', 'Low', 'Close', 'Volume'],
                    stock_list=[code], start_time=start, end_time=end,
                    dividend_type='front', period='1d', fill_data=True,
                )
                local = pd.DataFrame({
                    'Open':   pd.to_numeric(df_real['Open'][code],   errors='coerce'),
                    'High':   pd.to_numeric(df_real['High'][code],   errors='coerce'),
                    'Low':    pd.to_numeric(df_real['Low'][code],    errors='coerce'),
                    'Close':  pd.to_numeric(df_real['Close'][code],  errors='coerce'),
                    'Volume': pd.to_numeric(df_real['Volume'][code], errors='coerce'),
                }).sort_index()
                print(f"[TQ] {code}  拉到 {len(local)} 行  {local.index[0].date()} → {local.index[-1].date()}")
                if code == stock_code:
                    stock_df = local
                else:
                    index_df = local
            tq.close()
            return stock_df, index_df
        except Exception as e:
            print(f"[TQ] 拉取失败 ({type(e).__name__}: {e})")
            print("[TQ] 完整 traceback ↓")
            traceback.print_exc()
            print("[TQ] 自动回退到本地 CSV\n")

    # ---- 本地回退 ----
    stock_df = _try_local(stock_code)
    index_df = _try_local(index_code)
    return stock_df, index_df


# ---------- 主流程 ----------
print("=" * 70)
stock_df, index_df = load_pair(STOCK_CODE, INDEX_CODE, LOOKBACK_YEARS, USE_TQ)
if stock_df is None or index_df is None:
    print(f"❌ 缺少 {STOCK_CODE} 或 {INDEX_CODE} 数据,无法继续")
    sys.exit(1)

# 1. 日期对齐 + 构造 6 通道(stock×2 + index×2 + relative×2)
merged = pd.merge(
    stock_df[['Close', 'Volume']].rename(columns={'Close': 'stock_close', 'Volume': 'stock_volume'}),
    index_df[['Close', 'Volume']].rename(columns={'Close': 'idx_close',  'Volume': 'idx_volume'}),
    left_index=True, right_index=True, how='inner',
)
merged['rel_close']  = merged['stock_close'] / merged['idx_close']
merged['rel_volume'] = merged['stock_volume'] / (merged['idx_volume'] + 1)   # 防 0

print(f"\n对齐后交易日 {len(merged)} 行  |  "
      f"{merged.index[0].date()} → {merged.index[-1].date()}")

# 2. long format:每根 bar 6 行 (id, time, kind, value),tsfresh 自动按 kind 分别提特征
records = []
for i, (date, row) in enumerate(merged.iterrows()):
    for kind in ['stock_close', 'stock_volume', 'idx_close', 'idx_volume', 'rel_close', 'rel_volume']:
        records.append((STOCK_CODE, i, kind, row[kind]))

long_df = pd.DataFrame(records, columns=['id', 'time', 'kind', 'value'])
long_df['value'] = pd.to_numeric(long_df['value'], errors='coerce')
print(f"long format 形状:{long_df.shape}\n")

# 3. 滑动窗口
print(f"[1/4] roll_time_series (window={WINDOW})...")
rolled = roll_time_series(
    long_df, column_id='id', column_sort='time',
    max_timeshift=WINDOW - 1, min_timeshift=WINDOW - 1,
    n_jobs=0, disable_progressbar=True,
)
print(f"   → {rolled['id'].nunique()} 个窗口\n")

# 4. 提取特征(6 通道全量)
print("[2/4] extract_features (~1~3 分钟)...")
X_all = extract_features(
    rolled, column_id='id', column_sort='time', column_kind='kind', column_value='value',
    n_jobs=0, disable_progressbar=False,
)
impute(X_all)
print(f"   → 特征矩阵 {X_all.shape} (6 通道 × ~700)\n")

# 5. 标签:窗口结束日后 5 日 个股 是否跑赢 大盘
print(f"[3/4] 构造标签(未来 {HORIZON} 日相对大盘是否跑赢)...")
stock_arr = merged['stock_close'].values
idx_arr   = merged['idx_close'].values
labels = []
keep_idx = []
for idx in X_all.index:
    end_t = idx[1]
    if end_t + HORIZON >= len(merged):
        continue
    stock_fwd = stock_arr[end_t + HORIZON] / stock_arr[end_t] - 1
    idx_fwd   = idx_arr[end_t + HORIZON]   / idx_arr[end_t]   - 1
    labels.append(1 if stock_fwd > idx_fwd else 0)
    keep_idx.append(idx)
y = pd.Series(labels, index=keep_idx).astype(int)
X = X_all.loc[keep_idx]
print(f"   → 有效样本 {len(y)} 个  |  跑赢大盘 {y.sum()} 个  |  占比 {y.mean():.1%}\n")

# 6. 特征筛选(FDR=0.05;若失败放宽到 0.20)
print("[4/4] select_features (FDR=0.05)...")
X_sel = select_features(X, y, n_jobs=0, fdr_level=0.05)
if X_sel.shape[1] == 0:
    print("   ⚠️  0 特征通过,放宽到 FDR=0.20...")
    X_sel = select_features(X, y, n_jobs=0, fdr_level=0.20)
print(f"   → 显著特征 {X_sel.shape[1]} 列\n")

# 7. 训练 LR(用全部样本,因为只是打分,不做 walk-forward)
scaler = StandardScaler().fit(X_sel.values)
clf = LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced', random_state=42).fit(
    scaler.transform(X_sel.values), y.values
)

# 8. 对"最新一个窗口"打分(代表当前择时观点)
latest_idx = X_sel.index[-1]
end_t = latest_idx[1]
latest_features = X_sel.loc[[latest_idx]]
win_end_date = merged.index[end_t]
cur_rel = merged['rel_close'].iloc[end_t]
p_up = clf.predict_proba(scaler.transform(latest_features.values))[0, 1]

# 过去 5 日相对强弱(已发生回看)
past5_rel = merged['rel_close'].iloc[end_t - HORIZON] if end_t >= HORIZON else None
past5_pct = (cur_rel / past5_rel - 1) * 100 if past5_rel else None

result = pd.DataFrame([{
    'stock':         STOCK_CODE,
    'index':         INDEX_CODE,
    'win_end_date':  win_end_date.strftime('%Y-%m-%d'),
    'cur_rel_close': round(float(cur_rel), 4),
    'up_proba':      round(float(p_up), 4),
    'past_5d_rel_ret(%)': round(past5_pct, 2) if past5_pct is not None else None,
    'real_fwd_ret':  None,        # 未来未发生
    'real_fwd_dir':  None,
}])
print("=" * 70)
print("=== 多通道联合择时打分 ===")
print("=" * 70)
print(result.to_string(index=False))
print(f"\n  up_proba>0.5  => 模型判定:未来 {HORIZON} 日 个股相对大盘**跑赢**")

# 9. Top 20 显著特征(看大盘相关因子占比)
print("\n=== Top 20 显著特征(按 |corr(y)| 排序)===")
corr = X_sel.corrwith(y).rename('corr_with_y').sort_values(key=lambda s: s.abs(), ascending=False)
top20 = corr.head(20).reset_index()
top20.columns = ['feature', 'corr_with_y']
print(top20.to_string(index=False))

# 通道分布统计
top20['channel'] = top20['feature'].str.split('__').str[0]
print(f"\nTop 20 中各通道占比:")
print(top20['channel'].value_counts().to_string())

# 10. 保存
result.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到 {out_csv}")

# 训练集 OOF(就是训练 acc,参考用)
y_pred = clf.predict(scaler.transform(X_sel.values))
acc = (y_pred == y.values).mean()
print(f"\n  训练集 acc = {acc:.1%}(参考用,非样本外验证)")

print(f"\n  [WARN] 严格意义上,real_fwd_ret 要等 {HORIZON} 日后才能填,目前留空。")
