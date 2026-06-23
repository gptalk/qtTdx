# tsfresh 多通道联合特征提取:个股 + 大盘 + 相对强弱
# 标签:未来 5 日个股相对大盘是否跑赢
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import tsfresh_config as C
import tsfresh_pipeline as P

STOCK_CODE = '002475.SZ'
INDEX_CODE = '000001.SH'   # 上证指数
out_csv    = f'backtrace/tsfresh_multichannel_{STOCK_CODE.replace(".", "_")}_vs_{INDEX_CODE.replace(".", "_")}.csv'

# 1. 加载个股 + 大盘
print("=" * 70)
stock_df = P.load_ohlcv(STOCK_CODE, verbose=True)
index_df = P.load_ohlcv(INDEX_CODE, verbose=True)
if stock_df is None or index_df is None:
    print(f"[FAIL] 缺少 {STOCK_CODE} 或 {INDEX_CODE} 数据,无法继续")
    raise SystemExit(1)

# 2. 构造 6 通道(stock×2 + index×2 + relative×2)
merged = pd.merge(
    stock_df[['Close', 'Volume']].rename(columns={'Close': 'stock_close', 'Volume': 'stock_volume'}),
    index_df[['Close', 'Volume']].rename(columns={'Close': 'idx_close',  'Volume': 'idx_volume'}),
    left_index=True, right_index=True, how='inner',
)
merged['rel_close']  = merged['stock_close'] / merged['idx_close']
merged['rel_volume'] = merged['stock_volume'] / (merged['idx_volume'] + 1)   # 防 0
print(f"\n对齐后交易日 {len(merged)} 行  |  "
      f"{merged.index[0].date()} -> {merged.index[-1].date()}")

# 3. long format + 滑窗 + 提特征
long_df = P.to_long_format(
    merged,
    channels=['stock_close', 'stock_volume', 'idx_close', 'idx_volume', 'rel_close', 'rel_volume'],
    id_value=STOCK_CODE,
)
print(f"long format 形状:{long_df.shape}\n")
print(f"[1/4] roll_time_series (window={C.WINDOW})...")
X = P.extract_window_features(long_df, use_kind=True, verbose=False)
print(f"   -> 特征矩阵 {X.shape} (6 通道 x ~700)")

# 4. 标签(相对大盘)
print(f"\n[3/4] 构造标签(未来 {C.HORIZON} 日相对大盘是否跑赢)...")
y, X = P.make_labels(
    X, merged['stock_close'].values, ref_arr=merged['idx_close'].values, verbose=False
)
print(f"   -> 有效样本 {len(y)} 个  |  跑赢大盘 {y.sum()} 个 ({y.mean():.1%})")

# 5. FDR 筛选
X_sel = P.select_relevant(X, y)

# 6. 训练 + 最新窗口打分
scaler, clf = P.fit_logreg(X_sel, y, verbose=False)
latest_idx = X_sel.index[-1]
end_t = latest_idx[1]
win_end_date = merged.index[end_t]
cur_rel = merged['rel_close'].iloc[end_t]
p_up = clf.predict_proba(scaler.transform(X_sel.loc[[latest_idx]].values))[0, 1]

past5_rel = merged['rel_close'].iloc[end_t - C.HORIZON] if end_t >= C.HORIZON else None
past5_pct = (cur_rel / past5_rel - 1) * 100 if past5_rel else None

result = pd.DataFrame([{
    'stock':         STOCK_CODE,
    'index':         INDEX_CODE,
    'win_end_date':  win_end_date.strftime('%Y-%m-%d'),
    'cur_rel_close': round(float(cur_rel), 4),
    'up_proba':      round(float(p_up), 4),
    'past_5d_rel_ret(%)': round(past5_pct, 2) if past5_pct is not None else None,
    'real_fwd_ret':  None,
    'real_fwd_dir':  None,
}])
print("=" * 70)
print("=== 多通道联合择时打分 ===")
print("=" * 70)
print(result.to_string(index=False))
print(f"\n  up_proba>0.5  => 模型判定:未来 {C.HORIZON} 日 个股相对大盘**跑赢**")

# 7. Top 20 特征通道分布
print("\n=== Top 20 显著特征(按 |corr(y)| 排序) ===")
corr = X_sel.corrwith(y).rename('corr_with_y').sort_values(key=lambda s: s.abs(), ascending=False)
top20 = corr.head(20).reset_index()
top20.columns = ['feature', 'corr_with_y']
print(top20.to_string(index=False))
top20['channel'] = top20['feature'].str.split('__').str[0]
print(f"\nTop 20 中各通道占比:")
print(top20['channel'].value_counts().to_string())

# 8. 保存
result.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到 {out_csv}")

# 训练集 acc(参考用,非样本外)
y_pred = clf.predict(scaler.transform(X_sel.values))
acc = (y_pred == y.values).mean()
print(f"\n  训练集 acc = {acc:.1%}(参考用,非样本外验证)")
print(f"  [WARN] 严格意义上,real_fwd_ret 要等 {C.HORIZON} 日后才能填,目前留空。")
