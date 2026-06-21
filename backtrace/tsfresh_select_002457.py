# 使用 tsfresh 提取个股日线特征 + select_features 筛选显著特征
# 流程:滑动窗口(30日) → 每窗口提一组特征 → 标签=窗口结束后5日收益正负 → select_features 卡 p 值
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute, roll_time_series

# ========================= 配置 =========================
stock_code = '002475.SZ'
csv_path   = f'backtrace/{stock_code.replace(".", "_")}_daily.csv'
out_csv    = f'backtrace/tsfresh_selected_{stock_code.replace(".", "_")}.csv'

WINDOW     = 30   # 每个样本用最近 30 个交易日
HORIZON    = 5    # 标签:窗口结束后 5 个交易日的收益
# ========================================================

# 1. 读 CSV → OHLCV
raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)
raw = raw.sort_index()
print(f"{stock_code} 日线 {len(raw)} 行  |  {raw.index[0].date()} → {raw.index[-1].date()}\n")

# 2. 整理成 tsfresh long format,time 用整数序号
raw = raw.reset_index(drop=True)
raw['id'] = stock_code           # 全部属于同一只股票
raw['time'] = raw.index          # 整数时间戳

# 选 5 个通道
ts = raw[['id', 'time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
    ts[c] = pd.to_numeric(ts[c], errors='coerce')

# 3. 用 roll_time_series 把单条长序列切成多个 (id, end_t) 子样本
#    每个子样本长度 = min_timeshift+1 ~ max_timeshift+1 之间的滚动窗口
print(f"步骤 1/3:roll_time_series 切窗口(window={WINDOW})...")
rolled = roll_time_series(
    ts,
    column_id='id',
    column_sort='time',
    max_timeshift=WINDOW - 1,
    min_timeshift=WINDOW - 1,    # 强制固定长度,丢弃前面不足 30 行的窗口
    n_jobs=0,
    disable_progressbar=True,
)
# rolled 的 id 变成 (stock_code, window_end_time) 元组,每个窗口 WINDOW 行
n_windows = rolled['id'].nunique()
print(f"   → 生成 {n_windows} 个滑动窗口样本\n")

# 4. 提取特征(每个窗口 → 一行特征)
print("步骤 2/3:extract_features(约 1~3 分钟)...")
X = extract_features(
    rolled,
    column_id='id',
    column_sort='time',
    n_jobs=0,
    disable_progressbar=False,
)
impute(X)
print(f"   → 特征矩阵 X 形状 = {X.shape}\n")

# 5. 构造标签 y:窗口结束日 close → 5 日后 close 的收益率正负
close_full = pd.to_numeric(raw['Close'], errors='coerce').values

def label_for_window_end(end_t):
    """end_t 是窗口最后一个 bar 的整数 time。看 end_t+HORIZON 是否上涨。"""
    target_t = end_t + HORIZON
    if target_t >= len(close_full):
        return np.nan
    return 1 if close_full[target_t] > close_full[end_t] else 0

# X.index 是 (stock_code, end_t) 二元组
y_raw = pd.Series(
    {idx: label_for_window_end(idx[1]) for idx in X.index},
    index=X.index,
)

# 去掉未来不可知(label=NaN)的窗口
valid = y_raw.notna()
X = X.loc[valid]
y = y_raw.loc[valid].astype(int)
print(f"步骤 3/3:有效样本 {len(y)} 个  |  正样本(未来{HORIZON}日上涨){y.sum()} 个  |"
      f"  正样本占比 {y.mean():.1%}\n")

# 6. select_features:基于 Benjamini-Yekutieli FDR 控制的多重假设检验
print(f"select_features 筛选(原始 {X.shape[1]} 列)...")
X_selected = select_features(X, y, n_jobs=0, fdr_level=0.05)
print(f"   → 显著特征 {X_selected.shape[1]} 列(FDR=0.05)\n")

# 7. 保存:特征名 → 与 y 的 Pearson 相关 + 简单统计
if X_selected.shape[1] == 0:
    print("⚠️  当前样本量太少,5% FDR 下没特征通过。放宽到 fdr_level=0.20 重试...")
    X_selected = select_features(X, y, n_jobs=0, fdr_level=0.20)
    print(f"   → 放宽后显著特征 {X_selected.shape[1]} 列\n")

if X_selected.shape[1] > 0:
    corr = X_selected.corrwith(y).rename('pearson_corr_with_y')
    desc = X_selected.describe().T[['mean', 'std', 'min', 'max']]
    summary = pd.concat([corr, desc], axis=1).sort_values(
        'pearson_corr_with_y', key=lambda s: s.abs(), ascending=False
    )
    summary.to_csv(out_csv, encoding='utf-8-sig')
    print(f"=== 显著特征 Top 15(按 |corr(y)| 降序) ===")
    print(summary.head(15).to_string())
    print(f"\n完整结果已保存到 {out_csv}")
else:
    print("⚠️  即使放宽到 20% FDR 仍没特征通过,样本量({})可能太少。".format(len(y)))
    print("    建议:把 csv_path 换成更长历史的数据,或缩短 WINDOW。")
