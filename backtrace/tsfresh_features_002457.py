# 使用 tsfresh 提取个股日线特征(整段历史 → 1 个样本 → N 个特征)
# 数据源:本地 CSV(因为 TQ 客户端未启动,演示用 002475_SZ;TQ 起来后把 stock_code 改成 002457.SZ 重跑即可)
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

# ========================= 配置 =========================
stock_code = '002475.SZ'                                          # 想跑 002457 时改这里 + 数据源
csv_path   = f'backtrace/{stock_code.replace(".", "_")}_daily.csv'
out_csv    = f'backtrace/tsfresh_features_{stock_code.replace(".", "_")}.csv'
# ========================================================

# 1. 读 CSV → OHLCV 时间序列
raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)
print(f"{stock_code} 日线样本数:{len(raw)}")
print(f"日期范围:{raw.index[0].date()}  ~  {raw.index[-1].date()}\n")

# 2. 整理成 tsfresh "long format":列 = id / time / kind / value
records = []
for kind in ['Open', 'High', 'Low', 'Close', 'Volume']:
    s = raw[kind].reset_index(drop=True)
    for t, v in s.items():
        records.append((stock_code, t, kind.lower(), v))

long_df = pd.DataFrame(records, columns=['id', 'time', 'kind', 'value'])
long_df['value'] = pd.to_numeric(long_df['value'], errors='coerce')

# 3. 提取全量默认特征(每个 kind 约 780 个,5 个 kind 共约 3900 列)
print("=" * 70)
print("开始提取 tsfresh 特征(整段历史 → 1 个样本)...")
print("=" * 70)

features = extract_features(
    long_df,
    column_id='id',
    column_sort='time',
    column_kind='kind',
    column_value='value',
    n_jobs=0,           # 单进程,Windows 下避免 multiprocessing 卡死
    disable_progressbar=False,
)

# 4. 把 inf/-inf/NaN 替换成有限值
impute(features)

print(f"\n" + "=" * 70)
print(f"提取完成:形状 = {features.shape}  (1 行 × {features.shape[1]} 列特征)")
print("=" * 70)

# 5. 保存:转置成"特征名 → 值"两列,便于人工阅读
out_df = features.T.reset_index()
out_df.columns = ['feature', 'value']
out_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n所有特征已保存到 {out_csv}")

# 6. 各通道按"绝对值最大"排序,看一眼每个 kind 最有信息量的前 8 个特征
print("\n=== 各通道 |value| 最大的前 8 个特征 ===")
for kind in ['open', 'high', 'low', 'close', 'volume']:
    sub = out_df[out_df['feature'].str.startswith(f'{kind}__')].copy()
    sub['abs'] = sub['value'].abs()
    top = sub.sort_values('abs', ascending=False).head(8)[['feature', 'value']]
    print(f"\n--- {kind} ---")
    print(top.to_string(index=False))
