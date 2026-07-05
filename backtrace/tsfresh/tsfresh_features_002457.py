import sys, os
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)

# 使用 tsfresh 提取个股日线特征(整段历史 -> 1 个样本 -> N 个特征)
# 数据源:本地 CSV → TQ(失败回退)
import warnings
warnings.filterwarnings('ignore')

from common import tsfresh_pipeline as P

STOCK_CODE = '002475.SZ'   # 想跑 002457 时改这里
out_csv    = P.csv_path('features', STOCK_CODE)

# 1. 加载 + long format + 全量特征(整段历史当 1 个样本)
df = P.load_ohlcva(STOCK_CODE, verbose=True)
print(f"{STOCK_CODE} 日线 {len(df)} 行  |  {df.index[0].date()} -> {df.index[-1].date()}\n")

long_df = P.to_long_format(df, id_value=STOCK_CODE)
features = P.extract_window_features(long_df, use_kind=True, roll=False, verbose=True)   # 整段历史当 1 个样本
print(f"\n  -> 特征形状 {features.shape}\n")

# 2. 保存(转置,特征名 -> 值 两列)
out_df = features.T.reset_index()
# 提取特征名(0 列)和 value(1 列)
out_df = out_df.iloc[:, :2]
out_df.columns = ['feature', 'value']
out_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"已保存到 {out_csv}")

# 3. 各通道 |value| 最大的前 8 个特征
print("\n=== 各通道 |value| 最大的前 8 个特征 ===")
for kind in ['open', 'high', 'low', 'close', 'volume']:
    sub = out_df[out_df['feature'].str.startswith(f'{kind}__')].copy()
    sub['abs'] = sub['value'].abs()
    top = sub.sort_values('abs', ascending=False).head(8)[['feature', 'value']]
    print(f"\n--- {kind} ---")
    print(top.to_string(index=False))
