import sys, os
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)

# 使用 tsfresh 提取个股日线特征(整段历史 -> 1 个样本 -> N 个特征)
# 数据源:本地 CSV → TQ(失败回退)
# 输出:tsfresh_features_<code>.csv(数百列 tsfresh 特征,feature/value 两列)
#      tsfresh_features_<code>_wide.csv(便于多股票横向拼接:一行这只股票,列是特征)
# 用途:EDA / 看看 tsfresh 能提什么;不直接用于交易
#
# 写法注意:
#  1. roll=False — 整段历史当 1 个样本,提取的是「这只股票整个历史阶段的统计画像」
#     (区别于 roll=True 的「每个时间点用过去 N 天窗口提特征 → 时序特征矩阵」)
#  2. Top8 按 |value| 排序时,**先剔除量纲爆炸特征**(abs_energy / sum_values / mean 等),
#     否则「Top 特征」基本都会被「这只股票均价多少」这类量纲特征霸占,看不出形态信号
#  3. NaN 必报 — 停牌/常数序列会导致 tsfresh 静默返 NaN,先看到 NaN 比例再谈特征解读
import warnings
warnings.filterwarnings('ignore')

from common import tsfresh_pipeline as P

STOCK_CODE = '002475.SZ'   # 想跑 002457 时改这里
out_csv    = P.csv_path('features', STOCK_CODE)

# 量纲爆炸特征 — 这类特征数值随股价/成交量的绝对水平缩放,跨股票不可比。
# 展示「Top 重要特征」时先剔除,否则排行榜会被「这只股票均价多少」霸占,看不出形态信号。
#
# 三档量纲敏感度:
#   量纲³(立方级):c3(三阶自相关)、time_reversal_asymmetry_statistic(三次方项)
#   量纲²(平方级):abs_energy、fft_coefficient__coeff_0(恒等于 sum_values)
#   量纲¹(线性级):sum_values、mean、median、maximum、minimum 等
# fft_coefficient 不能简单按前缀整体排除 — attr_"angle" 是相位角(量纲无关,值域 [-π, π]),
#   排除掉可惜;所以下方 _is_scale_dep 单独精确匹配 coeff_0 + abs/real 这两个会重复 sum_values 的特例
SCALE_DEPENDENT = (
    'sum_values', 'abs_energy', 'mean', 'median',
    'maximum', 'minimum', 'sum_of_reoccurring_data_points',
    'sum_of_reoccurring_values',
    'mean_abs_change', 'mean_change',
    'c3',                                    # 三阶自相关,量纲是原始值的三次方
    'time_reversal_asymmetry_statistic',    # 同样含三次方项,volume 通道能炸到 1e22
)

# 1. 加载 + long format + 全量特征(整段历史当 1 个样本)
df = P.load_ohlcva(STOCK_CODE, verbose=True)
print(f"{STOCK_CODE} 日线 {len(df)} 行  |  {df.index[0].date()} -> {df.index[-1].date()}\n")

long_df = P.to_long_format(df, id_value=STOCK_CODE)
features = P.extract_window_features(long_df, use_kind=True, roll=False, verbose=True)   # 整段历史当 1 个样本
print(f"\n  -> 特征形状 {features.shape}\n")

# 2. 保存(转置,特征名 -> 值 两列)
out_df = features.T.reset_index()
out_df = out_df.iloc[:, :2]
out_df.columns = ['feature', 'value']
out_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"已保存到 {out_csv}")

# 2b. 额外存一份宽表(一行 = 一只股票,列 = 特征名,stock 列打头)
#     多股票拼接 / 跨股票横向对比 / 聚类分析时直接 concat 即可,
#     不必每次重新从两列长表 pivot
wide_df = features.reset_index(drop=True)
wide_df.insert(0, 'stock', STOCK_CODE)
wide_out = out_csv.replace('.csv', '_wide.csv')
wide_df.to_csv(wide_out, index=False, encoding='utf-8-sig')
print(f"宽表格式已保存到 {wide_out}(便于多股票拼接)\n")

# 3. NaN 比例盘点 — 停牌 / 常数序列会让 tsfresh 静默返 NaN,先看清
#    比例再谈特征解读;vol=NaN 偏低是常态(停牌期)
print("=== 各通道 NaN 特征数量 ===")
for kind in ['open', 'high', 'low', 'close', 'volume']:
    sub = out_df[out_df['feature'].str.startswith(f'{kind}__')]
    n_nan = sub['value'].isna().sum()
    print(f"  {kind:<8} {n_nan:>4}/{len(sub)} NaN ({n_nan / max(len(sub), 1):.1%})")
print()

# 4. 各通道 |value| 最大的前 8 个特征 — **剔除量纲爆炸** 后
print("=== 各通道 |value| 最大的前 8 个特征(已剔除量纲相关) ===")
def _is_scale_dep(feature_name):
    """feature_name 是 tsfresh 完整名,形如 '{kind}__{feature_func}' 或带参数 '...__attr_<X>__coeff_<N>'"""
    parts = feature_name.split('__')
    if len(parts) < 2:
        return False
    fn = parts[1]
    # 精确排除 fft_coefficient 的 coeff_0(abs/real 同样恒等于 sum_values);保留 angle(相位角,量纲无关)
    if fn == 'fft_coefficient' and len(parts) >= 4:
        attr = parts[2]    # 形如 attr_"abs" / attr_"real" / attr_"angle"
        coeff = parts[3]   # 形如 coeff_0
        if attr in ('attr_"abs"', 'attr_"real"') and coeff == 'coeff_0':
            return True
        return False
    return any(fn.startswith(s) for s in SCALE_DEPENDENT)

for kind in ['open', 'high', 'low', 'close', 'volume']:
    sub = out_df[out_df['feature'].str.startswith(f'{kind}__')].copy()
    sub = sub[~sub['feature'].apply(_is_scale_dep)]
    sub = sub.dropna(subset=['value'])
    sub['abs'] = sub['value'].abs()
    top = sub.sort_values('abs', ascending=False).head(8)[['feature', 'value']]
    print(f"\n--- {kind} ---")
    print(top.to_string(index=False))
