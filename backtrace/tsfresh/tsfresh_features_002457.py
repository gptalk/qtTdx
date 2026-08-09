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
#  2. **OHLCV 先 z-score 标准化**(喂给 tsfresh 之前)— 这是从源头解决跨股票量纲
#     不可比问题,而不是事后在排除列表里打地鼠。标准化后,mean/median/abs_energy/
#     variance/spkt_welch_density/fft_coefficient 都变常量或量纲无关,Top 8 才有
#     机会浮现 autocorrelation/number_peaks 之类描述「形状」的特征
#  3. NaN 必报 — 停牌/常数序列会导致 tsfresh 静默返 NaN,先看到 NaN 比例再谈特征解读
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from common import tsfresh_pipeline as P

STOCK_CODE = '002475.SZ'   # 想跑 002457 时改这里
out_csv    = P.csv_path('features', STOCK_CODE)

# 标准化后的「恒等于常量」特征,Top 8 列表里没必要展示(它们已成常数,无形状信息)。
# zscore 后 mean=0 / sum_values≈0 / variance=1 / abs_energy=N / median≈0;
# fft_coefficient__coeff_0 恒等于 mean,归到这里一起排除。
# 角速度相关的 c3 / time_reversal_asymmetry 在标准化下变成「标准化三次矩 / skewness」,
# **不再量纲敏感** — 不需要排除。
# variation_coefficient = std / mean — 标准化后 mean ≈ 1e-16(浮点噪声),std/mean → 1e16,
# 数学上不稳定,Top 8 里只看会误导。
SCALE_DEPENDENT = (
    'mean', 'median', 'sum_values', 'abs_energy', 'variance',
    'variation_coefficient',
)


def zscore_series(s):
    """单列 z-score 标准化。常数序列(σ=0,如整段停牌期)→ 全 0,但保留 NaN/Inf 位置。
    注:标准化后 mean=0 / variance=1 / sum_values=N 是常数,所以这几个特征不再
    提供信息,但被 tsfresh 提出来也只是冗余 — 不需要专门排除。
    """
    s = s.astype(float)
    mu, sigma = s.mean(), s.std()
    if not np.isfinite(sigma) or sigma == 0:
        return s * 0
    return (s - mu) / sigma


# 1. 加载 + 标准化 + long format + 全量特征(整段历史当 1 个样本)
df = P.load_ohlcva(STOCK_CODE, verbose=True)
print(f"{STOCK_CODE} 日线 {len(df)} 行  |  {df.index[0].date()} -> {df.index[-1].date()}\n")

# **关键**:在 to_long_format 前对每列做 z-score。后续所有「带量纲」特征
# (abs_energy / variance / fft_coefficient / c3 / time_reversal_asymmetry / 等)
# 都会变成「标准化的矩」或「标准化的功率」的相对度量,跨股票可比
df_norm = df.copy()
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df_norm[col] = zscore_series(df[col])

long_df = P.to_long_format(df_norm, id_value=STOCK_CODE)
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
