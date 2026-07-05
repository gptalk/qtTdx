import sys, os
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)

# 使用 tsfresh 提取个股日线特征 + select_features 筛选显著特征
# 流程:滑动窗口(30日) -> 每窗口提一组特征 -> 标签=窗口结束后5日收益正负 -> select 卡 p 值
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from common import tsfresh_config as C
from common import tsfresh_pipeline as P

STOCK_CODE = '002475.SZ'
out_csv    = P.csv_path('selected', STOCK_CODE)

# 1. 加载 + 切窗口 + 提特征
df = P.load_ohlcva(STOCK_CODE, verbose=True)
print()
long_df = P.to_long_format(df, id_value=STOCK_CODE)
X = P.extract_window_features(long_df, use_kind=True, verbose=True)
print()

# 2. 构造标签(绝对收益)
y, X = P.make_labels(X, df['Close'].values)
print()

# 3. FDR 筛选
X_sel = P.select_relevant(X, y)
print()

# 4. 保存(特征名 + 与 y 的 Pearson 相关 + 描述统计)
if X_sel.shape[1] > 0:
    corr = X_sel.corrwith(y).rename('pearson_corr_with_y')
    desc = X_sel.describe().T[['mean', 'std', 'min', 'max']]
    summary = pd.concat([corr, desc], axis=1).sort_values(
        'pearson_corr_with_y', key=lambda s: s.abs(), ascending=False
    )
    summary.to_csv(out_csv, encoding='utf-8-sig')
    print(f"=== 显著特征 Top 15(按 |corr(y)| 降序) ===")
    print(summary.head(15).to_string())
    print(f"\n完整结果已保存到 {out_csv}")
else:
    print("[WARN] 0 特征通过,样本量({})可能太少".format(len(y)))
    print("       建议:把 csv_path 换成更长历史的数据,或缩短 WINDOW")
