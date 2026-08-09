import sys, os
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)

# 使用 tsfresh 提取个股日线特征 + select_features 筛选显著特征
# 流程:滑动窗口(30日) -> 每窗口提一组特征 -> 标签=窗口结束后5日收益正负 -> select 卡 p 值
# 输出:tsfresh_selected_<code>.csv(FDR 通过的显著特征列 + 样本外相关性)
#
# 用法:`python tsfresh/select_002457.py` → 看哪些 tsfresh 特征对未来 5 日涨跌有预测力
#
# 写法注意:
#  1. **train/test 拆分** — 用前 70% 选特征,后 30% 算 OOS 相关性
#     否则「筛选后算相关」天然有 selection bias(在筛选集上 corrwith(y) 会偏高)
#  2. **可抽稀**(THIN_STEP) — 滑动窗口样本重叠度高,违反 BH FDR 的独立性假设,
#     设 THIN_STEP=5 让窗口起点步进 5 日,相邻样本不再共线,但样本量会从 ~1200 掉到 ~240,
#     FDR 可能过严掉到 0 特征 → 默认 1(原始行为)
#  3. **索引对齐 assert** — make_labels 内部已保证 X.loc[keep] 与 y 同索引,
#     加 assert 是防回归(以后改 make_labels 时这条会先报错)
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from common import tsfresh_config as C
from common import tsfresh_pipeline as P

STOCK_CODE = '002475.SZ'
out_csv    = P.csv_path('selected', STOCK_CODE)

# ============== 配置 ==============
SPLIT_RATIO = 0.70   # 前 70% 用于特征筛选,后 30% 用于 OOS 验证
THIN_STEP   = 1      # 滑动窗口抽稀步长;1=不抽稀(原始);5=每 5 日取一个窗口起点
# ================================

# 1. 加载 + 切窗口 + 提特征
df = P.load_ohlcva(STOCK_CODE, verbose=True)
print()
long_df = P.to_long_format(df, id_value=STOCK_CODE)
X = P.extract_window_features(long_df, use_kind=True, verbose=True)
print()

# 2. 构造标签(绝对收益)
y, X = P.make_labels(X, df['Close'].values)
# 防回归:make_labels 内部 X.loc[keep] 已经保证对齐,这里加断言挡后续改动
assert X.index.equals(y.index), "X 和 y 时间索引必须完全对齐(make_labels 防回归)"
print()

# 2b. (可选) 抽稀滑动窗口,降低样本重叠度 → BH FDR 独立性假设更稳
if THIN_STEP > 1:
    X = X.iloc[::THIN_STEP]
    y = y.iloc[::THIN_STEP]
    print(f"[thinning] step={THIN_STEP} → 剩余 {len(X)} 样本\n")

# 3. train/test 拆分 — 只用 train 选特征,test 留给 OOS 验证
split_idx = int(len(X) * SPLIT_RATIO)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"[split] train={len(X_train)} (前 {SPLIT_RATIO:.0%})  test={len(X_test)} (后 {1 - SPLIT_RATIO:.0%})")

# 4. FDR 筛选(只在 train 上做 → 避免 selection bias)
X_sel_train = P.select_relevant(X_train, y_train)
selected_cols = X_sel_train.columns.tolist()
print()

# 5. OOS 验证 + 报告
if len(selected_cols) > 0 and len(X_test) > 0:
    X_sel_test = X_test[selected_cols]
    # 两边都算 corrwith(y):train 看 selection bias 的真实幅度,test 看真实预测力
    corr_train = X_sel_train.corrwith(y_train).rename('corr_in_sample')
    corr_test  = X_sel_test.corrwith(y_test).rename('corr_out_of_sample')
    summary = pd.concat([corr_train, corr_test], axis=1)
    summary['corr_decay'] = summary['corr_out_of_sample'] - summary['corr_in_sample']
    # 按 OOS 相关性的绝对值排序 — 找出真正能跨样本预测的特征
    summary = summary.sort_values('corr_out_of_sample', key=lambda s: s.abs(), ascending=False)

    summary.to_csv(out_csv, encoding='utf-8-sig')
    print(f"=== 显著特征 Top 15(按 |corr_out_of_sample| 降序) ===")
    print(summary.head(15).to_string())
    print(f"\n完整结果已保存到 {out_csv}")
    print(f"\n解读:")
    print(f"  - corr_in_sample  = 在筛选集上的相关性(天然有 selection bias,会被拉高)")
    print(f"  - corr_out_of_sample = **真实预测力** — 没参与筛选,跨期,|corr| > 0.05 算有信号")
    print(f"  - corr_decay = OOS - IS(诊断项):")
    print(f"      ≈ 0  : 两期一致,特征稳健")
    print(f"      ≪ 0  : 可能 selection bias 主导(IS 虚高),或 OOS 期市场状态变了")
    print(f"      ≫ 0  : 特征在 OOS 反而更强(说明 IS 因样本噪音没充分体现)")
    print(f"  - 注意:269 个 FDR 显著特征里,只看 |corr_out_of_sample| 前 20 才有信号;")
    print(f"          后面 249 个排不进榜的特征对模型贡献接近零,建议直接丢弃,别全喂给模型")
elif len(selected_cols) == 0:
    print("[WARN] 0 特征通过,样本量({})可能太少或 fdr 过严".format(len(y_train)))
    print("       建议:换更长历史数据 / 缩短 WINDOW / 调低 fdr_level")
else:
    print("[WARN] test 集为空(样本数 {} < 所需),跳过 OOS 验证".format(len(X)))
