import sys, os, argparse
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)

# 使用 tsfresh 提取个股日线特征 + select_features 筛选显著特征
# 流程:滑动窗口(30日) -> 每窗口提一组特征 -> 标签=窗口结束后5日收益正负 -> select 卡 p 值
# 输出:tsfresh_selected_<code>.csv(FDR 通过的显著特征列 + 样本外相关性)
#
# 用法:`python tsfresh/select_002457.py [--code 002475.SZ]` → 看哪些 tsfresh 特征对未来 5 日涨跌有预测力
#
# 写法注意(按重要性排):
#  1. **train/test 拆分 + purge gap** — 用前 70% 选特征,后 30% 算 OOS 相关性。
#     两件事分开做防两类偏差:
#     a) Selection bias — 同一份数据既筛选又算相关,IS corr 必然虚高。
#     b) **窗口重叠泄漏** — 滑动窗口样本天然重叠(相邻窗口共享 W-1 天原始数据),
#        切分点附近的 train 末样本和 test 首样本共享近整段历史 → 真实数据泄漏,
#        OOS corr 被系统性拉高,purge gap = WINDOW-1 的缓冲区可彻底消除。
#  2. **可抽稀**(THIN_STEP) — 滑动窗口样本重叠带来两个独立问题:
#     a) BH FDR 独立性假设被违反 → FPR 偏高(每个假设不再近似独立)
#     b) 上面提到的窗口重叠泄漏(切分点更明显)
#     设 THIN_STEP=5 让窗口起点步进 5 日,两个问题都能缓解;只有 THIN_STEP >= WINDOW
#     才能彻底消除问题 b。样本量从 ~1200 掉到 ~240,FDR 可能过严到 0 特征 → 默认 1。
#  3. **索引对齐 assert** — make_labels 内部已保证 X.loc[keep] 与 y 同索引,
#     加 assert 是防回归(以后改 make_labels 时这条会先报错)
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from common import tsfresh_config as C
from common import tsfresh_pipeline as P

# ============== CLI 参数 ==============
ap = argparse.ArgumentParser(description='tsfresh 单股特征显著性筛选 + OOS 相关性验证')
ap.add_argument('--code', default='002475.SZ',
                help='股票代码(默认 002475.SZ);支持 600118.SH、300750.SZ 等')
args = ap.parse_args()
STOCK_CODE = args.code
out_csv    = P.csv_path('selected', STOCK_CODE)

# ============== 配置 ==============
SPLIT_RATIO = 0.70   # 前 70% 用于特征筛选,后 30% 用于 OOS 验证
THIN_STEP   = 1      # 滑动窗口抽稀步长;1=不抽稀(原始);5=每 5 日取一个窗口起点
# purge gap 默认从 tsfresh_config.WINDOW 算;若改 C.WINDOW 这里自动跟上
PURGE_GAP   = (C.WINDOW - 1) if THIN_STEP < C.WINDOW else 0
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
#    切分点附近留 PURGE_GAP 的缓冲区(对称地砍两端),保证 train/test 之间零窗口重叠
split_idx = int(len(X) * SPLIT_RATIO)
half_gap  = PURGE_GAP // 2
train_end  = split_idx - half_gap
test_start = split_idx + half_gap

X_train, X_test = X.iloc[:train_end], X.iloc[test_start:]
y_train, y_test = y.iloc[:train_end], y.iloc[test_start:]
print(f"[split] train={len(X_train)}  purge_gap={PURGE_GAP}  test={len(X_test)}")

# 4. FDR 筛选(只在 train 上做 → 避免 selection bias)
X_sel_train = P.select_relevant(X_train, y_train)
selected_cols = X_sel_train.columns.tolist()
n_selected    = len(selected_cols)
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
    print(f"  - 注意:{n_selected} 个 FDR 显著特征里,只看 |corr_out_of_sample| 前 20 才有信号;")
    print(f"          后面 {max(n_selected - 20, 0)} 个排不进榜的特征对模型贡献接近零,建议直接丢弃,别全喂给模型")
elif len(selected_cols) == 0:
    print("[WARN] 0 特征通过,样本量({})可能太少或 fdr 过严".format(len(y_train)))
    print("       建议:换更长历史数据 / 缩短 WINDOW / 调低 fdr_level")
else:
    print("[WARN] test 集为空(样本数 {} < 所需),跳过 OOS 验证".format(len(X)))
