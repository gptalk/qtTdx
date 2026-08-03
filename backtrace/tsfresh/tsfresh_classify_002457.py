import sys, os
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)

# 用 23 个 tsfresh 显著特征训练分类器 + walk-forward 时序回测 + 最新窗口打分
# 输出:tsfresh_model_<code>.csv(每折的训练/测试指标)
# 用途:验证显著特征单独做 LR 是否真有样本外预测力(基线对比)
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from common import tsfresh_config as C
from common import tsfresh_pipeline as P

STOCK_CODE = '002475.SZ'
N_FOLDS    = 5
model_csv  = P.csv_path('model', STOCK_CODE)

# 1. 加载 + 切窗口 + 提特征 + 标签
df = P.load_ohlcva(STOCK_CODE, verbose=True)
print()
long_df = P.to_long_format(df, id_value=STOCK_CODE)
X = P.extract_window_features(long_df, use_kind=True, verbose=True)
y, X = P.make_labels(X, df['Close'].values)
print(f"\n   -> 样本 {len(X)} 个  |  正样本占比 {y.mean():.1%}\n")

# 2. Walk-forward 时序回测
print(f"步骤 3/4:Walk-forward 回测(共 {N_FOLDS} 折,递增训练)...")
n = len(X)
fold_size = n // (N_FOLDS + 1)
train_end = N_FOLDS * fold_size
oof_proba = np.full(n, np.nan)
fold_metrics = []

for i in range(N_FOLDS):
    test_start = train_end + i * fold_size
    test_end   = test_start + fold_size if i < N_FOLDS - 1 else n
    if test_start >= n:
        break
    X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
    X_te, y_te = X.iloc[test_start:test_end], y.iloc[test_start:test_end]

    scaler, clf = P.fit_logreg(X_tr, y_tr, verbose=False)
    p = clf.predict_proba(scaler.transform(X_te.values))[:, 1]
    oof_proba[test_start:test_end] = p

    acc = accuracy_score(y_te, (p > 0.5).astype(int))
    auc = roc_auc_score(y_te, p)
    fold_metrics.append({'fold': i+1, 'train_size': len(X_tr),
                         'test_size': len(X_te), 'acc': acc, 'auc': auc})
    print(f"  Fold {i+1}: 训练 {len(X_tr):3d}  测试 {len(X_te):3d}  "
          f"Acc={acc:.2%}  AUC={auc:.3f}")

# 3. 汇总
oof_mask = ~np.isnan(oof_proba)
y_oof = y.values[oof_mask]
acc_all = accuracy_score(y_oof, (oof_proba[oof_mask] > 0.5).astype(int))
auc_all = roc_auc_score(y_oof, oof_proba[oof_mask])
print(f"\n  OOF 整体:Acc={acc_all:.2%}  AUC={auc_all:.3f}")

baseline = max(y_oof.mean(), 1 - y_oof.mean())
print(f"  随机猜基线 Acc={baseline:.2%}  ->  模型提升 {(acc_all-baseline)*100:+.1f} pp\n")

pd.DataFrame(fold_metrics).to_csv(model_csv, index=False, encoding='utf-8-sig')
print(f"各折指标已保存到 {model_csv}")

# 4. 实战:用全部数据训,对最新窗口打分
print(f"\n步骤 4/4:最新窗口打分(给 {STOCK_CODE} 当前态势打分)...")
scaler, clf = P.fit_logreg(X, y, verbose=False)
latest = X.iloc[[-1]]
p_now = clf.predict_proba(scaler.transform(latest.values))[0, 1]
print(f"  最新窗口(结束日 = raw 第 {X.index[-1][1]} 根) "
      f"-> 未来 {C.HORIZON} 日上涨概率 = {p_now:.1%}")
if p_now > 0.6:
    print("  [BULL] 模型判定:偏多")
elif p_now < 0.4:
    print("  [BEAR] 模型判定:偏空")
else:
    print("  [NEUTRAL] 模型判定:中性")
