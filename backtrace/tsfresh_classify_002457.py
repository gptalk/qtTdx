# 用 23 个 tsfresh 显著特征训练分类器 + walk-forward 时序回测 + 板块应用
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute, roll_time_series

# ========================= 配置 =========================
stock_code     = '002475.SZ'
csv_path       = f'backtrace/{stock_code.replace(".", "_")}_daily.csv'
selected_csv   = f'backtrace/tsfresh_selected_{stock_code.replace(".", "_")}.csv'   # 上一步的筛选结果
model_csv      = f'backtrace/tsfresh_model_{stock_code.replace(".", "_")}.csv'

WINDOW         = 30     # 滑窗
HORIZON        = 5      # 标签:窗口结束后 5 日
N_FOLDS        = 5      # walk-forward 折数(数据尾部留 1 折,前 N-1 折递增训练)
# ========================================================

# 1. 读 CSV
raw = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index().reset_index(drop=True)
print(f"{stock_code} 日线 {len(raw)} 行  |  {raw.iloc[0,0]}  ←  CSV 索引(忽略,只看行数)")

# 2. 读取上一步选出的 23 个特征(第一列是无名索引列,真正的特征名在第一列)
selected = pd.read_csv(selected_csv, index_col=0)
feat_cols = selected.index.astype(str).tolist()
print(f"载入显著特征 {len(feat_cols)} 个(FDR=0.05 通过)\n")

# 3. 重新提全量特征(完整 3915 维)
raw['id']  = stock_code
raw['time'] = raw.index
ts = raw[['id', 'time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
    ts[c] = pd.to_numeric(ts[c], errors='coerce')

print("步骤 1/4:roll_time_series 切窗口...")
rolled = roll_time_series(
    ts, column_id='id', column_sort='time',
    max_timeshift=WINDOW - 1, min_timeshift=WINDOW - 1,
    n_jobs=0, disable_progressbar=True,
)
print("步骤 2/4:extract_features(全量,30 秒内)...")
X_all = extract_features(rolled, column_id='id', column_sort='time',
                         n_jobs=0, disable_progressbar=False)
impute(X_all)

# 4. 切回 23 维 + 标签
X = X_all[feat_cols].copy()
close_arr = pd.to_numeric(raw['Close'], errors='coerce').values
y = pd.Series(
    {idx: (1 if close_arr[idx[1] + HORIZON] > close_arr[idx[1]] else 0)
         if idx[1] + HORIZON < len(close_arr) else np.nan
     for idx in X.index},
    index=X.index
).astype(float)
valid = y.notna()
X, y = X.loc[valid], y.loc[valid].astype(int)
print(f"   → 样本 {len(X)} 个  |  正样本占比 {y.mean():.1%}\n")

# 5. Walk-forward 时序回测
#    切法:按时间均分 N_FOLDS 段,fold i 用 [0 : (N-i)*step] 训练,(N-i)*step 测试
#    简化:前 60% 训练,后 40% 滚动测试 5 次
print(f"步骤 3/4:Walk-forward 回测(共 {N_FOLDS} 折,递增训练)...")
n = len(X)
fold_size = n // (N_FOLDS + 1)  # 前 N_FOLDS*fold_size 训练一次,后面 N_FOLDS 段依次测试
# 训练起点 = N_FOLDS*fold_size 之前的全部数据
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

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # 正则化逻辑回归:小样本下比 GBDT 稳
    clf = LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced', random_state=42)
    clf.fit(X_tr_s, y_tr)

    p = clf.predict_proba(X_te_s)[:, 1]
    oof_proba[test_start:test_end] = p

    acc = accuracy_score(y_te, (p > 0.5).astype(int))
    auc = roc_auc_score(y_te, p)
    fold_metrics.append({'fold': i+1, 'train_size': len(X_tr),
                         'test_size': len(X_te), 'acc': acc, 'auc': auc})
    print(f"  Fold {i+1}: 训练 {len(X_tr):3d}  测试 {len(X_te):3d}  "
          f"Acc={acc:.2%}  AUC={auc:.3f}")

# 6. 汇总指标
oof_mask = ~np.isnan(oof_proba)
oof_pred = (oof_proba[oof_mask] > 0.5).astype(int)
y_oof = y.values[oof_mask]
acc_all = accuracy_score(y_oof, oof_pred)
auc_all = roc_auc_score(y_oof, oof_proba[oof_mask])
print(f"\n  OOF 整体:Acc={acc_all:.2%}  AUC={auc_all:.3f}\n")

# 7. 对比基线(随机猜,基线 = 正样本占比 0.5 上下)
baseline = max(y_oof.mean(), 1 - y_oof.mean())
print(f"  随机猜基线 Acc={baseline:.2%}  →  模型提升 {(acc_all-baseline)*100:+.1f} pp")

# 8. 导出
metrics_df = pd.DataFrame(fold_metrics)
metrics_df.to_csv(model_csv, index=False, encoding='utf-8-sig')
print(f"\n各折指标已保存到 {model_csv}")

# 9. 实战应用:用全部数据训模型,对"最新窗口"打分
print(f"\n步骤 4/4:最新窗口打分(给 {stock_code} 当前态势打分)...")
scaler = StandardScaler().fit(X.values)
clf = LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced', random_state=42).fit(
    scaler.transform(X.values), y.values
)
latest = X.iloc[[-1]]
p_now = clf.predict_proba(scaler.transform(latest.values))[0, 1]
print(f"  最新窗口(结束日 = raw 第 {X.index[-1][1]} 根) → 未来 {HORIZON} 日上涨概率 = {p_now:.1%}")
if p_now > 0.6:
    print("  🟢 模型判定:偏多")
elif p_now < 0.4:
    print("  🔴 模型判定:偏空")
else:
    print("  ⚪ 模型判定:中性")
