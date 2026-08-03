import sys, os
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)

# 用训练好的模型对多只股票打分 -> 输出选股 CSV
# 数据源:优先 TQ(通达信88) -> 自动回退到本地 CSV
# 输出:tsfresh_pick_stocks_YYYYMMDD_HHMMSS.csv(每票一行:proba / 当日 OHLC)
# 用途:研究时看哪些票被模型打高分;不直接用于下单
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from common import tsfresh_config as C
from common import tsfresh_pipeline as P

TRAIN_CODE  = '002475.SZ'   # 用哪只股票的数据重训 scaler+clf
out_csv     = P.timestamped_csv_path('pick_stocks')

# 1. 拉板块数据
print("=" * 70)
stock_data = P.load_sector(verbose=True)
print("=" * 70)
if TRAIN_CODE not in stock_data:
    print(f"[FAIL] 训练集 {TRAIN_CODE} 不在 stock_data 中,无法训练")
    raise SystemExit(1)

# 2. 训练集提特征 + 标签 + 训模型
selected_csv = P.csv_path('selected', TRAIN_CODE)
feat_cols = pd.read_csv(selected_csv, index_col=0).index.astype(str).tolist()
print(f"\n载入训练用显著特征 {len(feat_cols)} 个\n")

train_raw = stock_data[TRAIN_CODE]
long_train = P.to_long_format(train_raw, id_value=TRAIN_CODE)
X_train_all = P.extract_window_features(long_train, use_kind=True, verbose=True)
y_train, X_train = P.make_labels(
    X_train_all[feat_cols], train_raw['Close'].values, verbose=False
)
print(f"   -> 训练样本 {len(X_train)} 个")
scaler, clf = P.fit_logreg(X_train, y_train, verbose=False)
print(f"模型训练完成\n" + "=" * 70)

# 3. 遍历每只股票 -> 提最新窗口 -> 打分
results = []
for code, raw in stock_data.items():
    if len(raw) < C.WINDOW + C.HORIZON:
        print(f"  [WARN] {code} 样本不足 {C.WINDOW+C.HORIZON},跳过")
        continue

    win = raw.iloc[-C.WINDOW:].copy()
    long_win = P.to_long_format(win, id_value=code)
    print(f"提取 {code} 最新 {C.WINDOW} 日窗口特征...")
    X_win_all = P.extract_window_features(long_win, use_kind=True, verbose=False)
    # 兜底:缺失列填 0,保证列结构与训练集一致
    X_win = P.align_window_features(X_win_all, feat_cols).iloc[[-1]]

    p = clf.predict_proba(scaler.transform(X_win.values))[0, 1]
    cur_close = float(raw['Close'].iloc[-1])

    # 未来收益未知 -> 留空;过去 5 日回看
    real_fwd_ret, real_fwd_dir = None, None
    if len(raw) > C.HORIZON:
        past5_close = float(raw['Close'].iloc[-1 - C.HORIZON])
        past5_ret   = (cur_close / past5_close - 1) * 100
        past5_dir   = 'up' if past5_ret > 0 else 'down'
    else:
        past5_ret, past5_dir = None, None

    results.append({
        'stock':         code,
        'win_end_date':  raw.index[-1].strftime('%Y-%m-%d'),
        'cur_close':     round(cur_close, 2),
        'up_proba':      round(p, 4),
        'past_5d_ret':   round(past5_ret, 2) if past5_ret is not None else None,
        'past_5d_dir':   past5_dir,
        'real_fwd_ret':  real_fwd_ret,
        'real_fwd_dir':  real_fwd_dir,
    })

# 4. 输出
df = pd.DataFrame(results).sort_values('up_proba', ascending=False).reset_index(drop=True)
print(f"\n" + "=" * 70)
print("=== 选股打分结果(按 up_proba 降序) ===")
print("=" * 70)
print(df.to_string(index=False))

df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n已保存到 {out_csv}")

# 命中率回看(past_5d_dir 近似;真实未来收益要等 HORIZON 日后)
if df['past_5d_dir'].notna().all():
    hit = (df['up_proba'] > 0.5).astype(int).values == \
          (df['past_5d_dir'] == 'up').astype(int).values
    print(f"\n  注:真实未来收益尚未发生,以下用 past_5d_dir 近似评估:")
    print(f"  模型预测命中率(proba>0.5 -> past_5d_dir=up):"
          f"{hit.sum()}/{len(hit)} = {hit.mean():.1%}")
    print(f"  [WARN] 严格意义上,选股打分要等 HORIZON 日后才能验证。")
