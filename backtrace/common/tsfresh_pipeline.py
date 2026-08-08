# -*- coding: utf-8 -*-
"""
tsfresh pipeline 工具函数,被 5 个脚本共用。无 print,verbose 标志控制。

流程/约定/做法:
  1. 拉数据:`load_ohlcva` / `load_sector`(TQ 优先 → 本地 CSV 回退)
  2. 提特征:`to_long_format` → `extract_window_features`
  3. 打标签:`make_labels`(绝对 / 相对大盘二选一)
  4. 筛选特征:`select_relevant`(FDR 多重检验校正)
  5. 训练:`fit_logreg` → 用 `align_window_features` 对齐列后再 predict

输入/输出:
  - 输入:股票代码(code)或板块名(sector_name),TQ SDK 在 import 时按需懒加载
  - 输出:pd.DataFrame(每只票 1 个)或 dict {code: DataFrame};特征矩阵 X、标签 Series y

依赖:
  - tsfresh.extract_features / select_features / impute / roll_time_series
  - sklearn.linear_model.LogisticRegression + StandardScaler
  - C:/new_tdx_mock/PYPlugins/user/tqcenter.tq(运行时)

用法:
  from common import tsfresh_pipeline as P
  df = P.load_ohlcva('600118.SH', verbose=True)
"""
import sys
import os
import traceback
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute, roll_time_series

from common import tsfresh_config as C
from common import data_store


# ==================== 1. TQ 上下文 ====================
def init_tq_path():
    """tq.initialize 要求传脚本路径,__file__ 在 -c 模式或子进程里可能没定义"""
    # 兜底链:`__file__` → `sys.argv[0]` → `cwd`,至少返回一个可解析的绝对路径
    if '__file__' in globals() and __file__:
        return os.path.abspath(__file__)
    return os.path.abspath(sys.argv[0]) if sys.argv else os.getcwd()


# ==================== 2. 数据加载 ====================
def _try_local_csv(code):
    """回退路径:委托给 data_store(仓库根 data/)。

    历史教训:这里曾自己拼 backtrace/{code}_daily.csv,而写入方落在 backtrace/outputs/,
    路径分家导致回退长期静默失效。现在读写共用 data_store.csv_path()。
    """
    return data_store.load_daily(code)


def load_ohlcva(code, lookback_years=None, use_tq=True, verbose=False, include_amount=True):
    """
    TQ 优先 → 失败回退本地 CSV。
    返回 DataFrame(列 Open/High/Low/Close/Volume/Amount,DatetimeIndex)
    include_amount=False 时不取成交额(回退场景:CSV 没 Amount 列时设 False)
    """
    lookback_years = lookback_years or C.LOOKBACK_YEARS
    fields = ['Open', 'High', 'Low', 'Close', 'Volume']
    if include_amount:
        fields.append('Amount')

    if use_tq:
        try:
            sys.path.insert(0, C.TQ_PLUGINS_DIR)
            from tqcenter import tq
            tq.initialize(init_tq_path())
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=lookback_years * 365 + 30)).strftime("%Y%m%d")
            df_real = tq.get_market_data(
                field_list=fields, stock_list=[code],
                start_time=start, end_time=end,
                dividend_type='front', period='1d', fill_data=True,
            )
            out = pd.DataFrame({
                'Open':   pd.to_numeric(df_real['Open'][code],   errors='coerce'),
                'High':   pd.to_numeric(df_real['High'][code],   errors='coerce'),
                'Low':    pd.to_numeric(df_real['Low'][code],    errors='coerce'),
                'Close':  pd.to_numeric(df_real['Close'][code],  errors='coerce'),
                'Volume': pd.to_numeric(df_real['Volume'][code], errors='coerce'),
            }).sort_index()
            if include_amount and 'Amount' in df_real and code in df_real['Amount'].columns:
                out['Amount'] = pd.to_numeric(df_real['Amount'][code], errors='coerce')
            tq.close()
            if verbose:
                print(f"[TQ] {code}  {len(out)} 行  {out.index[0].date()} -> {out.index[-1].date()}")
            return out
        except Exception as e:
            if verbose:
                print(f"[TQ] 拉取失败 ({type(e).__name__}: {e})")
                traceback.print_exc()
                print("[TQ] 回退到本地 CSV")

    local = _try_local_csv(code)
    if local is not None and verbose:
        print(f"[CSV] {code}  {len(local)} 行  {local.index[0].date()} -> {local.index[-1].date()}")
    return local


def load_sector(sector_name=None, lookback_years=None, use_tq=True, verbose=False, include_amount=True):
    """TQ 拉板块所有成员日线 -> {code: DataFrame};失败回退本地 fallback 列表"""
    sector_name = sector_name or C.SECTOR_NAME
    lookback_years = lookback_years or C.LOOKBACK_YEARS
    stock_data = {}
    fields = ['Open', 'High', 'Low', 'Close', 'Volume']
    if include_amount:
        fields.append('Amount')

    if use_tq:
        try:
            sys.path.insert(0, C.TQ_PLUGINS_DIR)
            from tqcenter import tq
            tq.initialize(init_tq_path())
            codes = tq.get_stock_list_in_sector(sector_name) or []
            if not codes:
                raise RuntimeError(f"板块 {sector_name} 拉不到")
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=lookback_years * 365 + 30)).strftime("%Y%m%d")
            df_real = tq.get_market_data(
                field_list=fields, stock_list=codes,
                start_time=start, end_time=end,
                dividend_type='front', period='1d', fill_data=True,
            )
            for c in codes:
                if c in df_real['Close'].columns:
                    row = {
                        'Open':   pd.to_numeric(df_real['Open'][c],   errors='coerce'),
                        'High':   pd.to_numeric(df_real['High'][c],   errors='coerce'),
                        'Low':    pd.to_numeric(df_real['Low'][c],    errors='coerce'),
                        'Close':  pd.to_numeric(df_real['Close'][c],  errors='coerce'),
                        'Volume': pd.to_numeric(df_real['Volume'][c], errors='coerce'),
                    }
                    if include_amount and 'Amount' in df_real and c in df_real['Amount'].columns:
                        row['Amount'] = pd.to_numeric(df_real['Amount'][c], errors='coerce')
                    stock_data[c] = pd.DataFrame(row).sort_index()
            tq.close()
            if stock_data:
                if verbose:
                    extra = " (+Amount)" if include_amount else ""
                    print(f"[TQ] 板块 {sector_name} 拉到 {len(stock_data)} 只{extra}")
                return stock_data
        except Exception as e:
            if verbose:
                print(f"[TQ] 拉板块失败 ({type(e).__name__}: {e})")
                traceback.print_exc()

    for c in C.LOCAL_FALLBACK_CODES:
        local = _try_local_csv(c)
        if local is not None:
            stock_data[c] = local
    if verbose:
        print(f"[CSV] 本地回退拿到 {len(stock_data)} 只")
    return stock_data


# ==================== 3. 特征工程 ====================
def to_long_format(ohlcv_df, channels=None, id_value=0):
    """
    把 OHLCV DataFrame 转 tsfresh long format。
    channels: 要包含的列名列表,None=全 5 个。
    id_value: 该票的 id(单只票可固定 0;多票时改为股票代码字符串)
    返回 DataFrame [id, time, kind, value]
    """
    channels = channels or ['Open', 'High', 'Low', 'Close', 'Volume']
    records = []
    for t_idx, (_, row) in enumerate(ohlcv_df.iterrows()):
        for k in channels:
            records.append((id_value, t_idx, k.lower(), row[k]))
    out = pd.DataFrame(records, columns=['id', 'time', 'kind', 'value'])
    out['value'] = pd.to_numeric(out['value'], errors='coerce')   # coerce→NaN,tsfresh 内部再 impute
    return out


def extract_window_features(long_df, window=None, use_kind=False, roll=True, verbose=True):
    """
    tsfresh 提特征。
    long_df: 来自 to_long_format()
    window: 窗口大小(None=用 config.WINDOW)
    use_kind: 是否多通道(用 column_kind)
    roll: 是否滑动窗口(默认 True;False 时整段历史当 1 样本)
    返回 X: 索引是 id(或 (id, end_t) 元组),列是 tsfresh 特征名

    为什么用 impute:tsfresh 不允许 NaN/Inf,impute 用均值/中位数兜底,
      避免特征矩阵出现整列 NaN 后 select_features 把它当成"零方差"剔除掉。
    """
    window = window or C.WINDOW
    if roll:
        df_in = roll_time_series(
            long_df, column_id='id', column_sort='time',
            max_timeshift=window - 1, min_timeshift=window - 1,
            n_jobs=C.TSFRESH_N_JOBS, disable_progressbar=True,
        )
    else:
        df_in = long_df
    kwargs = dict(column_id='id', column_sort='time',
                  n_jobs=C.TSFRESH_N_JOBS, disable_progressbar=False)
    if use_kind:
        kwargs.update(column_kind='kind', column_value='value')
    X = extract_features(df_in, **kwargs)
    impute(X)
    if verbose:
        if roll:
            n_windows = df_in['id'].nunique()
            print(f"   -> {n_windows} 窗口 x {X.shape[1]} 特征")
        else:
            print(f"   -> 1 样本 x {X.shape[1]} 特征(整段历史)")
    return X


# ==================== 4. 标签构造 ====================
def make_labels(X, close_arr, horizon=None, ref_arr=None, verbose=True):
    """
    根据窗口结束时间生成标签。
    X: 特征矩阵,索引是 (id, end_t)
    close_arr: 主标的 close 数组
    ref_arr: 大盘 close 数组(若提供则用"相对收益"标签)
    horizon: 标签窗口(None=用 config.HORIZON)

    返回 (y: Series, X_filtered: DataFrame)
    """
    horizon = horizon or C.HORIZON
    labels, keep = [], []
    for idx in X.index:
        end_t = idx[1]
        if end_t + horizon >= len(close_arr):
            continue
        if ref_arr is not None and end_t + horizon >= len(ref_arr):
            continue
        fwd_main = close_arr[end_t + horizon] / close_arr[end_t] - 1
        if ref_arr is not None:
            fwd_ref = ref_arr[end_t + horizon] / ref_arr[end_t] - 1
            labels.append(1 if fwd_main > fwd_ref else 0)
        else:
            labels.append(1 if fwd_main > 0 else 0)
        keep.append(idx)
    y = pd.Series(labels, index=keep).astype(int)
    X = X.loc[keep]
    if verbose:
        mode = '相对大盘' if ref_arr is not None else '绝对'
        print(f"   -> 有效样本 {len(y)} 个  |  {mode}上涨 {y.sum()} 个 ({y.mean():.1%})")
    return y, X


# ==================== 5. 特征筛选 ====================
def select_relevant(X, y, fdr_level=None, verbose=True):
    """
    FDR(Benjamini-Hochberg)多重检验校正。
    参数:
      X: 特征矩阵
      y: 0/1 标签
      fdr_level: 显著性阈值,None=用 config.FDR_LEVEL(=0.05)
    返回:显著特征子集 X_sel

    为什么 0 特征自动放宽:样本少(几百行以下)+ 严格 FDR 容易全砍掉,
      此时放宽到 0.20 至少留点特征能训出模型(实战经验值)。
    """
    fdr_level = fdr_level or C.FDR_LEVEL
    X_sel = select_features(X, y, n_jobs=C.TSFRESH_N_JOBS, fdr_level=fdr_level)
    if X_sel.shape[1] == 0 and fdr_level < 0.20:
        if verbose:
            print("   [WARN] 0 特征通过,放宽到 fdr=0.20")
        X_sel = select_features(X, y, n_jobs=C.TSFRESH_N_JOBS, fdr_level=0.20)
    if verbose:
        print(f"   -> 显著特征 {X_sel.shape[1]} 列")
    return X_sel


# ==================== 6. 模型 ====================
def fit_logreg(X, y, verbose=True):
    """
    训练 StandardScaler + 平衡 LogisticRegression。
    参数:同 sklearn 接口
    返回:(scaler, clf)— predict 时记得先 scaler.transform(X)
    """
    scaler = StandardScaler().fit(X.values)
    clf = LogisticRegression(
        C=C.LR_C, max_iter=C.LR_MAX_ITER,
        class_weight=C.LR_CLASS_WEIGHT, random_state=C.LR_RANDOM_STATE,
    ).fit(scaler.transform(X.values), y.values)
    if verbose:
        print(f"   -> 训练样本 {len(X)} 个")
    return scaler, clf


# ==================== 7. 窗口特征列对齐 ====================
def align_window_features(X_win, ref_cols, fill_value=0.0):
    """
    把单只股票的最新窗口特征 X_win(1 行)对齐到训练集 ref_cols 列结构。
    缺失列填 fill_value,多余列丢弃。

    为什么需要:不同股票窗口长度可能不同、个别 tsfresh 特征可能因序列过短或常数被跳过。
    直接用 0 兜底是最简做法,更严谨的做法是用训练集均值或全市场均值填充(留待扩展)。
    """
    return X_win.reindex(columns=ref_cols, fill_value=fill_value)


# ==================== 8. 输出命名 ====================
def csv_path(kind, code):
    """生成 backtrace/outputs/{kind}_{code}.csv 路径(无时间戳,同 code 会覆盖)"""
    return os.path.join(C.OUTPUTS_DIR, f'tsfresh_{kind}_{code.replace(".", "_")}.csv')


def timestamped_csv_path(kind, ext='csv'):
    """生成带时间戳的输出路径 → backtrace/outputs/tsfresh_{kind}_{YYYYMMDD_HHMMSS}.{ext}(不会覆盖历史)"""
    name = f'tsfresh_{kind}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{ext}'
    return os.path.join(C.OUTPUTS_DIR, name)
