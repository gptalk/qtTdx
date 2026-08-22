# -*- coding: utf-8 -*-
"""
Spike: 1-min vs daily dynamics — Nyquist 不足是否成立?

目的(2026-08-22 用户假设):
  "动力学自然周期是日内的(小时级),日线分析可能太粗糙,超过运动周期"

Spike 边界(明确"扔掉什么,保留什么"):
  - 一次性脚本,不接入 common.tsfresh_pipeline / projection._projection_core
  - 不复用 projection 几何(Min-Max 归一化/投影到大盘向量),仅复用 ODE 的 OLS 数学
  - 输入:同 3 只票 + 各自大盘指数,TQ 拉最近 5 个交易日的 1min K 线
  - 计算双尺度的 (k̂, ĉ, ω_n),落到 CSV + 一张 console 表
  - 任何中间代码不视为可保留资产,后续真做迁移时重写

输出:
  - backtrace/outputs/spike_1min_nyquist/fit_summary.csv    (双尺度参数对照)
  - backtrace/outputs/spike_1min_nyquist/omega_compare.csv  (Nyquist vs 估计 ω_n)
  - backtrace/outputs/spike_1min_nyquist/path_compare.csv   (1 步预测残差)

执行:
  PYTHONIOENCODING=utf-8 python backtrace/spikes/spike_1min_nyquist.py
"""
import os
import sys
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import numpy as np
import pandas as pd

from common import data_store  # noqa: F401  (路径解析,即使本脚本不写)
from common import tsfresh_pipeline as P  # noqa: F401  (静默导入,后续可扩展)

from tqcenter import tq

# 1 分钟 spike 范围(刻意小,5 个交易日即可说明问题)
SPIKE_STOCKS = [
    ('600519.SH', '000001.SH', '贵州茅台 / 上证综指'),
    ('000001.SZ', '399001.SZ', '平安银行 / 深证成指'),
    ('300750.SZ', '399001.SZ', '宁德时代 / 深证成指'),
]
N_DAYS = 5   # 5 个交易日 = 5 × 240 = 1200 个 1min 样本/票

OUT_DIR = os.path.join(BACKTRACE_DIR, 'outputs', 'spike_1min_nyquist')
os.makedirs(OUT_DIR, exist_ok=True)


# ============== 数据 ==============
def fetch_1min(code, n_days):
    """TQ 拉最近 n_days 个交易日的 1 分钟 K 线。返回 DataFrame 含 Open/High/Low/Close/Volume/Amount。"""
    end = datetime.now().strftime('%Y%m%d')
    # 多请求 5 天,保证拿满 n_days 个交易日
    start = (datetime.now() - timedelta(days=n_days * 2 + 10)).strftime('%Y%m%d')
    tq.initialize(os.path.abspath(__file__))
    fields = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
    raw = tq.get_market_data(
        field_list=fields, stock_list=[code],
        start_time=start, end_time=end,
        dividend_type='front', period='1m', fill_data=True,
    )
    if raw is None or raw.get('Close') is None or raw['Close'].empty:
        raise RuntimeError(f"TQ 拉 {code} 1min 数据为空")
    out = pd.DataFrame({
        'Open':   pd.to_numeric(raw['Open'][code],   errors='coerce'),
        'High':   pd.to_numeric(raw['High'][code],   errors='coerce'),
        'Low':    pd.to_numeric(raw['Low'][code],    errors='coerce'),
        'Close':  pd.to_numeric(raw['Close'][code],  errors='coerce'),
        'Volume': pd.to_numeric(raw['Volume'][code], errors='coerce'),
    }).dropna(subset=['Close'])
    if 'Amount' in raw and code in raw['Amount'].columns:
        out['Amount'] = pd.to_numeric(raw['Amount'][code], errors='coerce')
    else:
        out['Amount'] = out['Volume'] * out['Close']  # 兜底
    # 只保留最近 n_days 个交易日的索引(按日期)
    out['__date__'] = out.index.date
    last_n_dates = sorted(out['__date__'].unique())[-n_days:]
    out = out[out['__date__'].isin(last_n_dates)].drop(columns='__date__')
    return out


def fetch_daily(code, n_days=None):
    """从本地 data/ 缓存拿最近 n_days 个交易日(已有,只读)。
       n_days=None → 全量缓存(500 天,跟日线生产代码一致)。
    """
    df = P.load_ohlcva(code, use_tq=False, verbose=False)
    if df is None or df.empty:
        raise RuntimeError(f"本地缓存缺 {code},请先跑 fetch_daily.py")
    return df.tail(n_days).copy() if n_days else df.copy()


# ============== ODE 数学(复用 parameter_fit 的形式) ==============
def state_2d(df):
    """把 OHLCV 转成 2D 状态向量(u_S, v_S, u_M, v_M):
       u_S = 标准化成交量(V/V_max), v_S = 标准化成交额(A/A_max)
       对大盘 index 同理。
       Δu_S[t] = u_S[t] - u_S[t-1]  (差分 → 速度)
       Δv_S 同理。
    """
    def norm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)

    return np.column_stack([
        norm(df['Volume'].to_numpy()),
        norm(df['Amount'].to_numpy()),
    ])


def fit_ode_2d(state_S, state_M, beta=1.0):
    """ODE 形式(同 projection.parameter_fit):
       Δu_S[t+1] - β·Δu_M[t+1] = -k·cum_u_S[t] - c·u_S[t] + F_self

       简化版:本 spike 不算 cum/displacement,只看 (Δu_S - β·Δu_M) 对 (u_S, u_M) 的回归。
       即 a = -k·u_S - c·u_M + F_self 的差分近似。

       输入:state_S/state_M 形状 (T, 2),输出 (k, c, F², n_valid, ω_n)
    """
    dS = np.diff(state_S, axis=0)
    dM = np.diff(state_M, axis=0)
    # Y = Δu_S - β·Δu_M,两个维度堆叠
    Y = np.concatenate([dS[:, 0] - beta * dM[:, 0], dS[:, 1] - beta * dM[:, 1]])
    # X = [-u_S_x, -u_M_x; -u_S_y, -u_M_y]
    n = dS.shape[0]
    X = np.zeros((2 * n, 2))
    X[:n, 0] = -state_S[:-1, 0]
    X[:n, 1] = -state_M[:-1, 0]
    X[n:, 0] = -state_S[:-1, 1]
    X[n:, 1] = -state_M[:-1, 1]
    # OLS
    theta, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    if theta.size == 0:
        return dict(k=0, c=0, f2=np.inf, n=n, omega=0, rank=0)
    k, c = float(theta[0]), float(theta[1])
    resid = Y - X @ theta
    f2 = float(np.mean(resid ** 2))
    # ω_n = √(k² - c²) (假设欠阻尼;若 k² < c² 则过阻尼,ω_n = 0)
    disc = max(k ** 2 - c ** 2, 0)
    omega = float(np.sqrt(disc))
    return dict(k=k, c=c, f2=f2, n=n, omega=omega, rank=int(rank))


# ============== 主流程 ==============
def run():
    rows_summary = []
    rows_omega = []
    rows_path = []

    for stock_code, index_code, label in SPIKE_STOCKS:
        print(f"\n========== {label} ==========")
        try:
            df_1m_S = fetch_1min(stock_code, N_DAYS)
            df_1m_M = fetch_1min(index_code, N_DAYS)
            # 公平对照:日线用全量缓存(500 天),不是 5 天
            df_d_S  = fetch_daily(stock_code)
            df_d_M  = fetch_daily(index_code)
        except Exception as e:
            print(f"  [SKIP] 数据拉取失败: {type(e).__name__}: {e}")
            continue

        print(f"  1min: {len(df_1m_S)} 行 / daily: {len(df_d_S)} 行")

        s1 = state_2d(df_1m_S)
        m1 = state_2d(df_1m_M)
        sD = state_2d(df_d_S)
        mD = state_2d(df_d_M)

        # β 估计:线性回归 Δu_S = β·Δu_M  → 用最小二乘
        def beta_hat(s, m):
            a = np.concatenate([np.diff(s[:, 0]), np.diff(s[:, 1])])
            b = np.concatenate([np.diff(m[:, 0]), np.diff(m[:, 1])])
            denom = float(np.dot(b, b))
            return float(np.dot(a, b) / denom) if denom > 1e-12 else 1.0

        b1 = beta_hat(s1, m1)
        bD = beta_hat(sD, mD)

        f1 = fit_ode_2d(s1, m1, beta=b1)
        fD = fit_ode_2d(sD, mD, beta=bD)

        # 时间归一化:1min Δt = 60 sec, daily Δt = 86400 sec
        dt_1m, dt_d = 60.0, 86400.0
        # ω_n_phys(单位:rad/sec)= (ω_n_step) / Δt
        omega_1m_phys = f1['omega'] / dt_1m
        omega_d_phys  = fD['omega'] / dt_d
        # Nyquist(单位:rad/sec):ω_N = π / Δt
        nyq_1m = np.pi / dt_1m
        nyq_d  = np.pi / dt_d

        print(f"  1min fit: k={f1['k']:.4f}, c={f1['c']:.4f}, ω_step={f1['omega']:.4f} rad/sample, "
              f"ω_phys={omega_1m_phys:.6e} rad/sec, F²={f1['f2']:.4e}, n={f1['n']}, β={b1:.3f}, rank={f1['rank']}")
        print(f"  daily fit: k={fD['k']:.4f}, c={fD['c']:.4f}, ω_step={fD['omega']:.4f} rad/sample, "
              f"ω_phys={omega_d_phys:.6e} rad/sec, F²={fD['f2']:.4e}, n={fD['n']}, β={bD:.3f}, rank={fD['rank']}")
        print(f"  Nyquist (1min)  = {nyq_1m:.6e} rad/sec")
        print(f"  Nyquist (daily) = {nyq_d:.6e} rad/sec")

        # === 关键诊断 ===
        # 1) 一致性:如果两个尺度都给出一致的 ω_phys,采样够;否则至少一边欠采样
        if omega_1m_phys > 1e-12:
            ratio = omega_d_phys / omega_1m_phys
        else:
            ratio = np.nan
        # 2) 日线欠采样:ω_d_phys < ω_1m_phys(欠采样会把高频成分 alias 到低频,看似"周期更长")
        #    或者:ω_1m_phys 远大于 Nyquist_daily
        undersampled_flag = omega_1m_phys > nyq_d * 1.5  # 留余量
        print(f"  ratio ω_d/ω_1m = {ratio:.3f}  (1.0 = 一致,< 0.5 = 日线严重低估 ω)")
        print(f"  日线欠采样?   = {undersampled_flag}  (ω_1m > 1.5×Nyquist_daily)")

        rows_summary.append({
            'label': label, 'stock': stock_code, 'index': index_code,
            'scale': '1min', 'n_obs': f1['n'],
            'k_hat': f1['k'], 'c_hat': f1['c'],
            'f2': f1['f2'], 'beta': b1, 'rank': f1['rank'],
        })
        rows_summary.append({
            'label': label, 'stock': stock_code, 'index': index_code,
            'scale': 'daily', 'n_obs': fD['n'],
            'k_hat': fD['k'], 'c_hat': fD['c'],
            'f2': fD['f2'], 'beta': bD, 'rank': fD['rank'],
        })
        rows_omega.append({
            'label': label, 'stock': stock_code,
            'omega_phys_1min_radps': omega_1m_phys,
            'omega_phys_daily_radps': omega_d_phys,
            'nyquist_1min_radps': nyq_1m,
            'nyquist_daily_radps': nyq_d,
            'ratio_d_over_1m': ratio,
            'daily_undersampled': undersampled_flag,
            'period_1min_sec': (2 * np.pi / omega_1m_phys) if omega_1m_phys > 1e-12 else np.inf,
            'period_daily_sec': (2 * np.pi / omega_d_phys) if omega_d_phys > 1e-12 else np.inf,
        })

        # === 一致性诊断:用日线 (k̂, ĉ) 当"已知常数"代入 1 分钟 ODE ===
        #   1min ODE 1-step 预测残差 vs 直接用 1min (k̂, ĉ) 的预测残差
        #   注:ODE 是 a = -k·u_S - c·u_M,所以"预测 dS[t+1] = dS[t] - k·u_S[t]·dt - c·u_M[t]·dt" 不严谨。
        #   这里更朴素的诊断:算"残差范数比",作为"日线参数在 1 分钟尺度下失配"的代理。
        # 计算日线参数下 1min 数据的 F²:用日线 (k, c) 算 1min 残差
        Y_1m = np.concatenate([np.diff(s1[:, 0]) - bD * np.diff(m1[:, 0]),
                               np.diff(s1[:, 1]) - bD * np.diff(m1[:, 1])])
        n = Y_1m.shape[0] // 2
        X_1m = np.zeros((2 * n, 2))
        X_1m[:n, 0] = -s1[:-1, 0]
        X_1m[:n, 1] = -m1[:-1, 0]
        X_1m[n:, 0] = -s1[:-1, 1]
        X_1m[n:, 1] = -m1[:-1, 1]
        theta_daily = np.array([fD['k'], fD['c']])
        resid_1m_using_daily = Y_1m - X_1m @ theta_daily
        f2_1m_using_daily = float(np.mean(resid_1m_using_daily ** 2))

        rows_path.append({
            'label': label, 'stock': stock_code,
            'f2_1min_native_fit': f1['f2'],
            'f2_1min_using_daily_params': f2_1m_using_daily,
            'ratio_daily_over_native': f2_1m_using_daily / f1['f2'] if f1['f2'] > 1e-12 else np.inf,
        })
        print(f"  F²(1min, 用日线参数) / F²(1min, 用 1min 参数) = "
              f"{f2_1m_using_daily / f1['f2'] if f1['f2'] > 1e-12 else float('inf'):.2f}")

    # ============== 写出 ==============
    pd.DataFrame(rows_summary).to_csv(
        os.path.join(OUT_DIR, 'fit_summary.csv'), index=False)
    pd.DataFrame(rows_omega).to_csv(
        os.path.join(OUT_DIR, 'omega_compare.csv'), index=False)
    pd.DataFrame(rows_path).to_csv(
        os.path.join(OUT_DIR, 'path_compare.csv'), index=False)

    # ============== 终判 ==============
    print("\n\n========== Spike 总结 ==========")
    if not rows_omega:
        print("无有效结果")
        return
    df_omega = pd.DataFrame(rows_omega)
    print(df_omega.to_string(index=False))

    n_undersampled = int(df_omega['daily_undersampled'].sum())
    n_total = len(df_omega)
    print(f"\n[Nyquist 不足] {n_undersampled}/{n_total} 只票日线欠采样")
    if n_undersampled == n_total:
        print("  → 全票命中,日线 ω_d 显著低估(高频 alias),支持迁移到日内分辨率")
    elif n_undersampled > 0:
        print("  → 部分命中,日线分辨率对小部分票不足,可考虑分批迁移")
    else:
        print("  → 未命中,日线分辨率在本数据上未显示 Nyquist 不足;")

    print(f"\nCSV 落盘到 {OUT_DIR}/")


if __name__ == '__main__':
    try:
        run()
    finally:
        try:
            tq.close()
        except Exception:
            pass