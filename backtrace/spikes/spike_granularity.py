# -*- coding: utf-8 -*-
"""
Spike #2: 找最小充分分辨率(1min vs 5min vs 15min)

上轮结论:1 分钟数据揭示 10-25 分钟级固有振荡,日线 Nyquist 严重不足。
本轮:在更宽样本(20 票 × 30 个 1 分钟交易日)和中间粒度(5min / 15min)上验证:
  - 1 分钟 vs 5 分钟:ω_n_phys 是否一致?一致 → 1 分钟 overkill,可降级
  - 5 分钟 vs 15 分钟:ω_n_phys 是否一致?一致 → 5 分钟 overkill,可降到 15 分钟
  - 三档 vs 日线:日线 ω_n 是否再次出现低估?

输出:outputs/spike_granularity/granularity_compare.csv
     outputs/spike_granularity/cross_scale_predict.csv
     outputs/spike_granularity/REPORT.md  (摘要)

边界:
  - 仍是一次性脚本,不接入 projection 模块(数学跟 spike #1 同源但加了多档粒度)
  - 任何代码视为可丢弃;真迁移重写
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

from common import tsfresh_pipeline as P
from tqcenter import tq

# === 配置 ===
SPIKE_STOCKS = [
    # 大盘蓝筹
    ('600519.SH', '000001.SH', '贵州茅台'),
    ('000001.SZ', '399001.SZ', '平安银行'),
    ('300750.SZ', '399001.SZ', '宁德时代'),
    ('601318.SH', '000001.SH', '中国平安'),
    ('600036.SH', '000001.SH', '招商银行'),
    ('000858.SZ', '399001.SZ', '五粮液'),
    ('601398.SH', '000001.SH', '工商银行'),
    ('601988.SH', '000001.SH', '中国银行'),
    # 中盘
    ('002594.SZ', '399001.SZ', '比亚迪'),
    ('600276.SH', '000001.SH', '恒瑞医药'),
    ('000333.SZ', '399001.SZ', '美的集团'),
    ('002475.SZ', '399001.SZ', '立讯精密'),
    ('300760.SZ', '399001.SZ', '迈瑞医疗'),
    # 小盘(各行业)
    ('600438.SH', '000001.SH', '通威股份'),
    ('002714.SZ', '399001.SZ', '牧原股份'),
    ('300015.SZ', '399001.SZ', '爱尔眼科'),
    ('601012.SH', '000001.SH', '隆基绿能'),
    ('600900.SH', '000001.SH', '长江电力'),
    ('002230.SZ', '399001.SZ', '科大讯飞'),
    ('600887.SH', '000001.SH', '伊利股份'),
]

N_DAYS = 30  # 30 个交易日
GRANULARITIES = ['1m', '5m', '15m']  # TQ 支持的三档
DT_SEC = {'1m': 60, '5m': 300, '15m': 900}  # 每根 bar 的实际秒数

OUT_DIR = os.path.join(BACKTRACE_DIR, 'outputs', 'spike_granularity')
os.makedirs(OUT_DIR, exist_ok=True)


# ============== 数据 ==============
def fetch_at_granularity(code, n_days, period):
    """TQ 拉最近 n_days 个交易日的 period K 线。"""
    end = datetime.now().strftime('%Y%m%d')
    # 多请求 1.5× 自然日,确保拿满 n_days 个交易日
    start = (datetime.now() - timedelta(days=int(n_days * 1.8) + 10)).strftime('%Y%m%d')
    fields = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
    raw = tq.get_market_data(
        field_list=fields, stock_list=[code],
        start_time=start, end_time=end,
        dividend_type='front', period=period, fill_data=True,
    )
    if raw is None or raw.get('Close') is None or raw['Close'].empty:
        raise RuntimeError(f"TQ 拉 {code} {period} 数据为空")
    out = pd.DataFrame({
        'Close':  pd.to_numeric(raw['Close'][code],  errors='coerce'),
        'Volume': pd.to_numeric(raw['Volume'][code], errors='coerce'),
    }).dropna(subset=['Close'])
    if 'Amount' in raw and code in raw['Amount'].columns:
        out['Amount'] = pd.to_numeric(raw['Amount'][code], errors='coerce')
    else:
        out['Amount'] = out['Volume'] * out['Close']
    out['__date__'] = out.index.date
    last_n_dates = sorted(out['__date__'].unique())[-n_days:]
    out = out[out['__date__'].isin(last_n_dates)].drop(columns='__date__')
    return out


# ============== ODE 数学 ==============
def state_2d(df):
    def norm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)
    return np.column_stack([norm(df['Volume'].to_numpy()),
                            norm(df['Amount'].to_numpy())])


def fit_ode_2d(s, m, beta):
    dS, dM = np.diff(s, axis=0), np.diff(m, axis=0)
    Y = np.concatenate([dS[:, 0] - beta * dM[:, 0], dS[:, 1] - beta * dM[:, 1]])
    n = dS.shape[0]
    X = np.zeros((2 * n, 2))
    X[:n, 0] = -s[:-1, 0]
    X[:n, 1] = -m[:-1, 0]
    X[n:, 0] = -s[:-1, 1]
    X[n:, 1] = -m[:-1, 1]
    theta, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    if theta.size == 0:
        return dict(k=0, c=0, f2=np.inf, n=n, omega=0, rank=0)
    k, c = float(theta[0]), float(theta[1])
    resid = Y - X @ theta
    f2 = float(np.mean(resid ** 2))
    disc = max(k ** 2 - c ** 2, 0)
    return dict(k=k, c=c, f2=f2, n=n, omega=float(np.sqrt(disc)), rank=int(rank))


def beta_hat(s, m):
    a = np.concatenate([np.diff(s[:, 0]), np.diff(s[:, 1])])
    b = np.concatenate([np.diff(m[:, 0]), np.diff(m[:, 1])])
    denom = float(np.dot(b, b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 1.0


# ============== 主流程 ==============
def run():
    tq.initialize(os.path.abspath(__file__))
    rows_gran = []
    rows_xscale = []

    print(f"配置: {len(SPIKE_STOCKS)} 票 × {len(GRANULARITIES)} 粒度 × {N_DAYS} 个交易日\n")

    for stock_code, index_code, name in SPIKE_STOCKS:
        print(f"\n========== {name} ({stock_code}) ==========")
        # 一次拉完 3 个粒度,避免重复调度
        data = {}
        for g in GRANULARITIES:
            try:
                data[g] = fetch_at_granularity(stock_code, N_DAYS, g)
            except Exception as e:
                print(f"  [SKIP {g}] {type(e).__name__}: {e}")
                data[g] = None

        # 大盘指数同样拉
        for g in GRANULARITIES:
            if data.get(g) is not None and f'idx_{g}' not in data:
                try:
                    data[f'idx_{g}'] = fetch_at_granularity(index_code, N_DAYS, g)
                except Exception as e:
                    print(f"  [SKIP idx {g}] {type(e).__name__}: {e}")
                    data[f'idx_{g}'] = None

        for g in GRANULARITIES:
            df_S = data.get(g)
            df_M = data.get(f'idx_{g}')
            if df_S is None or df_M is None or len(df_S) < 50:
                print(f"  [{g}] 数据不足,跳过")
                continue

            s = state_2d(df_S)
            m = state_2d(df_M)
            b = beta_hat(s, m)
            f = fit_ode_2d(s, m, beta=b)

            omega_phys = f['omega'] / DT_SEC[g]
            nyq_phys = np.pi / DT_SEC[g]
            period_sec = (2 * np.pi / omega_phys) if omega_phys > 1e-12 else np.inf

            print(f"  [{g}] n={f['n']}, k={f['k']:.3f}, c={f['c']:.3f}, "
                  f"ω_phys={omega_phys:.4e}, T={period_sec:.0f}s, F²={f['f2']:.3e}, β={b:.2f}")

            rows_gran.append({
                'name': name, 'stock': stock_code, 'granularity': g,
                'n_obs': f['n'], 'k_hat': f['k'], 'c_hat': f['c'],
                'f2': f['f2'], 'beta': b, 'rank': f['rank'],
                'omega_step_radpsample': f['omega'],
                'omega_phys_radps': omega_phys,
                'period_sec': period_sec,
                'nyquist_phys_radps': nyq_phys,
                'sampling_ratio_omega_over_nyq': omega_phys / nyq_phys if nyq_phys > 0 else np.nan,
            })

        # === 跨粒度预测测试 ===
        # 用上一粒度的 (k̂, ĉ) 代入下一粒度的数据,看 F² 是不是爆炸
        for src_g, tgt_g in [('15m', '5m'), ('5m', '1m'), ('1m', '1m')]:
            df_S_src = data.get(src_g)
            df_M_src = data.get(f'idx_{src_g}')
            df_S_tgt = data.get(tgt_g)
            df_M_tgt = data.get(f'idx_{tgt_g}')
            if any(x is None for x in [df_S_src, df_M_src, df_S_tgt, df_M_tgt]):
                continue
            s_src = state_2d(df_S_src)
            m_src = state_2d(df_M_src)
            f_src = fit_ode_2d(s_src, m_src, beta=beta_hat(s_src, m_src))
            s_tgt = state_2d(df_S_tgt)
            m_tgt = state_2d(df_M_tgt)
            # 用 src 粒度的 (k, c) 算 tgt 粒度的残差
            dS_t, dM_t = np.diff(s_tgt, axis=0), np.diff(m_tgt, axis=0)
            Y_t = np.concatenate([dS_t[:, 0] - beta_hat(s_tgt, m_tgt) * dM_t[:, 0],
                                   dS_t[:, 1] - beta_hat(s_tgt, m_tgt) * dM_t[:, 1]])
            n = dS_t.shape[0]
            X_t = np.zeros((2 * n, 2))
            X_t[:n, 0] = -s_tgt[:-1, 0]; X_t[:n, 1] = -m_tgt[:-1, 0]
            X_t[n:, 0] = -s_tgt[:-1, 1]; X_t[n:, 1] = -m_tgt[:-1, 1]
            theta_src = np.array([f_src['k'], f_src['c']])
            resid_t = Y_t - X_t @ theta_src
            f2_using_src = float(np.mean(resid_t ** 2))
            # tgt 自己的 F²
            f_tgt = fit_ode_2d(s_tgt, m_tgt, beta=beta_hat(s_tgt, m_tgt))
            ratio = f2_using_src / f_tgt['f2'] if f_tgt['f2'] > 1e-12 else np.inf

            print(f"  cross-scale [{src_g}→{tgt_g}]: F²(用 src)/F²(用 tgt) = {ratio:.2f}")
            rows_xscale.append({
                'name': name, 'stock': stock_code,
                'src_granularity': src_g, 'tgt_granularity': tgt_g,
                'f2_native_target': f_tgt['f2'],
                'f2_using_source_params': f2_using_src,
                'f2_ratio': ratio,
            })

    tq.close()

    # ============== 写出 + 报告 ==============
    df_gran = pd.DataFrame(rows_gran)
    df_xs = pd.DataFrame(rows_xscale)
    df_gran.to_csv(os.path.join(OUT_DIR, 'granularity_compare.csv'), index=False)
    df_xs.to_csv(os.path.join(OUT_DIR, 'cross_scale_predict.csv'), index=False)

    print("\n\n========== 总结 ==========")
    if df_gran.empty:
        print("无有效结果")
        return

    # 按粒度聚合
    print("\n[各粒度的 ω_phys 中位数 ± IQR]")
    for g in GRANULARITIES:
        sub = df_gran[df_gran['granularity'] == g]
        if sub.empty:
            continue
        omega_med = sub['omega_phys_radps'].median()
        omega_p25 = sub['omega_phys_radps'].quantile(0.25)
        omega_p75 = sub['omega_phys_radps'].quantile(0.75)
        T_med = sub['period_sec'].median()
        print(f"  {g}: ω_phys = {omega_med:.4e} [{omega_p25:.4e}, {omega_p75:.4e}], "
              f"中位周期 = {T_med:.0f}s ({T_med/60:.1f} min), "
              f"ω/Nyquist 比中位数 = {sub['sampling_ratio_omega_over_nyq'].median():.3f}")

    print("\n[跨粒度 F² 退化(中位数)]")
    if not df_xs.empty:
        for pair in df_xs.groupby(['src_granularity', 'tgt_granularity']):
            sub = pair[1]
            print(f"  {pair[0][0]} → {pair[0][1]}: F² 比中位数 = {sub['f2_ratio'].median():.2f}")

    # 写入 markdown 报告
    md = ["# Spike #2:粒度对比报告\n"]
    md.append(f"**配置**: {len(SPIKE_STOCKS)} 票 × {N_DAYS} 个交易日 × {len(GRANULARITIES)} 粒度\n\n")
    md.append("## 各粒度的物理 ω 与推断周期\n\n")
    md.append(df_gran.groupby('granularity').agg(
        n=('name', 'count'),
        omega_med=('omega_phys_radps', 'median'),
        omega_p25=('omega_phys_radps', lambda x: x.quantile(0.25)),
        omega_p75=('omega_phys_radps', lambda x: x.quantile(0.75)),
        period_med_sec=('period_sec', 'median'),
        period_med_min=('period_sec', lambda x: x.median() / 60),
    ).round(4).to_markdown())
    md.append("\n\n## 跨粒度 F² 比(中位数)\n\n")
    if not df_xs.empty:
        md.append(df_xs.groupby(['src_granularity', 'tgt_granularity'])['f2_ratio'].median().round(3).to_markdown())
    md.append("\n\n## 完整数据\n\n粒度参数:`granularity_compare.csv`\n\n跨粒度预测:`cross_scale_predict.csv`\n")
    with open(os.path.join(OUT_DIR, 'REPORT.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f"\nCSV/MD 落盘到 {OUT_DIR}/")


if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print(f"FATAL: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
    finally:
        try:
            tq.close()
        except Exception:
            pass