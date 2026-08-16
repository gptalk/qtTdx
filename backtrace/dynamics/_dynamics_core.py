# -*- coding: utf-8 -*-
# _dynamics_core.py — 离散动力系统入口(描述层 + 力模型 + N 步轨迹模拟)
#
# 数学源头:backtrace/projection/_projection_core.py
#   - compute_dynamics / classify_states / build_dynamics_df   → 描述层
#   - compute_forces    / build_forces_df                       → 力模型
#   - STATE_LABELS / STATE_COLORS / STATE_LABELS_CN            → 状态配色
#
# 本模块新增 3 个真正属于"动力系统"的功能:
#   1. predict_next_state  — 1 步预测(扩展 prediction_ode.py 的散装函数为可调用 API)
#   2. simulate_trajectory — N 步前向模拟(已知未来大盘 + β + 残差,个股怎么演化)
#   3. build_simulation_df — 模拟结果 DataFrame
#
# 不重写任何 projection 层的数学 — 单一来源真理。
import os
import sys
import numpy as np
import pandas as pd

# 把 projection/ 加进 path,导入其数学
_PROJECTION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'projection')
if _PROJECTION_DIR not in sys.path:
    sys.path.insert(0, _PROJECTION_DIR)

# 复用 projection 层的所有动力相关函数
from _projection_core import (
    # 描述层
    compute_dynamics,
    classify_states,
    build_dynamics_df,
    # 力模型
    compute_forces,
    build_forces_df,
    # 状态分类配色
    STATE_LABELS,
    STATE_COLORS,
    STATE_LABELS_CN,
)

# === 1 步预测(用户 prompt §19) ==============================================
def predict_next_state(
    v_S_now: np.ndarray,        # (2,) 当前个股速度
    a_M_now: np.ndarray,        # (2,) 当前大盘加速度
    beta_now: float,            # 当前 β
    d_now: np.ndarray,          # (2,) 当前位置偏离
    u_now: np.ndarray,          # (2,) 当前速度偏离
    F_self_now: np.ndarray | None = None,  # (2,) 当前残差;None 时按 F_self = a_S - β·a_M 推
    a_S_now: np.ndarray | None = None,     # (2,) 当前个股加速度(F_self=None 时必传)
    k: float = 0.0,
    c: float = 0.0,
) -> tuple:
    """1 步预测下一个交易日的个股加速度 / 速度 / 位置增量。

    模型(用户 prompt §14-19):
        a_pred = β·a_M - k·d - c·u + F_self
        v_pred = v_S + a_pred          (Δt = 1)
        ΔS_pred = v_pred               (速度直接 = 下日 ΔS,因为 v ≡ ΔS/Δt, Δt=1)

    Args:
        v_S_now:  当前 v_S(2-D 向量,ΔVol/ΔAmt 量纲)
        a_M_now:  当前 a_M(2-D 向量)
        beta_now: 当前 β
        d_now:    当前 d(2-D 位置偏离累积)
        u_now:    当前 u(2-D 速度偏离)
        F_self_now: (可选)外部给定的残差;若 None 则由 a_S_now 推 F_self = a_S - β·a_M + k·d + c·u
        a_S_now:   (F_self_now=None 时必传)当前 a_S
        k: 恢复系数
        c: 阻尼系数

    Returns:
        (a_pred, v_pred, delta_S_pred),都是 (2,) ndarray
    """
    if F_self_now is None:
        if a_S_now is None:
            raise ValueError("F_self_now=None 时必须传 a_S_now 才能推残差")
        F_self_now = a_S_now - beta_now * a_M_now + k * d_now + c * u_now
    a_pred = beta_now * a_M_now - k * d_now - c * u_now + F_self_now
    v_pred = v_S_now + a_pred                  # Δt = 1
    delta_S_pred = v_pred                      # 下日 ΔS 预测 = 下日 v_S 预测(Δt=1)
    return a_pred, v_pred, delta_S_pred


# === N 步前向模拟(用户 prompt §19 + §14-18) =================================
def simulate_trajectory(
    v_S_init: np.ndarray,           # (2,) 起点速度(取末日真实 v_S)
    v_M_seq: np.ndarray,            # (N, 2) 未来 N 天大盘速度
    beta_seq: np.ndarray,           # (N,)   未来 N 天 β
    F_self_seq: np.ndarray,         # (N, 2) 残差序列(t=0..N-1)
    d_init: np.ndarray,             # (2,)   起点位置偏离
    u_init: np.ndarray,             # (2,)   起点速度偏离
    k: float = 0.0,
    c: float = 0.0,
    q_t_seq: np.ndarray | None = None,   # (N,) 锚定强度;None 时默认 1
    classify_thresholds: tuple = (0.10, 0.50, np.deg2rad(30), np.deg2rad(90)),
) -> dict:
    """N 步前向模拟(Oracle 模式:未来大盘/β 已知,残差可外推)。

    链(对应用户 prompt §19):
        for t in range(N):
            a_M(t) = v_M_seq[t+1] - v_M_seq[t]      # 末步 NaN
            a_t = β_t·a_M(t) - k·d_t - c·u_t + F_self(t)   (q_t 阻尼 a_M)
            v_{t+1} = v_t + a_t
            u_{t+1} = v_{t+1} - β_{t+1}·v_M_seq[t+1]  (用下日 β;末步用 β_t 兜底)
            d_{t+1} = d_t + u_{t+1}              (位置偏离累计)
        返回 v_seq / a_seq / d_seq / u_seq / E_* / R / θ / state

    Args:
        v_S_init: (2,) 起点速度(末日真实 v_S)
        v_M_seq:  (N, 2) 未来 N 天大盘速度
        beta_seq: (N,)   未来 N 天 β
        F_self_seq: (N, 2) 残差;长度 N 即 0..N-1(步 t 用 F_self_seq[t])
        d_init / u_init: (2,) 起点状态
        k / c: 力模型系数
        q_t_seq: (N,) 锚定强度,None 时默认全 1(无阻尼);与 description 层 λ_q 同语义
        classify_thresholds: 4 元组,与 classify_states 同

    Returns:
        dict with keys(均为长度 N+1,index 0=起点):
            v_seq / d_seq / u_seq: ndarray (N+1, 2)
            a_seq:                 ndarray (N+1, 2) 末行 NaN
            E_total / E_market / E_self: ndarray (N+1,)
            R:                     ndarray (N+1,)
            theta:                 ndarray (N+1,) 弧度
            state:                 list[str] (N+1,)
            v_M_seq_used:          ndarray (N, 2) 回放(便于 caller 画图)
            beta_seq_used:         ndarray (N,)
            F_market / F_restore / F_damp / F_self: ndarray (N+1,) 各力模长(末行 NaN for F_market/Self)
    """
    N = v_M_seq.shape[0]
    if beta_seq.shape[0] != N:
        raise ValueError(f"beta_seq 长度 {beta_seq.shape[0]} != v_M_seq {N}")
    if F_self_seq.shape != (N, 2):
        raise ValueError(f"F_self_seq 形状 {F_self_seq.shape} != ({N}, 2)")
    if q_t_seq is None:
        q_t_seq = np.ones(N)
    elif q_t_seq.shape[0] != N:
        raise ValueError(f"q_t_seq 长度 {q_t_seq.shape[0]} != v_M_seq {N}")

    v_seq = np.zeros((N + 1, 2))
    d_seq = np.zeros((N + 1, 2))
    u_seq = np.zeros((N + 1, 2))
    a_seq = np.full((N + 1, 2), np.nan)
    F_market = np.full(N + 1, np.nan)
    F_restore = np.full(N + 1, np.nan)
    F_damp = np.full(N + 1, np.nan)
    F_self = np.full(N + 1, np.nan)

    # t=0 = 起点
    v_seq[0] = v_S_init
    d_seq[0] = d_init
    u_seq[0] = u_init
    # t=0 的力:用 v_S_init 等于刚走完的上一步,初值力 = (q*β*a_M, -k*d, -c*u, F_self)
    # 末步观察的外推 — 上一步 a_M 用 v_M_seq[0] - 0(假设 0 之前的市场速度)
    # 这里我们只对 t=0 设近似力:
    if N >= 1:
        F_market[0] = abs(q_t_seq[0] * beta_seq[0] * 0.0)  # 无前一步 → 0
        F_restore[0] = abs(k * d_init[0]) if hasattr(k * d_init, '__len__') else abs(k * np.linalg.norm(d_init))
        F_damp[0] = abs(c * np.linalg.norm(u_init))
        F_self[0] = abs(np.linalg.norm(F_self_seq[0]))

    # 大盘加速度 a_M(t) = v_M_seq[t+1] - v_M_seq[t],长度 N-1;末步 a_M(N-1) = NaN
    a_M_seq = np.full((N, 2), np.nan)
    if N >= 2:
        a_M_seq[:-1] = np.diff(v_M_seq, axis=0)

    # 主体循环
    for t in range(N):
        d_t = d_seq[t]
        u_t = u_seq[t]
        v_t = v_seq[t]
        # a_M(t) 末步 NaN — 此时模型也只产出 NaN 加速,后续 v_{N} 标 NaN
        a_M_t = a_M_seq[t]
        if not np.isfinite(a_M_t).all():
            # 末步:无 a_M 输入 → 市场驱动力 = 0
            a_pred = -k * d_t - c * u_t + F_self_seq[t]
            F_market_t = 0.0
        else:
            # q_t 阻尼 a_M(与 description 层 compute_dynamics 一致)
            a_pred = (
                q_t_seq[t] * beta_seq[t] * a_M_t
                - k * d_t - c * u_t
                + F_self_seq[t]
            )
            F_market_t = float(np.linalg.norm(q_t_seq[t] * beta_seq[t] * a_M_t))
        a_seq[t] = a_pred
        # 力分解(取模长)— F_self = F_self_seq[t] 给定(模型外生)
        F_market[t] = F_market_t
        F_restore[t] = float(np.linalg.norm(k * d_t))
        F_damp[t] = float(np.linalg.norm(c * u_t))
        F_self[t] = float(np.linalg.norm(F_self_seq[t]))

        # 步 t → t+1
        v_seq[t + 1] = v_t + a_pred
        # 下日 β;末步用 β_t 兜底(避免越界)
        beta_next = beta_seq[t + 1] if t + 1 < N else beta_seq[t]
        v_M_next = v_M_seq[t + 1] if t + 1 < N else v_M_seq[t]
        u_seq[t + 1] = v_seq[t + 1] - beta_next * v_M_next
        d_seq[t + 1] = d_t + u_seq[t + 1]

    # 末步 (t=N) 力的"回声":用末态估算
    F_restore[N] = float(np.linalg.norm(k * d_seq[N]))
    F_damp[N] = float(np.linalg.norm(c * u_seq[N]))

    # 派生量:R / θ / E / state
    v_S_mag = np.linalg.norm(v_seq, axis=1)
    v_resi = u_seq                                  # 残差 = 速度偏离(物理上 = v_S - β·v_M)
    v_resi_mag = np.linalg.norm(v_resi, axis=1)
    # 数值上更稳的拆分:正交分量模长平方相减
    v_proj_mag_sq = np.maximum(v_S_mag ** 2 - v_resi_mag ** 2, 0.0)
    E_market = 0.5 * v_proj_mag_sq
    E_self = 0.5 * v_resi_mag ** 2
    E_total = 0.5 * v_S_mag ** 2
    # R = ‖v_resi‖² / ‖v_S‖² ∈ [0, 1];变 β 下正交分解退化,clip 防 > 1
    R_raw = np.divide(
        v_resi_mag ** 2, v_S_mag ** 2,
        out=np.zeros_like(v_resi_mag),
        where=(v_S_mag ** 2 > 1e-12) & np.isfinite(v_S_mag ** 2),
    )
    R = np.clip(R_raw, 0.0, 1.0)
    # θ — t=0..N-2 用 (v_t, v_M_t),t=N-1 无 v_M_N → NaN,t=N 不算(末行)
    v_M_mag = np.linalg.norm(v_M_seq, axis=1)
    cos_theta = np.full(N + 1, np.nan, dtype=float)
    for t in range(N):
        denom = v_S_mag[t] * v_M_mag[t]
        if denom > 1e-12 and np.isfinite(denom):
            cos = np.dot(v_seq[t], v_M_seq[t]) / denom
            cos_theta[t] = float(np.clip(cos, -1.0, 1.0))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # 状态分类 — R/theta 在 t=0..N-1 都有限,直接 classify
    R_low, R_high, theta_following, theta_against = classify_thresholds
    state = classify_states(
        R[:N], theta[:N], E_self[:N],
        (R_low, R_high, theta_following, theta_against),
    )
    # t=N 的 state = 'none'(无对应分类输入)
    state = state + ['none']

    return {
        'v_seq': v_seq,
        'a_seq': a_seq,
        'd_seq': d_seq,
        'u_seq': u_seq,
        'v_S_mag': v_S_mag,
        'v_M_mag': np.concatenate([v_M_mag, [v_M_mag[-1]]]),  # 长度对齐 N+1
        'E_total': E_total,
        'E_market': E_market,
        'E_self': E_self,
        'R': R,
        'theta': theta,
        'state': state,
        'v_M_seq_used': v_M_seq,
        'beta_seq_used': beta_seq,
        'q_t_seq_used': q_t_seq,
        'F_market': F_market,
        'F_restore': F_restore,
        'F_damp': F_damp,
        'F_self': F_self,
        'k_restore': k,
        'c_damp': c,
    }


# === 模拟结果 DataFrame 组装 =================================================
def build_simulation_df(sim: dict, dates, index_tag: str, stock_tag: str) -> pd.DataFrame:
    """把 simulate_trajectory() 的输出组装成 CSV-ready DataFrame。

    Args:
        sim:  simulate_trajectory() 返回的 dict
        dates: (N+1,) 日期序列;None 时用 0..N 整数占位
        index_tag / stock_tag: 列名后缀

    Returns:
        pd.DataFrame,长度 N+1,14 列:
            Date, Sim_a_{idx}, Sim_a_{st}, Sim_v_{st}_Vol, Sim_v_{st}_Amt,
            Sim_d_{st}, Sim_u_{st}, Sim_R, Sim_Theta, Sim_E_Total,
            Sim_E_Market, Sim_E_Self, Sim_State
    """
    N_plus_1 = sim['v_seq'].shape[0]
    if dates is None:
        dates = list(range(N_plus_1))
    elif len(dates) != N_plus_1:
        raise ValueError(
            f"dates 长度 {len(dates)} != N+1 = {N_plus_1}"
        )

    return pd.DataFrame({
        'Date': dates,
        f'Sim_v_M_{index_tag}': np.concatenate([
            np.linalg.norm(sim['v_M_seq_used'], axis=1),  # (N,)
            [np.nan],                                     # pad 到 N+1
        ]),
        f'Sim_a_{stock_tag}': np.linalg.norm(sim['a_seq'], axis=1),
        f'Sim_v_{stock_tag}_Vol': sim['v_seq'][:, 0],
        f'Sim_v_{stock_tag}_Amt': sim['v_seq'][:, 1],
        f'Sim_d_{stock_tag}': np.linalg.norm(sim['d_seq'], axis=1),
        f'Sim_u_{stock_tag}': np.linalg.norm(sim['u_seq'], axis=1),
        'Sim_R': sim['R'],
        'Sim_Theta': sim['theta'],
        'Sim_E_Total': sim['E_total'],
        'Sim_E_Market': sim['E_market'],
        'Sim_E_Self': sim['E_self'],
        f'Sim_F_Market_{index_tag}': sim['F_market'],
        f'Sim_F_Restore_{stock_tag}': sim['F_restore'],
        f'Sim_F_Damp_{stock_tag}': sim['F_damp'],
        f'Sim_F_Self_{stock_tag}': sim['F_self'],
        f'Sim_State_{stock_tag}': sim['state'],
        # 调试用 — k / c 末行对齐(标量)
        'Sim_k_restore': [sim['k_restore']] * N_plus_1,
        'Sim_c_damp': [sim['c_damp']] * N_plus_1,
    })


# === 公开 re-export ==========================================================
__all__ = [
    # 复用 projection 层
    'compute_dynamics',
    'classify_states',
    'build_dynamics_df',
    'compute_forces',
    'build_forces_df',
    'STATE_LABELS',
    'STATE_COLORS',
    'STATE_LABELS_CN',
    # 新增
    'predict_next_state',
    'simulate_trajectory',
    'build_simulation_df',
]
