# -*- coding: utf-8 -*-
# backtrace/dynamics — 离散动力系统入口
#
# 复用 backtrace/projection/_projection_core.py 的全部数学(描述层 + 力模型 + 状态分类),
# 在此之上提供面向"动力系统"的 API + CLI 入口:
#   - 1 步预测:_dynamics_core.predict_next_state
#   - N 步轨迹模拟:_dynamics_core.simulate_trajectory
#   - 模拟结果 CSV:_dynamics_core.build_simulation_df
#   - 单股端到端 CLI:dynamics_system.py
#   - 批量 CLI:dynamics_batch.py
#
# 数学源头 docs:
#   - docs/superpowers/specs/2026-08-16-market-stock-dynamics-design.md  (描述层)
#   - docs/superpowers/specs/2026-08-16-dynamics-system-design.md          (本目录增量)
from ._dynamics_core import (
    # 复用 projection 层
    compute_dynamics,
    classify_states,
    build_dynamics_df,
    compute_forces,
    build_forces_df,
    STATE_LABELS,
    STATE_COLORS,
    STATE_LABELS_CN,
    # 新增
    predict_next_state,
    simulate_trajectory,
    build_simulation_df,
)

__all__ = [
    'compute_dynamics',
    'classify_states',
    'build_dynamics_df',
    'compute_forces',
    'build_forces_df',
    'STATE_LABELS',
    'STATE_COLORS',
    'STATE_LABELS_CN',
    'predict_next_state',
    'simulate_trajectory',
    'build_simulation_df',
]
