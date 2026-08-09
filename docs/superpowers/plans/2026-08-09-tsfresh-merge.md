# tsfresh 脚本合并实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 抽 2 个 `backtrace/common/` 模块消除 3 个 tsfresh vbt 回测脚本的 ~70% 重复代码,所有近期修复(INIT_TRAIN_SIZE FDR 限制、bfill、80% 拒单 warning、friction_loss_pp 符号检查)统一为 common 模块的默认行为。

**Architecture:** 2 个新 pure-utility 模块承担公共逻辑(无 print,verbose 标志控制,跟 `tsfresh_pipeline.py` 风格一致)。3 个原脚本瘦身成「配置 + STRATEGIES 列表 + 编排循环」的薄壳,只调 common 模块不重写逻辑。4 个独立 commit,每个 commit 后跑一次冒烟(改 STOCK_CODE 跑 1 只票)确认 schema 不变。

**Tech Stack:** Python 3 + pandas + numpy + vectorbt + scikit-learn + tsfresh + pytest(测试在 `tests/` 下,沿用 `test_data_store.py` 的 fixture 风格)。不引入新依赖。

## Global Constraints

1. **不修改** `backtrace/common/tsfresh_pipeline.py`、`jhzq_fees.py`、`data_store.py`、`tsfresh_config.py`(已稳定,改动牵连所有脚本)
2. **不修改** 3 个原脚本的输出 CSV schema(用户已基于历史输出做对比)
3. **TQ 客户端不在 common 模块 import** — 跟现有 `tsfresh_pipeline.py` 一致(懒加载 + 委托调用方 `tq.initialize(__file__)`)
4. **保留 vbt_combo.py:174-176 的 pf_zero reuse caveat** — 在 `B.run_vbt_backtest` 的 docstring 顶部写明
5. **保留 with_ma_channel.py:7-15 的阈值偏差警告** — 在 `W.tsfresh_walkforward_proba` 的 docstring 顶部写明
6. **`PYTHONIOENCODING=utf-8`** 仍然必须在命令行加(Windows GBK 终端)— 见 README
7. **新增模块不引入新依赖**(vectorbt / sklearn / tsfresh 都已在 requirements 里)
8. **每个 commit 后跑冒烟**:`PYTHONIOENCODING=utf-8 python backtrace/tsfresh/tsfresh_vbt_combo.py` 不崩 + 输出 CSV schema 列名一致

---

## File Structure

```
backtrace/
├── common/
│   ├── tsfresh_walkforward.py     ← 新增(纯函数,无 TQ 依赖)
│   └── vbt_jhzq_backtest.py       ← 新增(纯函数,无 TQ 依赖)
├── tsfresh/
│   ├── tsfresh_vbt_combo.py       ← 精简:274 行 → ~140 行
│   ├── tsfresh_with_ma_channel.py ← 精简:273 行 → ~140 行
│   └── tsfresh_with_ma_grid_sector.py ← 精简:195 行 → ~120 行
tests/
├── test_tsfresh_walkforward.py    ← 新增
└── test_vbt_jhzq_backtest.py      ← 新增
```

每个文件一个职责:
- `tsfresh_walkforward.py` — 给 tsfresh 特征 + walk-forward 训练,产出 proba(同时承担 MA 通道 + 通道构成报告这两个「给 proba 准备数据」的小工具)
- `vbt_jhzq_backtest.py` — 给 proba + 价格序列,跑 vbt + 真实扣费 + warning + 格式化(承担「proba 之后所有事情」)
- 3 个原脚本只剩「策略定义 + 编排」

---

### Task 1: 新增 `backtrace/common/tsfresh_walkforward.py`

**Files:**
- Create: `backtrace/common/tsfresh_walkforward.py`
- Create: `tests/test_tsfresh_walkforward.py`

**Interfaces:**
- Consumes: `P.to_long_format` / `P.extract_window_features` / `P.make_labels` / `P.select_relevant` / `P.fit_logreg` from `tsfresh_pipeline`(已有)
- Produces:
  - `add_ma_channels(ohlcv_df, windows=(5,10,20), add_rel=True) -> pd.DataFrame`
  - `report_channel_composition(X_sel, label='') -> None`(打印,不返回)
  - `tsfresh_walkforward_proba(ohlcv_df, channels, *, init_train_size=200, step=50, fillna='bfill', id_value=None, verbose=True) -> tuple[pd.Series, pd.DataFrame]`

- [ ] **Step 1: 写失败的测试 — `add_ma_channels`**

`tests/test_tsfresh_walkforward.py`:

```python
import pandas as pd
import numpy as np
from common.tsfresh_walkforward import add_ma_channels, report_channel_composition


def _sample_ohlcv(n=30):
    idx = pd.date_range('2026-01-01', periods=n, freq='D')
    return pd.DataFrame({
        'Open':   np.linspace(10, 15, n),
        'High':   np.linspace(10.5, 15.5, n),
        'Low':    np.linspace(9.5, 14.5, n),
        'Close':  np.linspace(10, 15, n),
        'Volume': np.linspace(1e6, 2e6, n),
    }, index=idx)


def test_add_ma_channels_adds_four_columns():
    df = _sample_ohlcv()
    out = add_ma_channels(df)
    assert {'ma5', 'ma10', 'ma20', 'rel_ma5'}.issubset(out.columns)
    assert len(out) == len(df)


def test_add_ma_channels_does_not_mutate_input():
    df = _sample_ohlcv()
    cols_before = set(df.columns)
    _ = add_ma_channels(df)
    assert set(df.columns) == cols_before


def test_add_ma_channels_uses_bfill_not_zero_fill():
    """早期段(头 19 天 ma20 是 NaN)必须用 bfill,不能是 0.0"""
    df = _sample_ohlcv(n=30)
    out = add_ma_channels(df)
    # ma20 第 1 天 NaN 应被第 2 天值 bfill,而不是被 0 填充
    assert not np.isclose(out['ma20'].iloc[0], 0.0), \
        f"ma20.iloc[0]={out['ma20'].iloc[0]} 应为 bfill 后的实值,非 0"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd c:/Users/yellow/mcp/qtTdx && PYTHONIOENCODING=utf-8 python -m pytest tests/test_tsfresh_walkforward.py -v`
Expected: `ModuleNotFoundError: No module named 'common.tsfresh_walkforward'`

- [ ] **Step 3: 写最小实现 — `add_ma_channels`**

`backtrace/common/tsfresh_walkforward.py`:

```python
# -*- coding: utf-8 -*-
"""tsfresh walk-forward proba 生成 + MA 通道工具。

约定(写在本模块 docstring 顶部,3 个原脚本迁移后共享):
  1. **bfill 而非 fillna(0.0)** — 避免序列开头 0 → 实值跳变被 tsfresh 当成「突变」特征。
     理由见原 with_ma_channel.py:96-98 注释。
  2. **FDR 严格在 init_train_size 段筛** — 防止特征选择时用上 walk-forward 期内的未来样本
     (原 grid_sector 在 X_all 上筛有泄漏风险,迁移后自动修好)。
  3. **X_sel 列对齐** — 在全期 X_all[selected_cols] 上做,索引仍覆盖全期。
"""
import numpy as np
import pandas as pd
import vectorbt as vbt

from common import tsfresh_pipeline as P


def add_ma_channels(ohlcv_df, windows=(5, 10, 20), add_rel=True):
    """原地加 ma5/ma10/ma20 + rel_ma5(Close 相对 ma5 的偏离度)。
    复制 df 后修改,避免污染上游调用方。
    使用 bfill(替代原 grid_sector 的 fillna(0.0))——见 with_ma_channel.py:96-98 注释。
    """
    out = ohlcv_df.copy()
    for w in windows:
        out[f'ma{w}'] = vbt.MA.run(out['Close'], window=w).ma.bfill()
    if add_rel:
        out['rel_ma5'] = ((out['Close'] - out['ma5']) / out['ma5'].replace(0, np.nan)).bfill()
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd c:/Users/yellow/mcp/qtTdx && PYTHONIOENCODING=utf-8 python -m pytest tests/test_tsfresh_walkforward.py::test_add_ma_channels_adds_four_columns -v`
Expected: PASS

- [ ] **Step 5: 加 `report_channel_composition` 测试**

`tests/test_tsfresh_walkforward.py` 追加:

```python
def test_report_channel_composition_warns_when_ma_dominates(capsys):
    """当 ma* 通道入选特征占比 > 33% 时,必须打印 [WARN] 冗余风险"""
    cols = [f'close__f{i}' for i in range(10)] + [f'ma5__f{i}' for i in range(15)]
    X_sel = pd.DataFrame(np.random.randn(5, len(cols)), columns=cols)
    report_channel_composition(X_sel, label='test')
    out = capsys.readouterr().out
    assert '[WARN]' in out and '冗余' in out


def test_report_channel_composition_no_warn_when_basic(capsys):
    cols = [f'close__f{i}' for i in range(20)] + [f'volume__f{i}' for i in range(5)]
    X_sel = pd.DataFrame(np.random.randn(5, len(cols)), columns=cols)
    report_channel_composition(X_sel, label='test')
    out = capsys.readouterr().out
    assert '[WARN]' not in out
```

- [ ] **Step 6: 写 `report_channel_composition` 实现**

`backtrace/common/tsfresh_walkforward.py` 追加(在 `add_ma_channels` 之后):

```python
def report_channel_composition(X_sel, label=''):
    """统计 X_sel 各通道入选特征数。若 ma*/rel_ma5 占比 > 33% 打印 [WARN] 冗余风险。
    从原 with_ma_channel.py:_report_channel_composition 搬过来。
    """
    if X_sel.shape[1] == 0:
        return
    channels = pd.Series([col.split('__', 1)[0] for col in X_sel.columns])
    counts = channels.value_counts()
    prefix = f'[{label}] ' if label else ''
    print(f'   {prefix}各通道入选特征数:')
    for ch, n in counts.items():
        pct = n / len(channels) * 100
        print(f'     {ch:<10} {n:>4}  ({pct:5.1f}%)')
    ma_total = sum(counts.get(c, 0) for c in counts.index
                   if c.startswith('ma') or c == 'rel_ma5')
    if ma_total > 0 and ma_total / len(channels) > 0.33:
        print(f'   [WARN] MA 相关通道占 {ma_total/len(channels):.0%},'
              f'可能与 Close 通道冗余')
```

- [ ] **Step 7: 跑测试确认通过**

Run: `cd c:/Users/yellow/mcp/qtTdx && PYTHONIOENCODING=utf-8 python -m pytest tests/test_tsfresh_walkforward.py -v`
Expected: 5 PASS

- [ ] **Step 8: 加 `tsfresh_walkforward_proba` 测试(纯逻辑部分)**

`tests/test_tsfresh_walkforward.py` 追加:

```python
def test_tsfresh_walkforward_proba_returns_proba_and_xsel():
    """小数据集(30 天 + 5 通道)— 验证返回类型与 FDR 限制逻辑"""
    df = _sample_ohlcv(n=60)
    # 此测试需要 TQ 关闭时也能跑,所以用 fillna='zero' 跳过 tsfresh
    # 改测 init_train_size 行为:若样本不足,应抛 ValueError 而不是默默返回空
    import pytest
    with pytest.raises(ValueError):
        from common.tsfresh_walkforward import tsfresh_walkforward_proba
        tsfresh_walkforward_proba(df, channels=['Close'], init_train_size=200)
```

> 注:这个测试只验证 init_train_size > 样本数时抛 ValueError,不真跑 tsfresh(那要 ~30 秒 + TQ)。其他行为靠冒烟测试(Task 5 的 Step 4)验证。

- [ ] **Step 9: 写 `tsfresh_walkforward_proba` 最小实现**

`backtrace/common/tsfresh_walkforward.py` 追加:

```python
def tsfresh_walkforward_proba(ohlcv_df, channels, *,
                              init_train_size=200, step=50,
                              fillna='bfill', id_value=None, verbose=True):
    """跑 tsfresh 全流程 → 返回 (proba, X_sel)。

    行为约定:
      - **FDR 在 X.iloc[:init_train_size] 段筛**(防泄漏)
      - **bfill 替代 fillna(0.0)**(防早期跳变特征)
      - **walk-forward**:初始 init_train_size,之后每 step 重训一次
      - **proba 在 date_index[end_t] 当日计算**,下游 shift(1) 视作次日开盘成交

    Raises:
      ValueError: 样本数 < init_train_size(无法满足 FDR 限制 + 留 walk-forward 起点)
    """
    if len(ohlcv_df) < init_train_size + step:
        raise ValueError(
            f'样本数 {len(ohlcv_df)} < init_train_size({init_train_size}) + step({step}),'
            f'无法跑 walk-forward'
        )

    df_fill = ohlcv_df[channels].copy()
    if fillna == 'bfill':
        df_fill = df_fill.bfill()
    elif fillna == 'zero':
        df_fill = df_fill.fillna(0.0)
    else:
        raise ValueError(f"fillna 必须是 'bfill' 或 'zero',收到 {fillna!r}")

    if verbose:
        print(f'   通道 {len(channels)} 个: {channels}')

    id_val = id_value if id_value is not None else (ohlcv_df.name or 'X')
    long_df = P.to_long_format(df_fill, channels=channels, id_value=id_val)
    X_all = P.extract_window_features(long_df, use_kind=True, verbose=False)
    y_all, X_all = P.make_labels(X_all, ohlcv_df['Close'].values, verbose=False)
    if verbose:
        print(f'   样本 {len(y_all)} 个  |  正样本 {y_all.mean():.1%}  |  '
              f'特征 {X_all.shape[1]} 列')

    # FDR 严格限制在前 init_train_size 段(防未来信息泄漏)
    X_train0 = X_all.iloc[:init_train_size]
    y_train0 = y_all.iloc[:init_train_size]
    X_sel_initial = P.select_relevant(X_train0, y_train0, verbose=False)
    selected_cols = X_sel_initial.columns.tolist()
    if len(selected_cols) == 0:
        if verbose:
            print(f'   [WARN] FDR=0,用全量 {X_all.shape[1]} 特征')
        X_sel = X_all
    else:
        X_sel = X_all[selected_cols]
    if verbose:
        print(f'   FDR 显著 {X_sel.shape[1]} 列 (前 {init_train_size} 个样本筛)')

    date_index = pd.DatetimeIndex(ohlcv_df.index)
    proba_records = []
    scaler_w = clf_w = None
    for pos, idx in enumerate(X_sel.index):
        end_t = idx[1]
        if end_t >= len(date_index):
            continue
        if pos < init_train_size:
            proba_records.append((date_index[end_t], np.nan))
            continue
        if pos == init_train_size or (pos - init_train_size) % step == 0:
            scaler_w, clf_w = P.fit_logreg(X_sel.iloc[:pos], y_all.iloc[:pos],
                                            verbose=False)
        p = float(clf_w.predict_proba(
            scaler_w.transform(X_sel.iloc[[pos]].values))[0, 1])
        proba_records.append((date_index[end_t], p))

    proba = pd.Series([v for _, v in proba_records],
                      index=pd.DatetimeIndex([d for d, _ in proba_records]),
                      name='proba').sort_index()
    proba = proba[~proba.index.duplicated(keep='last')].dropna()
    if verbose:
        print(f'   proba {len(proba)} 个  |  mean={proba.mean():.3f}')
    return proba, X_sel
```

- [ ] **Step 10: 跑全部测试**

Run: `cd c:/Users/yellow/mcp/qtTdx && PYTHONIOENCODING=utf-8 python -m pytest tests/test_tsfresh_walkforward.py -v`
Expected: 6 PASS

- [ ] **Step 11: 冒烟(用本地 CSV,不开 TQ)**

Run:
```bash
cd c:/Users/yellow/mcp/qtTdx && PYTHONIOENCODING=utf-8 python -c "
import pandas as pd, numpy as np
from common.tsfresh_walkforward import add_ma_channels, report_channel_composition, tsfresh_walkforward_proba
idx = pd.date_range('2024-01-01', periods=400, freq='D')
np.random.seed(42)
df = pd.DataFrame({
    'Open':   10 + np.cumsum(np.random.randn(400) * 0.1),
    'High':   10 + np.cumsum(np.random.randn(400) * 0.1) + 0.5,
    'Low':    10 + np.cumsum(np.random.randn(400) * 0.1) - 0.5,
    'Close':  10 + np.cumsum(np.random.randn(400) * 0.1),
    'Volume': np.random.uniform(1e6, 2e6, 400),
}, index=idx)
df.name = 'TEST.SH'
df2 = add_ma_channels(df)
print('ma5 head:', df2['ma5'].head(3).tolist())
print('rel_ma5 head:', df2['rel_ma5'].head(3).tolist())
proba, X_sel = tsfresh_walkforward_proba(df, channels=['Open','High','Low','Close','Volume'], verbose=True)
report_channel_composition(X_sel, label='smoke')
print('proba len:', len(proba), 'min:', proba.min(), 'max:', proba.max())
"
```
Expected: `proba len > 100`,`X_sel.shape[1] > 0`,`report_channel_composition` 打印 5 通道(无 [WARN])

- [ ] **Step 12: 提交**

```bash
cd c:/Users/yellow/mcp/qtTdx
git add backtrace/common/tsfresh_walkforward.py tests/test_tsfresh_walkforward.py
git commit -m "feat(common): 新增 tsfresh_walkforward — walk-forward proba + MA 通道 + 通道构成报告

- add_ma_channels:原地加 ma5/10/20 + rel_ma5,bfill 而非 fillna(0.0)
- report_channel_composition:ma*/rel_ma5 占比 > 33% 时打印 [WARN] 冗余风险
- tsfresh_walkforward_proba:FDR 严格限制在前 init_train_size 段(防泄漏),
  walk-forward 重训循环,bfill 默认行为(消除早期 0 跳变)

从原 vbt_combo / with_ma_channel / with_ma_grid_sector 三脚本抽出。"
```

---

### Task 2: 新增 `backtrace/common/vbt_jhzq_backtest.py`

**Files:**
- Create: `backtrace/common/vbt_jhzq_backtest.py`
- Create: `tests/test_vbt_jhzq_backtest.py`

**Interfaces:**
- Consumes: `F.summary_after_fees` from `jhzq_fees`(已有)、`vectorbt`(已有)
- Produces:
  - `compute_shares_per_trade(init_cash, max_pos_pct, init_open) -> int`
  - `build_proba_signals(proba, bar_index, *, entry_th, exit_th, shift_for_next_open=True) -> tuple[pd.Series, pd.Series]`
  - `run_vbt_backtest(ohlcv_df, entries, exits, stock_code, *, init_cash=100_000, max_pos_pct=0.95, upon_long_conflict='exit', print_rejection_warning=True) -> dict`
  - `fmt_money(x) -> str`、`fmt_pct(x) -> str`、`fmt_pp(x) -> str`

- [ ] **Step 1: 写失败的测试 — `compute_shares_per_trade`**

`tests/test_vbt_jhzq_backtest.py`:

```python
import pandas as pd
import numpy as np
import pytest

from common.vbt_jhzq_backtest import (
    compute_shares_per_trade,
    build_proba_signals,
    fmt_money, fmt_pct, fmt_pp,
)


def test_compute_shares_basic():
    # 100000 * 0.95 / 30 = 3166.67 → 31 手 → 3100 股
    assert compute_shares_per_trade(100_000, 0.95, 30.0) == 3100


def test_compute_shares_returns_zero_below_one_lot():
    assert compute_shares_per_trade(100_000, 0.95, 100_000.0) == 0


def test_compute_shares_handles_invalid_open():
    assert compute_shares_per_trade(100_000, 0.95, 0.0) == 0
    assert compute_shares_per_trade(100_000, 0.95, float('nan')) == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd c:/Users/yellow/mcp/qtTdx && PYTHONIOENCODING=utf-8 python -m pytest tests/test_vbt_jhzq_backtest.py -v`
Expected: `ModuleNotFoundError: No module named 'common.vbt_jhzq_backtest'`

- [ ] **Step 3: 写最小实现 — `compute_shares_per_trade`**

`backtrace/common/vbt_jhzq_backtest.py`:

```python
# -*- coding: utf-8 -*-
"""vbt + jhzq_fees 单次回测 + 格式化工具。

约定(写在本模块 docstring 顶部):
  1. **pf_zero reuse** — 复用 fees=0+slippage=0 的 pf_zero 拿 trades,
     jhzq_fees 后置单独算扣费。前提:策略的 entry/exit 判定逻辑与交易费用无关
     (signal 按价格穿越触发)。若未来新增"预期收益需覆盖手续费才 entry"类策略,
     需单独跑一次有费率 portfolio(原 vbt_combo.py:174-176 caveat 注释)。
  2. **80% 拒单 warning** — 当实际成交笔数 < 信号数 * 0.8 时打印 [WARN],
     原因:MAX_POS_PCT=0.95 + 固定 shares,股价上涨后资金不足导致 vbt 静默拒单
     (原 vbt_combo.py:212-222)。
  3. **friction_loss_pp 符号检查** — zero_friction_ret - net_ret 应恒 ≥ 0,
     负值说明 zero/net_ret 口径不一致或费率 bug(原 vbt_combo.py:234-235)。
"""
import numpy as np
import pandas as pd
import vectorbt as vbt

from common import jhzq_fees as F


def compute_shares_per_trade(init_cash, max_pos_pct, init_open):
    """每笔固定股数 = floor(init_cash * max_pos_pct / open0 / 100) * 100。
    返回 0 表示价格/仓位下没有 100 股整手(调用方应跳过该票)。"""
    if not np.isfinite(init_open) or init_open <= 0:
        return 0
    raw = init_cash * max_pos_pct / init_open
    if not np.isfinite(raw) or raw < 100:
        return 0
    return int(np.floor(raw / 100) * 100)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd c:/Users/yellow/mcp/qtTdx && PYTHONIOENCODING=utf-8 python -m pytest tests/test_vbt_jhzq_backtest.py::test_compute_shares_basic -v`
Expected: PASS

- [ ] **Step 5: 加 `build_proba_signals` 测试**

`tests/test_vbt_jhzq_backtest.py` 追加:

```python
def test_build_proba_signals_basic():
    idx = pd.date_range('2026-01-01', periods=10, freq='D')
    proba = pd.Series([0.4, 0.6, 0.7, 0.3, 0.8, 0.55, 0.2, 0.65, 0.45, 0.75], index=idx)
    bar_index = idx
    entries, exits = build_proba_signals(proba, bar_index, entry_th=0.55, exit_th=0.50)
    # shift(1) → 第 1 天对应 proba 第 0 天 = 0.4 → 不 entry
    # 第 2 天对应 proba 第 1 天 = 0.6 > 0.55 → entry
    assert entries.iloc[1] == True
    assert entries.iloc[0] == False
    # 第 7 天对应 proba 第 6 天 = 0.2 < 0.50 → exit
    assert exits.iloc[7] == True


def test_build_proba_signals_all_nan_returns_false():
    idx = pd.date_range('2026-01-01', periods=5, freq='D')
    proba = pd.Series([np.nan] * 5, index=idx)
    bar_index = idx
    entries, exits = build_proba_signals(proba, bar_index, entry_th=0.55, exit_th=0.50)
    assert not entries.any()
    assert not exits.any()
```

- [ ] **Step 6: 写 `build_proba_signals` 实现**

`backtrace/common/vbt_jhzq_backtest.py` 追加:

```python
def build_proba_signals(proba, bar_index, *, entry_th, exit_th,
                        shift_for_next_open=True):
    """proba reindex 到 bar_index 上 → 生成 (entries, exits) 布尔 Series,
    shift(1) 视作次日开盘成交(默认开启)。

    边界:aligned 全 NaN → 返回 (全 False, 全 False),不会崩。
    """
    aligned = proba.reindex(bar_index)
    entries = (aligned > entry_th).fillna(False).astype(bool)
    exits = (aligned < exit_th).fillna(False).astype(bool)
    if shift_for_next_open:
        entries = entries.shift(1).fillna(False).astype(bool)
        exits = exits.shift(1).fillna(False).astype(bool)
    return entries, exits
```

- [ ] **Step 7: 加格式化函数测试**

`tests/test_vbt_jhzq_backtest.py` 追加:

```python
def test_fmt_money_basic():
    assert fmt_money(1234.5) == '    1,234.50'


def test_fmt_money_handles_nan():
    assert 'N/A' in fmt_money(float('nan'))


def test_fmt_pct_basic():
    out = fmt_pct(0.123)
    assert '%' in out


def test_fmt_pct_handles_inf():
    assert fmt_pct(float('inf')) == '     inf'


def test_fmt_pp_basic():
    out = fmt_pp(2.5)
    assert 'pp' in out
```

- [ ] **Step 8: 写 3 个格式化函数**

`backtrace/common/vbt_jhzq_backtest.py` 追加:

```python
def fmt_money(x):
    """'   1,234.50' 格式;NaN → '          N/A'"""
    return f'{x:>12,.2f}' if pd.notna(x) else '          N/A'


def fmt_pct(x):
    """'   12.30%' 格式;inf → '     inf'"""
    if pd.notna(x) and x != float('inf'):
        return f'{x:>7.2%}'
    return '     inf'


def fmt_pp(x):
    """'   2.5pp' 格式;NaN → '  N/A'"""
    return f'{x:>6.1f}pp' if pd.notna(x) else '  N/A'
```

- [ ] **Step 9: 加 `run_vbt_backtest` 测试(纯逻辑部分)**

`tests/test_vbt_jhzq_backtest.py` 追加:

```python
def test_run_vbt_backtest_returns_dict_with_expected_keys():
    """小数据集 — 验证返回 dict 的 schema"""
    idx = pd.date_range('2026-01-01', periods=100, freq='D')
    np.random.seed(42)
    df = pd.DataFrame({
        'Open':   10 + np.cumsum(np.random.randn(100) * 0.1),
        'High':   10 + np.cumsum(np.random.randn(100) * 0.1) + 0.5,
        'Low':    10 + np.cumsum(np.random.randn(100) * 0.1) - 0.5,
        'Close':  10 + np.cumsum(np.random.randn(100) * 0.1),
        'Volume': np.random.uniform(1e6, 2e6, 100),
    }, index=idx)
    entries = pd.Series(False, index=idx)
    exits = pd.Series(False, index=idx)
    entries.iloc[10] = True
    exits.iloc[20] = True
    entries.iloc[40] = True
    exits.iloc[50] = True
    entries.iloc[70] = True
    exits.iloc[80] = True
    summary = run_vbt_backtest(df, entries, exits, 'TEST.SH',
                               init_cash=100_000, max_pos_pct=0.5)
    expected_keys = {'strategy', 'trades', 'gross_pnl', 'total_stamp',
                     'total_transfer', 'net_pnl', 'avg_net_per_trade',
                     'net_ret', 'win_rate', 'profit_factor', 'zero_friction_ret'}
    assert expected_keys.issubset(summary.keys())


def test_run_vbt_backtest_no_signals_returns_zero_summary():
    idx = pd.date_range('2026-01-01', periods=30, freq='D')
    df = pd.DataFrame({'Open': 10.0, 'High': 10.0, 'Low': 10.0,
                       'Close': 10.0, 'Volume': 1e6}, index=idx)
    entries = pd.Series(False, index=idx)
    exits = pd.Series(False, index=idx)
    summary = run_vbt_backtest(df, entries, exits, 'TEST.SH')
    assert summary['trades'] == 0
    assert summary['zero_friction_ret'] == 0.0
```

- [ ] **Step 10: 写 `run_vbt_backtest` 实现**

`backtrace/common/vbt_jhzq_backtest.py` 追加:

```python
def run_vbt_backtest(ohlcv_df, entries, exits, stock_code, *,
                     init_cash=100_000, max_pos_pct=0.95,
                     upon_long_conflict='exit',
                     print_rejection_warning=True):
    """跑 vbt + jhzq_fees 真实扣费的单次回测。

    返回 summary dict(11 列):
      strategy, trades, gross_pnl, total_stamp, total_transfer,
      net_pnl, avg_net_per_trade, net_ret, win_rate, profit_factor,
      zero_friction_ret

    副作用:
      - 当实际成交笔数 < 信号数 * 0.8 时打印 [WARN] 拒单警告
      - 当 friction_loss_pp < 0 时打印 [WARN] 口径不一致警告
    """
    base = {'strategy': stock_code, 'trades': 0,
            'gross_pnl': 0.0, 'total_stamp': 0.0, 'total_transfer': 0.0,
            'net_pnl': 0.0, 'avg_net_per_trade': 0.0,
            'net_ret': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0,
            'zero_friction_ret': 0.0}

    entry_signals = int(entries.sum())
    if entry_signals == 0:
        return base

    init_open = float(ohlcv_df['Open'].iloc[0])
    shares = compute_shares_per_trade(init_cash, max_pos_pct, init_open)
    if shares == 0:
        return base

    close = ohlcv_df['Close']
    open_ = ohlcv_df['Open']

    # ===== A. 零摩擦 portfolio(fees=0, slippage=0)=====
    pf_zero = vbt.Portfolio.from_signals(
        close=close, entries=entries, exits=exits, price=open_,
        init_cash=init_cash, fees=0, slippage=0, freq='D',
        size=shares, size_type='amount', size_granularity=100,
        upon_long_conflict=upon_long_conflict,
    )
    base['zero_friction_ret'] = pf_zero.total_return()

    # ===== B. 复用 pf_zero 的 trades 算 jhzq_fees =====
    # 前提:策略的 entry/exit 判定逻辑与交易费用无关
    # 若未来新增"预期收益需覆盖手续费才 entry"类策略,需单独跑有费率 portfolio
    trades = pf_zero.trades.records_readable
    if len(trades) == 0:
        return base

    summary = F.summary_after_fees(trades, stock_code)
    summary['strategy'] = stock_code
    summary['zero_friction_ret'] = base['zero_friction_ret']

    pnl_col = next(c for c in trades.columns if 'PnL' in c and '扣' not in c)
    wins = (trades[pnl_col] > 0).sum()
    summary['win_rate'] = wins / len(trades) if len(trades) > 0 else 0.0
    summary['profit_factor'] = (
        trades[pnl_col][trades[pnl_col] > 0].sum() /
        abs(trades[pnl_col][trades[pnl_col] < 0].sum())
        if (trades[pnl_col] < 0).sum() > 0 else float('inf')
    )
    summary['net_ret'] = summary['net_pnl'] / init_cash

    # 80% 拒单 warning
    actual = int(summary.get('trades', 0))
    if print_rejection_warning and actual < entry_signals * 0.8:
        print(f'   [WARN] 信号 {entry_signals} 个 → 实际成交 {actual} 笔 '
              f'({(1 - actual / entry_signals):.0%} 被拒)')
        print(f'          可能因 MAX_POS_PCT={max_pos_pct} 时股价上涨后资金不足;')
        print(f'          收益对比会失真,降 MAX_POS_PCT 或加现金补充')

    # friction_loss_pp 符号检查
    friction_loss_pp = (base['zero_friction_ret'] - summary['net_ret']) * 100
    if friction_loss_pp < 0:
        print(f'   [WARN] friction_loss_pp={friction_loss_pp:.1f} 负值,'
              f'检查 zero_friction_ret 与 net_ret 口径是否一致')

    return summary
```

- [ ] **Step 11: 跑全部测试**

Run: `cd c:/Users/yellow/mcp/qtTdx && PYTHONIOENCODING=utf-8 python -m pytest tests/test_vbt_jhzq_backtest.py -v`
Expected: 12 PASS

- [ ] **Step 12: 提交**

```bash
cd c:/Users/yellow/mcp/qtTdx
git add backtrace/common/vbt_jhzq_backtest.py tests/test_vbt_jhzq_backtest.py
git commit -m "feat(common): 新增 vbt_jhzq_backtest — vbt + jhzq_fees 单次回测 + 格式化

- compute_shares_per_trade:floor(init_cash * max_pos_pct / open0 / 100) * 100
- build_proba_signals:proba reindex → (entries, exits) 布尔 Series,shift(1) 默认开启
- run_vbt_backtest:11 列 summary,统一 pf_zero reuse + 80% 拒单 warning + friction_loss_pp 符号检查
- fmt_money / fmt_pct / fmt_pp:3 个格式化函数

从原 vbt_combo / with_ma_channel / with_ma_grid_sector 三脚本抽出。"
```

---

### Task 3: 重构 `backtrace/tsfresh/tsfresh_vbt_combo.py`

**Files:**
- Modify: `backtrace/tsfresh/tsfresh_vbt_combo.py`(274 行 → ~140 行)

**Interfaces consumed:**
- `W = common.tsfresh_walkforward`(from Task 1)
- `B = common.vbt_jhzq_backtest`(from Task 2)
- `P = common.tsfresh_pipeline`(已有)
- `tq` = `tqcenter.tq`(已有)

- [ ] **Step 1: 备份当前文件做参考(只读)**

Run: `cp c:/Users/yellow/mcp/qtTdx/backtrace/tsfresh/tsfresh_vbt_combo.py /tmp/tsfresh_vbt_combo.before.py`

- [ ] **Step 2: 完整重写脚本**

**删除:**
- L 47-118 数据加载 + walk-forward 全套循环(72 行)
- L 124-138 `build_signals()` 自定义(15 行)
- L 142-144 `shares_per_trade` 计算(3 行)
- L 147-197 `run_vbt_backtest()` 自定义(51 行)
- L 209-222 拒单 warning 手工实现(改走 common)(14 行)

**重写后的脚本(完整代码):**

`backtrace/tsfresh/tsfresh_vbt_combo.py`:

```python
# -*- coding: utf-8 -*-
"""tsfresh + VectorBT 集成 demo

4 组合网格:MA5 基线 + tsfresh 三个阈值,全部走 jhzq_fees 真实扣费
输出:tsfresh_vbt_grid_<code>_<start>_<end>.csv(每策略 trades + 净 PnL)
用法:验证 tsfresh 信号当 vbt entry/exit 是否比纯 MA5 基线多赚(扣费后)

瘦身后:walk-forward proba、vbt 回测、80% 拒单 warning 都走 common 模块。
"""
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import numpy as np
import pandas as pd
import vectorbt as vbt
from tqcenter import tq

from common import tsfresh_config as C
from common import tsfresh_pipeline as P
from common import tsfresh_walkforward as W
from common import vbt_jhzq_backtest as B
from common import jhzq_fees as F

tq.initialize(__file__)

# ============== 配置 ==============
STOCK_CODE   = '688318.SH'             # 沪市股,验证 SH 过户费分支
TARGET_START = '20250101'
TARGET_END   = '20251231'
WINDOW_MA    = 5
INIT_CASH    = 100_000
MAX_POS_PCT  = 0.95
INIT_TRAIN_SIZE = 200
STEP            = 50

# 4 组合网格
STRATEGIES = [
    {'name': 'MA5_baseline', 'kind': 'ma',     'entry_th': 0,  'exit_th': 0},
    {'name': 'tsfresh_p50',  'kind': 'proba',  'entry_th': 0.50, 'exit_th': 0.50},
    {'name': 'tsfresh_p55',  'kind': 'proba',  'entry_th': 0.55, 'exit_th': 0.50},
    {'name': 'tsfresh_p60',  'kind': 'proba',  'entry_th': 0.60, 'exit_th': 0.50},
]
# ===================================

# ---------- 1. 数据 ----------
print("=" * 70)
df = P.load_ohlcva(STOCK_CODE, verbose=True)
if df is None or len(df) == 0:
    print("❌ 数据缺失"); tq.close(); raise SystemExit(1)

df_demo = df.loc[TARGET_START:TARGET_END].copy()
print(f"\n回测区间:{df_demo.index[0].date()} → {df_demo.index[-1].date()}  ({len(df_demo)} 交易日)")

# ---------- 2. walk-forward proba(bfill 默认)----------
print("\n" + "=" * 70)
print("tsfresh + walk-forward proba...")
proba, X_sel = W.tsfresh_walkforward_proba(
    df, channels=['Open', 'High', 'Low', 'Close', 'Volume'],
    init_train_size=INIT_TRAIN_SIZE, step=STEP, verbose=True,
)
W.report_channel_composition(X_sel, label='OHLCV')

# ---------- 3. 4 套 entry/exit ----------
def build_signals(strategy):
    name = strategy['name']
    if strategy['kind'] == 'ma':
        ma = vbt.MA.run(df['Close'], window=WINDOW_MA).ma.ffill()
        entries = df['Close'].vbt.crossed_above(ma).shift(1).fillna(False).astype(bool)
        exits = df['Close'].vbt.crossed_below(ma).shift(1).fillna(False).astype(bool)
        return name, entries, exits
    entries, exits = B.build_proba_signals(
        proba, df.index,
        entry_th=strategy['entry_th'], exit_th=strategy['exit_th'],
    )
    return name, entries, exits


# ---------- 4. 跑 4 组合网格 ----------
print("=" * 70)
print("=== 4 组合网格回测 ===")
print("=" * 70)

results = []
for strat in STRATEGIES:
    name, entries, exits = build_signals(strat)
    summary = B.run_vbt_backtest(
        df, entries, exits, name,
        init_cash=INIT_CASH, max_pos_pct=MAX_POS_PCT,
    )
    results.append(summary)

# ---------- 5. 输出对比表 ----------
results_df = pd.DataFrame(results)
cols = ['strategy', 'trades', 'zero_friction_ret',
        'gross_pnl', 'total_stamp', 'total_transfer',
        'net_pnl', 'net_ret', 'win_rate', 'profit_factor', 'avg_net_per_trade']
results_df = results_df[cols].sort_values('net_pnl', ascending=False).reset_index(drop=True)
results_df['friction_loss_pp'] = (results_df['zero_friction_ret'] - results_df['net_ret']) * 100

print("\n" + "=" * 90)
print("=== 4 组合对比表 ===")
print("=" * 90)
print(f"{'策略':<16} {'笔数':>5} {'零摩擦':>8} {'毛收益':>12} "
      f"{'印花税':>8} {'过户费':>8} {'净收益':>12} {'净收益率':>8} "
      f"{'胜率':>6} {'盈亏比':>7} {'摩擦吃损':>8}")
print("-" * 120)
for _, r in results_df.iterrows():
    print(f"{r['strategy']:<16} {int(r['trades']):>5} "
          f"{B.fmt_pct(r['zero_friction_ret'])} "
          f"{B.fmt_money(r['gross_pnl'])} "
          f"{-r['total_stamp']:>8.2f} {-r['total_transfer']:>8.2f} "
          f"{B.fmt_money(r['net_pnl'])} {B.fmt_pct(r['net_ret'])} "
          f"{B.fmt_pct(r['win_rate'])} {B.fmt_pct(r['profit_factor'])} {B.fmt_pp(r['friction_loss_pp'])}")

# ---------- 6. 保存 ----------
out_csv = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/outputs",
    f'tsfresh_vbt_grid_{STOCK_CODE.replace(".", "_")}_{TARGET_START}_{TARGET_END}.csv',
)
results_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到 {out_csv}")

tq.close()
```

- [ ] **Step 3: 行数 + 导入检查**

Run:
```bash
wc -l c:/Users/yellow/mcp/qtTdx/backtrace/tsfresh/tsfresh_vbt_combo.py
# 期望:~140 行(原 274 行)
grep -n "from common import" c:/Users/yellow/mcp/qtTdx/backtrace/tsfresh/tsfresh_vbt_combo.py
# 期望:看到 tsfresh_walkforward + vbt_jhzq_backtest 两条新 import
```

- [ ] **Step 4: 冒烟测试(冒烟跑 1 只票)**

Run: `cd c:/Users/yellow/mcp/qtTdx && PYTHONIOENCODING=utf-8 python backtrace/tsfresh/tsfresh_vbt_combo.py`
Expected:
- 不抛异常
- 输出 CSV 落到 `backtrace/outputs/tsfresh_vbt_grid_688318_SH_20250101_20251231.csv`
- CSV 列名:`strategy,trades,zero_friction_ret,gross_pnl,total_stamp,total_transfer,net_pnl,net_ret,win_rate,profit_factor,avg_net_per_trade,friction_loss_pp`(12 列)
- 4 行(MA5_baseline + tsfresh_p50/55/60)

- [ ] **Step 5: CSV schema 对比(跟迁移前版本)**

Run:
```bash
cd c:/Users/yellow/mcp/qtTdx
python -c "
import pandas as pd
new = pd.read_csv('backtrace/outputs/tsfresh_vbt_grid_688318_SH_20250101_20251231.csv')
print('new cols:', list(new.columns))
print('new shape:', new.shape)
"
```

Expected: 12 列(包含 avg_net_per_trade 和 friction_loss_pp),4 行

- [ ] **Step 6: 提交**

```bash
cd c:/Users/yellow/mcp/qtTdx
git add backtrace/tsfresh/tsfresh_vbt_combo.py
git commit -m "refactor(tsfresh): tsfresh_vbt_combo 改用 common 模块(274 → 140 行)

- walk-forward proba 改调 W.tsfresh_walkforward_proba()(默认 bfill)
- vbt + jhzq_fees 改调 B.run_vbt_backtest()(80% warning 自动)
- build_signals 简化:tsfresh 模式走 B.build_proba_signals()
- 格式化函数改走 B.fmt_*

输出 CSV schema 12 列不变。"
```

---

### Task 4: 重构 `backtrace/tsfresh/tsfresh_with_ma_channel.py`

**Files:**
- Modify: `backtrace/tsfresh/tsfresh_with_ma_channel.py`(273 行 → ~140 行)

**Interfaces consumed:**
- `W = common.tsfresh_walkforward`(from Task 1)
- `B = common.vbt_jhzq_backtest`(from Task 2)
- `P = common.tsfresh_pipeline`(已有)

- [ ] **Step 1: 备份当前文件**

Run: `cp c:/Users/yellow/mcp/qtTdx/backtrace/tsfresh/tsfresh_with_ma_channel.py /tmp/tsfresh_with_ma_channel.before.py`

- [ ] **Step 2: 完整重写脚本**

**删除:**
- L 57-60 MA 通道手工计算(4 行)
- L 65-139 `build_tsfresh_proba()` + `_report_channel_composition()`(75 行)
- L 143-153 2 套 PROBA_CACHE 构建(改调 `W.tsfresh_walkforward_proba`)(10 行)
- L 161-182 `build_signals()`(22 行)
- L 186-220 `run_one()`(35 行)

**重写后的脚本(完整代码):**

`backtrace/tsfresh/tsfresh_with_ma_channel.py`:

```python
# -*- coding: utf-8 -*-
"""tsfresh + MA5 作独立通道 → vbt + jhzq_fees 真实扣费

核心创新:把 ma5 / ma10 / ma20 + close/ma5 偏离度 也作为通道喂给 tsfresh
对比 3 个 tsfresh 通道方案 + 1 个 MA5 基线
输出:tsfresh_with_ma_<code>_<start>_<end>.csv

⚠️ 已知陷阱:
  1. MA 通道与 Close 结构性共线 — 详见 W.report_channel_composition 自动诊断
  2. 阈值选择偏差 — 0.60 是从同一段 5 年回测网格搜索出的「最优」,二次拟合风险大。
     默认改用 0.55(非拟合中点);想保留 0.60 见 --entry-th 参数(已删除 CLI,
     当前直接改 STRATEGIES 配置)
  3. bfill 而非 fillna(0.0) — 由 W.tsfresh_walkforward_proba 默认保证
"""
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import numpy as np
import pandas as pd
import vectorbt as vbt
from tqcenter import tq

from common import tsfresh_config as C
from common import tsfresh_pipeline as P
from common import tsfresh_walkforward as W
from common import vbt_jhzq_backtest as B
from common import jhzq_fees as F

tq.initialize(__file__)

# ============== 配置 ==============
STOCK_CODE   = '688318.SH'
TARGET_START = '20250101'
TARGET_END   = '20251231'
INIT_CASH    = 100_000
MAX_POS_PCT  = 0.95
INIT_TRAIN_SIZE = 200
STEP = 50
EXIT_TSF = 0.50
ENTRY_TSF = 0.55   # 0.55 是非拟合中点,避开 0.60 的二次拟合偏差
# ===================================

# ---------- 1. 数据 ----------
print("=" * 70)
df = P.load_ohlcva(STOCK_CODE, verbose=True)
if df is None or len(df) == 0:
    print("❌ 数据缺失"); tq.close(); raise SystemExit(1)
df_demo = df.loc[TARGET_START:TARGET_END].copy()
print(f"\n回测区间:{df_demo.index[0].date()} → {df_demo.index[-1].date()}  ({len(df_demo)} 交易日)")

# ---------- 2. 计算 MA 通道(走 common)----------
df = W.add_ma_channels(df)

# ---------- 3. 2 套通道 proba ----------
PROBA_CACHE = {}
proba_basic, X_sel_basic = W.tsfresh_walkforward_proba(
    df, channels=['Open', 'High', 'Low', 'Close', 'Volume'],
    init_train_size=INIT_TRAIN_SIZE, step=STEP, verbose=True,
)
W.report_channel_composition(X_sel_basic, label='basic(5 通道 OHLCV)')
PROBA_CACHE['basic'] = proba_basic

proba_with_ma, X_sel_ma = W.tsfresh_walkforward_proba(
    df, channels=['Open', 'High', 'Low', 'Close', 'Volume',
                  'ma5', 'ma10', 'ma20', 'rel_ma5'],
    init_train_size=INIT_TRAIN_SIZE, step=STEP, verbose=True,
)
W.report_channel_composition(X_sel_ma, label='with_ma(8 通道 + MA)')
PROBA_CACHE['with_ma'] = proba_with_ma


# ---------- 4. 4 策略 ----------
def build_signals(strategy):
    name = strategy['name']
    if strategy['kind'] == 'ma':
        ma = vbt.MA.run(df['Close'], window=5).ma.ffill()
        entries = df['Close'].vbt.crossed_above(ma).shift(1).fillna(False).astype(bool)
        exits = df['Close'].vbt.crossed_below(ma).shift(1).fillna(False).astype(bool)
        return name, entries, exits

    proba = PROBA_CACHE[strategy['proba_key']]
    if strategy['mode'] == 'pure':
        entries, exits = B.build_proba_signals(
            proba, df.index, entry_th=ENTRY_TSF, exit_th=EXIT_TSF,
        )
    elif strategy['mode'] == 'confirmed':
        # MA5 触发 + tsfresh 确认(双条件 AND)
        ma = vbt.MA.run(df['Close'], window=5).ma.ffill()
        ma_entry = df['Close'].vbt.crossed_above(ma).shift(1).fillna(False).astype(bool)
        tsf_entry, _ = B.build_proba_signals(
            proba, df.index, entry_th=ENTRY_TSF, exit_th=EXIT_TSF,
        )
        entries = ma_entry & tsf_entry
        exits = df['Close'].vbt.crossed_below(ma).shift(1).fillna(False).astype(bool)
    return name, entries, exits


STRATEGIES = [
    {'name': 'MA5_baseline',          'kind': 'ma'},
    {'name': 'tsfresh_basic_p60',     'kind': 'tsfresh', 'proba_key': 'basic',   'mode': 'pure'},
    {'name': 'tsfresh_with_ma_p60',   'kind': 'tsfresh', 'proba_key': 'with_ma', 'mode': 'pure'},
    {'name': 'MA5_AND_with_ma_p60',   'kind': 'tsfresh', 'proba_key': 'with_ma', 'mode': 'confirmed'},
]

# ---------- 5. 跑 4 策略 ----------
print("=" * 70)
print("=== 4 策略对比 ===")
print("=" * 70)
results = []
for strat in STRATEGIES:
    name, entries, _ = build_signals(strat)
    summary = B.run_vbt_backtest(
        df, entries,
        pd.Series(False, index=df.index),  # exits 已在 build_signals 内部处理
        name,
        init_cash=INIT_CASH, max_pos_pct=MAX_POS_PCT,
    )
    # 注意:上面的简化丢了 exits 参数 — 修正:build_signals 应该返回 (entries, exits)
    results.append(summary)
```

Wait — there's a design issue with `run_vbt_backtest`. The function takes `entries, exits` but in the original, exits varies by strategy (ma vs tsfresh). I need to fix build_signals to return both.

Let me rewrite Step 2 correctly:

```python
def build_signals(strategy):
    """返回 (name, entries, exits)"""
    name = strategy['name']
    if strategy['kind'] == 'ma':
        ma = vbt.MA.run(df['Close'], window=5).ma.ffill()
        entries = df['Close'].vbt.crossed_above(ma).shift(1).fillna(False).astype(bool)
        exits = df['Close'].vbt.crossed_below(ma).shift(1).fillna(False).astype(bool)
        return name, entries, exits

    proba = PROBA_CACHE[strategy['proba_key']]
    if strategy['mode'] == 'pure':
        entries, exits = B.build_proba_signals(
            proba, df.index, entry_th=ENTRY_TSF, exit_th=EXIT_TSF,
        )
    elif strategy['mode'] == 'confirmed':
        ma = vbt.MA.run(df['Close'], window=5).ma.ffill()
        ma_entry = df['Close'].vbt.crossed_above(ma).shift(1).fillna(False).astype(bool)
        tsf_entry, _ = B.build_proba_signals(
            proba, df.index, entry_th=ENTRY_TSF, exit_th=EXIT_TSF,
        )
        entries = ma_entry & tsf_entry
        exits = df['Close'].vbt.crossed_below(ma).shift(1).fillna(False).astype(bool)
    return name, entries, exits


# ... in main loop:
for strat in STRATEGIES:
    name, entries, exits = build_signals(strat)
    summary = B.run_vbt_backtest(df, entries, exits, name,
                                  init_cash=INIT_CASH, max_pos_pct=MAX_POS_PCT)
    results.append(summary)
```

Let me redo Step 2 with the correct code.

- [ ] **Step 2: 完整重写脚本(corrected)**

`backtrace/tsfresh/tsfresh_with_ma_channel.py`:

```python
# -*- coding: utf-8 -*-
"""tsfresh + MA5 作独立通道 → vbt + jhzq_fees 真实扣费

核心创新:把 ma5 / ma10 / ma20 + close/ma5 偏离度 也作为通道喂给 tsfresh
对比 3 个 tsfresh 通道方案 + 1 个 MA5 基线
输出:tsfresh_with_ma_<code>_<start>_<end>.csv

⚠️ 已知陷阱:
  1. MA 通道与 Close 结构性共线 — 详见 W.report_channel_composition 自动诊断
  2. 阈值选择偏差 — 0.60 是从同一段 5 年回测网格搜索出的「最优」,二次拟合风险大。
     默认改用 0.55(非拟合中点)
  3. bfill 而非 fillna(0.0) — 由 W.tsfresh_walkforward_proba 默认保证

瘦身后:MA 通道、walk-forward proba、通道构成报告、vbt 回测都走 common 模块。
"""
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import numpy as np
import pandas as pd
import vectorbt as vbt
from tqcenter import tq

from common import tsfresh_config as C
from common import tsfresh_pipeline as P
from common import tsfresh_walkforward as W
from common import vbt_jhzq_backtest as B
from common import jhzq_fees as F

tq.initialize(__file__)

# ============== 配置 ==============
STOCK_CODE   = '688318.SH'
TARGET_START = '20250101'
TARGET_END   = '20251231'
INIT_CASH    = 100_000
MAX_POS_PCT  = 0.95
INIT_TRAIN_SIZE = 200
STEP = 50
EXIT_TSF = 0.50
ENTRY_TSF = 0.55
# ===================================

# ---------- 1. 数据 ----------
print("=" * 70)
df = P.load_ohlcva(STOCK_CODE, verbose=True)
if df is None or len(df) == 0:
    print("❌ 数据缺失"); tq.close(); raise SystemExit(1)
df_demo = df.loc[TARGET_START:TARGET_END].copy()
print(f"\n回测区间:{df_demo.index[0].date()} → {df_demo.index[-1].date()}  ({len(df_demo)} 交易日)")

# ---------- 2. 计算 MA 通道(走 common)----------
df = W.add_ma_channels(df)

# ---------- 3. 2 套通道 proba(bfill 默认)----------
PROBA_CACHE = {}
for key, chs in [
    ('basic',   ['Open', 'High', 'Low', 'Close', 'Volume']),
    ('with_ma', ['Open', 'High', 'Low', 'Close', 'Volume',
                 'ma5', 'ma10', 'ma20', 'rel_ma5']),
]:
    proba, X_sel = W.tsfresh_walkforward_proba(
        df, channels=chs, init_train_size=INIT_TRAIN_SIZE, step=STEP, verbose=True,
    )
    W.report_channel_composition(X_sel, label=f'{key}({len(chs)} 通道)')
    PROBA_CACHE[key] = proba


# ---------- 4. 4 策略 ----------
def build_signals(strategy):
    """返回 (name, entries, exits)"""
    name = strategy['name']
    if strategy['kind'] == 'ma':
        ma = vbt.MA.run(df['Close'], window=5).ma.ffill()
        entries = df['Close'].vbt.crossed_above(ma).shift(1).fillna(False).astype(bool)
        exits = df['Close'].vbt.crossed_below(ma).shift(1).fillna(False).astype(bool)
        return name, entries, exits

    proba = PROBA_CACHE[strategy['proba_key']]
    if strategy['mode'] == 'pure':
        entries, exits = B.build_proba_signals(
            proba, df.index, entry_th=ENTRY_TSF, exit_th=EXIT_TSF,
        )
    elif strategy['mode'] == 'confirmed':
        ma = vbt.MA.run(df['Close'], window=5).ma.ffill()
        ma_entry = df['Close'].vbt.crossed_above(ma).shift(1).fillna(False).astype(bool)
        tsf_entry, _ = B.build_proba_signals(
            proba, df.index, entry_th=ENTRY_TSF, exit_th=EXIT_TSF,
        )
        entries = ma_entry & tsf_entry
        exits = df['Close'].vbt.crossed_below(ma).shift(1).fillna(False).astype(bool)
    return name, entries, exits


STRATEGIES = [
    {'name': 'MA5_baseline',          'kind': 'ma'},
    {'name': 'tsfresh_basic_p60',     'kind': 'tsfresh', 'proba_key': 'basic',   'mode': 'pure'},
    {'name': 'tsfresh_with_ma_p60',   'kind': 'tsfresh', 'proba_key': 'with_ma', 'mode': 'pure'},
    {'name': 'MA5_AND_with_ma_p60',   'kind': 'tsfresh', 'proba_key': 'with_ma', 'mode': 'confirmed'},
]

# ---------- 5. 跑 4 策略 ----------
print("=" * 70)
print("=== 4 策略对比 ===")
print("=" * 70)
results = []
for strat in STRATEGIES:
    name, entries, exits = build_signals(strat)
    summary = B.run_vbt_backtest(
        df, entries, exits, name,
        init_cash=INIT_CASH, max_pos_pct=MAX_POS_PCT,
    )
    results.append(summary)

# ---------- 6. 输出对比表 ----------
results_df = pd.DataFrame(results)
cols = ['strategy', 'trades', 'zero_friction_ret',
        'gross_pnl', 'total_stamp', 'total_transfer',
        'net_pnl', 'net_ret', 'win_rate', 'profit_factor']
results_df = results_df[cols].sort_values('net_pnl', ascending=False).reset_index(drop=True)
results_df['friction_loss_pp'] = (results_df['zero_friction_ret'] - results_df['net_ret']) * 100

print("\n" + "=" * 110)
print("=== 4 策略对比表(basic vs with_ma vs MA5 复合) ===")
print("=" * 110)
print(f"{'策略':<22} {'笔数':>5} {'零摩擦':>8} "
      f"{'净收益':>12} {'净收益率':>8} {'胜率':>6} {'盈亏比':>8} {'摩擦吃损':>10}")
print("-" * 110)
for _, r in results_df.iterrows():
    print(f"{r['strategy']:<22} {int(r['trades']):>5} "
          f"{B.fmt_pct(r['zero_friction_ret'])} "
          f"{B.fmt_money(r['net_pnl'])} {B.fmt_pct(r['net_ret'])} "
          f"{B.fmt_pct(r['win_rate'])} {B.fmt_pct(r['profit_factor'])} "
          f"{B.fmt_pp(r['friction_loss_pp'])}")

# ---------- 7. 保存 ----------
out_csv = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/outputs",
    f'tsfresh_with_ma_grid_{STOCK_CODE.replace(".", "_")}_{TARGET_START}_{TARGET_END}.csv',
)
results_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到 {out_csv}")

tq.close()
```

- [ ] **Step 3: 行数 + 导入检查**

Run:
```bash
wc -l c:/Users/yellow/mcp/qtTdx/backtrace/tsfresh/tsfresh_with_ma_channel.py
# 期望:~140 行
grep -n "from common import" c:/Users/yellow/mcp/qtTdx/backtrace/tsfresh/tsfresh_with_ma_channel.py
# 期望:看到 tsfresh_walkforward + vbt_jhzq_backtest 两条新 import
```

- [ ] **Step 4: 冒烟测试**

Run: `cd c:/Users/yellow/mcp/qtTdx && PYTHONIOENCODING=utf-8 python backtrace/tsfresh/tsfresh_with_ma_channel.py`
Expected:
- 不抛异常
- 输出 4 策略对比
- 通道构成报告显示
- 输出 CSV: `backtrace/outputs/tsfresh_with_ma_grid_688318_SH_20250101_20251231.csv`

- [ ] **Step 5: CSV schema 对比**

Run:
```bash
cd c:/Users/yellow/mcp/qtTdx
python -c "
import pandas as pd
new = pd.read_csv('backtrace/outputs/tsfresh_with_ma_grid_688318_SH_20250101_20251231.csv')
print('new cols:', list(new.columns))
print('new shape:', new.shape)
"
```
Expected: 11 列(strategy/trades/zero_friction_ret/gross_pnl/total_stamp/total_transfer/net_pnl/net_ret/win_rate/profit_factor/friction_loss_pp),4 行

- [ ] **Step 6: 提交**

```bash
cd c:/Users/yellow/mcp/qtTdx
git add backtrace/tsfresh/tsfresh_with_ma_channel.py
git commit -m "refactor(tsfresh): with_ma_channel 改用 common 模块(273 → 140 行)

- MA 通道计算改调 W.add_ma_channels()
- walk-forward 2 套通道(basic + with_ma)改调 W.tsfresh_walkforward_proba()(bfill 默认)
- 通道构成报告改调 W.report_channel_composition()
- 'confirmed' 模式(MA5 AND tsfresh)简化后只剩 5 行
- vbt + jhzq_fees 改调 B.run_vbt_backtest()

输出 CSV schema 11 列不变。"
```

---

### Task 5: 重构 `backtrace/tsfresh/tsfresh_with_ma_grid_sector.py`

**Files:**
- Modify: `backtrace/tsfresh/tsfresh_with_ma_grid_sector.py`(195 行 → ~120 行)

**Interfaces consumed:**
- `W = common.tsfresh_walkforward`(from Task 1)
- `B = common.vbt_jhzq_backtest`(from Task 2)
- `P = common.tsfresh_pipeline`(已有)

**Important side effect:** grid_sector 当前在 X_all 上筛 FDR(泄漏),迁移后通过 `W.tsfresh_walkforward_proba` 自动获得 INIT_TRAIN_SIZE 限制 — 这是免费修好的潜在 bug。

- [ ] **Step 1: 备份当前文件**

Run: `cp c:/Users/yellow/mcp/qtTdx/backtrace/tsfresh/tsfresh_with_ma_grid_sector.py /tmp/tsfresh_with_ma_grid_sector.before.py`

- [ ] **Step 2: 完整重写脚本**

**删除:**
- L 40-69 `tsfresh_walkforward_proba()` 自定义(30 行)
- L 72-102 `run_backtest()` 自定义(31 行)

**重写后的脚本(完整代码):**

`backtrace/tsfresh/tsfresh_with_ma_grid_sector.py`:

```python
# -*- coding: utf-8 -*-
"""tsfresh_with_ma 跨通达信88 板块验证

对每只股票跑 basic vs with_ma 两套方案,看 with_ma 是否稳定胜出
输出:tsfresh_with_ma_grid_<code>_<start>_<end>.csv
用法:板块层面跑一遍,验证 with_ma 方案在多数票上是否真的更稳(不是过拟合单只票)

瘦身后:每只票的 walk-forward proba、MA 通道计算、vbt 回测都走 common 模块。
⚠️ 自动修复:grid_sector 原本在 X_all 上筛 FDR(泄漏!),迁移后自动获得
INIT_TRAIN_SIZE 限制。
"""
import warnings
warnings.filterwarnings('ignore')

import sys, os, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import numpy as np
import pandas as pd
from tqcenter import tq

from common import tsfresh_config as C
from common import tsfresh_pipeline as P
from common import tsfresh_walkforward as W
from common import vbt_jhzq_backtest as B
from common import jhzq_fees as F

tq.initialize(__file__)

# ============== 配置 ==============
SECTOR_NAME   = '通达信88'
TARGET_START  = '20250101'
TARGET_END    = '20251231'
INIT_CASH     = 100_000
MAX_POS_PCT   = 0.95

INIT_TRAIN_SIZE = 200
STEP = 50
ENTRY_TSF = 0.60
EXIT_TSF  = 0.50

CHANNELS_BASIC = ['Open', 'High', 'Low', 'Close', 'Volume']
CHANNELS_MA    = CHANNELS_BASIC + ['ma5', 'ma10', 'ma20', 'rel_ma5']
# ===================================


def run_one_strategy(df, proba, stock_code):
    """单只股票跑 vbt + jhzq_fees,返回 summary dict"""
    entries, exits = B.build_proba_signals(
        proba, df.index, entry_th=ENTRY_TSF, exit_th=EXIT_TSF,
    )
    return B.run_vbt_backtest(
        df, entries, exits, stock_code,
        init_cash=INIT_CASH, max_pos_pct=MAX_POS_PCT,
        print_rejection_warning=False,  # 88 票循环,关掉单只 warning 避免刷屏
    )


# ---------- 1. 拉全板块 ----------
print("=" * 70)
print(f"[{SECTOR_NAME}] 拉板块全部成员...")
stock_data = P.load_sector(sector_name=SECTOR_NAME, verbose=True)
print("=" * 70)
if not stock_data:
    print("❌ 板块数据为空"); tq.close(); raise SystemExit(1)

# ---------- 2. 对每只股票跑 2 方案 ----------
results = []
total = len(stock_data)
t_start = time.time()

for i, (code, raw) in enumerate(stock_data.items(), 1):
    df = raw.loc[TARGET_START:TARGET_END].copy()
    if len(df) < 100:
        print(f"  [{i:2d}/{total}] {code} 样本不足 ({len(df)}),跳过")
        continue

    df = W.add_ma_channels(df)

    print(f"  [{i:2d}/{total}] {code} ({len(df)} 交易日) ...")
    rec = {'stock': code, 'n_days': len(df)}

    # basic
    try:
        proba_basic, _ = W.tsfresh_walkforward_proba(
            df, channels=CHANNELS_BASIC,
            init_train_size=INIT_TRAIN_SIZE, step=STEP, verbose=False,
        )
        r_basic = run_one_strategy(df, proba_basic, code)
        rec['basic_net_ret'] = r_basic.get('net_ret')
        rec['basic_trades']  = r_basic.get('trades', 0)
        rec['basic_winrate'] = r_basic.get('win_rate')
    except Exception as e:
        print(f"     basic 失败:{e}")
        rec['basic_net_ret'] = None

    # with_ma
    try:
        proba_ma, _ = W.tsfresh_walkforward_proba(
            df, channels=CHANNELS_MA,
            init_train_size=INIT_TRAIN_SIZE, step=STEP, verbose=False,
        )
        r_ma = run_one_strategy(df, proba_ma, code)
        rec['with_ma_net_ret'] = r_ma.get('net_ret')
        rec['with_ma_trades']  = r_ma.get('trades', 0)
        rec['with_ma_winrate'] = r_ma.get('win_rate')
    except Exception as e:
        print(f"     with_ma 失败:{e}")
        rec['with_ma_net_ret'] = None

    rec['with_ma_wins'] = (
        (rec.get('with_ma_net_ret') or -1) > (rec.get('basic_net_ret') or -1)
        if rec.get('with_ma_net_ret') is not None and rec.get('basic_net_ret') is not None
        else None
    )
    results.append(rec)

    elapsed = time.time() - t_start
    eta = elapsed / i * (total - i)
    print(f"     basic={rec.get('basic_net_ret', 'NA')}, with_ma={rec.get('with_ma_net_ret', 'NA')},"
          f" 胜={rec.get('with_ma_wins', 'NA')} | 累计 {elapsed:.0f}s,ETA {eta:.0f}s")

# ---------- 3. 汇总 ----------
df_res = pd.DataFrame(results)
df_res['alpha_diff_pp'] = (df_res['with_ma_net_ret'] - df_res['basic_net_ret']) * 100

print("\n" + "=" * 90)
print("=== 88 板块 basic vs with_ma 汇总 ===")
print("=" * 90)
both_valid = df_res.dropna(subset=['basic_net_ret', 'with_ma_net_ret'])
print(f"有效股票数:{len(both_valid)} / {len(df_res)}")
print(f"with_ma 胜出(>basic):{(both_valid['with_ma_wins'] == True).sum()} 只"
      f"  ({(both_valid['with_ma_wins'] == True).mean():.1%})")
print(f"with_ma 跑赢平均提升:{both_valid['alpha_diff_pp'].mean():+.2f} pp(中位数 {both_valid['alpha_diff_pp'].median():+.2f} pp)")
print(f"\n平均净收益:basic={both_valid['basic_net_ret'].mean():.2%}  with_ma={both_valid['with_ma_net_ret'].mean():.2%}")
print(f"中位净收益:basic={both_valid['basic_net_ret'].median():.2%}  with_ma={both_valid['with_ma_net_ret'].median():.2%}")

top = both_valid.sort_values('with_ma_net_ret', ascending=False).head(10)
print("\n=== Top 10 with_ma 净收益最高 ===")
print(top[['stock', 'basic_net_ret', 'with_ma_net_ret', 'alpha_diff_pp']].to_string(index=False))

# ---------- 4. 保存 ----------
out_csv = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/outputs",
    f'tsfresh_with_ma_sector_{SECTOR_NAME}_{TARGET_START}_{TARGET_END}.csv',
)
df_res.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到 {out_csv}")

tq.close()
```

- [ ] **Step 3: 行数 + 导入检查**

Run:
```bash
wc -l c:/Users/yellow/mcp/qtTdx/backtrace/tsfresh/tsfresh_with_ma_grid_sector.py
# 期望:~120 行
grep -n "from common import" c:/Users/yellow/mcp/qtTdx/backtrace/tsfresh/tsfresh_with_ma_grid_sector.py
```

- [ ] **Step 4: 冒烟测试(限 5 只票加速)**

为节省时间,临时改 `SECTOR_NAME` 或限制跑 5 只。最简办法:在脚本里把 `for i, (code, raw) in enumerate(stock_data.items(), 1):` 改成 `for i, (code, raw) in enumerate(list(stock_data.items())[:5], 1):`,跑完恢复。

Run:
```bash
cd c:/Users/yellow/mcp/qtTdx
# 临时改 5 只限制(用 sed 替换循环起始行)
python -c "
import re
p = 'backtrace/tsfresh/tsfresh_with_ma_grid_sector.py'
s = open(p).read()
s = s.replace('for i, (code, raw) in enumerate(stock_data.items(), 1):',
              'for i, (code, raw) in enumerate(list(stock_data.items())[:5], 1):')
open(p, 'w').write(s)
"
PYTHONIOENCODING=utf-8 python backtrace/tsfresh/tsfresh_with_ma_grid_sector.py
# 跑完恢复
python -c "
p = 'backtrace/tsfresh/tsfresh_with_ma_grid_sector.py'
s = open(p).read()
s = s.replace('for i, (code, raw) in enumerate(list(stock_data.items())[:5], 1):',
              'for i, (code, raw) in enumerate(stock_data.items(), 1):')
open(p, 'w').write(s)
"
```
Expected:
- 不抛异常
- 跑完 5 只票,每只 basic + with_ma 都跑通
- 输出 CSV 落到 `backtrace/outputs/tsfresh_with_ma_sector_通达信88_20250101_20251231.csv`

- [ ] **Step 5: CSV schema 对比**

Run:
```bash
cd c:/Users/yellow/mcp/qtTdx
python -c "
import pandas as pd
new = pd.read_csv('backtrace/outputs/tsfresh_with_ma_sector_通达信88_20250101_20251231.csv')
print('new cols:', list(new.columns))
print('new shape:', new.shape)
"
```
Expected: 列名与原 schema 一致(stock / n_days / basic_net_ret / basic_trades / basic_winrate / with_ma_net_ret / with_ma_trades / with_ma_winrate / with_ma_wins / alpha_diff_pp)

- [ ] **Step 6: 验证 INIT_TRAIN_SIZE 修复生效**

跑 grid_sector 时,在脚本里某次跑输出应该有「FDR 显著 N 列 (前 200 个样本筛)」日志(由 common 模块保证)。

Run: 跑 `tsfresh_with_ma_grid_sector.py` 时观察日志,确认出现「FDR 显著 X 列 (前 200 个样本筛)」字样。

Expected: 看到该日志(原来 grid_sector 没有这行 — 是迁移后新增的行为)

- [ ] **Step 7: 提交**

```bash
cd c:/Users/yellow/mcp/qtTdx
git add backtrace/tsfresh/tsfresh_with_ma_grid_sector.py
git commit -m "refactor(tsfresh): with_ma_grid_sector 改用 common 模块(195 → 120 行)

- 88 票循环里 walk-forward 改调 W.tsfresh_walkforward_proba()
- MA 通道计算改调 W.add_ma_channels()
- vbt + jhzq_fees 改调 B.run_vbt_backtest()(print_rejection_warning=False 避免刷屏)
- 自动修复:FDR 现在限制在前 INIT_TRAIN_SIZE 段(原版在 X_all 上筛有泄漏)

输出 CSV schema 不变。"
```

---

### Task 6: 全量回归测试

**Files:** 不修改,只跑测试。

- [ ] **Step 1: 跑所有单元测试**

Run: `cd c:/Users/yellow/mcp/qtTdx && PYTHONIOENCODING=utf-8 python -m pytest tests/ -v`
Expected: 全部 PASS(原 test_data_store + test_fallback + test_fetch_helpers + 新 test_tsfresh_walkforward + test_vbt_jhzq_backtest)

- [ ] **Step 2: 跑 3 个原脚本冒烟(每次 1 只票)**

```bash
cd c:/Users/yellow/mcp/qtTdx
PYTHONIOENCODING=utf-8 python backtrace/tsfresh/tsfresh_vbt_combo.py
PYTHONIOENCODING=utf-8 python backtrace/tsfresh/tsfresh_with_ma_channel.py
PYTHONIOENCODING=utf-8 python backtrace/tsfresh/tsfresh_with_ma_grid_sector.py
```
Expected: 3 个脚本都不抛异常,各自输出 CSV

- [ ] **Step 3: 对比 3 个 CSV 的 schema 列**

```bash
cd c:/Users/yellow/mcp/qtTdx
python -c "
import pandas as pd
for name, path in [
    ('vbt_combo', 'backtrace/outputs/tsfresh_vbt_grid_688318_SH_20250101_20251231.csv'),
    ('with_ma_channel', 'backtrace/outputs/tsfresh_with_ma_grid_688318_SH_20250101_20251231.csv'),
    ('with_ma_sector', 'backtrace/outputs/tsfresh_with_ma_sector_通达信88_20250101_20251231.csv'),
]:
    df = pd.read_csv(path)
    print(f'{name:<20} cols={list(df.columns)}')
    print(f'{\"\":<20} shape={df.shape}')
"
```
Expected: 列名跟设计 spec 一致(vbt_combo 12 列 / with_ma_channel 11 列 / with_ma_sector 10 列)

- [ ] **Step 4: git log 确认提交历史**

Run: `cd c:/Users/yellow/mcp/qtTdx && git log --oneline -8`
Expected:
```
xxxxxx refactor(tsfresh): with_ma_grid_sector 改用 common 模块
xxxxxx refactor(tsfresh): with_ma_channel 改用 common 模块
xxxxxx refactor(tsfresh): vbt_combo 改用 common 模块
xxxxxx feat(common): 新增 vbt_jhzq_backtest
xxxxxx feat(common): 新增 tsfresh_walkforward
```

- [ ] **Step 5: 完结,等待用户验收**

跑完所有冒烟后告知用户:实施完成,4 个 commit 已 push 到本地,等待用户跑全量 88 票回归(预计 ~3.5 小时)。

---

## Self-Review

### Spec coverage check:

- ✅ `add_ma_channels` (spec §1.1) → Task 1 Steps 3, 4
- ✅ `report_channel_composition` (spec §1.1) → Task 1 Steps 5, 6
- ✅ `tsfresh_walkforward_proba` (spec §1.1) → Task 1 Steps 8, 9
- ✅ bfill 约定 (spec §1.2) → Task 1 Step 3 (add_ma_channels bfill) + Step 9 (tsfresh_walkforward_proba bfill default)
- ✅ FDR INIT_TRAIN_SIZE 限制 (spec §1.2) → Task 1 Step 9 (`X_train0 = X_all.iloc[:init_train_size]`)
- ✅ X_sel 列对齐 (spec §1.2) → Task 1 Step 9 (`X_sel = X_all[selected_cols]`)
- ✅ `compute_shares_per_trade` (spec §2.1) → Task 2 Steps 1, 3
- ✅ `run_vbt_backtest` (spec §2.1) → Task 2 Steps 9, 10
- ✅ 80% 拒单 warning (spec §2.1) → Task 2 Step 10
- ✅ friction_loss_pp 符号检查 (spec §2.1) → Task 2 Step 10
- ✅ `build_proba_signals` (spec §2.1) → Task 2 Steps 5, 6
- ✅ 3 个 fmt 函数 (spec §2.1) → Task 2 Steps 7, 8
- ✅ vbt_combo 瘦身 274→140 (spec §3.1) → Task 3
- ✅ with_ma_channel 瘦身 273→140 (spec §3.2) → Task 4
- ✅ grid_sector 瘦身 195→120 (spec §3.3) → Task 5
- ✅ grid_sector 泄漏修复 (spec §3.3 + §5.3) → Task 5 Step 6
- ✅ pf_zero reuse caveat 注释 (Global Constraint #4) → Task 2 Step 10 docstring 顶部
- ✅ 阈值偏差警告 (Global Constraint #5) → Task 1 Step 9 docstring 顶部
- ✅ 每个 commit 后冒烟 (Global Constraint #8) → Task 3/4/5 Step 4, Task 6 Step 2

### Placeholder scan:

- ❌ 无 "TBD"/"TODO"/"implement later" — 已检查
- ❌ 无 "add appropriate error handling" — 错误处理具体写在了 ValueError 抛出 + pytest.raises
- ❌ 无 "similar to Task N" — 每个 task 都是完整代码
- ❌ 无 "write tests for the above" — 测试代码完整提供

### Type consistency check:

- ✅ `W.tsfresh_walkforward_proba` 返回 `tuple[pd.Series, pd.DataFrame]` 在 Task 1 定义,Task 3/4/5 调用一致
- ✅ `B.run_vbt_backtest` 接受 `(ohlcv_df, entries, exits, stock_code, *, ...)` 在 Task 2 定义,Task 3/4/5 调用一致
- ✅ `B.build_proba_signals(proba, bar_index, *, entry_th, exit_th)` 在 Task 2 定义,Task 4 多次调用一致
- ✅ `B.fmt_money/fmt_pct/fmt_pp` 在 Task 2 定义,Task 3/4 调用一致
- ✅ `W.add_ma_channels(ohlcv_df)` 在 Task 1 定义,Task 4/5 调用一致
- ✅ `W.report_channel_composition(X_sel, label=...)` 在 Task 1 定义,Task 3/4 调用一致

无 type 一致性问题。