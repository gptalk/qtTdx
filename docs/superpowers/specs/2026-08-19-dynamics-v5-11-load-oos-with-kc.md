# Spec v5.11 — `load_oos_predictions` × parameter_fit integration

> **Date:** 2026-08-19
> **Base:** v5.10 full-market OOS distribution (`c50b248`)
> **Branch:** new (from `main` HEAD = `c50b248`)

## 1. 问题

v5.9 `dynamics_oos_viz.py::load_oos_predictions` 返回 `k_used: float` 和 `c_used: float`,但当前实现是**占位符**:

```python
# dynamics_oos_viz.py:115-116
k_used = float(k) if k is not None else 0.0
c_used = float(c) if c is not None else 0.0
```

只有 caller 显式传 `--k`/`--c` 时才有非零值;否则永远 0.0。

**业务后果**:
- v5.10 README §4.1.9 已 caveat:`k_used` / `c_used` 永远 0.0 — 实际值需 `parameter_fit.py` 集成
- v5.10 `compute_oos_metrics` 把这俩字段透传给 `metrics_list`,但永远是 0.0
- v5.10 dashboard 不能直接展示"哪些股票是 k 主导 / c 主导 / 平衡"
- 业务决策时被迫假设所有股票 k=c=0(纯 F_self 驱动),而**真实拟合值可能让排序完全反转**

**已有基础设施**:
- `backtrace/projection/parameter_fit.py` 已跑出 `data/projection/kc_estimates.csv`(每只票 `(k̂, ĉ)`)
- v5.2 已把 `kc_estimates.csv` 接入 `dynamics_forced_response.py`(4 函数 + `--from-kc-estimates` flag),但只用于行业级聚合,不喂回单股 `load_oos_predictions`
- v5.9 `load_oos_predictions` 签名已接受 `k=None, c=None`,但**没有"从 CSV 查表"的入口**

## 2. 目标

**核心**:给 `load_oos_predictions` 加 `kc_estimates_path: str | None = None` 参数。当提供时,函数按 `stock_code` 在 CSV 里查 `(k̂, ĉ)`,命中就用真实拟合值(覆盖 `k=None` / `c=None` 的 0.0 fallback);未命中保留原 fallback 行为。

**业务价值**:v5.10 全市场 OOS 分布的每个点都有真实的 (k̂, ĉ) 标注,业务方可以一眼看出"高频共振"vs"过阻尼"vs"纯动量"股票的预测质量差异。

**非目标(YAGNI)**:
- ❌ 不动 `parameter_fit.py` 本身 — v5.11 只**消费**它的输出
- ❌ 不做行业级聚合 — `dynamics_forced_response.py` v5.2 已覆盖
- ❌ 不做滚动拟合 `kc_rolling_*.csv` 查询 — 只用 full-sample
- ❌ 不改 v5.10 dashboard 视觉(2×2 + top-N) — 只让 k_used/c_used 字段变实
- ❌ 不在 `compute_oos_metrics` 加新列(列 schema 不变)
- ❌ 不做参数稳定性检查(filter 掉 c<0 之类的异常值)— 用户应自己保证 CSV 干净

**理由**:
- v5.9 函数签名已具备扩展点(可选 `k`/`c`),v5.11 复用该扩展点
- v5.2 在 `dynamics_forced_response.py` 已有 `load_kc_estimates()` 模板,v5.11 复制该模式(不耦合 import)
- 单文件改动 + 1 CLI flag + 1 test,scope 最小

## 3. 设计

### 3.1 架构

```
                ┌────────────────────────────────────────┐
                │ dynamics_oos_viz.py (v5.11 扩展)       │
                │                                        │
                │  load_oos_predictions(                 │
                │      stock_code,                       │
                │      days,                             │
                │      ...,                              │
                │      kc_estimates_path=None, ← NEW     │
                │  ):                                    │
                │    1) 拉数据 (load_pair)               │
                │    2) motion projection                │
                │    3) compute_dynamics                 │
                │    4) k/c 查找:                        │
                │       if kc_estimates_path:            │
                │         lookup_kc_for_code(            │
                │             kc_estimates_path,         │
                │             stock_code,                │
                │         ) → (k_fit, c_fit)            │
                │       if found: k_used, c_used = fit   │
                │       elif k/c 参数显式传: 用 k, c    │
                │       else: 0.0 (现有 fallback)        │
                │    5) 1 步预测主循环                   │
                │    6) return dict (含真实 k_used, c_used)│
                └────────────────────────────────────────┘
```

### 3.2 查找优先级(显式契约)

| `--k` / `--c` | `kc_estimates_path` | CSV 命中 | 结果 |
|---|---|---|---|
| ✅ 传了 | — | — | 用 `k`, `c` (现有行为) |
| ❌ 未传 | ✅ 传了 | ✅ | 用 CSV 的 (k̂, ĉ) |
| ❌ 未传 | ✅ 传了 | ❌ | 0.0 + WARNING log(查不到 code) |
| ❌ 未传 | ❌ 未传 | — | 0.0 (现有 fallback) |

**注**:caller 显式传 `--k`/`--c` 始终优先(v5.9 既定行为不变)。

### 3.3 v5.11 新 API

```python
def lookup_kc_for_code(
    kc_csv_path: str,
    stock_code: str,
) -> tuple[float, float] | None:
    """从 parameter_fit kc_estimates.csv 查单只票的 (k̂, ĉ)。

    Args:
        kc_csv_path: kc_estimates.csv 路径
        stock_code: e.g. '600118.SH'

    Returns:
        (k_hat, c_hat) if found and status='ok'
        None if not found OR status != 'ok' OR 必需列缺失 OR 文件不存在

    注:
        - 不抛异常(契约:None = "查不到,继续 fallback")
        - 只匹配 `code == stock_code AND status == 'ok'` 的行
        - 必需列:code, k_hat, c_hat, status(其他列可选)
    """
```

```python
def load_oos_predictions(
    stock_code: str,
    days: int = DEFAULTS['days'],
    *,
    prefer_industry: bool = DEFAULTS['prefer_industry'],
    k: float | None = None,
    c: float | None = None,
    lambda_q: float | None = None,
    f_self_window: int = 10,
    kc_estimates_path: str | None = None,  # ← v5.11 NEW
) -> dict:
    """[v5.9 docstring retained] + v5.11 kc_estimates_path:
        若提供,按查找优先级表解析 (k_used, c_used)。
    """
```

### 3.4 CLI 扩展

`dynamics_oos_viz.py` main() 加 1 个 flag:

```python
p.add_argument('--kc-estimates-csv', dest='kc_estimates_csv', type=str, default=None,
               help='v5.11: parameter_fit kc_estimates.csv 路径(为 None 则用现有 0.0 fallback)')
```

### 3.5 传播到 v5.10

`compute_oos_metrics` (v5.10 `dynamics_oos_batch.py`) **0 修改**:它已经透传 `k, c` 给 `load_oos_predictions`。当 `k=None, c=None` 时,加 `--kc-estimates-csv` 到 v5.10 main() 让所有股票都用真实 (k̂, ĉ)。

`v5.10 main()` 增 1 个 flag:

```python
p.add_argument('--kc-estimates-csv', dest='kc_estimates_csv', type=str, default=None,
               help='v5.11: 透传给 compute_oos_metrics → load_oos_predictions')
```

`compute_oos_metrics` 签名加 1 个 keyword-only 参数透传:

```python
def compute_oos_metrics(
    stock_code: str,
    days: int = 250,
    *,
    prefer_industry: bool = True,
    k: float | None = None,
    c: float | None = None,
    f_self_window: int = 10,
    kc_estimates_path: str | None = None,  # ← v5.11 NEW
) -> dict:
```

### 3.6 数据流总结

```
parameter_fit.py (已存在)
  ↓ writes kc_estimates.csv
data/projection/kc_estimates.csv
  ↓ read by lookup_kc_for_code (v5.11 NEW)
load_oos_predictions (v5.9 + v5.11 扩展)
  ↓ returns k_used, c_used (now real when CSV provided)
compute_oos_metrics (v5.10 + v5.11 透传)
  ↓ returns {k_used, c_used} in metrics_list
build_full_market_oos_html (v5.10, 0 修改)
  ↓ HTML output — k_used/c_used 现在是真实值
```

## 4. 验证

### 4.1 单元测试

```python
# tests/test_dynamics_oos_viz.py 或 test_dynamics_eigen.py 新增 test

def test_lookup_kc_for_code(tmp_path):
    """v5.11 — lookup_kc_for_code 单元测试 (无 subprocess,快速)"""
    # 1. 写 mock kc_estimates.csv
    csv = tmp_path / 'kc.csv'
    csv.write_text('code,index_code,k_hat,c_hat,status\n'
                   '600118.SH,801010.SH,0.5,0.3,ok\n'
                   '000001.SZ,801020.SH,0.8,0.4,ok\n'
                   '999999.SH,801030.SH,0.1,0.2,failed\n', encoding='utf-8')

    # 2. 命中 ok
    k, c = lookup_kc_for_code(str(csv), '600118.SH')
    assert (k, c) == (0.5, 0.3)

    # 3. status != 'ok' → None
    assert lookup_kc_for_code(str(csv), '999999.SH') is None

    # 4. code 不存在 → None
    assert lookup_kc_for_code(str(csv), '000777.SZ') is None

    # 5. 文件不存在 → None(不抛)
    assert lookup_kc_for_code(str(tmp_path / 'missing.csv'), '600118.SH') is None

    # 6. 缺必需列 → None
    bad = tmp_path / 'bad.csv'
    bad.write_text('code,foo,bar\n600118.SH,1,2\n', encoding='utf-8')
    assert lookup_kc_for_code(str(bad), '600118.SH') is None
```

### 4.2 集成测试 (F3 inverted tolerance)

扩展现有 `test_cli_oos_viz_mode` + 新增 `test_cli_oos_viz_with_kc`:

```python
def test_cli_oos_viz_with_kc(tmp_path):
    """v5.11 — load_oos_predictions 用真实 kc_estimates 跑 E2E。

    流程:
        1. 写 mock kc_estimates.csv 到 tmp_path
        2. 用 --kc-estimates-csv 跑 CLI
        3. 验证生成的 HTML 包含 k̂ / ĉ 标题(而非 0.0000)
    """
    # ... subprocess.run with --kc-estimates-csv tmp/kc.csv
    # ... F3 inverted tolerance (本地缓存缺失 skip)
    # ... 断言 HTML contains 'k̂' 或 'k̂' (k + combining circumflex)
```

### 4.3 CLI 端到端(冒烟)

```bash
# 1. 先跑 parameter_fit 产生真实 CSV
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe \
    backtrace/projection/parameter_fit.py --limit 5

# 2. 用真实 CSV 跑 v5.11 OOS 可视化
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe \
    backtrace/dynamics/dynamics_oos_viz.py --code 000001.SZ --days 60 \
    --kc-estimates-csv data/projection/kc_estimates.csv \
    --output backtrace/outputs/_smoke_v5_11_oos.html

# 3. 用真实 CSV 跑 v5.10 全市场
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe \
    backtrace/dynamics/dynamics_oos_batch.py --days 60 --limit 5 --top-n 3 \
    --kc-estimates-csv data/projection/kc_estimates.csv \
    --output backtrace/outputs/_smoke_v5_11_batch.html
```

期望:HTML 标题/副标题里出现真实 (k̂, ĉ) 值(非 0.0000)。

## 5. 约束

- 0 modifications to 11 protected files + `dynamics_oos_batch.py` (v5.10)
- 0 new dependencies(pandas 已在)
- 1 new file:无
- 2 modified files:`backtrace/dynamics/dynamics_oos_viz.py` (核心), `tests/test_dynamics_eigen.py` (+1 test)
- 1 propagated change:`backtrace/dynamics/dynamics_oos_batch.py` main() + compute_oos_metrics 签名透传 kc_estimates_path

## 6. 与 v5.x 系列的关系

| 版 | commit | 主题 | k_used/c_used 状态 |
|---|---|---|---|
| v5.9 | d692860 | load_oos_predictions + 4-row plotly | 0.0 fallback(占位) |
| v5.10 | c50b248 | 全市场分布 + top-N | 同上 + README §4.1.9 caveat |
| **v5.11** | (本次) | **kc_estimates 接入单股** | **真实 (k̂, ĉ)** |

v5.11 是 v5.9 框架的"数据接入"层,对应 v5.2 在 `dynamics_forced_response.py` 的同款升级。

## 7. 不在范围 / 后续

- v5.12: dashboard 加 "k 主导 vs c 主导 vs 平衡" 颜色编码(基于真实 k_used/c_used)
- v5.13: 滚动拟合 `kc_rolling_*.csv` 接入(`days=250` 时取最匹配窗口)
- v5.14: M1 tsfresh 根因 fix(整个 dynamics/ CLI 共享的 shadow 问题)
- v5.15: 状态转移热图(P(state[t+1] | state[t]))