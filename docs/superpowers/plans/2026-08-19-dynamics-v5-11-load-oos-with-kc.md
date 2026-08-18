# v5.11 Implementation Plan — `load_oos_predictions` × parameter_fit integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close v5.10 README §4.1.9 caveat (`k_used`/`c_used` 永远 0.0) by letting `load_oos_predictions` look up real (k̂, ĉ) from `parameter_fit`'s `kc_estimates.csv` when `--kc-estimates-csv` is provided.

**Architecture:** Add `lookup_kc_for_code()` helper + `kc_estimates_path: str | None = None` keyword to `load_oos_predictions` (v5.9). When provided, function looks up the stock's (k̂, ĉ) and uses them for prediction. Propagate through `compute_oos_metrics` (v5.10) + 1 CLI flag on each of `dynamics_oos_viz.py` and `dynamics_oos_batch.py`.

**Tech Stack:** Python 3.x, pandas (already installed), `parameter_fit` CSV output, `load_oos_predictions` from v5.9.

## Global Constraints

- 0 modifications to 11 protected files (named in spec §6)
- 0 modifications to `dynamics_oos_batch.py` core (only signature extension + CLI flag pass-through allowed)
- 0 new dependencies (pandas already installed)
- 0 new files (extend existing)
- 1 new test in `tests/test_dynamics_eigen.py` (or new `test_dynamics_oos_viz.py`)
- M1 tsfresh shadow tolerated as documented limitation
- F3 inverted tolerance: only documented failures skip

## File Structure

| File | Change | Net lines |
|---|---|---|
| `backtrace/dynamics/dynamics_oos_viz.py` | +1 helper + signature + main flag | ~+60 |
| `backtrace/dynamics/dynamics_oos_batch.py` | +1 sig param + 1 CLI flag | ~+5 |
| `tests/test_dynamics_eigen.py` | +1 unit test (lookup_kc) | ~+50 |
| `backtrace/dynamics/README.md` | §4.1.10 added | ~+25 |
| `docs/superpowers/specs/...v5-11...md` | status footer | +5 |
| `docs/superpowers/plans/...v5-11...md` | status footer | +5 |
| Memory file + MEMORY.md | new | +50 |

Total: 6 files, ~+200 lines

---

## Task 1: `lookup_kc_for_code` helper + signature extension on `load_oos_predictions`

**Files:**
- Modify: `backtrace/dynamics/dynamics_oos_viz.py:80-187` (signature + body)
- Modify: `backtrace/dynamics/dynamics_oos_viz.py:212-220` (`__all__`)

**Interfaces:**
- Consumes: nothing (new helper)
- Produces: `lookup_kc_for_code(kc_csv_path: str, stock_code: str) -> tuple[float, float] | None`
- Produces: `load_oos_predictions` accepts new keyword `kc_estimates_path: str | None = None`

**Required additions (verbatim):**

### 1a. New helper (insert AFTER `_label_from_a`)

```python
def lookup_kc_for_code(
    kc_csv_path: str,
    stock_code: str,
) -> tuple[float, float] | None:
    """从 parameter_fit kc_estimates.csv 查单只票的 (k̂, ĉ)。

    契约:返回 None = 查不到(caller 继续 fallback)。不抛异常。

    Returns:
        (k_hat, c_hat) if found AND status == 'ok'
        None if file missing, code not found, status != 'ok', or 必需列缺失

    必需列:code, k_hat, c_hat, status
    """
    import os
    REQUIRED = ['code', 'k_hat', 'c_hat', 'status']
    if not os.path.exists(kc_csv_path):
        log.warning(f"[v5.11] kc_estimates 文件不存在: {kc_csv_path}")
        return None
    try:
        df = pd.read_csv(kc_csv_path, encoding='utf-8-sig')
    except Exception as e:
        log.warning(f"[v5.11] kc_estimates 读失败 ({type(e).__name__}): {e}")
        return None
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        log.warning(f"[v5.11] kc_estimates 缺必需列: {missing}")
        return None
    rows = df[df['code'] == stock_code]
    rows = rows[rows['status'] == 'ok']
    if len(rows) == 0:
        return None
    row = rows.iloc[0]
    try:
        return (float(row['k_hat']), float(row['c_hat']))
    except (ValueError, TypeError) as e:
        log.warning(f"[v5.11] kc_estimates 解析 (k, c) 失败: {e}")
        return None
```

### 1b. Modify `load_oos_predictions` signature

Add keyword `kc_estimates_path: str | None = None` after `f_self_window`:

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
    kc_estimates_path: str | None = None,  # v5.11 NEW
) -> dict:
```

### 1c. Modify k/c fallback block (lines 114-116)

Replace:
```python
# 4) k / c 兜底(brief §6.4-6.5;dyn 不返回 k_hat/c_hat,默认 0)
k_used = float(k) if k is not None else 0.0
c_used = float(c) if c is not None else 0.0
```

With:
```python
# 4) k / c 查找优先级(spec §3.2):
#    显式 k/c > kc_estimates 命中 > 0.0 fallback
k_used = float(k) if k is not None else 0.0
c_used = float(c) if c is not None else 0.0
if (k_used == 0.0 and c_used == 0.0) and kc_estimates_path:
    fit = lookup_kc_for_code(kc_estimates_path, stock_code)
    if fit is not None:
        k_used, c_used = fit
        log.info(f"[v5.11] {stock_code}: 使用 (k̂={k_used:.4f}, ĉ={c_used:.4f}) from {kc_estimates_path}")
```

### 1d. Add `__all__` entry

Add `'lookup_kc_for_code'` to the `__all__` list (line 213-220).

### 1e. Verify imports

`pd` and `log` are already imported in this file. `os` is already imported (top). Confirm — if not, add `import os` (should be already there from v5.9 setup).

### Smoke verification (Step 1.5)

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -c "
import sys, os
sys.path.insert(0, 'c:/Users/yellow/mcp/qtTdx/backtrace')
import tempfile, pandas as pd
from dynamics.dynamics_oos_viz import lookup_kc_for_code

# mock CSV
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
    f.write('code,index_code,k_hat,c_hat,status\n')
    f.write('600118.SH,801010.SH,0.5,0.3,ok\n')
    f.write('000001.SZ,801020.SH,0.8,0.4,ok\n')
    f.write('999999.SH,801030.SH,0.1,0.2,failed\n')
    path = f.name

print('hit ok:', lookup_kc_for_code(path, '600118.SH'))
print('hit ok2:', lookup_kc_for_code(path, '000001.SZ'))
print('status!=ok:', lookup_kc_for_code(path, '999999.SH'))
print('not found:', lookup_kc_for_code(path, '000777.SZ'))
print('missing file:', lookup_kc_for_code('/tmp/nonexistent_xyz.csv', '600118.SH'))

os.unlink(path)
"
```

Expected:
```
hit ok: (0.5, 0.3)
hit ok2: (0.8, 0.4)
status!=ok: None
not found: None
missing file: None
```

### Report contract

```
# Task 1 Report

## Status
DONE | DONE_WITH_CONCERNS | NEEDS_CONTEXT | BLOCKED

## Commit
<hash> — feat(dynamics): v5.11 — lookup_kc_for_code helper + kc_estimates_path param

## Test summary
<one-line: "OK 5/5 mock cases (hit/ok/status!=ok/missing file/missing col)">

## Concerns (if any)
```

---

## Task 2: CLI `--kc-estimates-csv` flag on `dynamics_oos_viz.py`

**Files:**
- Modify: `backtrace/dynamics/dynamics_oos_viz.py:420-460` (main() + parse_args)

**Interfaces:**
- Consumes: `load_oos_predictions` (v5.11) with new `kc_estimates_path` param
- Produces: 1 new CLI flag `--kc-estimates-csv`

### 2a. Add CLI flag in parse_args (find the argparse block in `main()`)

Locate the existing `parse_args()` call in main() — it's inside `main()`, not a separate function. Add this flag:

```python
p.add_argument('--kc-estimates-csv', dest='kc_estimates_csv', type=str, default=None,
               help='v5.11: parameter_fit kc_estimates.csv 路径(为 None 则用现有 0.0 fallback)')
```

### 2b. Pass through to load_oos_predictions

In the `main()` body, locate the call to `load_oos_predictions(...)` and add the new kwarg:

```python
data = load_oos_predictions(
    args.code,
    days=args.days,
    prefer_industry=args.prefer_industry,
    kc_estimates_path=args.kc_estimates_csv,  # v5.11 NEW
)
```

(Other params may exist; just add the new one.)

### 2c. Log line update

If main() prints a summary line, augment to include `(k̂, ĉ)` if non-zero:

```python
log.info(f"[v5.11/v5.9] {args.code}: k_used={data['k_used']:.4f}, c_used={data['c_used']:.4f}, n_oos={len(data['common_idx'])}")
```

### Smoke verification (Step 2.2)

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe backtrace/dynamics/dynamics_oos_viz.py --help
```

Expected: `--kc-estimates-csv PATH` appears in help output.

If `data/projection/kc_estimates.csv` exists from a prior parameter_fit run:

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe backtrace/dynamics/dynamics_oos_viz.py \
    --code 000001.SZ --days 60 \
    --kc-estimates-csv data/projection/kc_estimates.csv \
    --output backtrace/outputs/_smoke_v5_11_oos.html
```

Expected: log line shows non-zero `(k̂, ĉ)` values; HTML generated.

If `kc_estimates.csv` does NOT exist: smoke should still run with `k_used=c_used=0.0` and log warning about missing file.

### Report contract

```
# Task 2 Report

## Status
DONE | DONE_WITH_CONCERNS | NEEDS_CONTEXT | BLOCKED

## Commit
<hash> — feat(dynamics): v5.11 — --kc-estimates-csv CLI flag on dynamics_oos_viz

## Test summary
<one-line>

## Concerns (if any)
```

---

## Task 3: Propagate `kc_estimates_path` through `compute_oos_metrics` + v5.10 CLI flag

**Files:**
- Modify: `backtrace/dynamics/dynamics_oos_batch.py` (compute_oos_metrics signature + main() flag)

**Interfaces:**
- Consumes: `load_oos_predictions` (v5.11) with new `kc_estimates_path` param
- Produces: `compute_oos_metrics` accepts new keyword `kc_estimates_path: str | None = None`
- Produces: 1 new CLI flag `--kc-estimates-csv` on `dynamics_oos_batch.py`

### 3a. Modify `compute_oos_metrics` signature

Add keyword `kc_estimates_path: str | None = None` after `f_self_window`:

```python
def compute_oos_metrics(
    stock_code: str,
    days: int = 250,
    *,
    prefer_industry: bool = True,
    k: float | None = None,
    c: float | None = None,
    f_self_window: int = 10,
    kc_estimates_path: str | None = None,  # v5.11 NEW
) -> dict:
```

### 3b. Pass through to `load_oos_predictions`

In the body, locate the call to `load_oos_predictions(...)` and add the new kwarg:

```python
d = load_oos_predictions(
    stock_code=code,
    days=days,
    prefer_industry=prefer_industry,
    kc_estimates_path=kc_estimates_path,  # v5.11 NEW
)
```

### 3c. Add CLI flag in main() (v5.10 batch CLI)

Locate the `argparse.ArgumentParser` block in `main()`. Add:

```python
p.add_argument('--kc-estimates-csv', dest='kc_estimates_csv', type=str, default=None,
               help='v5.11: 透传给 compute_oos_metrics → load_oos_predictions')
```

### 3d. Pass through in main() loop

In the `for idx, code in enumerate(codes, ...)` loop, add to the `compute_oos_metrics` call:

```python
m = compute_oos_metrics(
    stock_code=code,
    days=args.days,
    prefer_industry=args.prefer_industry,
    kc_estimates_path=args.kc_estimates_csv,  # v5.11 NEW
)
```

### 3e. Log line update

In the same loop, augment the log to include k_used/c_used if non-zero:

```python
log.info(f"[{idx}/{len(codes)}] {code}: hit={m['hit_rate']:.3f}, RMSE={m['rmse']:.4f}, "
         f"k̂={m['k_used']:.4f}, ĉ={m['c_used']:.4f}")
```

### Smoke verification (Step 3.3)

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe backtrace/dynamics/dynamics_oos_batch.py --help
```

Expected: `--kc-estimates-csv PATH` appears.

If `kc_estimates.csv` exists:
```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe backtrace/dynamics/dynamics_oos_batch.py \
    --days 60 --limit 3 --top-n 2 \
    --kc-estimates-csv data/projection/kc_estimates.csv \
    --output backtrace/outputs/_smoke_v5_11_batch.html
```

Expected: log line shows non-zero `(k̂, ĉ)` per stock; both HTMLs generated.

### Report contract

```
# Task 3 Report

## Status
DONE | DONE_WITH_CONCERNS | NEEDS_CONTEXT | BLOCKED

## Commit
<hash> — feat(dynamics): v5.11 — propagate kc_estimates_path through v5.10

## Test summary
<one-line>

## Concerns (if any)
```

---

## Task 4: Test `test_lookup_kc_for_code` + full suite + README §4.1.10 + final review + push + memory

**Files:**
- Modify: `tests/test_dynamics_eigen.py` (+1 unit test, ~50 lines)
- Modify: `backtrace/dynamics/README.md` (+§4.1.10, ~25 lines)
- Modify: `docs/superpowers/specs/2026-08-19-dynamics-v5-11-load-oos-with-kc.md` (status footer)
- Modify: `docs/superpowers/plans/2026-08-19-dynamics-v5-11-load-oos-with-kc.md` (status footer)
- Create: `C:\Users\yellow\.claude\projects\c--Users-yellow-mcp-qtTdx\memory\dynamics-v5-11-load-oos-with-kc.md`
- Modify: `C:\Users\yellow\.claude\projects\c--Users-yellow-mcp-qtTdx\memory\MEMORY.md` (+1 line)

### 4a. Add `test_lookup_kc_for_code` to `tests/test_dynamics_eigen.py`

Append at end of file:

```python
def test_lookup_kc_for_code(tmp_path):
    """v5.11 — lookup_kc_for_code 单元测试 (no subprocess, fast)."""
    from backtrace.dynamics.dynamics_oos_viz import lookup_kc_for_code

    # 1. mock kc_estimates.csv
    csv = tmp_path / 'kc.csv'
    csv.write_text('code,index_code,k_hat,c_hat,status\n'
                   '600118.SH,801010.SH,0.5,0.3,ok\n'
                   '000001.SZ,801020.SH,0.8,0.4,ok\n'
                   '999999.SH,801030.SH,0.1,0.2,failed\n', encoding='utf-8')

    # 2. hit ok
    assert lookup_kc_for_code(str(csv), '600118.SH') == (0.5, 0.3)
    # 3. status != 'ok' → None
    assert lookup_kc_for_code(str(csv), '999999.SH') is None
    # 4. code 不存在 → None
    assert lookup_kc_for_code(str(csv), '000777.SZ') is None
    # 5. 文件不存在 → None
    assert lookup_kc_for_code(str(tmp_path / 'missing.csv'), '600118.SH') is None
    # 6. 缺必需列 → None
    bad = tmp_path / 'bad.csv'
    bad.write_text('code,foo,bar\n600118.SH,1,2\n', encoding='utf-8')
    assert lookup_kc_for_code(str(bad), '600118.SH') is None
```

### 4b. README §4.1.10 (after §4.1.9)

```markdown
### 4.1.10 v5.11 — `load_oos_predictions` × parameter_fit integration

**File:** `backtrace/dynamics/dynamics_oos_viz.py` (extended)

**Goal:** Close §4.1.9 caveat (`k_used`/`c_used`=0). When `--kc-estimates-csv PATH` is provided, `load_oos_predictions` looks up real (k̂, ĉ) from `parameter_fit`'s output CSV and uses them for 1-step prediction.

**New helper:** `lookup_kc_for_code(kc_csv_path, stock_code) -> tuple[float, float] | None`
- Returns None (no exception) when file missing, code not found, status != 'ok', or 必需列缺失.

**Lookup priority** (spec §3.2):
1. Explicit `--k`/`--c` (caller override)
2. `--kc-estimates-csv` hit (real (k̂, ĉ))
3. 0.0 fallback (existing behavior)

**CLI:**
```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_oos_viz.py \
    --code 000001.SZ --days 60 \
    --kc-estimates-csv data/projection/kc_estimates.csv \
    --output backtrace/outputs/dynsys_oos_000001.html
```

**v5.10 propagation:** `dynamics_oos_batch.py::compute_oos_metrics` accepts `kc_estimates_path: str | None = None` + new CLI flag `--kc-estimates-csv`. All stocks in batch get real (k̂, ĉ) when CSV provided.

**Test:** `tests/test_dynamics_eigen.py::test_lookup_kc_for_code` (6 cases, no subprocess).
```

### 4c. Spec status footer (append to end of spec)

```markdown

## Status: ✅ DONE — 2026-08-19

4 tasks complete:
- Task 1: `lookup_kc_for_code` helper + signature — `<commit>`
- Task 2: `--kc-estimates-csv` flag on `dynamics_oos_viz` — `<commit>`
- Task 3: propagate through `compute_oos_metrics` + v5.10 CLI — `<commit>`
- Task 4: test + README + final review + push — `<commit>`

Final: 78 PASS + 0 SKIP (was 77 + 1 new test), 0 modifications to 11 protected files + `dynamics_oos_batch.py`, 0 new dependencies.
```

### 4d. Plan status footer (append to end of plan)

```markdown

## Status: ✅ DONE — 2026-08-19

All 4 tasks complete. Pushed to origin/main in commit `<push_commit_hash>`. See memory file `dynamics-v5-11-load-oos-with-kc.md`.
```

### 4e. Final code review (opus)

Generate full diff:
```bash
git diff 56f7799..HEAD > .superpowers/sdd/2026-08-19-dynamics-v5-11-load-oos-with-kc/final-review-package.txt
```

Dispatch opus final reviewer (same format as v5.10). Wait for PASS.

### 4f. Push to origin/main

```bash
git push origin main
```

### 4g. Memory file

`C:\Users\yellow\.claude\projects\c--Users-yellow-mcp-qtTdx\memory\dynamics-v5-11-load-oos-with-kc.md`:

```markdown
---
name: dynamics-v5-11-load-oos-with-kc
description: v5.11 load_oos_predictions 接 parameter_fit kc_estimates.csv 真实 (k̂, ĉ),关闭 v5.10 §4.1.9 占位符 caveat,78 tests PASS
metadata:
  type: project
---

v5.11 在 v5.9 `load_oos_predictions` 上加 `kc_estimates_path: str | None = None` keyword:传 CSV 路径时按 stock_code 查 `parameter_fit` 输出的真实 (k̂, ĉ) 用于 1 步预测,关闭 v5.10 README §4.1.9「k_used/c_used 永远 0.0」caveat。

**1 新 helper**:`lookup_kc_for_code(kc_csv_path, stock_code) -> tuple[float, float] | None`
- 契约:返回 None = 查不到(caller 继续 fallback),**不抛异常**
- 处理 4 种 None 情况:文件不存在 / code 不在 / status != 'ok' / 必需列缺失
- 必需列:code, k_hat, c_hat, status(其他列可选)

**查找优先级**(spec §3.2):
1. 显式 `--k` / `--c`(caller 最高优先)
2. `--kc-estimates-csv` 命中(真实拟合)
3. 0.0 fallback(v5.9 既有行为)

**2 个 CLI flag(都加,1 个 propagation)**:
- `dynamics_oos_viz.py --kc-estimates-csv PATH`
- `dynamics_oos_batch.py --kc-estimates-csv PATH`(v5.10 main 透传给 compute_oos_metrics)
- `compute_oos_metrics` 签名加 `kc_estimates_path: str | None = None`

**1 新 test**:`test_lookup_kc_for_code`,6 case (hit ok / status != ok / code not found / missing file / missing cols) — 无 subprocess,快速

**与 v5.2 的对称**:v5.2 在 `dynamics_forced_response.py` 已加同款 4 helper + `--from-kc-estimates` flag,但只用于行业级 overlay;v5.11 把同款思路应用到 v5.9 单股 OOS 预测。

**关联**:[[dynamics-v5-9-oos-prediction-html]] / [[dynamics-v5-10-full-market-oos-distribution]] / [[dynamics-v5-2-parameter-fit-integration]]

**Why:** 让 v5.10 全市场 OOS 分布每个点都有真实 (k̂, ĉ) — 业务方看 dashboard 时直接知道哪些股票是 k 主导(共振风险)vs c 主导(过阻尼稳定)vs 平衡(v5.9 之前的占位 0.0 框架里看不到这层)。

**How to apply:** 任何用 v5.9 / v5.10 的脚本想升级到真实参数,加 `--kc-estimates-csv data/projection/kc_estimates.csv`(前提是先跑 `parameter_fit.py` 产生该 CSV)。
```

Then add to MEMORY.md:
```
- [dynamics-v5-11-load-oos-with-kc](dynamics-v5-11-load-oos-with-kc.md) — v5.11 load_oos_predictions 接 parameter_fit 真实 (k̂, ĉ),78 tests
```

### Commit sequence

After all 4 tasks reviewed, commit + push in this order:
1. README + spec/plan status + memory (single commit):
   ```
   docs(dynamics): v5.11 — README §4.1.10 + spec/plan/memory
   
   Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
   ```
2. Push to origin/main

### Report contract

```
# Task 4 Report

## Status
DONE | DONE_WITH_CONCERNS | NEEDS_CONTEXT | BLOCKED

## Final state
- Commits: <list>
- Push: <commit hash on origin/main>
- Test count: 78 PASS + 0 SKIP
- Final review verdict: PASS

## Concerns (if any)
```

### Global Constraints (recap)

- 0 modifications to 11 protected files
- 0 modifications to `dynamics_oos_batch.py` core (only sig + flag pass-through, NO logic change)
- 0 new dependencies
- M1 tsfresh shadow tolerated
- F3 inverted tolerance in any new subprocess test (none added for v5.11 — only unit test for lookup)

## Status: ✅ DONE — 2026-08-19

All 4 tasks + 1 mid-stream fix complete. Pushed to origin/main in 8 commits `c50b248..5882a4b`:
- 75fc840 (Task 1): `lookup_kc_for_code` helper + `kc_estimates_path` param
- 7dd55d1 (Task 2): `--kc-estimates-csv` CLI flag on `dynamics_oos_viz`
- 0850679 (v5.11.1 fix): status filter for verbose format
- f6ba836 (Task 3): propagate `kc_estimates_path` through `compute_oos_metrics` + v5.10 CLI
- 6717d0f (Task 4): `test_lookup_kc_for_code` (6 cases)
- 33e5b08 (Task 4): README §4.1.10
- 0ac1730 (Task 4): spec/plan status + memory file
- 5882a4b (post-review touchup): §4.1.9 caveat update + orphan test line cleanup

Final: 78 PASS + 0 SKIP, 0 modifications to 11 protected files + `dynamics_oos_viz.py` core logic, 0 new dependencies. v5.10 §4.1.9 caveat closed. Opus final review PASS. See memory file `dynamics-v5-11-load-oos-with-kc.md`.