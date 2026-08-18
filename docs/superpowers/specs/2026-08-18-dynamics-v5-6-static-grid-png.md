# Spec v5.6 — Static 2D Grid (matplotlib PNG export)

> **Date:** 2026-08-18
> **Base:** v5.5 Regime Color Coding (`179a579` on main)
> **Branch:** modification only to `backtrace/dynamics/dynamics_si_freq_response.py`

## 1. 问题

v5.3-v5.5 把 `dynamics_si_freq_response` 从"单帧 overlay"扩到"动画 dual-pane + regime 颜色" — 业务可在浏览器拖 slider 看时序漂移。

但业务**写报告 / 演示**时需要**静态 PNG**(嵌 PDF / PPT)。当前:
- ❌ HTML 不能直接嵌入 PDF
- ❌ plotly 静态导出需 `kaleido` 依赖(env 没装)
- ❌ 业务只能截图(糊,不可重复)
- ❌ 多 date 时需要多张图,布局难保持

v5.6 用 **matplotlib**(已装, v5 tests 间接验证)生成**2D 网格 PNG** — **rows = 各 asof_date, cols = (|H| dB, ∠H deg)**, 1 张图把全部 dates 的 Bode 摆出来, 业务打印或嵌 PDF 即可。

## 2. 目标

**核心**: 新函数 `build_static_bode_grid(pairs_per_date, omega_grid, output_path, ...)` 用 matplotlib 生成 2D 网格 PNG — rows = unique asof_dates, cols = (|H(jω)| dB, ∠H(jω) deg), 每格画 N industries 的 Bode 曲线(颜色 = regime,复用 v5.5 4 色)。

**非目标(YAGNI)**:
- ❌ 不装 `kaleido` — matplotlib 已足够,避免新依赖
- ❌ 不做 SVG 输出 — PNG 嵌 PDF/PPT 够用,SVG 是炫技
- ❌ 不做交互(双击跳转等)— 静态图,纯展示
- ❌ 不做动图(GIF/MP4)— 已用 HTML slider
- ❌ 不导出 EPS/PDF — print 用 PNG 够
- ❌ 不做 hover tooltip — 静态图无交互

**理由**:
- 0 新依赖(env 已有 matplotlib 3.10.6)
- 业务价值: 报告/演示可嵌入静态图(SVG/PNG 自由)
- 复用 v5.5 regime 颜色逻辑(4 色 hex 不变)
- 1 函数 mod(新)+ 1 CLI flag + 1 test 新增

## 3. 设计

### 3.1 架构

```
backtrace/dynamics/dynamics_si_freq_response.py
  [v5.5 已有]  build_animated_overlay_html(...)  → HTML
  [v5.6 新增]  build_static_bode_grid(...)      → PNG (matplotlib)
  [v5.6 新增]  import matplotlib.pyplot / matplotlib.dates
  [v5.6 复用]  classify_response_type (v5 已有) → regime 分类
  [v5.6 复用]  magnitude_phase (v5 已有)         → 频率响应
  [v5.6 复用]  v5.5 _regime_color 字典           → 4 色
```

新增 1 个 module-level dict `REGIME_COLORS` (与 v5.5 闭包同 dict, 4 hex), `build_static_bode_grid` 用之。`build_animated_overlay_html` 内的 `_regime_color` 闭包不变(避免交叉修改)。

### 3.2 2D 网格布局

```
            |H(jω)| dB              ∠H(jω) deg
Date 1     [Bode curves]            [Phase curves]
Date 2     [Bode curves]            [Phase curves]
Date 3     [Bode curves]            [Phase curves]
...
```

- `n_rows = len(unique_dates)` (从 `pairs_per_date` 派生)
- `n_cols = 2` (|H| + ∠H)
- `figsize = (12, 4 * n_rows)` (高 4 inches / row, 12 inches wide)
- `dpi = 100` (默认, 1000×N px)
- 每格共享 x 轴(`sharex=True`), 共享 col y 轴(`sharey='col'`)

### 3.3 v5.6 修改范围

**新增 1 函数 + 1 CLI flag + main() 1 行**:

**File 1**: `backtrace/dynamics/dynamics_si_freq_response.py`
- 新 import: `import matplotlib.pyplot as plt`, `import matplotlib` (date axis 不需要)
- 新 module-level dict: `REGIME_COLORS = {'overdamped': '#2ca02c', 'critical': '#ff7f0e', 'underdamped': '#d62728', 'anti_damped': '#9467bd'}`
- 新函数 `build_static_bode_grid(pairs_per_date, omega_grid, output_path, title='...', dpi=100)`:
  - 派生 unique_dates
  - 创建 figure + subplots(n_rows, 2, sharex=True, figsize=(12, 4*n_rows))
  - 每个 date 一行, col 0 = |H| dB, col 1 = ∠H deg
  - 每个 industry 1 条曲线, 颜色 = `classify_response_type(k, c)` → `REGIME_COLORS`
  - 第 1 行 col 0 加 legend (industry label)
  - 第 1 行 col 1 加 plotly-style 注释(4 色 ↔ regime 映射)
  - 标题 + xlabel (只在底行)+ ylabel (只在左列)
  - `fig.tight_layout()`
  - `fig.savefig(output_path, dpi=dpi, bbox_inches='tight')`
- 新 CLI flag `--static-output PATH` (default: `backtrace/outputs/dynsys_si_freq_response_static.png`)
- main() 末尾 1 行: `build_static_bode_grid(pairs, omega_grid, args.static_output)`

**File 2**: `tests/test_dynamics_eigen.py`
- 新增 1 test `test_cli_static_grid_mode`:
  - 复用现有 fixture 模式(3 dates × 2 industries)
  - 跑 CLI subprocess 加 `--static-output` arg
  - 验证 PNG 文件存在 + size > 5000 bytes + header 是 PNG 字节(`b'\\x89PNG'`)

**File 3**: `backtrace/dynamics/README.md`
- §4.1.5 加 v5.6 footnote (1 段)

### 3.4 CLI 扩展

```bash
# v5.5 (不变)
python backtrace/dynamics/dynamics_si_freq_response.py

# v5.6 新增 static PNG export
python backtrace/dynamics/dynamics_si_freq_response.py --static-output PATH
# 默认: backtrace/outputs/dynsys_si_freq_response_static.png
```

### 3.5 输出(全 gitignored, 与 v5.5 并列)

| 路径 | 触发 | 大小 |
|---|---|---|
| HTML | 默认 | ~400KB (v5.5 不变) |
| TXT | 默认 | ~5KB (v5.5 不变) |
| CSV | 默认 | ~3KB (v5.5 不变) |
| **PNG** | **新增** `--static-output` | **~50-200KB** (n_rows 决定) |

### 3.6 测试

**新增 1 test**: `test_cli_static_grid_mode` (CLI subprocess, 复用 v5.3 fixture 模式)
- 合成 3 dates × 2 industries CSV → 跑 CLI 加 `--static-output` → 验证 PNG 文件存在 + PNG 字节头 + 文件大小 > 5KB

**72 → 73 tests pass**(1 新增)

## 4. 约束兑现

- ❌ `_dynamics_core.py` 0 行修改
- ❌ `dynamics_forced_response.py` 0 行修改(magnitude_phase / classify_response_type 不变)
- ❌ v5+v5.1+v5.2+v5.3+v5.4+v5.5 已有 6 函数签名 0 修改
- ❌ `parse_args()` 签名 0 变化(只加 1 个 flag)
- ❌ 3 caller + 4 v4.x CLI + `parameter_fit.py` 0 修改
- ✓ v5.6 是**1 新函数** + 1 CLI flag + main() 1 行
- ✓ 测试 1 新增, 72 → 73 tests pass
- ✓ 输出全部 gitignored

## 5. 关键文件

- **修改**: `backtrace/dynamics/dynamics_si_freq_response.py` — +1 import + `REGIME_COLORS` dict + 1 新函数 + 1 CLI flag + main() 1 行
- **修改**: `tests/test_dynamics_eigen.py` — +1 test
- **修改**: `backtrace/dynamics/README.md` — §4.1.5 加 v5.6 footnote
- 0 新建文件

## 6. 与 v5.5 的关系

v5.5 给 HTML 加 regime 颜色编码。v5.6 给同一数据加 **静态 PNG 导出**(业务报告用)。

| 版 | commit | 主题 |
|---|---|---|
| v5.4 | `7e02782` | 双子图 (|H| + ∠H) |
| v5.5 | `179a579` | regime color coding |
| **v5.6** | **(本次)** | **static 2D grid PNG export (matplotlib)** |

v5.6 是**输出载体**扩展(HTTP 输出 → HTTP + PNG), 数据层 + 颜色逻辑 0 重复。

## 7. 已知风险

| 风险 | 缓解 |
|---|---|
| matplotlib 字体(中文乱码) | v5.x 没用中文(matplotlib labels 英文), 不影响; 中文说明在 TXT |
| n_rows > 10 时图太大 | `--max-dates` 12 默认限制, 真实场景 ≤ 12 |
| industries 同名叠加 | v5.5 已处理(legendgroup), 这里按 (date, industry) 配对 |
| matplotlib 颜色 vs plotly 颜色 | 同 hex (`#2ca02c` / `#ff7f0e` / `#d62728` / `#9467bd`), 一致 |
| PNG 大小 | ≤ 12 rows × 4 inches × 100 dpi = 4800×1200 px, ~150KB, OK |

## 8. 验证

```bash
# 1. 73 tests pass
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v
# 2. 端到端
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_si_freq_response.py \
    --kc-time-csv data/projection/kc_estimates_time.csv --top-n-industries 5
# 期待: 4 个 gitignored 输出 (HTML + TXT + CSV + PNG)
# 3. 浏览器无法打开 PNG, 用图片查看器 / 嵌 PDF 验证
```

## 9. 验证清单

- [ ] `_dynamics_core.py` 0 修改
- [ ] `dynamics_forced_response.py` 0 修改
- [ ] v5+v5.1+v5.2+v5.3+v5.4+v5.5 已有 6 函数签名 0 修改
- [ ] parse_args 0 变化(只加 1 flag)
- [ ] 3 caller + 4 v4.x CLI + parameter_fit.py 0 修改
- [ ] `build_static_bode_grid` 新函数 + REGIME_COLORS dict + 1 import
- [ ] 1 CLI flag `--static-output`, 默认值 OK
- [ ] main() 末尾 1 行调用新函数
- [ ] 1 新 test, 72 → 73 tests pass
- [ ] README §4.1.5 加 v5.6 footnote
- [ ] 4 种 regime 颜色与 v5.5 视觉一致
