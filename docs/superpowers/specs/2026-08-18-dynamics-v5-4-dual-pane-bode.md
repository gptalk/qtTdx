# Spec v5.4 — Dual-Pane Bode Overlay (|H(jω)| + ∠H(jω))

> **Date:** 2026-08-18
> **Base:** v5.3 Real SI Frequency Response (`f1d17f7`)
> **Branch:** modification only to `backtrace/dynamics/dynamics_si_freq_response.py`

## 1. 问题

v5.3 `build_animated_overlay_html` 画**单子图**|H(jω)| dB vs ω。Spec §3.2 narrative 最初说"上子图 |H| + 下子图 arg H",但 v5.3 实现只画了 |H|(spec §3.5 code example binding + 我在 v5.3 spec drift fix 中已改成"phase 子图留 v5.4+")。

业务上其实**幅度 + 相位**是 Bode 图的标准双面板 — 单独看 |H| 看不到**相位失真**对 industry coupling 的提示。v5.4 把这一层补上,完成 v5.3 spec narrative。

## 2. 目标

**核心**:扩展 `build_animated_overlay_html` 用 plotly `make_subplots` 画**双子图**(上 |H(jω)| dB,下 ∠H(jω) deg),共享 x 轴(ω),都通过 `animation_frame` 联动时序。其他 5 函数(load / aggregate / select_top_n / 2 writers)0 修改,main() 0 修改,CLI 0 修改,测试只更新 1 个。

**非目标(YAGNI)**:
- ❌ 不画 3D surface — 不需要
- ❌ 不做相位展开(unwrapping)— KISS,business 看的是相对趋势
- ❌ 不画 group delay 派生面 — YAGNI
- ❌ 不画相位裕度 / 增益裕度 marker — 后续再说
- ❌ 不改 omega_grid、animation duration、slider 行为 — 一致性

**理由**:
- v5.3 已经搭好 animation 基础设施,`go.Frame` 改成 2 traces per industry 而不是 1 实现成本低
- 维持 single-mode main() 不变(business 调用方式 0 变化)
- 6 函数中的 5 已有签名 0 修改,职责清晰

## 3. 设计

### 3.1 架构

```
backtrace/dynamics/dynamics_si_freq_response.py
  [v5.3 已有]  build_animated_overlay_html(pairs_per_date, omega_grid, output_path, title)
  [v5.4 扩展]  同一函数内部:
                  - 改 go.Figure → go.Figure(make_subplots(2, 1, shared_xaxes=True))
                  - 改每帧 trace 1 条 → 2 条(magnitude + phase)
                  - 改 frame.data 同步 2 子图
                  - 改 slider → 联动 2 子图
                  - 改 updatemenus → 联动 2 子图
```

### 3.2 v5.4 修改范围

**仅修改 `build_animated_overlay_html` 一个函数**(signature 0 修改):
- 加 `import plotly.subplots` → `make_subplots`
- 上子图(原 |H(jω)| dB)保留 + title |H(jω)| dB
- 下子图(新 ∠H(jω) deg)via `magnitude_phase(z, k, c)[1]`
- 共享 x 轴(ω rad/day)+ 下子图标题 ω (rad/day)
- `animation_frame` 联动 2 子图(plotly 自动)

### 3.3 CLI 扩展

**0 改动** — CLI flag 一致,用户感知 0 变化,只是 HTML 渲染从单图变双图。

### 3.4 输出(全 gitignored,与 v5.3 相同)

| 路径 | 默认值 |
|---|---|
| HTML | `backtrace/outputs/dynsys_si_freq_response.html`(变大 ~2×, ~400KB) |
| TXT | `backtrace/outputs/dynsys_si_freq_response_summary.txt`(0 变化) |
| CSV | `data/dynamics/si_freq_response_pairs.csv`(0 变化) |

### 3.5 测试更新

**仅更新 1 个 test**:`test_cli_si_freq_response_mode`:
- HTML 仍需 `addFrames` / `Plotly.animate` / `animation_frame` / `frames` ✓
- HTML 现在还需 `xy` / `xaxis` / `yaxis` 多次出现(2 子图)
- HTML size > 2000(原 1000,扩大因 2 子图)
- 加 assertion: HTML 包含 `∠H` 或 `phase` 或 `arg` 关键词(确认相位子图存在)

**0 新测试**(v5.4 是已有测试的扩展,不是新功能)

**72 → 72 tests pass**(1 test 更新,总数不变)

## 4. 约束兑现

- ❌ `_dynamics_core.py` 0 行修改
- ❌ `dynamics_forced_response.py` 0 行修改
- ❌ v5+v5.1+v5.2+v5.3 已有 5 函数(`load_kc_time_series` / `aggregate_by_industry_per_date` / `select_top_n_per_date` / `write_animated_summary_txt` / `write_animated_pairs_csv`)签名 0 修改
- ❌ `parse_args()` 0 修改
- ❌ `main()` 0 修改
- ❌ 3 caller + 4 v4.x CLI + `parameter_fit.py` 0 修改
- ✓ v5.4 是 `build_animated_overlay_html` 内部扩展,签名 0 修改
- ✓ 测试 1 个更新,总数 0 变化
- ✓ 输出全部 gitignored

## 5. 关键文件

- **修改**:`backtrace/dynamics/dynamics_si_freq_response.py` — `build_animated_overlay_html` 内部从单子图 → 双子图
- **修改**:`tests/test_dynamics_eigen.py` — `test_cli_si_freq_response_mode` 1 个 assertion 扩充
- **修改**:`docs/superpowers/specs/2026-08-18-dynamics-v5-3-si-freq-response.md` — §3.2 docstring narrative 已正确(phase 已 leave v5.4+),无需改
- **修改**:`backtrace/dynamics/README.md` — §4.1.2 加 1 行 v5.4 提示("phase 子图 v5.4+")
- 0 新建文件

## 6. 与 v5.3 的关系

v5.3 narrative 最初说"两子图"但只实现一子图。v5.4 兑现 narrative。

| 版 | commit | 主题 |
|---|---|---|
| v5.3 | `f1d17f7` | 时序动画 single-pane (|H|) |
| **v5.4** | **(本次)** | **时序动画 dual-pane (|H| + ∠H)** |

## 7. 已知风险

| 风险 | 缓解 |
|---|---|
| plotly v3.x `make_subplots` 兼容 | v5.3 已经在用 plotly v3.x,`make_subplots` 是 v3.x 基础 API |
| HTML size 翻倍 | 仍 < 400KB,浏览器仍秒开 |
| 相位 unwrap 缺失 | KISS,business 看相对趋势;v5.5+ 再补 |
| 动画双面板同步 | plotly `animation_frame` 自动同步 所有 subplot |

## 8. 验证

```bash
# 1. 72 tests pass(无新增)
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v
# 2. 端到端
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_si_freq_response.py \
    --kc-time-csv data/projection/kc_estimates_time.csv --top-n-industries 5
# 3. 浏览器打开 HTML,确认 双子图(magnitude + phase) + animation slider
```

## 9. 验证清单

- [ ] `_dynamics_core.py` 0 修改
- [ ] v5+v5.1+v5.2+v5.3 已有 5 函数签名 0 修改
- [ ] `parse_args` + `main` 0 修改
- [ ] 3 caller + 4 v4.x CLI + `parameter_fit.py` 0 修改
- [ ] `build_animated_overlay_html` 签名 0 修改,内部从单 → 双子图
- [ ] 1 test 扩充,72 tests pass(总数 0 变化)
- [ ] README §4.1.2 + 1 行 v5.4 提示
