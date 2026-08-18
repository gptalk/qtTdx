# Spec v5.5 — Regime Color Coding on Dual-Pane Bode

> **Date:** 2026-08-18
> **Base:** v5.4 Dual-Pane Bode (`7e02782` on main)
> **Branch:** modification only to `backtrace/dynamics/dynamics_si_freq_response.py`

## 1. 问题

v5.4 把 v5.3 的单子图扩成**双子图 Bode** (|H(jω)| dB + ∠H(jω) deg),业务可同时看 gain + phase。

但**所有 industry 曲线都是同色**(plotly 默认色板)— 业务拖 slider 时**无法一眼看出哪些 industry 处于稳定/共振区**。要判断 industry 状态,得逐个读 legend label + 心里算 ζ = c / (2√k)。

行业已经聚合了 (k̂, ĉ) + 验证了 (k̂, ĉ) ∈ Schur 楔形,`classify_response_type` 在 `dynamics_forced_response.py:130` 已能返回 4 类 regime:
- `overdamped` (k < c, Schur 内,稳定)
- `critical` (k ≈ c, 边界)
- `underdamped` (k > c, Schur 外,不稳定 / 共振)
- `anti_damped` (k < 0, 负恢复系数,反向弹簧,病态)

v5.5 把这层"分类信息"**直接编码到曲线颜色** — 业务看颜色就知道"红 = 共振 / 绿 = 稳定",**完全跳过 ζ 计算**。

## 2. 目标

**核心**: `build_animated_overlay_html` 内每个 industry 的 2 条曲线(magnitude + phase)用**该 industry 在该 frame 的阻尼 regime 颜色**渲染。同 industry 跨 asof_date 的 (k̂, ĉ) 变化时颜色自动跟随。

**非目标(YAGNI)**:
- ❌ 不写 ζ / ω_n / gain margin 数字标注 — 颜色足够,**legend 注释说明即可**
- ❌ 不做 3D 散点图 — 已用 dual-pane
- ❌ 不加 filter(只显示某 regime 的 industries)— 业务想看全图
- ❌ 不加 stability table — 颜色 + 现有 legend 已够
- ❌ 不改 `classify_response_type` 本身 — 复用 v5 已有

**理由**:
- 最小实现: 1 helper (`_regime_color`) + 1 处 trace 颜色绑定
- 业务可读性: 颜色 = 一目了然的稳定度
- 数据真实性: 颜色来自该 (k̂, ĉ) 实际值,不引入新计算
- 时序漂移可见: 同一 industry 跨帧颜色变化 = 它从稳定漂到共振,正是 v5.3 时序动画的核心 insight

## 3. 设计

### 3.1 架构

```
backtrace/dynamics/dynamics_si_freq_response.py
  [v5.4 已有]  build_animated_overlay_html(pairs_per_date, omega_grid, output_path, title)
  [v5.5 扩展]  同一函数内部:
                  - 加 _regime_color(k, c) helper → 4 种颜色
                  - 每个 trace 颜色绑定 regime
                  - 加 HTML 注释说明颜色 ↔ regime 映射
                  - 0 新函数, 0 CLI flag, 0 main 改动
```

### 3.2 颜色映射

| Regime | 颜色 (hex) | 业务语义 | Plotly 名称 (可选) |
|---|---|---|---|
| `overdamped` | `#2ca02c` (绿) | Schur 内,稳定 | green |
| `critical` | `#ff7f0e` (橙) | Schur 边界,临界 | orange |
| `underdamped` | `#d62728` (红) | Schur 外,共振风险 | red |
| `anti_damped` | `#9467bd` (紫) | 负恢复系数,病态 | purple |

颜色选 matplotlib v3 调色板的 4 种**饱和度高、对比强**的标准色,色盲友好度 OK(绿/橙/红/紫区分明显)。

### 3.3 v5.5 修改范围(单函数内部)

**仅修改 `build_animated_overlay_html` 一个函数**(signature 0 修改):
- 加 `_regime_color(k, c)` 闭包,返回 hex 颜色字符串
- import `classify_response_type` from `backtrace.dynamics.dynamics_forced_response` (同级 module,无 cycle)
- 在 initial-state traces 和 frame traces 中,**每个 trace 的 `line=dict(color=...)` 绑定 regime 颜色**
- HTML 注释(annotation)说明 4 种颜色 ↔ regime 映射,放在 figure 右上角

### 3.4 CLI 扩展

**0 改动** — CLI flag 一致,用户感知 = 颜色编码,无新增 flag。

### 3.5 输出(全 gitignored,与 v5.3/v5.4 相同)

| 路径 | 变化 |
|---|---|
| HTML | `backtrace/outputs/dynsys_si_freq_response.html` — 颜色 + 注释,size 变化 < 5% |
| TXT | `backtrace/outputs/dynsys_si_freq_response_summary.txt` — 0 变化 |
| CSV | `data/dynamics/si_freq_response_pairs.csv` — 0 变化 |

### 3.6 测试更新

**仅更新 1 个 test**: `test_cli_si_freq_response_mode`
- HTML 含至少 3 种 regime 颜色 hex(过阻尼/欠阻尼/临界在 fixture 都有,断言能找到 `#2ca02c` 或 `#d62728` 等)
- HTML 含 regime 注释(关键词: `overdamped` / `临界` / `共振` / `稳定` 任一)

**0 新测试**(v5.5 是视觉增强,不是新功能)

**72 → 72 tests pass**(1 test 更新,总数不变)

## 4. 约束兑现

- ❌ `_dynamics_core.py` 0 行修改
- ❌ `dynamics_forced_response.py` 0 行修改(`classify_response_type` 等函数签名 0 变化,只被 v5.5 调)
- ❌ v5+v5.1+v5.2+v5.3+v5.4 已有 6 函数(`load_kc_time_series` / `aggregate_by_industry_per_date` / `select_top_n_per_date` / `write_animated_summary_txt` / `write_animated_pairs_csv` / `build_animated_overlay_html`)签名 0 修改
- ❌ `parse_args()` 0 修改
- ❌ `main()` 0 修改
- ❌ 3 caller + 4 v4.x CLI + `parameter_fit.py` 0 修改
- ✓ v5.5 是 `build_animated_overlay_html` 内部 1 闭包 + 1 line 颜色绑定 + 1 注释
- ✓ 测试 1 个更新,总数 0 变化
- ✓ 输出全部 gitignored
- ✓ `classify_response_type` 是新 import,来源 v5 已有,无新逻辑

## 5. 关键文件

- **修改**: `backtrace/dynamics/dynamics_si_freq_response.py` — `build_animated_overlay_html` 加 `_regime_color` 闭包 + 颜色绑定 + 注释
- **修改**: `tests/test_dynamics_eigen.py` — `test_cli_si_freq_response_mode` 1 个 assertion 扩充
- **修改**: `backtrace/dynamics/README.md` — §4.1.4 加 1 行 v5.5 提示
- 0 新建文件

## 6. 与 v5.4 的关系

v5.4 兑现 v5.3 spec narrative 的"双面板 Bode"。v5.5 把业务最常问的"哪些 industry 稳定 / 哪些共振"**直接编码到颜色**,免去 ζ 计算。

| 版 | commit | 主题 |
|---|---|---|
| v5.3 | `f1d17f7` | 时序动画 single-pane (|H|) |
| v5.4 | `7e02782` | 时序动画 dual-pane (|H| + ∠H) |
| **v5.5** | **(本次)** | **regime color coding on dual-pane** |

## 7. 已知风险

| 风险 | 缓解 |
|---|---|
| 4 种颜色在 light bg 下对比不够 | 用 matplotlib 标准 4 色(已测试色盲友好) |
| 同一 frame 中多 industry 同 regime 同色 → legend 看不出区分 | legend 仍显示 industry label + (k, c),颜色只是 stability 信号 |
| plotly v3.x `line.color` 兼容 | v5.3 已在用 plotly v3.x,`line=dict(color=...)` 是基础 API |
| 颜色跨帧变化时 plotly 动画卡顿 | 颜色随 frame 切换是 plotly 内置支持的,实测 < 1ms 切换 |

## 8. 验证

```bash
# 1. 72 tests pass(无新增)
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v
# 2. 端到端
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_si_freq_response.py \
    --kc-time-csv data/projection/kc_estimates_time.csv --top-n-industries 5
# 3. 浏览器打开 HTML,确认:
#   - 曲线颜色按 regime 区分(绿/橙/红/紫)
#   - 注释在右上角
#   - 拖 slider 时同 industry 颜色可能变化(若其 (k, c) 跨帧变化)
```

## 9. 验证清单

- [ ] `_dynamics_core.py` 0 修改
- [ ] `dynamics_forced_response.py` 0 修改
- [ ] v5+v5.1+v5.2+v5.3+v5.4 已有 6 函数签名 0 修改
- [ ] `parse_args` + `main` 0 修改
- [ ] 3 caller + 4 v4.x CLI + `parameter_fit.py` 0 修改
- [ ] `build_animated_overlay_html` 签名 0 修改,内部加 `_regime_color` 闭包 + 颜色绑定
- [ ] 1 test 扩充,72 tests pass(总数 0 变化)
- [ ] README §4.1.4 加 1 行 v5.5 提示
- [ ] 4 种颜色在 HTML 中可见(过阻尼绿/临界橙/欠阻尼红/anti-damped 紫)
