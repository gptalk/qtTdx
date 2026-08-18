# Spec v5.1 — Industry G(ω) Frequency Response Comparison

> **Date:** 2026-08-18
> **Base:** v5 受迫系统 + G(ω) 频率响应(`0ce3014`)
> **Branch:** new(从 `main` HEAD = `0ce3014`)

## 1. 问题

v5 提供了**单对 (k, c)** 的 Bode 图 + Schur 楔形热图。但业务场景核心问题是"对比"——
不同行业的 (k̂, ĉ) 不同,频率响应差异巨大。一次性画单条曲线没法回答:

- 哪个行业对 β 强迫**最敏感**(|H(jω_n)| 最大)?
- 哪个行业是**低通过滤器**(Schur 楔形内,k ≪ c)?
- 哪个行业**最危险**(Schur 楔形外,k > c 接近边界)?

## 2. 目标

**核心**:在 v5 已有 `transfer_function` / `natural_frequency` / `magnitude_phase` 上加
`bode_overlay()` 函数,支持多个 (k, c, label) 元组在一张 Bode 图上叠加显示。

**非目标(YAGNI)**:
- ❌ 不直接读 `parameter_fit` 输出 — 让用户传 (k, c, label)
- ❌ 不做行业筛选 / 排序 / 自动推荐 — 让用户控制输入
- ❌ 不做 Nyquist 多曲线(只 Bode 幅频 + 相频)
- ❌ 不做交互式 legend(legend 静态显示 label)

**理由**:
- 与 v5 解耦 — 不依赖 `parameter_fit` 输出格式
- 与 v4.x 解耦 — 不碰 SI / IC 评估
- 单一职责 — 只加可视化层,数学层用 v5 现有函数

## 3. 设计

### 3.1 架构

```
┌─────────────────────────────────────────────────────────────┐
│  dynamics_forced_response.py(v5 文件,扩展)                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  [v5 已有] transfer_function / magnitude_phase      │    │
│  │  natural_frequency / classify_response_type         │    │
│  │  bode_plot / stability_heatmap / write_summary     │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  [v5.1 新增] bode_overlay(k_c_pairs, ...)           │    │
│  │  write_overlay_summary(k_c_pairs, ...)              │    │
│  │  --overlay CLI flag                                  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 新 API

```python
def bode_overlay(
    omega_grid: np.ndarray,
    k_c_pairs: list[tuple[float, float, str]],   # [(k1, c1, "label1"), ...]
    output_path: str,
    title: str = "Industry G(ω) Frequency Response Comparison",
) -> None:
    """多对 (k, c) Bode plot 叠加对比。

    Args:
        omega_grid: 角频率数组(共享),shape (N,)
        k_c_pairs: [(k, c, label), ...] 列表
        output_path: HTML 输出路径
        title: 图表标题

    行为:
        - 2 子图:上面 |H(jω)| dB vs ω,下面 arg H(jω) vs ω
        - 每对 (k, c, label) 一条曲线,共享 omega_grid,不同颜色 + 实线
        - legend 显示 label + (k, c) 值
        - 每对额外标出 ω_n(如果有)+ 该点的 |H(jω_n)|
        - HTML 通过 plotly CDN 渲染(include_plotlyjs='cdn')

    Raises:
        ValueError: 空列表 / label 重复 / k 或 c 非正
    """
```

```python
def write_overlay_summary(
    omega_grid: np.ndarray,
    k_c_pairs: list[tuple[float, float, str]],
    output_path: str,
) -> None:
    """多对 (k, c) 的 UTF-8 中文汇总表。

    每对一行,展示:
        - 行业/时间窗 label
        - (k, c) + 响应类型(classify_response_type)
        - ω_n + |H(jω_n)|(若有)
        - |H(j0)| (DC 增益)
        - |H(jπ)| (Nyquist)
        - Schur 楔形内/外
        - 一句业务解读("最敏感"/"低通过滤"/"危险")
    """
```

### 3.3 CLI 扩展

```bash
# 单对模式(v5 默认,不变)
python backtrace/dynamics/dynamics_forced_response.py --k 2.0 --c 1.5

# 多对 overlay 模式(v5.1 新增)
python backtrace/dynamics/dynamics_forced_response.py \
    --overlay "0.5,1.5,Industry A; 2.0,1.5,Industry B; 3.5,0.5,Industry C"

# 解析规则:
# - 分号 `;` 分隔不同对
# - 逗号 `,` 分隔 k / c / label
# - label 可含空格(逗号解析只取前两个)
# - 例:"2.0,1.5,Bank Index" → (k=2.0, c=1.5, label="Bank Index")
```

### 3.4 输出(全 gitignored)

| 路径 | 触发条件 | 内容 |
|---|---|---|
| `backtrace/outputs/dynsys_bode_overlay.html` | `--overlay` 模式 | 多对 Bode plot |
| `backtrace/outputs/dynsys_bode_overlay_summary.txt` | `--overlay` 模式 | 多对中文汇总表 |

**单对模式下不写 overlay 文件**(只在 main() 检测 `--overlay` 参数非空时调用)。

## 4. 测试

### 4.1 单元测试(`tests/test_dynamics_eigen.py` 新增 4 个)

```python
def test_bode_overlay_creates_html(tmp_path):
    """bode_overlay 调用产生 HTML 文件 + 文件非空。"""

def test_bode_overlay_validates_empty_list(tmp_path):
    """空 k_c_pairs → ValueError('k_c_pairs 不能为空')。"""

def test_bode_overlay_validates_negative_k(tmp_path):
    """k < 0 → ValueError('k 必须 > 0')。"""

def test_bode_overlay_mixed_stability(tmp_path):
    """5 对混合 stable / unstable / boundary,所有曲线都画出来,legend 含所有 label。"""
```

### 4.2 测试数据

| 测试对 | 物理意义 |
|---|---|
| (k=0.5, c=2.0, "Strong damping") | Schur 楔形深内,单调滚降 |
| (k=2.0, c=1.5, "Mild damping") | Schur 楔形外,中等共振 |
| (k=2.01, c=2.0, "Near boundary") | 边界共振,peak ≈ 200 |
| (k=4.0, c=0.5, "Weak damping") | 强共振,peak 巨大 |
| (k=0.1, c=4.0, "Very overdamped") | 几乎无响应 |

5 对 5 种物理类型覆盖 + 1 个 Critical test 验证 overlay 流程(3 对足够)。

### 4.3 回归保护

- v5 已有 5 个测试 + v4.x 48 个测试,**全部不动**
- 53 → 53+4 = **57 tests pass**(目标)

## 5. 约束兑现

- ❌ `_dynamics_core.py` 0 行修改
- ❌ 3 caller (`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`) 0 行修改
- ❌ v5 已有函数(`transfer_function` / `natural_frequency` / `magnitude_phase` / `classify_response_type`)签名 0 修改
- ❌ v5 单对模式 main() 流程不变(只在末尾加 `if args.overlay:` 分支)
- ❌ v5 单对模式 CLI 输出不变(2 CSV + 2 HTML + 1 TXT)
- ❌ 4 个 v4.x CLI 0 修改
- ✓ 所有新增输出 gitignored(`backtrace/outputs/dynsys_bode_overlay*`)

## 6. 关键文件

- 修改:[`backtrace/dynamics/dynamics_forced_response.py`](backtrace/dynamics/dynamics_forced_response.py) — 加 `bode_overlay` + `write_overlay_summary` + `--overlay` CLI + 3 处 main() 分支
- 修改:[`backtrace/dynamics/README.md`](backtrace/dynamics/README.md) — §4 v5 加 §4.1 v5.1 子节
- 修改:[`tests/test_dynamics_eigen.py`](tests/test_dynamics_eigen.py) — 加 4 个 test

## 7. 与 v5 的关系

| 版 | commit | 主题 |
|---|---|---|
| v5 | `0ce3014` | 受迫系统 + G(ω) 单对频率响应 |
| **v5.1** | **(本次)** | **多对 (k, c) overlay 对比**(同一 Bode 上画多条曲线) |

v5.1 是 v5 的**可视化层扩展**,不动数学层。v5.2 候选:与 `parameter_fit` 集成,自动从历史 (k̂, ĉ) 序列选 top-N 行业画 overlay。

## 8. 已知风险

| 风险 | 缓解 |
|---|---|
| `--overlay` 字符串解析歧义(label 含 `;` 或 `,`) | label 用最后一段(逗号 split 后取 rest joined by `,`),分号 split 优先 |
| 曲线颜色冲突(plotly 默认 10 色,>10 对 label 难看) | 文档明示推荐 ≤ 10 对,>10 对用 `plotly.express.colors.qualitative.Light24` |
| Bode overlay 与单对 bode_plot 共存(main() 重复写 HTML) | 分支隔离:`if args.overlay: bode_overlay(...) else: bode_plot(...)` |

## 9. 演示 / 复现

```bash
git log --oneline 0ce3014..HEAD
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v
# 期待:57 passed(53 v5 + 4 v5.1)

# 端到端
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_forced_response.py \
    --overlay "0.5,2.0,Strong damping; 2.0,1.5,Mild damping; 2.01,2.0,Near boundary"
# 期待:backtrace/outputs/dynsys_bode_overlay.html + dynsys_bode_overlay_summary.txt
```

## 10. 验证清单

- [ ] `transfer_function` / `natural_frequency` / `magnitude_phase` / `classify_response_type` 0 修改
- [ ] `_dynamics_core.py` 0 修改
- [ ] 3 caller (`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`) 0 修改
- [ ] v5 单对模式 main() 输出不变(2 CSV + 2 HTML + 1 TXT)
- [ ] v5 单对模式 CLI flags 全部兼容
- [ ] 4 个 v4.x CLI 0 修改
- [ ] 新增 `dynsys_bode_overlay*` 全部 gitignored
- [ ] 4 个新测试 + 53 个旧测试 = 57 tests pass
- [ ] README §4 加 §4.1 v5.1 子节