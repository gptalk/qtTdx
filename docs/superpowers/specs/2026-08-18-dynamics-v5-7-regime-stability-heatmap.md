# v5.7 — Regime Stability Heatmap

**Date:** 2026-08-18
**Status:** Draft
**Base:** v5.6 (HEAD `74199d8`)
**Author:** Brainstorming output (user authorized: "按计划和推荐执行，不用问我")

## 1. Goal

给 v5.6 (静态 PNG 2D Bode grid) 再加一种 **dashboard 视图**:**2D regime heatmap** —— rows = unique asof_date (sorted ascending), cols = unique industry (sorted by label), 每个 cell 一个 regime 颜色 + 4 字 abbreviation。

**业务读法**:扫一眼就能看到"哪些行业在共振(红)/ 哪些稳定(绿)/ 哪些病态(紫)",无需解读频率响应曲线。

## 2. Why now

v5.3 → v5.6 都是"曲线图"模态(时序动画 / 双子图 / 颜色编码 / PNG 静态化)。Heatmap 是**结构上不同**的模态 —— cell-based grid,不是 line plot。它和 v5.5/v5.6 **互补**:
- v5.5 HTML:交互看曲线细节
- v5.6 PNG:静态看曲线细节
- **v5.7 PNG:静态看 regime 分布** <- new

业务报告里 v5.7 是"执行摘要"(谁是稳定板块),v5.5/v5.6 是"技术细节"(为什么稳定)。两者并列贴周到/月报。

## 3. Design

### 3.1 架构

复用 v5.6 已有的所有基础设施:

- `REGIME_COLORS` (v5.6 module-level dict,4 色 hex)
- `classify_response_type(k, c)` (from `dynamics_forced_response`)
- `matplotlib.use('Agg')` backend (v5.6 设置)
- `mpatches.Patch` legend(顶部 4 色横向图例)
- `--kc-time-csv` / `--top-n-industries`(v5.6 CLI flags)
- `n_rows == 1` axis wrapping(v5.6 pattern)

### 3.2 函数

```python
def build_regime_heatmap(
    pairs_per_date: list,  # [(asof_date, k_hat, c_hat, label), ...]
    output_path: str,
    title: str = 'Industry Regime Stability — Heatmap',
    dpi: int = 100,
) -> None:
    """Render regime for each (date, industry) as a 2D heatmap.

    Rows: unique asof_date (sorted ascending, top = oldest)
    Cols: unique industry label (sorted ascending, left = first)
    Cell: REGIME_COLORS.get(classify_response_type(k, c), '#7f7f7f')
    Cell text: 4-letter abbreviation (over/crit/under/anti)

    Raises:
        ValueError: pairs_per_date 为空
    """
```

**输入**:`pairs_per_date` 来自 v5.6 `select_top_n_per_date` (已是 select_top_n 的 per-date 版本)。

**输出**:
- PNG at `output_path`
- 2D 网格: rows = dates, cols = industries
- 每个 cell: 背景色 = regime color,中心文字 = 4 字 abbreviation
- 顶部图例:4 colored patches (mpatches.Patch,v5.6 I-1 模式)
- 行/列标签:date / industry label
- 顶部 suptitle:title string

### 3.3 CLI

新增 1 个 flag:

```python
parser.add_argument(
    '--heatmap-output',
    type=str,
    default='backtrace/outputs/dynsys_regime_heatmap.png',
    help='Regime heatmap PNG 输出路径',
)
```

`main()` 末尾追加:

```python
build_regime_heatmap(pairs, args.heatmap_output)
print(f'[v5.7] regime heatmap 已写入 {args.heatmap_output}')
```

**注意**:`heatmap` 与 v5.6 `static` 互不依赖,可同时跑出 2 张 PNG。

### 3.4 排放顺序(细)

1. 计算 unique dates (sorted ascending) + unique industries (sorted by label)
2. 对每个 (date, industry) cell:
   - 找到对应的 `k_hat, c_hat`
   - `classify_response_type(k, c)` → regime
   - cell color = `REGIME_COLORS.get(regime, '#7f7f7f')`
   - cell text = `{'overdamped': 'over', 'critical': 'crit', 'underdamped': 'under', 'anti_damped': 'anti'}.get(regime, '?')`
3. 用 `matplotlib.patches.Rectangle` 绘制 cell(不用 `imshow`,因为 imshow 不支持 cell-level text)
4. 顶部 mpatches.Patch legend (4 patches)
5. fig.suptitle + tight_layout(rect=[0, 0, 1, 0.94])
6. os.makedirs + savefig + plt.close

### 3.5 颜色 / 字体

- Cell 颜色:`#2ca02c` (over 绿) / `#ff7f0e` (crit 橙) / `#d62728` (under 红) / `#9467bd` (anti 紫) / `#7f7f7f` (unknown 灰)
- Cell 文字颜色:黑 (cell 颜色都是亮色,黑字 contrast 够)
- Cell 文字 size:matplotlib 自动缩放(根据 n_rows × n_cols)
- Date row label:YYYY-MM-DD (str)
- Industry col label:industry label (str)

### 3.6 数据 fallback

- 某个 (date, industry) 没数据 → 画灰色 cell + text "?"
- 全空 → `ValueError('pairs_per_date 为空,无法构建 heatmap')`

## 4. Files modified

| File | Type | Lines |
|---|---|---|
| `backtrace/dynamics/dynamics_si_freq_response.py` | modify | +50 |
| `tests/test_dynamics_eigen.py` | modify | +44 |
| `backtrace/dynamics/README.md` | modify | +15 |
| `docs/superpowers/specs/2026-08-18-dynamics-v5-7-regime-stability-heatmap.md` | new | (this) |
| `docs/superpowers/plans/2026-08-18-dynamics-v5-7-regime-stability-heatmap.md` | new | (TBD) |

Total: ~250 lines, 5 files (1 spec + 1 plan + 3 code/test/doc)

## 5. 测试

新增 1 test:

```python
def test_cli_regime_heatmap_mode(tmp_path):
    """v5.7: CLI regime heatmap mode — 验证 build_regime_heatmap 输出 PNG."""
    pytest.importorskip("matplotlib")

    import subprocess
    import sys
    import os

    # 合成 3 dates × 2 industries CSV (复用 v5.6 fixture)
    csv_path = tmp_path / 'kc_time.csv'
    rows = []
    for date_str in ['2024-09-30', '2024-10-31', '2024-11-30']:
        for code, k, c in [('AAA', 0.5, 2.0), ('BBB', 3.5, 0.5)]:
            rows.append({
                'code': code, 'index_code': f'Industry_{code}',
                'asof_date': date_str, 'k_hat': k, 'c_hat': c,
                'status': 'ok', 'n_valid_days': 200,
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')

    heatmap_png = tmp_path / 'heatmap.png'
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cli_script = os.path.join(repo_root, 'backtrace', 'dynamics', 'dynamics_si_freq_response.py')
    cmd = [
        sys.executable, cli_script,
        '--kc-time-csv', str(csv_path),
        '--top-n-industries', '2',
        '--heatmap-output', str(heatmap_png),
    ]
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(cmd, capture_output=True, env=env, timeout=60)
    assert result.returncode == 0, f'CLI failed: {result.stderr.decode("utf-8", errors="ignore")}'

    # 验证 PNG 存在 + 字节头 + size
    assert heatmap_png.exists(), f'PNG not created: {heatmap_png}'
    assert heatmap_png.stat().st_size > 5000, f'PNG too small: {heatmap_png.stat().st_size}'
    with open(heatmap_png, 'rb') as fh:
        header = fh.read(8)
    assert header.startswith(b'\x89PNG'), f'Not a valid PNG: header={header!r}'
```

合计:73 → 74 tests pass(1 new test)。

## 6. 验证

### 行为

- 输入 3 dates × 2 industries → 输出 3 rows × 2 cols PNG
- Cell 颜色看 `(k_hat, c_hat)` 而定:`(0.5, 2.0)` overdamped 绿,`(3.5, 0.5)` underdamped 红
- 顶部图例显示 4 种 regime color

### CLI

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_si_freq_response.py \
    --kc-time-csv backtrace/outputs/dynsys_kc_time.csv \
    --top-n-industries 5 \
    --heatmap-output backtrace/outputs/dynsys_regime_heatmap.png
```

## 7. 兼容性 / 不破坏

- 0 protected file modifications(`_dynamics_core.py` / `dynamics_forced_response.py` / `dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py` / `dynamics_si_ic.py` / `dynamics_si_timeseries.py` / `dynamics_si_lagged_ic.py` / `dynamics_eigen_analysis.py` / `projection/parameter_fit.py`)
- 0 新依赖
- 现有 6 个函数 + `parse_args` 签名 0 变化(只加 1 flag)
- v5.5 `build_animated_overlay_html` + v5.6 `build_static_bode_grid` 0 改动
- 1 个新 test + 1 个新 function + 1 个新 CLI flag + 1 个新 main() 行

## 8. Risk

| Risk | Mitigation |
|---|---|
| Cell 文字超出 cell 大小(industry 数 > 20) | matplotlib 默认 auto-shrink,留 M3(minor)余地 |
| 字体显示中文乱码 | Cell text 全英文(over/crit/under/anti),行/列 label 走 ASCII(industry label 已是英文) |
| Heatmap 与 v5.6 static grid 视觉上混淆 | 文档 §4.1.6 明确"v5.7 不画曲线,只画 cell" |
| Rectangles 数量大时性能 | 最多 ~20 dates × 20 industries = 400 cells,Rectangles 性能足够 |

## 9. v5.7 vs v5.5 vs v5.6 关系

| 版本 | 模态 | 输出 | 业务用例 |
|---|---|---|---|
| v5.5 | 交互曲线(HTML) | `dynsys_si_freq_response_overlay.html` | 浏览器拖 slider 看时序演化 |
| v5.6 | 静态曲线(PNG) | `dynsys_si_freq_response_static.png` | 嵌 PDF / 周报 |
| **v5.7** | **静态 cells(PNG)** | `dynsys_regime_heatmap.png` | **dashboard 视图,执行摘要** |

3 个功能**互补**,共用同一份 `kc_time.csv` + `REGIME_COLORS` + `classify_response_type`,业务可三选一或全要。

## 10. Future / out of scope

- v5.8:animated GIF (matplotlib.animation + Pillow) — 业务表示"想看时序演化但不想拖 slider"
- v5.9:per-pair PNG (single (date, industry) 高清) — 业务表示"想贴 PPT"
- v5.10:HTML static (no-slider) — 业务表示"想内嵌 Wiki"

本次 v5.7 只做 heatmap,其他按需。
