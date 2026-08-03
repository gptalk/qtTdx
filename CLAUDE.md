# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **API 表见 [docs/api.md](docs/api.md)。** 修改公开函数时请同步更新。

## 仓库本质

A 股量化研究 / 回测项目,围绕**通达信 TQ 接口**(`C:/new_tdx_mock/PYPlugins/user/tqcenter.py`)+ **vectorbt 回测**展开。所有脚本都在 `backtrace/` 下,统一拉 TQ 数据 → 跑研究 → 输出 CSV / HTML / 推送到通达信客户端。

VSCode multi-root workspace 配置文件: [qtTdxs.code-workspace](qtTdxs.code-workspace)

输出产物默认 gitignored(见 [.gitignore](.gitignore))。详见 [README.md](README.md)(TQ 平台说明) 和 [backtrace/gp_factor_mining/README.md](backtrace/gp_factor_mining/README.md)(GP 因子挖掘子项目)。

## 跑脚本的最快姿势

**所有脚本在 `backtrace/` 下子目录里**(已按功能分组),从仓库根运行即可:

```bash
PYTHONIOENCODING=utf-8 python backtrace/<subdir>/<script>.py
```

**`PYTHONIOENCODING=utf-8` 必须加** — Windows GBK 终端会让中文 print 直接 UnicodeEncodeError。

如果脚本有后台跑需求(像 `tsfresh_with_ma_grid_sector.py` 跑 3.5 小时),用 `run_in_background=true`,监控用 TaskOutput。

## backtrace/ 目录结构(按功能分组)

```
backtrace/
├── __init__.py
├── common/                  ← 公共模块(其他脚本 import)
│   ├── tsfresh_config.py
│   ├── tsfresh_pipeline.py
│   └── jhzq_fees.py
├── tsfresh/                 ← tsfresh 系列(10 个脚本)
│   ├── tsfresh_features_002457.py     ← EDA 提特征
│   ├── tsfresh_select_002457.py       ← FDR 筛选
│   ├── tsfresh_classify_002457.py     ← walk-forward 训模型
│   ├── tsfresh_pick_stocks.py         ← 板块选股
│   ├── tsfresh_multichannel_pick.py   ← 多通道
│   ├── tsfresh_vbt_combo.py           ← +vbt 集成
│   ├── tsfresh_with_ma_channel.py     ← MA 作通道
│   ├── tsfresh_with_ma_grid_sector.py ← 88 板块网格
│   ├── tsfresh_eval_indicators.py     ← 11 个指标 IC
│   └── tsfresh_top1_industry.py       ← 双重跑赢 + TQ 推送
├── vbt/                     ← vbt 系列
│   ├── vbt_simple_backtest.py
│   └── vbt_jhzq_backtest.py
├── alpha/                   ← 双层 α 选股
│   ├── two_layer_industry_strength.py
│   ├── two_layer_industry_strength_live.py
│   └── two_layer_relative_strength.py
├── talib/                   ← K 线形态
│   ├── talib_pattern_backtest.py
│   └── talib_pattern_verify.py
├── legacy/                  ← 过时 TQ 信号模板(已不推荐)
│   ├── ma_cross_signals.py
│   ├── price_rise_monitor.py
│   ├── stock_picker.py
│   ├── kline_chart.py
│   └── projection_2d.py
├── gp_factor_mining/        ← GP 因子挖掘子项目(独立 README)
└── outputs/                 ← 全部 CSV/HTML 输出
```

## 数据接入统一入口 — [docs/api.md](docs/api.md)

**所有脚本都用 [backtrace/common/tsfresh_pipeline.py](backtrace/common/tsfresh_pipeline.py)**(在 `common/` 下,其他脚本 `from common import tsfresh_pipeline as P`),不要自己写 TQ 拉数。

完整 API 表(11 个函数 + 5 个费率常量 + 14 个配置常量)见 [docs/api.md](docs/api.md)。常用:

- `P.load_ohlcva(code, ...)` / `P.load_sector(sector_name, ...)` — TQ 优先 → CSV 回退
- `P.extract_window_features(...)` / `P.select_relevant(...)` — tsfresh 提特征 + FDR 筛选
- `F.adjust_trades_pnl(trades, code)` / `F.summary_after_fees(trades, code)` — vbt 真实扣费

## 关键配置 — [backtrace/common/tsfresh_config.py](backtrace/common/tsfresh_config.py)

改这里 = 改所有脚本。`WINDOW=30` / `HORIZON=5` / `LR_C=0.5` / `FDR_LEVEL=0.05` 等。**全部 14 个常量见 [docs/api.md §tsfresh_config](docs/api.md#backtracecommontsfresh_configpy)**。

**注意**:`OUTPUTS_DIR` 自动指向 `backtrace/outputs/`,所以脚本的输出 CSV / HTML 都落到那里。

## 实盘费率模块 — [backtrace/common/jhzq_fees.py](backtrace/common/jhzq_fees.py)

江海证券真实费率:**佣金 万 0.85(免 5)/ 印花税 卖出 万 5 / 沪市过户费 万 0.1**。费率参数在 `jhzq/交易凭据.md`,**不要把资金账号 / 密码写进任何代码**。

`vbt` 的 `fees` 参数只能双边费率,无法表达"单边印花税 / SH-SZ 差异化过户费" → 必须 `fees=0` + `jhzq_fees.adjust_trades_pnl` 后置。**输出 5 列中文 schema 详见 [docs/api.md §adjust_trades_pnl](docs/api.md#adjust_trades_pnl-输出-schema新增-5-列)**。

## TQ 接口踩过的坑(写代码前必看)

1. **`get_stock_list_in_sector(code, block_type=0)` 必须传 `'880xxx.SH'` 带 `.SH`** — 不带返回 0;`block_type=1` 在我们环境拿不到成分股
2. **`get_sector_list(list_type=1)` 返回 588 个混合板块**;`get_stock_list('11', list_type=1)` 返回 **128 个申万二级真行业**(干净) — 优先用 `'11'`
3. **`vbt.MACD.run / STOCH.run / ADX.run` 在某些 dtype 上会报 `'ewms' int → PyBool` 错** — 直接用 pandas `ewm/rolling` 手算绕过
4. **`vbt.from_signals` 在 vbt 0.27 不支持 `short_enabled` 参数** — 去掉
5. **`tq.get_market_data` 一次拉 5000+ 只会很慢** — 缩到 Top 20 行业的成分股 (~600 只) 即可

## 安全

- `jhzq/交易凭据.md` 含**资金账号 / 密码**,**不要写进代码或 git**
- 推送只调 `send_message / send_user_block / send_warn` — 这些是**信号推送,不自动下单**,用户需手动决策
- 每次推送会创建带日期后缀的板块(如 `TSFresh候选_20260705`),板块列表会累积,需要定期清理

## 推送到通达信客户端的"信号"模式

研究脚本只产生信号,推送走 `tqcenter.tq`:
- `tq.create_sector(block_code, block_name)` → 创建板块
- `tq.send_user_block(block_code, stocks)` → 推股票到板块
- `tq.send_message(msg)` → 策略管理器显示
- `tq.send_warn(...)` → 预警信号

参考实现:[backtrace/tsfresh/tsfresh_top1_industry.py](backtrace/tsfresh/tsfresh_top1_industry.py) 的 `[5/5]` 和 `[6/6]` 段。

## backtrace/ 脚本分类

| 类别 | 路径 | 备注 |
|---|---|---|
| **公共模块** | [backtrace/common/](backtrace/common/) | 改一处 = 改所有 |
| **基础策略**(legacy) | [backtrace/legacy/](backtrace/legacy/) | TQ 信号推送基础模板 |
| **tsfresh 系列** | [backtrace/tsfresh/](backtrace/tsfresh/) | EDA → 选特征 → 训模型 → 应用(10 个脚本) |
| **vbt 系列** | [backtrace/vbt/](backtrace/vbt/) | 真实下单回测 |
| **双层 α 选股** | [backtrace/alpha/](backtrace/alpha/) | 行业 → 个股双层筛选 |
| **K 线形态** | [backtrace/talib/](backtrace/talib/) | TALib 形态回测 / 验证 |
| **GP 因子挖掘** | [backtrace/gp_factor_mining/](backtrace/gp_factor_mining/) | 遗传规划 + 因子库 |

## 已知陷阱(踩过无数次的)

- **Windows GBK 终端**:中文 print 报错 → 用 `PYTHONIOENCODING=utf-8` 或 print 全 ASCII
- **TQ 客户端**必须启动,否则 `tq.initialize` 直接 `RuntimeError`
- **TQ 批量拉** `get_market_data(stock_list=[6000 只])` 会 timeout;实际"未启动客户端"也会"假装成功"返回空数据 → 检查 `df.shape[1]` 必须 > 0
- **tsfresh 默认 `n_jobs=-1`** 在 Windows 下 multiprocessing 卡死 → **强制 `n_jobs=0`**(单进程)
- **NaN 在 tsfresh long format 中不被允许**(`ValueError: Column must not contain NaN values`);`to_long_format` 已经强制 `pd.to_numeric(..., errors='coerce')`,但下游用 `fillna(0.0)` 兜底更稳
- **回测数据集时间要 `df.loc[start:end]` 切片**,否则 vbt 会用全期数据

## 新增脚本的最小模板(放在子目录里)

```python
import warnings
warnings.filterwarnings('ignore')
import sys, os
# 把 backtrace/ 加到 path 才能 from common import ...
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import pandas as pd
import vectorbt as vbt
from datetime import datetime
from tqcenter import tq
from common import tsfresh_config as C, tsfresh_pipeline as P, jhzq_fees as F

tq.initialize(__file__)

# 拉数
df = P.load_ohlcva('600118.SH', verbose=True)
# ... 业务逻辑 ...
# vbt + 真实扣费
pf = vbt.Portfolio.from_signals(...)
trades = pf.trades.records_readable
summary = F.summary_after_fees(trades, '600118.SH')
# 推送
tq.send_user_block('TQI_TOP', ['600118.SH'])
tq.close()
```

## 调试技巧

- **TQ 没数据先看 `df.shape`**:空数据 = 客户端没启动 或 代码前缀错(`.SH`)
- **vbt 0 笔交易**:检查 `entries.sum()` / `exits.sum()`,常见原因 proba 全 NaN / size=0
- **回测结果异常**:对照 `zero_friction_ret`(零摩擦 alpha 上限)和 `net_ret`(扣费后),差距大 = 摩擦吃掉了大部分 alpha
- **tsfresh FDR=0 特征**:样本量少(几百以下)+ FDR=0.05 过严时常见,我们的 `select_relevant` 已自动放宽到 0.20