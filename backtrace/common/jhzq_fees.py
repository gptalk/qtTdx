# -*- coding: utf-8 -*-
"""
江海证券 A 股真实手续费计算器。

费率来源:jhzq/交易凭据.md
  - 佣金:万 0.85,**免 5**(无最低 5 元限制)
  - 印花税:卖出 万 5
  - 过户费:仅沪市,双边 万 0.1;深市 0

资金账号 / 密码 不在本模块。
"""
import pandas as pd

# ===== 费率常量 =====
STOCK_COMMISSION = 0.000085       # 双向 万 0.85,**免 5**(无最低收费);vbt 内部只按双边 fee 收,我们要补完整费率
STAMP_TAX_SELL   = 0.0005         # 仅卖出 万 5(凭据明确);买入 0
TRANSFER_SH      = 0.00001        # 沪市双边 万 0.1;深市为 0
# ETF / 转债(暂未用,留接口)
ETF_COMMISSION   = 0.00005
BOND_COMMISSION  = 0.00005


def is_sh(code: str) -> bool:
    """沪市股票要收过户费,深市不收;判断依据:代码后缀 .SH"""
    return code.upper().endswith(".SH")


def calc_single_fee(amount: float, is_sell: bool, is_sh_market: bool):
    """
    单笔成交完整费用计算(双笔金额 × 单次费率)。
    amount: 一次成交金额(买入金额或卖出金额,各算各的)
    is_sell: 是否卖出(印花税仅卖出收)
    is_sh_market: 是否沪市(沪市双边收过户费,深市 0)
    返回 (总费用, 佣金, 印花税, 过户费)— 4 元组
    """
    comm     = amount * STOCK_COMMISSION                # 双向,免 5(无最低)
    stamp    = amount * STAMP_TAX_SELL if is_sell else 0.0
    transfer = amount * TRANSFER_SH if is_sh_market else 0.0
    return comm + stamp + transfer, comm, stamp, transfer


def adjust_trades_pnl(trades_df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    """
    修正 VBT 原始成交记录,补全完整手续费,生成净盈亏。
    适配 VBT 多种列名版本(records_readable / records / 不同 vbt 版本)。
    入口:trades_df = pf.trades.records_readable(每个 portfolio 自己取)
    出口:在原 df 上新增 5 列(见下方 Schema)

    Schema(新增列):
      佣金_实扣     : float, 买卖双边佣金之和(= entry 佣金 + exit 佣金)
      印花税_实扣   : float, 仅卖出端的印花税
      过户费_实扣   : float, 仅沪市双边过户费,深市为 0
      总手续费      : float, 上述三项合计(= 总费用 - VBT 已扣 + VBT 已扣,详见实现注释)
      净盈亏_扣费后 : float, VBT 原 PnL 扣减总手续费后的净 PnL
    """
    if trades_df is None or trades_df.empty:
        return trades_df

    df = trades_df.copy()
    sh_flag = is_sh(stock_code)

    entry_col = next((c for c in df.columns if 'Entry' in c and 'Price' in c), None)
    exit_col  = next((c for c in df.columns if 'Exit'  in c and 'Price' in c), None)
    size_col  = next((c for c in df.columns if c == 'Size'), None)
    pnl_col   = next((c for c in df.columns if 'PnL' in c), None)
    if not (entry_col and exit_col and size_col and pnl_col):
        raise KeyError(
            f"VBT trades 表缺关键列,现有列: {list(df.columns)}\n"
            f"需要: Entry*Price / Exit*Price / Size / PnL"
        )

    entry_amt = df[entry_col].abs() * df[size_col].abs()       # 买入金额
    exit_amt  = df[exit_col].abs()  * df[size_col].abs()       # 卖出金额

    # 双笔各自的佣金、过户费;印花税只算卖出
    comm_entry, comm_exit, stamp_exit, transfer = [], [], [], []
    if sh_flag:
        transfer_amt = (entry_amt + exit_amt) * TRANSFER_SH
        transfer = [float(x) for x in transfer_amt]
    else:
        transfer = [0.0] * len(df)

    comm_entry = (entry_amt * STOCK_COMMISSION).astype(float).tolist()
    comm_exit  = (exit_amt  * STOCK_COMMISSION).astype(float).tolist()
    stamp_exit = (exit_amt  * STAMP_TAX_SELL).astype(float).tolist()

    total_fee = [comm_entry[i] + comm_exit[i] + stamp_exit[i] + transfer[i]
                 for i in range(len(df))]
    # VBT 内部已按比例扣过佣金(双边),我们这里加"完整费用 - VBT 已扣"
    # 简化:把 VBT 的 PnL 当作"未扣费前的 PnL",完整费用做减项
    net_pnl = [df[pnl_col].iloc[i] - total_fee[i] + (comm_entry[i] + comm_exit[i])  # 还原 + 扣完整
               for i in range(len(df))]

    df['佣金_实扣']       = [c + ce for c, ce in zip(comm_entry, comm_exit)]
    df['印花税_实扣']     = stamp_exit
    df['过户费_实扣']     = transfer
    df['总手续费']        = total_fee
    df['净盈亏_扣费后']   = net_pnl
    return df


def summary_after_fees(trades_df: pd.DataFrame, stock_code: str) -> dict:
    """
    汇总扣费后整体指标(分母用 INIT_CASH 由调用方传入,本函数不接触资金)。

    返回 dict 6 个 key:
      trades           : int,   成交笔数
      gross_pnl        : float, 扣费前 PnL 合计(VBT 原 PnL 列)
      total_stamp      : float, 印花税合计
      total_transfer   : float, 过户费合计
      net_pnl          : float, 净盈亏合计(扣完整手续费)
      avg_net_per_trade: float, 单笔平均净盈亏
    """
    df = adjust_trades_pnl(trades_df, stock_code)
    if df is None or df.empty:
        return {"trades": 0, "gross_pnl": 0.0, "total_stamp": 0.0,
                "total_transfer": 0.0, "net_pnl": 0.0, "avg_net_per_trade": 0.0}
    return {
        "trades":           len(df),
        "gross_pnl":        float(df[next(c for c in df.columns if 'PnL' in c and '扣' not in c)].sum()),
        "total_stamp":      float(df['印花税_实扣'].sum()),
        "total_transfer":   float(df['过户费_实扣'].sum()),
        "net_pnl":          float(df['净盈亏_扣费后'].sum()),
        "avg_net_per_trade": float(df['净盈亏_扣费后'].mean()),
    }
