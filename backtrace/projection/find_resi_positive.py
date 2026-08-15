# -*- coding: utf-8 -*-
"""扫描 data/projection/ 下所有配对 CSV,选出指定日期 State_Resi_Price > 0 的个股并推送到通达信自选股。

每个 CSV 文件名形如 projection_<codeA>_<codeB>.csv,内含一对股票的残差序列。
State_Resi_Price 是该配对在当日 state 投影模型拟合后的残差方向斜率
(>0 表示真实方向斜率偏离投影方向斜率正向)。

注:2026-08-15 列名从 `Resi_Price` 重命名为 `State_Resi_Price`(加 State_ 前缀),
跟 movement 投影的 `Move_Resi_Price` 列明确区分。

基线类型(codeA):
  - 000001 / 399001 → 大盘(上证综指 / 深证成指)
  - 881xxx          → 行业(申万二级)

参数:
  --date      YYYY-MM-DD   目标日期(默认:数据中最新一日)
  --type      all/大盘/行业  按基线类型过滤。默认 all(向后兼容)
  --data-dir  path         CSV 根目录。默认 data/projection
  --out       path         输出 CSV 路径
  --no-push                只扫描,不推送 TQ
  --exclude    codes       逗号分隔的排除代码(裸代码),默认 000001,399001

用法:
  PYTHONIOENCODING=utf-8 python backtrace/projection/find_resi_positive.py [--date 2026-08-07] [--no-push]

默认扫描 data/projection/,默认日期 = 数据中最新一日。
输出 CSV: backtrace/outputs/projection_resi_positive_<date>.csv
推送板块:PROJECTION / projection_<YYYYMMDD>(同名板块每日累积,需定期清理)
按 --type 过滤时,板块名加前缀:大盘_projection_<YYYYMMDD> / 行业_projection_<YYYYMMDD>
"""
import argparse
import glob
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TQ_PLUGIN = 'C:/new_tdx_mock/PYPlugins/user'
BLOCK_CODE = 'PROJECTION'
# 指数代码(投影中作主轴,不是真正的个股)默认排除
DEFAULT_EXCLUDE = {'000001', '399001'}

# 基线代码 → 类型
MARKET_INDEX_CODES = {'000001', '399001'}
SECTOR_INDEX_PREFIX = '881'


def classify_baseline(code_a: str) -> str:
    """由 codeA 判定基线类型:'大盘' / '行业' / '其他'。

    - 000001 / 399001 → 大盘
    - 881xxx          → 行业
    - 其它             → 其他(异常文件,跳过提醒)
    """
    if code_a in MARKET_INDEX_CODES:
        return '大盘'
    if code_a.startswith(SECTOR_INDEX_PREFIX):
        return '行业'
    return '其他'


def parse_filename(path: str):
    """从文件名解析 codeA / codeB;格式不合法返回 None。"""
    base = os.path.basename(path)
    if not base.startswith('projection_') or not base.endswith('.csv'):
        return None
    parts = base[len('projection_'):-len('.csv')].split('_')
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def add_suffix(code: str) -> str:
    """6 位代码补 .SH / .SZ(深市 0/3 开头,沪市 6/9 开头)。"""
    if '.' in code:
        return code
    if code.startswith(('6', '9')):
        return f"{code}.SH"
    if code.startswith(('0', '1', '2', '3')):
        return f"{code}.SZ"
    return code  # 5 位或其它,原样返回


def is_pushable(code: str) -> bool:
    """只有补得出 .SH/.SZ 的 6 位 A 股代码能进自选股。

    88xxxx 申万行业指数(投影里作基准轴)补不出后缀,必须在推送前剔除 ——
    tqcenter.convert_or_validate 撞上一个无后缀代码会丢弃整批返回 '',
    板块会被推成空的,而且不报错。
    """
    return '.' in add_suffix(code)


def latest_date_with_positive(data_dir: str) -> str:
    """扫描目录,返回最近一个有 State_Resi_Price>0 的日期 YYYY-MM-DD。"""
    paths = sorted(glob.glob(os.path.join(data_dir, 'projection_*.csv')))
    from collections import defaultdict
    pos = defaultdict(int)
    for p in paths:
        try:
            df = pd.read_csv(p, usecols=['Date', 'State_Resi_Price'])
        except Exception:
            continue
        for d, r in zip(df['Date'], df['State_Resi_Price']):
            if pd.notna(r) and r > 0:
                pos[d] += 1
    if not pos:
        return ''
    return max(pos)


def latest_date_any(data_dir: str) -> str:
    """扫描目录,返回数据中的最新日期(不论正负)。"""
    paths = sorted(glob.glob(os.path.join(data_dir, 'projection_*.csv')))
    latest = ''
    for p in paths:
        try:
            df = pd.read_csv(p, usecols=['Date'])
        except Exception:
            continue
        if len(df):
            d = df['Date'].iloc[-1]
            if d > latest:
                latest = d
    return latest


def scan(data_dir: str, target_date: str, baseline_type: str = 'all') -> pd.DataFrame:
    """扫描所有 projection CSV,挑出 target_date 当日 State_Resi_Price > 0 的配对。

    baseline_type:
      'all'   — 全部配对(向后兼容)
      '大盘' — 只保留 codeA ∈ {000001, 399001} 的配对
      '行业' — 只保留 codeA 以 881 开头的配对

    返回 DataFrame 含 baseline_type 列,方便按类型聚合/分类推送。
    """
    rows = []
    paths = sorted(glob.glob(os.path.join(data_dir, 'projection_*.csv')))
    for path in paths:
        parsed = parse_filename(path)
        if parsed is None:
            continue
        code_a, code_b = parsed
        btype = classify_baseline(code_a)
        if baseline_type != 'all' and btype != baseline_type:
            continue
        try:
            df = pd.read_csv(path, usecols=['Date', 'State_Resi_Price'])
        except Exception:
            continue
        hit = df[df['Date'] == target_date]
        if len(hit) == 0:
            continue
        resi = float(hit['State_Resi_Price'].iloc[0])
        if resi > 0:
            rows.append({
                'baseline_type': btype,
                'code_a': code_a,
                'code_b': code_b,
                'resi_price': resi,
                'file': os.path.basename(path),
            })
    return pd.DataFrame(rows)


def push_to_tq(stocks: list[str], target_date: str, dry_run: bool = False,
                block_prefix: str = ''):
    """把候选股推送到通达信板块 PROJECTION / projection_<日期>。

    block_prefix 为空:板块名 = projection_<YYYYMMDD>(向后兼容,混推)
    block_prefix 非空:板块名 = <prefix>_projection_<YYYYMMDD>(按基线类型分流)
    """
    if not stocks:
        print('[推送] 候选为空,跳过推送')
        return
    # 空推送客户端照收且 ErrorId=0,查返回值查不出来,只能推之前自己拦
    bad = [s for s in stocks if '.' not in s]
    if bad:
        print(f'[推送] 中止:{bad} 无市场后缀,会导致整批被丢弃、板块推成空的')
        return
    if dry_run:
        prefix_label = f'{block_prefix}_' if block_prefix else ''
        print(f'[推送 DRY-RUN] {len(stocks)} 只候选将推送到板块 {BLOCK_CODE}/{prefix_label}projection_<date>')
        print('           ', stocks[:20], '...' if len(stocks) > 20 else '')
        return
    if not os.path.isdir(TQ_PLUGIN):
        print(f'[推送] TQ 插件目录不存在:{TQ_PLUGIN},跳过推送')
        return
    if TQ_PLUGIN not in sys.path:
        sys.path.insert(0, TQ_PLUGIN)
    try:
        from tqcenter import tq
    except Exception as e:
        print(f'[推送] tqcenter 加载失败:{e},跳过推送')
        return

    suffix = target_date.replace("-", "")
    block_name = f'{block_prefix}_projection_{suffix}' if block_prefix else f'projection_{suffix}'
    try:
        tq.initialize(__file__)
        try:
            tq.create_sector(block_code=BLOCK_CODE, block_name=block_name)
            print(f'  [OK] 自定义板块 {block_name} 创建成功')
        except Exception as e:
            print(f'  [INFO] 板块可能已存在(可继续 send_user_block):{e}')
        tq.send_user_block(block_code=BLOCK_CODE, stocks=stocks)
        print(f'  [OK] 已推送 {len(stocks)} 只到板块 {block_name}')
        print(f'  [提示] 打开通达信客户端 → 自选股板块 → {block_name}')
        try:
            tq.send_message(
                f'projection({block_prefix or "all"}) 残差正向 {target_date}:{len(stocks)} 只候选推送完成 → {block_name}'
            )
            print('  [OK] 已发送 MSG 到策略管理器')
        except Exception as e:
            print(f'  [WARN] 策略管理器 MSG 失败:{e}')
        tq.close()
    except Exception as e:
        print(f'  [WARN] TQ 推送失败:{e}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--date', default=None, help='目标日期 YYYY-MM-DD(默认:数据中最新一日)')
    p.add_argument('--type', default='all', choices=['all', '大盘', '行业'],
                   help='按基线类型过滤。all=全部(默认)/大盘=仅 000001+399001/行业=仅 881xxx')
    p.add_argument('--data-dir', default='data/projection', help='CSV 根目录')
    p.add_argument('--out', default=None, help='输出 CSV 路径(自动按 --type 加前后缀)')
    p.add_argument('--no-push', action='store_true', help='只扫描,不推送 TQ')
    p.add_argument('--exclude', default=','.join(sorted(DEFAULT_EXCLUDE)),
                   help=f'逗号分隔的排除代码(裸代码,不带后缀),默认:{",".join(sorted(DEFAULT_EXCLUDE))}')
    args = p.parse_args()
    excludes = {x.strip() for x in args.exclude.split(',') if x.strip()}

    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(ROOT, args.data_dir)
    if args.out is None:
        type_tag = '' if args.type == 'all' else f'_{args.type}'
        args.out = os.path.join(
            ROOT, 'backtrace', 'outputs',
            f'projection_resi_positive{type_tag}_{(args.date or "latest").replace("-", "")}.csv',
        )
    elif not os.path.isabs(args.out):
        args.out = os.path.join(ROOT, args.out)

    target = args.date
    if not target:
        target = latest_date_any(args.data_dir)
        if not target:
            print('❌ 数据目录为空,无任何日期')
            sys.exit(1)
        print(f'[默认] 使用数据中最新日期:{target}')

    print(f'[过滤] 基线类型 = {args.type}')
    df = scan(args.data_dir, target, baseline_type=args.type)
    if len(df) == 0:
        scanned = len(glob.glob(os.path.join(args.data_dir, 'projection_*.csv')))
        print(f'❌ {target} 当日无 State_Resi_Price>0 的配对(过滤类型={args.type},共扫描 {scanned} 文件)')
        sys.exit(0)

    df = df.sort_values('resi_price', ascending=False).reset_index(drop=True)
    df.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f'✅ {target} State_Resi_Price>0 配对数:{len(df)}')
    print(f'   已保存:{args.out}')

    # 按基线类型统计(只对 all 模式有意义;过滤模式下只有一类)
    if args.type == 'all' and 'baseline_type' in df.columns:
        type_counts = df['baseline_type'].value_counts().to_dict()
        print(f'   按基线类型:{type_counts}')

    # 提取个股代码(去重,加 .SH/.SZ 后缀,排除指数)
    # 每只股可能在多对里出现,取它的最大 resi_price 作为代表值(推送的入选门槛 = 至少一对正向)
    raw_codes = set()
    code_to_max_resi = {}
    for _, r in df.iterrows():
        for raw in (r['code_a'], r['code_b']):
            if raw in excludes:
                continue
            raw_codes.add(raw)
            rpv = float(r['resi_price'])
            if (raw not in code_to_max_resi) or (rpv > code_to_max_resi[raw]):
                code_to_max_resi[raw] = rpv
    stocks = sorted(add_suffix(c) for c in raw_codes if is_pushable(c))
    dropped = sorted(c for c in raw_codes if not is_pushable(c))
    print(f'\n✅ 去重后个股数:{len(stocks)}(排除:{sorted(excludes)})')
    if dropped:
        print(f'   [剔除非个股] {dropped}')
    print('   Top 30:', stocks[:30], '...' if len(stocks) > 30 else '')

    # 推送前打印每只股的 resi_price(按 resi_price 降序,一只股可能在多对里 → 取最大)
    stock_resi_rows = sorted(
        ((add_suffix(c), code_to_max_resi[c]) for c in raw_codes if is_pushable(c)),
        key=lambda x: x[1], reverse=True,
    )
    print(f'\n=== 推送个股 resi_price 明细({len(stock_resi_rows)} 只) ===')
    detail_df = pd.DataFrame(stock_resi_rows, columns=['code', 'resi_price'])
    print(detail_df.to_string(index=False))

    if not args.no_push:
        print('\n=== 推送到通达信 ===')
        # --type 过滤时:板块名加前缀;all 模式:维持原板块名(向后兼容)
        block_prefix = '' if args.type == 'all' else args.type
        push_to_tq(stocks, target, dry_run=False, block_prefix=block_prefix)
    else:
        print('\n[--no-push] 跳过推送,只输出 CSV')

    print('\nTop 20 配对:')
    print(df.head(20).to_string(index=False))


if __name__ == '__main__':
    main()