# -*- coding: utf-8 -*-
"""扫描 data/projection/ 下所有配对 CSV,选出指定日期 Resi_Price > 0 的个股并推送到通达信自选股。

每个 CSV 文件名形如 projection_<codeA>_<codeB>.csv,内含一对股票的残差序列。
Resi_Price 是该配对在当日投影模型拟合后的残差(>0 表示真实价偏离投影价正向)。

用法:
    PYTHONIOENCODING=utf-8 python backtrace/projection/find_resi_positive.py [--date 2026-08-07] [--no-push]

默认扫描 data/projection/,默认日期 = 数据中最新一日。
输出 CSV: backtrace/outputs/projection_resi_positive_<date>.csv
推送板块:PROJECTION / projection_<YYYYMMDD>(同名板块每日累积,需定期清理)
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


def latest_date_with_positive(data_dir: str) -> str:
    """扫描目录,返回最近一个有 Resi_Price>0 的日期 YYYY-MM-DD。"""
    paths = sorted(glob.glob(os.path.join(data_dir, 'projection_*.csv')))
    from collections import defaultdict
    pos = defaultdict(int)
    for p in paths:
        try:
            df = pd.read_csv(p, usecols=['Date', 'Resi_Price'])
        except Exception:
            continue
        for d, r in zip(df['Date'], df['Resi_Price']):
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


def scan(data_dir: str, target_date: str) -> pd.DataFrame:
    """扫描所有 projection CSV,挑出 target_date 当日 Resi_Price > 0 的配对。"""
    rows = []
    paths = sorted(glob.glob(os.path.join(data_dir, 'projection_*.csv')))
    for path in paths:
        parsed = parse_filename(path)
        if parsed is None:
            continue
        code_a, code_b = parsed
        try:
            df = pd.read_csv(path, usecols=['Date', 'Resi_Price'])
        except Exception:
            continue
        hit = df[df['Date'] == target_date]
        if len(hit) == 0:
            continue
        resi = float(hit['Resi_Price'].iloc[0])
        if resi > 0:
            rows.append({
                'code_a': code_a,
                'code_b': code_b,
                'resi_price': resi,
                'file': os.path.basename(path),
            })
    return pd.DataFrame(rows)


def push_to_tq(stocks: list[str], target_date: str, dry_run: bool = False):
    """把候选股推送到通达信板块 PROJECTION / projection_<日期>。"""
    if not stocks:
        print('[推送] 候选为空,跳过推送')
        return
    if dry_run:
        print(f'[推送 DRY-RUN] {len(stocks)} 只候选将推送到板块 {BLOCK_CODE}')
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

    block_name = f'projection_{target_date.replace("-", "")}'
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
                f'projection 残差正向 {target_date}:{len(stocks)} 只候选推送完成 → {block_name}'
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
    p.add_argument('--data-dir', default='data/projection', help='CSV 根目录')
    p.add_argument('--out', default=None, help='输出 CSV 路径')
    p.add_argument('--no-push', action='store_true', help='只扫描,不推送 TQ')
    p.add_argument('--exclude', default=','.join(sorted(DEFAULT_EXCLUDE)),
                   help=f'逗号分隔的排除代码(裸代码,不带后缀),默认:{",".join(sorted(DEFAULT_EXCLUDE))}')
    args = p.parse_args()
    excludes = {x.strip() for x in args.exclude.split(',') if x.strip()}

    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(ROOT, args.data_dir)
    if args.out is None:
        args.out = os.path.join(
            ROOT, 'backtrace', 'outputs',
            f'projection_resi_positive_{(args.date or "latest").replace("-", "")}.csv',
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

    df = scan(args.data_dir, target)
    if len(df) == 0:
        print(f'❌ {target} 当日无 Resi_Price>0 的配对(共扫描 {len(glob.glob(os.path.join(args.data_dir, "projection_*.csv")))} 文件)')
        sys.exit(0)

    df = df.sort_values('resi_price', ascending=False).reset_index(drop=True)
    df.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f'✅ {target} Resi_Price>0 配对数:{len(df)}')
    print(f'   已保存:{args.out}')

    # 提取个股代码(去重,加 .SH/.SZ 后缀,排除指数)
    raw_codes = set()
    for _, r in df.iterrows():
        if r['code_a'] not in excludes:
            raw_codes.add(r['code_a'])
        if r['code_b'] not in excludes:
            raw_codes.add(r['code_b'])
    stocks = sorted(add_suffix(c) for c in raw_codes)
    print(f'\n✅ 去重后个股数:{len(stocks)}(排除:{sorted(excludes)})')
    print('   Top 30:', stocks[:30], '...' if len(stocks) > 30 else '')

    if not args.no_push:
        print('\n=== 推送到通达信 ===')
        push_to_tq(stocks, target, dry_run=False)
    else:
        print('\n[--no-push] 跳过推送,只输出 CSV')

    print('\nTop 20 配对:')
    print(df.head(20).to_string(index=False))


if __name__ == '__main__':
    main()