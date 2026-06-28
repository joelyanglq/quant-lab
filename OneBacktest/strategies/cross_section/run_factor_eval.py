"""
因子评估流水线 v2 (Factor Evaluation CLI)

华泰金工风格: 预处理 → IC分析 → 回归分析 → 换手率分析 → 收益分析

Usage:
    cd OneBacktest

    # 单因子评估
    python -m strategies.cross_section.run_factor_eval --factor RS_12M

    # 全量因子
    python -m strategies.cross_section.run_factor_eval

    # rank+zscore 预处理
    python -m strategies.cross_section.run_factor_eval --factor RS_12M --preprocess rank_zscore

    # 手动指定 horizon
    python -m strategies.cross_section.run_factor_eval --factor RS_12M --horizon 21

    # 跳过中性化
    python -m strategies.cross_section.run_factor_eval --factor RS_12M --no-neutralize

    # 不出图
    python -m strategies.cross_section.run_factor_eval --factor RS_12M --no-plot
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

# Ensure OneBacktest on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.prices import load_index_symbols, load_price_panel
from data.fundamentals import build_shares_panel
from strategies.cross_section.factors import (
    compute_technical_factors,
    compute_reversal_factors,
    compute_fundamental_factors,
)
from strategies.cross_section.preprocessing import preprocess_factor
from strategies.cross_section.neutralize import load_gics_sector_map
from strategies.cross_section.eval import run_factor_eval

warnings.filterwarnings('ignore', category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser(description='因子评估流水线 v2')
    parser.add_argument('--factor', type=str, default=None,
                        help='单因子名 (不传=所有 1d 因子)')
    parser.add_argument('--start', type=str, default='2020-01-01')
    parser.add_argument('--end', type=str, default='2026-12-31')
    parser.add_argument('--n-symbols', type=int, default=0,
                        help='限制 symbol 数量 (0=全部)')
    parser.add_argument('--preprocess', type=str, default='mad_zscore',
                        choices=['mad_zscore', 'rank_zscore', 'none'],
                        help='预处理方法')
    parser.add_argument('--no-neutralize', action='store_true',
                        help='跳过行业/市值中性化')
    parser.add_argument('--rebalance-freq', type=str, default='M',
                        help='调仓频率 (M, W-FRI, Q)')
    parser.add_argument('--horizons', type=str, default='1,5,10,21,42,63',
                        help='IC horizons (逗号分隔)')
    parser.add_argument('--horizon', type=int, default=None,
                        help='手动指定评估 horizon (None=IC分析自动选择)')
    parser.add_argument('--no-plot', action='store_true',
                        help='不出图')
    parser.add_argument('--output-dir', type=str, default='output/factor_eval',
                        help='输出目录')
    return parser.parse_args()


def compute_1d_factors(panels, symbols):
    """计算所有只需 1d 数据的因子"""
    close = panels['close']
    high = panels['high']
    low = panels['low']
    open_price = panels['open']

    all_factors = {}

    print('  Computing technical factors...')
    tech = compute_technical_factors(close, high, low)
    all_factors.update(tech)

    print('  Computing reversal factors...')
    rev = compute_reversal_factors(close, open_price)
    all_factors.update(rev)

    print('  Computing fundamental factors...')
    try:
        fund = compute_fundamental_factors(symbols, close)
        all_factors.update(fund)
    except Exception as e:
        print(f'  [WARN] Fundamental factors failed: {e}')

    return all_factors


def main():
    args = parse_args()
    horizons = [int(h) for h in args.horizons.split(',')]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # Step 1: 加载数据
    # ═══════════════════════════════════════════════════════════
    print('Step 1: Loading data...')
    t0 = time.time()

    symbols = load_index_symbols()
    if args.n_symbols > 0:
        symbols = symbols[:args.n_symbols]
    print(f'  {len(symbols)} symbols')

    panels = load_price_panel(symbols, args.start, args.end)
    close = panels['close']

    # Shares (for mktcap)
    print('  Loading shares...')
    shares = None
    try:
        shares = build_shares_panel(symbols, close.index)
    except Exception:
        print('  [WARN] Shares data unavailable')

    # Sector map
    sector_map = load_gics_sector_map()
    print(f'  GICS: {len(sector_map) if sector_map is not None else 0} mappings')

    # SPY
    spy_close = None
    if 'SPY' in close.columns:
        spy_close = close['SPY']
    else:
        try:
            spy_panel = load_price_panel(['SPY'], args.start, args.end)
            spy_close = spy_panel['close']['SPY']
        except Exception:
            pass

    print(f'  Data loaded in {time.time() - t0:.1f}s')

    # ═══════════════════════════════════════════════════════════
    # Step 2: 计算因子
    # ═══════════════════════════════════════════════════════════
    print('\nStep 2: Computing factors...')
    t0 = time.time()

    all_factors = compute_1d_factors(panels, symbols)
    print(f'  {len(all_factors)} factors computed in {time.time() - t0:.1f}s')

    if args.factor:
        if args.factor not in all_factors:
            print(f'  [ERROR] Factor "{args.factor}" not found.')
            print(f'  Available: {sorted(all_factors.keys())}')
            sys.exit(1)
        all_factors = {args.factor: all_factors[args.factor]}

    # ═══════════════════════════════════════════════════════════
    # Step 3: 预处理
    # ═══════════════════════════════════════════════════════════
    if args.preprocess != 'none':
        print(f'\nStep 3: Preprocessing ({args.preprocess}, neutralize={not args.no_neutralize})...')
        t0 = time.time()

        mktcap = close * shares if shares is not None else None

        processed = {}
        for name, fac in all_factors.items():
            processed[name] = preprocess_factor(
                fac, method=args.preprocess,
                neutralize=not args.no_neutralize,
                mktcap=mktcap, sector_map=sector_map,
            )
        all_factors = processed
        print(f'  Preprocessing done in {time.time() - t0:.1f}s')

    # ═══════════════════════════════════════════════════════════
    # Step 4: 四步评估
    # ═══════════════════════════════════════════════════════════
    print('\nStep 4: Factor Evaluation...')

    for name, fac in all_factors.items():
        run_factor_eval(
            fac, close,
            factor_name=name,
            horizons=horizons,
            horizon=args.horizon,
            rebalance_freq=args.rebalance_freq,
            sector_map=sector_map,
            benchmark_close=spy_close,
            save_dir=str(output_dir),
            plot=not args.no_plot,
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
