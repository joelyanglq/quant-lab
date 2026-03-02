"""
因子研究入口脚本

Usage:
    cd OneBacktest
    python -m factor_research.run_factor_backtest                    # 全量 (510 symbols)
    python -m factor_research.run_factor_backtest --n-symbols 50     # 快速测试
    python -m factor_research.run_factor_backtest --factor ROE       # 只跑一个因子
    python -m factor_research.run_factor_backtest --no-plot          # 不画图
"""
import argparse
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings('ignore', 'Mean of empty slice')

from .data_loader import load_index_symbols, load_price_panel
from .factors_fundamental import compute_fundamental_factors
from .factors_technical import compute_technical_factors
from .backtest import build_monthly_rebalance
from .composite import build_composite_factor
from .analytics import compute_factor_metrics, format_metrics
from .plotting import plot_factor_report


def main():
    parser = argparse.ArgumentParser(description='Factor Research Backtest')
    parser.add_argument('--start', default='2019-01-01', help='Backtest start')
    parser.add_argument('--end', default='2025-12-31', help='Backtest end')
    parser.add_argument('--n-symbols', type=int, default=0,
                        help='Limit symbols (0=all)')
    parser.add_argument('--factor', type=str, default=None,
                        help='Run single factor only')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')
    parser.add_argument('--save-dir', default=None,
                        help='Save plots to directory')
    args = parser.parse_args()

    # ── 1. 加载数据 ────────────────────────────────
    symbols = load_index_symbols()
    if args.n_symbols > 0:
        symbols = symbols[:args.n_symbols]

    print(f'Loading price data for {len(symbols)} symbols...')
    t0 = time.time()
    panels = load_price_panel(symbols, args.start, args.end)
    close = panels['close']
    # 过滤掉全 NaN 的 symbol
    valid_syms = close.columns[close.notna().any()].tolist()
    close = close[valid_syms]
    print(f'  {close.shape[0]} dates x {close.shape[1]} symbols ({time.time()-t0:.1f}s)')

    # ── 2. 计算因子 ────────────────────────────────
    print('\nComputing fundamental factors...')
    t0 = time.time()
    fund_factors = compute_fundamental_factors(valid_syms, close)
    print(f'  8 fundamental factors ({time.time()-t0:.1f}s)')

    print('Computing technical factors...')
    t0 = time.time()
    tech_factors = compute_technical_factors(
        close, panels['high'][valid_syms], panels['low'][valid_syms]
    )
    print(f'  5 technical factors ({time.time()-t0:.1f}s)')

    all_factors = {**fund_factors, **tech_factors}

    # ── 3. 回测 ─────────────────────────────────────
    if args.factor:
        factor_names = [args.factor]
    else:
        factor_names = list(all_factors.keys())

    print(f'\nRunning backtests for {len(factor_names)} factors...')

    results = {}
    for name in factor_names:
        if name not in all_factors:
            print(f'  {name}: NOT FOUND, skipping')
            continue

        t0 = time.time()
        bt = build_monthly_rebalance(all_factors[name], close)
        met = compute_factor_metrics(bt)
        results[name] = {'backtest': bt, 'metrics': met}

        elapsed = time.time() - t0
        print(f'\n{format_metrics(met, name)}')
        print(f'  ({elapsed:.1f}s)')

    # ── 4. 复合因子 ─────────────────────────────────
    if not args.factor:
        print('\n--- Composite Factor ---')
        t0 = time.time()
        composite = build_composite_factor(all_factors)
        bt = build_monthly_rebalance(composite, close)
        met = compute_factor_metrics(bt)
        results['Composite'] = {'backtest': bt, 'metrics': met}
        print(f'\n{format_metrics(met, "Composite")}')
        print(f'  ({time.time()-t0:.1f}s)')

    # ── 5. 可视化 ───────────────────────────────────
    if not args.no_plot:
        save_dir = Path(args.save_dir) if args.save_dir else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        for name, res in results.items():
            save_path = str(save_dir / f'{name}.png') if save_dir else None
            plot_factor_report(res['backtest'], res['metrics'], name, save_path)

    # ── 6. 汇总 ─────────────────────────────────────
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    header = (f'{"Factor":<20s} {"L/S Ret":>8s} {"Vol":>7s} {"Sharpe":>7s} '
              f'{"WinRate":>8s} {"MaxDD":>7s} {"RankIC":>7s} {"ICIR":>6s} {"Mono":>6s}')
    print(header)
    print('-' * 80)
    for name, res in results.items():
        m = res['metrics']
        print(f'{name:<20s} {m["ls_annual_return"]:>7.1%} {m["ls_annual_vol"]:>7.1%} '
              f'{m["ls_sharpe"]:>7.2f} {m["ls_win_rate"]:>7.1%} '
              f'{m["ls_max_drawdown"]:>7.2%} {m["rank_ic"]:>7.4f} '
              f'{m["rank_icir"]:>6.2f} {m["monotonicity"]:>6.2f}')


if __name__ == '__main__':
    main()
