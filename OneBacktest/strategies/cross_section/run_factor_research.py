"""
因子研究流水线 (Factor Research CLI)

统一入口: 计算 → 诊断 → Alpha 测试 → 出图

Usage:
    cd OneBacktest

    # 全量因子研究 (所有 1d 因子)
    python -m strategies.cross_section.run_factor_research

    # 只研究一个因子
    python -m strategies.cross_section.run_factor_research --factor RS_12M

    # 只做诊断
    python -m strategies.cross_section.run_factor_research --stage diagnostics

    # 快速模式
    python -m strategies.cross_section.run_factor_research --n-symbols 50 --no-fama-macbeth

    # 不出图
    python -m strategies.cross_section.run_factor_research --no-plot
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure OneBacktest on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.prices import (
    load_index_symbols,
    load_price_panel,
)
from data.fundamentals import (
    build_shares_panel,
)
from strategies.cross_section.factors import (
    compute_technical_factors,
    compute_reversal_factors,
    compute_fundamental_factors,
)
from strategies.cross_section.diagnostics import (
    run_diagnostics,
    run_batch_diagnostics,
    build_style_factor_proxies,
    format_diagnostic_report,
)
from strategies.cross_section.alpha_test import (
    run_alpha_test,
    run_batch_alpha_test,
    load_gics_sectors,
    format_alpha_test_report,
)
from strategies.cross_section.diagnostics_plotting import (
    plot_diagnostic_summary,
    plot_alpha_test_summary,
    plot_ic_decay_curve,
    plot_fama_macbeth,
    plot_sub_sample_decomposition,
    plot_industry_distribution,
    plot_full_factor_report,
)

warnings.filterwarnings('ignore', category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser(description='因子研究流水线')
    parser.add_argument('--factor', type=str, default=None,
                        help='单因子名 (不传=所有 1d 因子)')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'diagnostics', 'alpha'],
                        help='执行阶段')
    parser.add_argument('--start', type=str, default='2020-01-01')
    parser.add_argument('--end', type=str, default='2026-12-31')
    parser.add_argument('--n-symbols', type=int, default=0,
                        help='限制 symbol 数量 (0=全部)')
    parser.add_argument('--rebalance-freq', type=str, default='M',
                        help='调仓频率 (M, W-FRI, Q)')
    parser.add_argument('--horizons', type=str, default='1,5,21,63',
                        help='IC horizons (逗号分隔)')
    parser.add_argument('--transaction-cost', type=float, default=10.0,
                        help='单边交易成本 (bps)')
    parser.add_argument('--no-fama-macbeth', action='store_true',
                        help='跳过 Fama-MacBeth 回归')
    parser.add_argument('--no-sub-sample', action='store_true',
                        help='跳过子样本分析')
    parser.add_argument('--no-plot', action='store_true',
                        help='不出图')
    parser.add_argument('--output-dir', type=str, default='output/factor_research',
                        help='输出目录')
    parser.add_argument('--save-csv', action='store_true',
                        help='保存汇总 CSV')
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
    volume = panels['volume']
    trading_dates = close.index

    # Shares (for mktcap)
    print('  Loading shares...')
    try:
        shares = build_shares_panel(symbols, trading_dates)
    except Exception:
        shares = None
        print('  [WARN] Shares data unavailable')

    # GICS
    gics_df = load_gics_sectors(symbols)
    print(f'  GICS: {len(gics_df)} mappings')

    # SPY (may not be in index constituents, load separately if needed)
    spy_close = None
    if 'SPY' in close.columns:
        spy_close = close['SPY']
    else:
        try:
            spy_panel = load_price_panel(['SPY'], args.start, args.end)
            spy_close = spy_panel['close']['SPY']
            print('  SPY loaded separately for regime analysis')
        except Exception:
            print('  [WARN] SPY data unavailable, sub-sample analysis will be skipped')

    print(f'  Data loaded in {time.time() - t0:.1f}s')

    # ═══════════════════════════════════════════════════════════
    # Step 2: 计算因子
    # ═══════════════════════════════════════════════════════════
    print('\nStep 2: Computing factors...')
    t0 = time.time()

    all_factors = compute_1d_factors(panels, symbols)
    print(f'  {len(all_factors)} factors computed in {time.time() - t0:.1f}s')

    # 如果指定单因子
    if args.factor:
        if args.factor not in all_factors:
            print(f'  [ERROR] Factor "{args.factor}" not found.')
            print(f'  Available: {sorted(all_factors.keys())}')
            sys.exit(1)
        all_factors = {args.factor: all_factors[args.factor]}

    # ═══════════════════════════════════════════════════════════
    # Step 3: 质量诊断
    # ═══════════════════════════════════════════════════════════
    diag_reports = {}
    if args.stage in ('all', 'diagnostics'):
        print('\nStep 3: Factor Diagnostics...')
        t0 = time.time()

        # 构建 ROE (如果可用)
        roe = all_factors.get('ROE')

        diag_reports = run_batch_diagnostics(
            all_factors,
            close=close, volume=volume,
            shares=shares, roe=roe,
            gics_df=gics_df,
        )

        # 打印摘要表
        print(f'\n{"Factor":<25s} {"Coverage":>8s} {"Drift":>6s} {"MaxCorr":>8s} '
              f'{"DomStyle":<12s} {"Turnover":>8s} {"HHI_L":>6s} {"Pass":>5s}')
        print('-' * 90)
        for name, r in sorted(diag_reports.items()):
            cov = f'{r.coverage.median_coverage:.0%}'
            drift = 'YES' if r.distribution.drift_detected else 'no'
            mc = f'{r.known_factors.max_abs_corr:.3f}' if r.known_factors else 'N/A'
            ds = r.known_factors.dominant_style if r.known_factors else 'N/A'
            turn = f'{r.turnover.mean_turnover:.0%}'
            hhi = f'{r.industry.hhi_top:.3f}' if r.industry else 'N/A'
            p = 'PASS' if r.pass_sanity else 'WARN'
            print(f'{name:<25s} {cov:>8s} {drift:>6s} {mc:>8s} {ds:<12s} {turn:>8s} {hhi:>6s} {p:>5s}')

        print(f'\n  Diagnostics done in {time.time() - t0:.1f}s')

        # 保存 CSV
        if args.save_csv:
            rows = []
            for name, r in diag_reports.items():
                row = {
                    'factor': name,
                    'coverage': r.coverage.median_coverage,
                    'drift': r.distribution.drift_detected,
                    'drift_p': r.distribution.drift_pvalue,
                    'max_style_corr': r.known_factors.max_abs_corr if r.known_factors else np.nan,
                    'dominant_style': r.known_factors.dominant_style if r.known_factors else '',
                    'turnover': r.turnover.mean_turnover,
                    'hhi_top': r.industry.hhi_top if r.industry else np.nan,
                    'hhi_bot': r.industry.hhi_bot if r.industry else np.nan,
                    'neutralized_ic_21d': r.industry.neutralized_ic if r.industry else np.nan,
                    'raw_ic_21d': r.industry.raw_ic_21d if r.industry else np.nan,
                    'pass': r.pass_sanity,
                    'warnings': '; '.join(r.warnings),
                }
                rows.append(row)
            pd.DataFrame(rows).to_csv(output_dir / 'summary_diagnostics.csv', index=False)
            print(f'  Saved {output_dir / "summary_diagnostics.csv"}')

    # ═══════════════════════════════════════════════════════════
    # Step 4: Alpha Test
    # ═══════════════════════════════════════════════════════════
    alpha_reports = {}
    if args.stage in ('all', 'alpha'):
        print('\nStep 4: Alpha Test...')
        t0 = time.time()

        alpha_reports = run_batch_alpha_test(
            all_factors,
            close=close,
            volume=volume,
            shares=shares,
            gics_df=gics_df,
            spy_close=spy_close,
            rebalance_freq=args.rebalance_freq,
            horizons=horizons,
            transaction_cost_bps=args.transaction_cost,
            do_fama_macbeth=not args.no_fama_macbeth,
            do_sub_sample=not args.no_sub_sample,
        )

        # 打印摘要表
        h_cols = '  '.join([f'IC_{h}d' for h in horizons])
        print(f'\n{"Factor":<25s} {h_cols}  {"ICIR_21d":>8s} {"NW_t_21d":>8s} '
              f'{"LS_Sharpe":>9s} {"Net_Sharpe":>10s}', end='')
        if not args.no_fama_macbeth:
            print(f' {"FM_t":>6s}', end='')
        print()
        print('-' * 120)

        for name, r in sorted(alpha_reports.items()):
            mic = r.multi_horizon_ic
            eq = r.enhanced_quantile
            ic_strs = '  '.join([f'{mic.mean_ic.get(h, np.nan):>+7.4f}' for h in horizons])
            icir_21 = f'{mic.icir.get(21, np.nan):>8.2f}'
            nw_21 = f'{mic.tstat_nw.get(21, np.nan):>8.2f}'
            ls_s = f'{eq.base_metrics["ls_sharpe"]:>9.2f}'
            net_s = f'{eq.ls_sharpe_net:>10.2f}'
            print(f'{name:<25s} {ic_strs}  {icir_21}  {nw_21} {ls_s} {net_s}', end='')
            if not args.no_fama_macbeth and r.fama_macbeth:
                print(f' {r.fama_macbeth.tstat:>6.2f}', end='')
            print()

        print(f'\n  Alpha Test done in {time.time() - t0:.1f}s')

        # 保存 CSV
        if args.save_csv:
            rows = []
            for name, r in alpha_reports.items():
                row = {'factor': name}
                for h in horizons:
                    row[f'ic_{h}d'] = r.multi_horizon_ic.mean_ic.get(h, np.nan)
                    row[f'icir_{h}d'] = r.multi_horizon_ic.icir.get(h, np.nan)
                    row[f'nw_t_{h}d'] = r.multi_horizon_ic.tstat_nw.get(h, np.nan)
                row['ic_half_life'] = r.multi_horizon_ic.ic_half_life
                row['ls_sharpe_gross'] = r.enhanced_quantile.base_metrics['ls_sharpe']
                row['ls_sharpe_net'] = r.enhanced_quantile.ls_sharpe_net
                row['ls_max_dd'] = r.enhanced_quantile.ls_max_dd
                row['monotonicity'] = r.enhanced_quantile.base_metrics['monotonicity']
                if r.fama_macbeth:
                    row['fm_slope'] = r.fama_macbeth.mean_slope
                    row['fm_tstat'] = r.fama_macbeth.tstat
                rows.append(row)
            pd.DataFrame(rows).to_csv(output_dir / 'summary_alpha_test.csv', index=False)
            print(f'  Saved {output_dir / "summary_alpha_test.csv"}')

    # ═══════════════════════════════════════════════════════════
    # Step 5: 出图
    # ═══════════════════════════════════════════════════════════
    if not args.no_plot and (diag_reports or alpha_reports):
        print('\nStep 5: Generating plots...')
        import matplotlib
        matplotlib.use('Agg')

        for name in all_factors:
            d_report = diag_reports.get(name)
            a_report = alpha_reports.get(name)

            factor_dir = output_dir / name
            factor_dir.mkdir(parents=True, exist_ok=True)

            if d_report:
                plot_diagnostic_summary(d_report, str(factor_dir / 'diagnostic_summary.png'))
                if d_report.industry is not None:
                    plot_industry_distribution(d_report, str(factor_dir / 'industry_distribution.png'))
            if a_report:
                plot_alpha_test_summary(a_report, str(factor_dir / 'alpha_test_summary.png'))
                plot_ic_decay_curve(a_report, str(factor_dir / 'ic_decay.png'))
                if a_report.fama_macbeth and len(a_report.fama_macbeth.slope_series) > 0:
                    plot_fama_macbeth(a_report, str(factor_dir / 'fama_macbeth.png'))
                if a_report.sub_sample and not a_report.sub_sample.regime_stats.empty:
                    plot_sub_sample_decomposition(a_report, str(factor_dir / 'sub_sample.png'))

        print(f'  Plots saved to {output_dir}/')

    # ═══════════════════════════════════════════════════════════
    # 完整因子报告 (单因子模式)
    # ═══════════════════════════════════════════════════════════
    if args.factor and args.factor in diag_reports and args.factor in alpha_reports:
        print('\n' + '=' * 60)
        try:
            print(format_diagnostic_report(diag_reports[args.factor]))
            print()
            print(format_alpha_test_report(alpha_reports[args.factor]))
        except UnicodeEncodeError:
            # Windows GBK console fallback
            import io, sys as _sys
            _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
            print(format_diagnostic_report(diag_reports[args.factor]))
            print()
            print(format_alpha_test_report(alpha_reports[args.factor]))

    print('\nDone.')


if __name__ == '__main__':
    main()
