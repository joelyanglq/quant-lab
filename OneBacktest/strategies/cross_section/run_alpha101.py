"""
101 Formulaic Alphas — 标准研究流程

对 Kakushadze (2016) 全部 101 个 alpha 进行标准因子研究:
  Phase 1: IC 初筛 (全部 101 个)
  Phase 2: 质量诊断 (top N)
  Phase 3: Alpha Test — 多周期 IC, 分位数回测, Fama-MacBeth, 子样本分析 (top N)
  Phase 4: 与现有因子对比 + 出图

用法:
    cd OneBacktest
    python -m strategies.cross_section.run_alpha101                        # 标准流程
    python -m strategies.cross_section.run_alpha101 --universe full        # 全市场
    python -m strategies.cross_section.run_alpha101 --top-n 30             # 前 30 走深度分析
    python -m strategies.cross_section.run_alpha101 --screen-only          # 只做 IC 初筛
    python -m strategies.cross_section.run_alpha101 --n-symbols 50         # 快速测试
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.prices import load_index_symbols, load_full_market_symbols, load_price_panel
from strategies.cross_section.alpha101 import Alphas

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
# IC Screening (Phase 1)
# ═══════════════════════════════════════════════════════════════

def compute_forward_returns(close: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Forward return: close[t+periods] / close[t] - 1."""
    return close.shift(-periods) / close - 1


def compute_rank_ic(factor: pd.DataFrame, fwd_ret: pd.DataFrame,
                    min_obs: int = 30) -> pd.Series:
    """Vectorized per-date Spearman rank IC (rank-then-Pearson)."""
    common_cols = factor.columns.intersection(fwd_ret.columns)
    common_dates = factor.index.intersection(fwd_ret.index)
    if len(common_cols) == 0 or len(common_dates) == 0:
        return pd.Series(dtype=float)

    f = factor.loc[common_dates, common_cols]
    r = fwd_ret.loc[common_dates, common_cols]

    f_rank = f.rank(axis=1)
    r_rank = r.rank(axis=1)

    valid = f_rank.notna() & r_rank.notna()
    n = valid.sum(axis=1)

    f_dm = f_rank.sub(f_rank.mean(axis=1), axis=0)
    r_dm = r_rank.sub(r_rank.mean(axis=1), axis=0)

    num = (f_dm * r_dm).sum(axis=1)
    denom = (f_dm ** 2).sum(axis=1).pow(0.5) * (r_dm ** 2).sum(axis=1).pow(0.5)

    ic = num / denom.replace(0, np.nan)
    ic[n < min_obs] = np.nan
    return ic.dropna()


def ic_summary(ic_series: pd.Series) -> dict:
    if len(ic_series) < 3:
        return {'mean_ic': np.nan, 'ic_std': np.nan, 'icir': np.nan,
                't_stat': np.nan, 'n_periods': 0}
    mean_ic = ic_series.mean()
    ic_std = ic_series.std()
    n = len(ic_series)
    icir = mean_ic / ic_std if ic_std > 0 else 0
    t = mean_ic / (ic_std / np.sqrt(n)) if ic_std > 0 else 0
    return {'mean_ic': mean_ic, 'ic_std': ic_std, 'icir': icir,
            't_stat': t, 'n_periods': n}


def run_ic_screening(all_alphas, close, horizons):
    """Phase 1: IC screening for all 101 alphas."""
    fwd_rets = {h: compute_forward_returns(close, h) for h in horizons}
    rows = []
    for name in sorted(all_alphas.keys()):
        factor = all_alphas[name]
        row = {'alpha': name}
        for h in horizons:
            ic_s = compute_rank_ic(factor, fwd_rets[h])
            stats = ic_summary(ic_s)
            row[f'IC_{h}d'] = stats['mean_ic']
            row[f'ICIR_{h}d'] = stats['icir']
            row[f't_{h}d'] = stats['t_stat']
        rows.append(row)
    return pd.DataFrame(rows).set_index('alpha')


def print_ic_table(results, horizons):
    """Print IC summary table."""
    results = results.sort_values(f'IC_{horizons[0]}d', key=abs, ascending=False)

    print(f'\n{"Alpha":<12s}', end='')
    for h in horizons:
        print(f'  {"IC_"+str(h)+"d":>8s}  {"ICIR":>6s}  {"t":>6s}', end='')
    print()
    print('-' * (12 + len(horizons) * 24))

    for name, row in results.iterrows():
        line = f'{name:<12s}'
        for h in horizons:
            ic = row.get(f'IC_{h}d', np.nan)
            icir = row.get(f'ICIR_{h}d', np.nan)
            t = row.get(f't_{h}d', np.nan)
            if pd.isna(ic):
                line += f'  {"N/A":>8s}  {"N/A":>6s}  {"N/A":>6s}'
            else:
                sig = '*' if abs(t) >= 2.0 else ' '
                line += f'  {ic:>+8.4f}  {icir:>+6.2f}  {t:>+5.1f}{sig}'
        print(line)

    sig_mask = results[f't_{horizons[0]}d'].abs() >= 2.0
    sig = results[sig_mask].sort_values(f'IC_{horizons[0]}d', ascending=False)
    n_pos = (sig[f'IC_{horizons[0]}d'] > 0).sum()
    n_neg = (sig[f'IC_{horizons[0]}d'] < 0).sum()
    print(f'\n  {len(sig)} significant (|t| >= 2 at {horizons[0]}d): '
          f'{n_pos} positive IC, {n_neg} negative IC')

    if not sig.empty:
        print(f'\n  Top 10 (positive IC):')
        for name, row in sig.head(10).iterrows():
            print(f'    {name:<12s}  IC={row[f"IC_{horizons[0]}d"]:+.4f}  '
                  f't={row[f"t_{horizons[0]}d"]:+.1f}')

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='101 Formulaic Alphas - Standard Research Pipeline')
    parser.add_argument('--universe', type=str, default='index',
                        choices=['index', 'full'],
                        help='Stock universe: index (S&P500+NDX100) or full (market-wide)')
    parser.add_argument('--min-adv', type=float, default=1e6,
                        help='Min avg daily dollar volume for full universe (default: 1e6)')
    parser.add_argument('--n-symbols', type=int, default=None,
                        help='Limit number of symbols (for quick test)')
    parser.add_argument('--start', type=str, default='2020-01-01',
                        help='Data start date')
    parser.add_argument('--screen-only', action='store_true',
                        help='Only run IC screening (Phase 1)')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Number of top alphas for deep analysis (default: 20)')
    parser.add_argument('--rebalance-freq', type=str, default='M',
                        help='Rebalance frequency for quantile backtest (M, W-FRI, Q)')
    parser.add_argument('--no-fama-macbeth', action='store_true',
                        help='Skip Fama-MacBeth regression')
    parser.add_argument('--no-sub-sample', action='store_true',
                        help='Skip sub-sample analysis')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plot generation')
    args = parser.parse_args()

    screen_horizons = [1, 5, 20]
    deep_horizons = [1, 5, 21, 63]

    out_dir = Path(__file__).resolve().parent.parent.parent / 'output'
    out_dir.mkdir(exist_ok=True)
    alpha101_dir = out_dir / 'alpha101'
    alpha101_dir.mkdir(exist_ok=True)

    suffix = '_full' if args.universe == 'full' else ''

    print()
    print('=' * 80)
    print('  101 Formulaic Alphas — Standard Research Pipeline')
    print('  Kakushadze (2016)')
    print('=' * 80)

    # ═══════════════════════════════════════════════════════════
    # Step 1: Load data
    # ═══════════════════════════════════════════════════════════
    print('\n[Step 1] Loading data ...')
    t0 = time.time()

    if args.universe == 'full':
        print(f'  Universe: full market (ADV >= ${args.min_adv:,.0f})')
        symbols = load_full_market_symbols(min_adv=args.min_adv)
        print(f'  Filtered: {len(symbols)} symbols')
    else:
        symbols = load_index_symbols()
    if args.n_symbols:
        symbols = symbols[:args.n_symbols]

    panels = load_price_panel(symbols, start=args.start)
    close = panels['close']
    open_ = panels['open']
    high = panels['high']
    low = panels['low']
    volume = panels['volume']
    print(f'  {close.shape[1]} symbols x {close.shape[0]} days  ({time.time()-t0:.1f}s)')

    # GICS
    gics_df = None
    sector = None
    try:
        from strategies.cross_section.alpha_test import load_gics_sectors
        gics_df = load_gics_sectors(symbols)
        if 'symbol' in gics_df.columns and 'gics_sector' in gics_df.columns:
            sector = gics_df.set_index('symbol')['gics_sector']
            sector = sector[sector.index.isin(close.columns)]
            print(f'  GICS sectors: {len(sector)} symbols, {sector.nunique()} sectors')
    except Exception:
        print('  GICS sectors: not available')

    # Market cap / shares
    shares = None
    cap = None
    try:
        from data.fundamentals import build_shares_panel
        shares = build_shares_panel(symbols, close.index)
        cap = close * shares.reindex(index=close.index, columns=close.columns)
        print(f'  Market cap: available')
    except Exception:
        print(f'  Market cap: not available')

    # SPY for regime analysis
    spy_close = None
    if 'SPY' in close.columns:
        spy_close = close['SPY']
    else:
        try:
            spy_panel = load_price_panel(['SPY'], start=args.start)
            spy_close = spy_panel['close']['SPY']
        except Exception:
            pass

    print(f'  Total load time: {time.time()-t0:.1f}s')

    # ═══════════════════════════════════════════════════════════
    # Step 2: Compute 101 alphas
    # ═══════════════════════════════════════════════════════════
    print(f'\n[Step 2] Computing 101 alphas ...')
    t0 = time.time()
    alphas_obj = Alphas(close, open_, high, low, volume,
                        cap=cap, sector=sector)
    all_alphas = alphas_obj.compute_all(verbose=True)
    print(f'\n  Computed: {len(all_alphas)}/101 alphas  ({time.time()-t0:.1f}s)')

    if not all_alphas:
        print('ERROR: No alphas computed.')
        return

    # ═══════════════════════════════════════════════════════════
    # Phase 1: IC Screening (all 101)
    # ═══════════════════════════════════════════════════════════
    print(f'\n{"="*80}')
    print(f'  Phase 1: IC Screening (all {len(all_alphas)} alphas)')
    print(f'{"="*80}')
    t0 = time.time()
    ic_results = run_ic_screening(all_alphas, close, screen_horizons)
    print(f'  IC screening done ({time.time()-t0:.1f}s)')

    print(f'\n  IC Summary (horizons: {screen_horizons})')
    ic_results = print_ic_table(ic_results, screen_horizons)

    # Save Phase 1
    csv_path = alpha101_dir / f'phase1_ic_screening{suffix}.csv'
    ic_results.to_csv(csv_path)
    print(f'\n  Saved {csv_path}')

    if args.screen_only:
        print(f'\n  --screen-only: stopping after Phase 1.')
        return

    # Select top-N by |IC| at 1d for deep analysis
    ic_results_sorted = ic_results.sort_values('IC_1d', key=abs, ascending=False)
    valid_mask = ic_results_sorted['t_1d'].abs() >= 2.0
    sig_alphas = ic_results_sorted[valid_mask]
    top_names = sig_alphas.head(args.top_n).index.tolist()
    top_factors = {n: all_alphas[n] for n in top_names if n in all_alphas}

    print(f'\n  Selected top {len(top_factors)} alphas for deep analysis')

    if not top_factors:
        print('  No significant alphas found. Stopping.')
        return

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Diagnostics (top N)
    # ═══════════════════════════════════════════════════════════
    print(f'\n{"="*80}')
    print(f'  Phase 2: Factor Diagnostics (top {len(top_factors)})')
    print(f'{"="*80}')
    t0 = time.time()

    from strategies.cross_section.diagnostics import (
        run_batch_diagnostics,
    )

    roe = None  # alpha101 factors are not fundamentals

    diag_reports = run_batch_diagnostics(
        top_factors,
        close=close, volume=volume,
        shares=shares, roe=roe,
        gics_df=gics_df,
    )

    # Print summary
    print(f'\n  {"Factor":<16s} {"Coverage":>8s} {"Drift":>6s} {"MaxCorr":>8s} '
          f'{"DomStyle":<12s} {"Turnover":>8s} {"HHI_L":>6s} {"Pass":>5s}')
    print('  ' + '-' * 80)
    for name, r in sorted(diag_reports.items()):
        cov = f'{r.coverage.median_coverage:.0%}'
        drift = 'YES' if r.distribution.drift_detected else 'no'
        mc = f'{r.known_factors.max_abs_corr:.3f}' if r.known_factors else 'N/A'
        ds = r.known_factors.dominant_style if r.known_factors else 'N/A'
        turn = f'{r.turnover.mean_turnover:.0%}'
        hhi = f'{r.industry.hhi_top:.3f}' if r.industry else 'N/A'
        p = 'PASS' if r.pass_sanity else 'WARN'
        print(f'  {name:<16s} {cov:>8s} {drift:>6s} {mc:>8s} '
              f'{ds:<12s} {turn:>8s} {hhi:>6s} {p:>5s}')

    n_pass = sum(1 for r in diag_reports.values() if r.pass_sanity)
    n_warn = len(diag_reports) - n_pass
    print(f'\n  Results: {n_pass} PASS, {n_warn} WARN  ({time.time()-t0:.1f}s)')

    # Save Phase 2
    diag_rows = []
    for name, r in diag_reports.items():
        diag_rows.append({
            'alpha': name,
            'coverage': r.coverage.median_coverage,
            'drift': r.distribution.drift_detected,
            'max_style_corr': r.known_factors.max_abs_corr if r.known_factors else np.nan,
            'dominant_style': r.known_factors.dominant_style if r.known_factors else '',
            'turnover': r.turnover.mean_turnover,
            'hhi_top': r.industry.hhi_top if r.industry else np.nan,
            'pass_sanity': r.pass_sanity,
            'warnings': '; '.join(r.warnings),
        })
    pd.DataFrame(diag_rows).to_csv(
        alpha101_dir / f'phase2_diagnostics{suffix}.csv', index=False)

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Alpha Test (top N)
    # ═══════════════════════════════════════════════════════════
    print(f'\n{"="*80}')
    print(f'  Phase 3: Alpha Test (top {len(top_factors)})')
    print(f'  Horizons: {deep_horizons}, Rebalance: {args.rebalance_freq}')
    if not args.no_fama_macbeth:
        print(f'  Fama-MacBeth: YES (controls: Momentum, Size, Sector)')
    if not args.no_sub_sample:
        print(f'  Sub-sample: YES (Bull/Bear, HighVol/LowVol, Walk-forward)')
    print(f'{"="*80}')
    t0 = time.time()

    from strategies.cross_section.alpha_test import (
        run_batch_alpha_test,
    )

    alpha_reports = run_batch_alpha_test(
        top_factors,
        close=close,
        volume=volume,
        shares=shares,
        gics_df=gics_df,
        spy_close=spy_close,
        rebalance_freq=args.rebalance_freq,
        horizons=deep_horizons,
        transaction_cost_bps=10.0,
        do_fama_macbeth=not args.no_fama_macbeth,
        do_sub_sample=not args.no_sub_sample,
    )

    # Print summary table
    h_cols = '  '.join([f'IC_{h}d' for h in deep_horizons])
    print(f'\n  {"Alpha":<16s} {h_cols}  {"ICIR_21d":>8s} {"NW_t_21d":>8s} '
          f'{"LS_Shrp":>7s} {"Net_Shrp":>8s}', end='')
    if not args.no_fama_macbeth:
        print(f' {"FM_t":>6s}', end='')
    print()
    print('  ' + '-' * 100)

    for name, r in sorted(alpha_reports.items(),
                          key=lambda x: abs(x[1].multi_horizon_ic.mean_ic.get(21, 0)),
                          reverse=True):
        mic = r.multi_horizon_ic
        eq = r.enhanced_quantile
        ic_strs = '  '.join([f'{mic.mean_ic.get(h, np.nan):>+7.4f}' for h in deep_horizons])
        icir_21 = f'{mic.icir.get(21, np.nan):>8.2f}'
        nw_21 = f'{mic.tstat_nw.get(21, np.nan):>8.2f}'
        ls_s = f'{eq.base_metrics["ls_sharpe"]:>7.2f}'
        net_s = f'{eq.ls_sharpe_net:>8.2f}'
        print(f'  {name:<16s} {ic_strs}  {icir_21}  {nw_21} {ls_s} {net_s}', end='')
        if not args.no_fama_macbeth and r.fama_macbeth:
            print(f' {r.fama_macbeth.tstat:>6.2f}', end='')
        print()

    print(f'\n  Alpha Test done in {time.time()-t0:.1f}s')

    # Save Phase 3
    alpha_rows = []
    for name, r in alpha_reports.items():
        row = {'alpha': name}
        for h in deep_horizons:
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
        # Yearly IC stability
        if r.sub_sample and r.sub_sample.yearly_stats is not None:
            ys = r.sub_sample.yearly_stats
            if not ys.empty:
                for yr in ys.index:
                    row[f'ic_yr_{yr}'] = ys.loc[yr, 'mean_ic']
                n_pos_years = (ys['mean_ic'] > 0).sum()
                row['ic_positive_years'] = f'{n_pos_years}/{len(ys)}'
        alpha_rows.append(row)
    pd.DataFrame(alpha_rows).to_csv(
        alpha101_dir / f'phase3_alpha_test{suffix}.csv', index=False)

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Plots
    # ═══════════════════════════════════════════════════════════
    if not args.no_plot and (diag_reports or alpha_reports):
        print(f'\n{"="*80}')
        print(f'  Phase 4: Generating plots ...')
        print(f'{"="*80}')
        import matplotlib
        matplotlib.use('Agg')

        from strategies.cross_section.diagnostics_plotting import (
            plot_diagnostic_summary,
            plot_alpha_test_summary,
            plot_ic_decay_curve,
            plot_fama_macbeth,
            plot_sub_sample_decomposition,
            plot_industry_distribution,
        )

        for name in top_factors:
            d_report = diag_reports.get(name)
            a_report = alpha_reports.get(name)

            factor_dir = alpha101_dir / name
            factor_dir.mkdir(parents=True, exist_ok=True)

            try:
                if d_report:
                    plot_diagnostic_summary(
                        d_report, str(factor_dir / 'diagnostic_summary.png'))
                    if d_report.industry is not None:
                        plot_industry_distribution(
                            d_report, str(factor_dir / 'industry_distribution.png'))
                if a_report:
                    plot_alpha_test_summary(
                        a_report, str(factor_dir / 'alpha_test_summary.png'))
                    plot_ic_decay_curve(
                        a_report, str(factor_dir / 'ic_decay.png'))
                    if a_report.fama_macbeth and len(a_report.fama_macbeth.slope_series) > 0:
                        plot_fama_macbeth(
                            a_report, str(factor_dir / 'fama_macbeth.png'))
                    if a_report.sub_sample:
                        has_yearly = (a_report.sub_sample.yearly_stats is not None
                                      and not a_report.sub_sample.yearly_stats.empty)
                        has_regime = not a_report.sub_sample.regime_stats.empty
                        if has_yearly or has_regime:
                            plot_sub_sample_decomposition(
                                a_report, str(factor_dir / 'sub_sample.png'))
            except Exception as e:
                print(f'  {name}: plot error ({e})')

        print(f'  Plots saved to {alpha101_dir}/')

    # ═══════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════
    print(f'\n{"="*80}')
    print(f'  FINAL SUMMARY')
    print(f'{"="*80}')
    print(f'  Universe:          {close.shape[1]} symbols x {close.shape[0]} days')
    print(f'  Alphas computed:   {len(all_alphas)}/101')
    sig_1d = (ic_results['t_1d'].abs() >= 2.0).sum()
    print(f'  Significant (1d):  {sig_1d}/{len(all_alphas)}')
    avg_ic = ic_results['IC_1d'].abs().mean()
    print(f'  Avg |IC| (1d):     {avg_ic:.4f}')
    print(f'  Deep analyzed:     {len(top_factors)} alphas')

    if alpha_reports:
        # Find best alpha by net Sharpe
        best_name = max(alpha_reports,
                        key=lambda n: alpha_reports[n].enhanced_quantile.ls_sharpe_net)
        best = alpha_reports[best_name]
        print(f'\n  Best alpha (by net L/S Sharpe):')
        print(f'    {best_name}')
        print(f'    IC_1d:  {best.multi_horizon_ic.mean_ic.get(1, np.nan):+.4f}')
        print(f'    IC_21d: {best.multi_horizon_ic.mean_ic.get(21, np.nan):+.4f}')
        print(f'    L/S Sharpe (net): {best.enhanced_quantile.ls_sharpe_net:.2f}')
        if best.fama_macbeth:
            print(f'    FM t-stat: {best.fama_macbeth.tstat:.2f}')

    print(f'\n  Output: {alpha101_dir}/')
    print('=' * 80)
    print()


if __name__ == '__main__':
    main()
