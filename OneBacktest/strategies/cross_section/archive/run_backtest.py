"""
统一回测 CLI

替代 run_factor_backtest.py / weekly_pipeline.py / run_pipeline.py /
daily_ml_predict.py / compare_normalization.py.

支持配置:
    - 调仓频率: D / W-FRI / M / Q
    - 因子选择: selected (17个) / auto (IC筛选) / all / 手动指定
    - 打分方式: equal_weight / ridge / rf / xgb / lgb

Usage:
    cd OneBacktest

    # 等权月度回测 (默认)
    python -m strategies.cross_section.run_backtest

    # ML 周度回测
    python -m strategies.cross_section.run_backtest --freq W-FRI --scorer ridge

    # 自动因子筛选
    python -m strategies.cross_section.run_backtest --factors auto

    # 手动指定因子
    python -m strategies.cross_section.run_backtest --factors RS_12M,ROE,ROIC

    # 输出最新选股
    python -m strategies.cross_section.run_backtest --live --top-n 20

    # 日频 ML
    python -m strategies.cross_section.run_backtest --freq D --scorer xgb
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

from data.prices import load_index_symbols, load_price_panel
from strategies.cross_section.factors import (
    compute_technical_factors,
    compute_reversal_factors,
    compute_fundamental_factors,
    compute_microstructure_factors,
    compute_battle_factors,
    compute_surge_factors,
    compute_regression_factors,
    compute_fuzzy_factors,
    compute_rebuild_factors,
    compute_tidal_factors,
    compute_jump_factors,
    RTH_START,
    RTH_END,
)
from strategies.cross_section.pick_stocks import (
    SELECTED_FACTORS,
    compute_selected_factors,
)
from strategies.cross_section.screening import (
    screen_factors,
    compute_ic_summary,
)
from strategies.cross_section.backtest import (
    build_periodic_rebalance,
    compute_forward_returns,
)
from strategies.cross_section.analytics import (
    compute_factor_metrics,
    format_metrics,
)
from strategies.cross_section.scorer import (
    score_equal_weight,
    score_ml_walk_forward,
    score_latest_ml,
    available_models,
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', 'Mean of empty slice')

BARS_1MIN_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / 'data' / 'processed' / 'bars_1min'
)

# 频率 → 默认持仓天数
FREQ_HOLDING = {
    'D': 1,
    'W-FRI': 5,
    'W': 5,
    'M': 21,
    'ME': 21,
    'Q': 63,
    'QE': 63,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='统一因子回测 CLI')

    parser.add_argument(
        '--freq', type=str, default='M',
        help='调仓频率: D, W-FRI, M, Q (default: M)')
    parser.add_argument(
        '--factors', type=str, default='selected',
        help='因子选择: selected (17个) | auto (IC筛选) | all | '
             'RS_12M,ROE,... (手动列表)')
    parser.add_argument(
        '--scorer', type=str, default='equal_weight',
        help='打分方式: equal_weight | ridge | rf | xgb | lgb '
             '(default: equal_weight)')

    parser.add_argument('--start', type=str, default='2020-01-01')
    parser.add_argument('--end', type=str, default='2026-12-31')
    parser.add_argument('--top-n', type=int, default=30,
                        help='Long-only top N 持仓数')
    parser.add_argument('--n-quantiles', type=int, default=5)
    parser.add_argument('--transaction-cost', type=float, default=10.0,
                        help='单边交易成本 (bps)')
    parser.add_argument('--no-1min', action='store_true',
                        help='跳过 1min 因子')
    parser.add_argument('--min-train', type=int, default=52,
                        help='ML walk-forward 最少训练期数')
    parser.add_argument('--n-pca', type=int, default=None,
                        help='PCA 降维维度 (default: 不降维)')

    parser.add_argument('--live', action='store_true',
                        help='输出最新 top-N 选股')
    parser.add_argument('--output-dir', type=str, default='output/backtest')
    parser.add_argument('--save-csv', action='store_true')
    parser.add_argument('--no-plot', action='store_true')

    return parser.parse_args()


def _compute_all_1d_factors(panels, symbols):
    """计算所有 1d 因子 (技术面 + 反转 + 基本面)."""
    close = panels['close']
    high = panels['high']
    low = panels['low']
    open_p = panels['open']

    all_factors = {}

    print('  Computing technical factors ...', end=' ')
    t0 = time.time()
    all_factors.update(compute_technical_factors(close, high, low))
    print(f'{time.time() - t0:.1f}s')

    print('  Computing reversal factors ...', end=' ')
    t0 = time.time()
    all_factors.update(compute_reversal_factors(close, open_p))
    print(f'{time.time() - t0:.1f}s')

    print('  Computing fundamental factors ...', end=' ')
    t0 = time.time()
    try:
        all_factors.update(compute_fundamental_factors(symbols, close))
        print(f'{time.time() - t0:.1f}s')
    except Exception as e:
        print(f'SKIP ({e})')

    return all_factors


def _compute_all_1min_factors(
    symbols, close, high, low, open_p, storage_1min, start_1min, end_1min,
):
    """Compute all available 1min factor groups from loaded intraday panels."""
    if storage_1min is None or not start_1min or not end_1min:
        return {}

    print('  Loading 1min data for all intraday factors ...', end=' ')
    t0 = time.time()
    raw_1m = storage_1min.load(
        symbols,
        pd.Timestamp(start_1min, tz='US/Eastern'),
        pd.Timestamp(end_1min, tz='US/Eastern'),
        '1m',
    )
    t = raw_1m.index.time
    raw_1m = raw_1m[(t >= RTH_START) & (t <= RTH_END)]
    close_1m = raw_1m.pivot_table(index=raw_1m.index, columns='symbol', values='close')
    vol_1m = raw_1m.pivot_table(index=raw_1m.index, columns='symbol', values='volume')
    print(f'{time.time() - t0:.1f}s  ({len(close_1m)} bars)')

    panels_1m = {}
    for f in ['open', 'high', 'low', 'close', 'volume']:
        panels_1m[f] = raw_1m.pivot_table(index=raw_1m.index, columns='symbol', values=f)

    intraday = {}
    factor_groups = [
        ('microstructure', lambda: compute_microstructure_factors(close_1m, vol_1m, close, open_p)),
        ('battle', lambda: compute_battle_factors(raw_1m)),
        ('surge', lambda: compute_surge_factors(close_1m, vol_1m)),
        ('regression', lambda: compute_regression_factors(close_1m, vol_1m)),
        ('fuzzy', lambda: compute_fuzzy_factors(close_1m, vol_1m)),
        ('rebuild', lambda: compute_rebuild_factors(panels_1m)),
        ('tidal', lambda: compute_tidal_factors(close_1m, vol_1m)),
        ('jump', lambda: compute_jump_factors(close_1m, close, high, low)),
    ]

    for label, fn in factor_groups:
        print(f'  Computing {label} intraday factors ...', end=' ')
        t0 = time.time()
        try:
            out = fn()
            intraday.update(out)
            print(f'+{len(out)} ({time.time() - t0:.1f}s)')
        except Exception as e:
            print(f'SKIP ({e})')

    return intraday


def _resolve_factors_and_directions(
    args, factors_computed, close,
):
    """
    根据 --factors 参数确定使用的因子和方向.

    Returns:
        (factors_subset, directions)
    """
    mode = args.factors.strip().lower()

    if mode == 'selected':
        # 使用预筛选的 17 因子
        selected = set(SELECTED_FACTORS.keys())
        subset = {k: v for k, v in factors_computed.items()
                  if k in selected}
        directions = {k: SELECTED_FACTORS[k]
                      for k in subset if k in SELECTED_FACTORS}
        missing = selected - set(subset.keys())
        if missing:
            print(f'  Missing factors: {sorted(missing)}')
        return subset, directions

    elif mode == 'auto':
        # IC 筛选 + 相关性去重
        print('  Running IC screening ...')
        holding = FREQ_HOLDING.get(args.freq, 21)
        fwd_ret = compute_forward_returns(close, holding)

        selected_names, ic_df = screen_factors(
            factors_computed, fwd_ret,
            min_abs_ic=0.005, max_corr=0.7)

        if not selected_names:
            print('  [WARN] No factors passed screening, using all.')
            selected_names = list(factors_computed.keys())
            ic_df = compute_ic_summary(factors_computed, fwd_ret)

        # 方向: IC 正 → +1, IC 负 → -1
        directions = {}
        for name in selected_names:
            if name in ic_df.index:
                directions[name] = +1 if ic_df.loc[name, 'mean_ic'] >= 0 else -1
            else:
                directions[name] = +1

        subset = {k: factors_computed[k] for k in selected_names}
        print(f'  Screening: {len(selected_names)} factors selected')
        return subset, directions

    elif mode == 'all':
        # 使用全部已计算因子
        holding = FREQ_HOLDING.get(args.freq, 21)
        fwd_ret = compute_forward_returns(close, holding)
        ic_df = compute_ic_summary(factors_computed, fwd_ret)

        directions = {}
        for name in factors_computed:
            if name in ic_df.index:
                directions[name] = +1 if ic_df.loc[name, 'mean_ic'] >= 0 else -1
            else:
                directions[name] = +1
        return factors_computed, directions

    else:
        # 手动指定: --factors RS_12M,ROE,...
        manual_names = [n.strip() for n in args.factors.split(',')]
        subset = {}
        for name in manual_names:
            if name in factors_computed:
                subset[name] = factors_computed[name]
            else:
                print(f'  [WARN] Factor "{name}" not found, skipping.')

        if not subset:
            print('  [ERROR] No valid factors. Available:',
                  sorted(factors_computed.keys()))
            sys.exit(1)

        # 方向: 用 SELECTED_FACTORS 如有, 否则计算 IC 推断
        holding = FREQ_HOLDING.get(args.freq, 21)
        fwd_ret = compute_forward_returns(close, holding)
        ic_df = compute_ic_summary(subset, fwd_ret)

        directions = {}
        for name in subset:
            if name in SELECTED_FACTORS:
                directions[name] = SELECTED_FACTORS[name]
            elif name in ic_df.index:
                directions[name] = +1 if ic_df.loc[name, 'mean_ic'] >= 0 else -1
            else:
                directions[name] = +1
        return subset, directions


def _print_live_picks(scores, top_n, factors, directions):
    """打印最新选股结果."""
    if isinstance(scores, pd.DataFrame):
        latest = scores.iloc[-1].dropna()
    else:
        latest = scores.dropna()

    top = latest.nlargest(top_n)
    today = pd.Timestamp.now().strftime('%Y-%m-%d')

    print()
    print('=' * 60)
    print(f'  Live Picks — {today}  (top {top_n})')
    print('=' * 60)
    print()
    print(f'  {"Rank":>4}  {"Symbol":<7}  {"Score":>8}')
    print('  ' + '-' * 25)
    for i, (sym, score) in enumerate(top.items(), 1):
        print(f'  {i:>4}  {sym:<7}  {score:>+8.4f}')
    print()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    # 验证 scorer
    if args.scorer != 'equal_weight':
        models = available_models()
        if args.scorer not in models:
            print(f'[ERROR] Model "{args.scorer}" not available.')
            print(f'  Installed: {models or ["none"]}')
            print('  Install: pip install scikit-learn xgboost lightgbm')
            sys.exit(1)

    # ═══════════════════════════════════════════════════════════
    # Step 1: 加载数据
    # ═══════════════════════════════════════════════════════════
    print('Step 1: Loading data ...')
    t0 = time.time()

    symbols = load_index_symbols()
    print(f'  Universe: {len(symbols)} symbols')

    panels = load_price_panel(symbols, args.start, args.end)
    close = panels['close']
    print(f'  1d bars: {close.shape[0]} days x {close.shape[1]} symbols  '
          f'({time.time() - t0:.1f}s)')

    # 1min storage (for --factors selected/auto/all with 1min factors)
    use_1min = not args.no_1min
    storage_1min = None
    start_1min = None
    end_1min = None

    if use_1min and BARS_1MIN_DIR.exists():
        from data.storage.parquet import ParquetStorage
        storage_1min = ParquetStorage(str(BARS_1MIN_DIR))
        start_1min = close.index[0].strftime('%Y-%m-%d')
        end_1min = close.index[-1].strftime('%Y-%m-%d')
        print(f'  1min range: {start_1min} ~ {end_1min}')
    elif use_1min:
        print(f'  1min data not found at {BARS_1MIN_DIR}, using 1d factors only')
        use_1min = False

    # ═══════════════════════════════════════════════════════════
    # Step 2: 计算因子
    # ═══════════════════════════════════════════════════════════
    print('\nStep 2: Computing factors ...')
    t0 = time.time()

    factor_mode = args.factors.strip().lower()

    if factor_mode == 'selected':
        # ????? 17 ??
        factors_computed = compute_selected_factors(
            symbols, close, panels['high'], panels['low'], panels['open'],
            storage_1min, start_1min, end_1min, use_1min)
    else:
        # ??? 1d ?? (auto/all/manual ???)
        factors_computed = _compute_all_1d_factors(panels, symbols)
        # all: add all intraday factor groups; auto/manual: keep selected intraday set.
        if use_1min and storage_1min is not None:
            try:
                if factor_mode == 'all':
                    all_1min = _compute_all_1min_factors(
                        symbols, close, panels['high'], panels['low'], panels['open'],
                        storage_1min, start_1min, end_1min,
                    )
                    for k, v in all_1min.items():
                        if k not in factors_computed:
                            factors_computed[k] = v
                else:
                    selected_1min = compute_selected_factors(
                        symbols, close, panels['high'], panels['low'],
                        panels['open'], storage_1min, start_1min, end_1min,
                        use_1min=True)
                    for k, v in selected_1min.items():
                        if k not in factors_computed:
                            factors_computed[k] = v
            except Exception as e:
                print(f'  [WARN] 1min factors failed: {e}')

    print(f'  {len(factors_computed)} factors computed  '
          f'({time.time() - t0:.1f}s)')

    if not factors_computed:
        print('[ERROR] No factors computed.')
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════
    # Step 3: 选择因子 & 确定方向
    # ═══════════════════════════════════════════════════════════
    print('\nStep 3: Selecting factors ...')
    factors_used, directions = _resolve_factors_and_directions(
        args, factors_computed, close)

    factor_names = sorted(factors_used.keys())
    print(f'  Using {len(factor_names)} factors: {factor_names[:10]}'
          f'{"..." if len(factor_names) > 10 else ""}')

    # ═══════════════════════════════════════════════════════════
    # Step 4: 打分
    # ═══════════════════════════════════════════════════════════
    print(f'\nStep 4: Scoring (method={args.scorer}, freq={args.freq}) ...')
    t0 = time.time()

    if args.scorer == 'equal_weight':
        composite = score_equal_weight(factors_used, directions)
    else:
        # ML walk-forward
        holding = FREQ_HOLDING.get(args.freq, 21)
        fwd_ret = compute_forward_returns(close, holding)

        # resample factors + fwd_ret 到目标频率
        # build_periodic_rebalance 内部也做这个, 但 ML 训练需要先做
        freq = args.freq
        rebalance_dates = close.resample(freq).last().dropna(how='all').index
        # 取交易日中最接近的日期
        trading_dates = close.index
        actual_dates = pd.DatetimeIndex([
            trading_dates[trading_dates <= d][-1]
            for d in rebalance_dates
            if len(trading_dates[trading_dates <= d]) > 0
        ])

        factors_resampled = {}
        for name, panel in factors_used.items():
            # 取 rebalance date 当天或之前最近的值
            factors_resampled[name] = panel.reindex(
                actual_dates, method='ffill')

        # Labels should not be forward-filled to avoid target contamination.
        fwd_ret_resampled = fwd_ret.reindex(actual_dates)

        composite = score_ml_walk_forward(
            factors_resampled, fwd_ret_resampled,
            method=args.scorer,
            min_train_periods=args.min_train,
            purge_periods=1,
            n_pca=args.n_pca,
        )

        if composite.empty:
            print('[ERROR] ML scoring produced no predictions.')
            sys.exit(1)

        # 回填到日频 (build_periodic_rebalance 需要日频输入)
        composite = composite.reindex(close.index, method='ffill')

    print(f'  Scoring done  ({time.time() - t0:.1f}s)')
    print(f'  Score panel: {composite.shape[0]} days x {composite.shape[1]} symbols')

    # ═══════════════════════════════════════════════════════════
    # Step 5: 回测
    # ═══════════════════════════════════════════════════════════
    print(f'\nStep 5: Backtesting (freq={args.freq}) ...')
    t0 = time.time()

    bt = build_periodic_rebalance(
        composite, close,
        rebalance_freq=args.freq,
        n_quantiles=args.n_quantiles,
    )

    metrics = compute_factor_metrics(bt)
    print(f'  Backtest done  ({time.time() - t0:.1f}s)')

    # ═══════════════════════════════════════════════════════════
    # Step 6: 输出
    # ═══════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(f'  Backtest Results')
    print(f'  Freq={args.freq}  Scorer={args.scorer}  '
          f'Factors={len(factor_names)}')
    print('=' * 60)
    print()
    print(format_metrics(metrics, factor_name='Composite'))

    # 交易成本调整
    cost_bps = args.transaction_cost
    if cost_bps > 0:
        turnover = metrics.get('turnover_mean', 0)
        ann = metrics['ann_factor']
        cost_drag = turnover * (cost_bps / 10000) * 2 * ann  # 2 legs
        net_sharpe = metrics['ls_sharpe']
        ls_vol = metrics['ls_annual_vol']
        if ls_vol > 0:
            net_ret = metrics['ls_annual_return'] - cost_drag
            net_sharpe = net_ret / ls_vol
        print(f'\nAfter costs ({cost_bps:.0f}bps):  '
              f'Net Return={net_ret:.2%}, Net Sharpe={net_sharpe:.2f}')

    # --live 选股
    if args.live:
        if args.scorer == 'equal_weight':
            _print_live_picks(composite, args.top_n, factors_used, directions)
        else:
            # 用全量数据训练, 预测最新截面
            print('\n  Training on full history for live prediction ...')
            holding = FREQ_HOLDING.get(args.freq, 21)
            fwd_ret = compute_forward_returns(close, holding)
            live_scores = score_latest_ml(
                factors_used, fwd_ret, method=args.scorer,
                n_pca=args.n_pca)
            if not live_scores.empty:
                _print_live_picks(live_scores, args.top_n,
                                  factors_used, directions)
            else:
                print('  [WARN] ML live scoring failed.')

    # --save-csv
    if args.save_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 回测指标
        metrics_flat = {k: v for k, v in metrics.items()
                        if not isinstance(v, dict)}
        pd.Series(metrics_flat).to_csv(
            output_dir / 'backtest_metrics.csv', header=['value'])
        print(f'\n  Saved {output_dir / "backtest_metrics.csv"}')

        # L/S 收益序列
        bt['long_short'].to_csv(output_dir / 'long_short_returns.csv')

        # 分位收益
        bt['quantile_returns'].to_csv(output_dir / 'quantile_returns.csv')

        # 因子 + 方向
        factor_info = pd.DataFrame([
            {'factor': name, 'direction': directions.get(name, 1)}
            for name in factor_names
        ])
        factor_info.to_csv(output_dir / 'factors_used.csv', index=False)

        if args.live:
            if args.scorer == 'equal_weight':
                latest = composite.iloc[-1].dropna().sort_values(ascending=False)
            else:
                latest = live_scores.sort_values(ascending=False)
            latest.head(args.top_n).to_csv(output_dir / 'live_picks.csv')
            print(f'  Saved {output_dir / "live_picks.csv"}')

    # --no-plot
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(
                f'Backtest: {args.scorer} | {args.freq} | '
                f'{len(factor_names)} factors',
                fontsize=14, fontweight='bold')

            # 分位累计收益
            ax = axes[0, 0]
            qret = bt['quantile_returns']
            n_q = qret.shape[1]
            colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_q))
            for q in range(1, n_q + 1):
                cum = (1 + qret[q].fillna(0)).cumprod()
                ax.plot(cum.index, cum.values, label=f'Q{q}',
                        color=colors[q - 1], linewidth=1.5)
            ax.set_title(f'Quintile Returns (Mono={metrics["monotonicity"]:.2f})')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)

            # L/S 累计
            ax = axes[0, 1]
            ls_cum = (1 + bt['long_short'].fillna(0)).cumprod()
            ax.plot(ls_cum.index, ls_cum.values, color='navy', linewidth=1.5)
            ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(f'Long-Short (Sharpe={metrics["ls_sharpe"]:.2f})')
            ax.grid(True, alpha=0.3)

            # IC 时序
            ax = axes[1, 0]
            ic = bt['ic_series']
            ax.bar(ic.index, ic.values, color='steelblue', alpha=0.5, width=5)
            if len(ic) >= 12:
                rolling_ic = ic.rolling(12).mean()
                ax.plot(rolling_ic.index, rolling_ic.values,
                        color='navy', linewidth=2)
            ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
            ax.set_title(f'Rank IC (mean={metrics["rank_ic"]:.4f}, '
                         f'ICIR={metrics["rank_icir"]:.2f})')
            ax.grid(True, alpha=0.3)

            # 分位年化收益 bar
            ax = axes[1, 1]
            q_rets = [metrics['quantile_metrics'][q]['annual_return']
                      for q in range(1, n_q + 1)]
            ax.bar(range(1, n_q + 1), q_rets, color=colors,
                   edgecolor='gray', alpha=0.8)
            for q, r in enumerate(q_rets, 1):
                ax.text(q, r, f'{r:+.1%}', ha='center',
                        va='bottom' if r >= 0 else 'top', fontsize=9)
            ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
            ax.set_title('Quantile Annual Returns')
            ax.set_xlabel('Quintile')
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()

            output_dir.mkdir(parents=True, exist_ok=True)
            fig_path = output_dir / 'backtest_summary.png'
            fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'\n  Plot saved to {fig_path}')
        except Exception as e:
            print(f'\n  [WARN] Plot generation failed: {e}')

    print('\nDone.')


if __name__ == '__main__':
    main()
