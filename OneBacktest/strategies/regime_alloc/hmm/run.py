"""
HMM Regime-Based Asset Allocation — 复现 Baitinger & Hoch (2024)

入口脚本: 对 n_states=2..7 逐一回测, 输出 Table 4 风格对比表 + 净值曲线.

Usage:
    cd OneBacktest
    python -m strategies.regime_alloc.hmm.run
    python -m strategies.regime_alloc.hmm.run --states 2 3 4
    python -m strategies.regime_alloc.hmm.run --oos-start 2021-01-01
    python -m strategies.regime_alloc.hmm.run --gamma 4
"""
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from data.storage.parquet import ParquetStorage

# ── 路径 ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
BARS_1D_DIR = _ROOT / 'data' / 'processed' / 'bars_1d'
TBILL_FILE = _ROOT / 'data' / 'processed' / 'rates' / 'tbill_3m.parquet'


def load_data():
    """加载 SPY 日线 + T-Bill 利率"""
    storage = ParquetStorage(str(BARS_1D_DIR))
    spy = storage.load(['SPY'], frequency='1d')
    spy_close = spy['close'].droplevel('symbol') if 'symbol' in spy.index.names else spy['close']
    # 确保 index 是普通 DatetimeIndex
    if hasattr(spy_close.index, 'get_level_values'):
        spy_close.index = spy_close.index.get_level_values(0)
    spy_close = spy_close.sort_index()
    spy_close.index = spy_close.index.tz_localize(None)

    tbill = pd.read_parquet(TBILL_FILE)
    tbill_rate = tbill['rate'].sort_index()

    return spy_close, tbill_rate


def print_table(all_metrics: dict, gamma: float, oos_start: str):
    """打印 Table 4 风格对比表"""
    print()
    print('=' * 80)
    print(f'HMM Regime-Based Allocation — OOS Results (daily data)')
    print(f'OOS: {oos_start} ~ present | γ = {gamma}')
    print('=' * 80)

    # 收集所有策略
    strategies = []
    for n_states, metrics in all_metrics.items():
        for label, m in metrics.items():
            name = f'{label}' if label.startswith('Bench') else f'HMM J={n_states}'
            strategies.append((name, m))

    # 去重 benchmark (只显示一次)
    seen = set()
    unique = []
    for name, m in strategies:
        if name in seen:
            continue
        seen.add(name)
        unique.append((name, m))

    # 表头
    cols = ['Ann Ret%', 'Ann Vol%', 'Sharpe', 'SR ATC', 'MaxDD%', 'Exp%', 'TV/M%', 'EndW']
    header = f'{"Strategy":<16}' + ''.join(f'{c:>10}' for c in cols)
    print(header)
    print('-' * len(header))

    for name, m in unique:
        row = (
            f'{name:<16}'
            f'{m["Ann Return (%)"]:>10.2f}'
            f'{m["Ann Vol (%)"]:>10.2f}'
            f'{m["Sharpe"]:>10.2f}'
            f'{m["Sharpe ATC"]:>10.2f}'
            f'{m["Max DD (%)"]:>10.2f}'
            f'{m["Avg Exposure (%)"]:>10.1f}'
            f'{m["Avg TV/Month (%)"]:>10.2f}'
            f'{m["End Wealth"]:>10.2f}'
        )
        print(row)

    print()


def plot_results(all_results: dict, gamma: float, oos_start: str):
    """画净值曲线和权重时序图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  (matplotlib not available, skipping plots)')
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # 上图: 净值曲线
    ax1 = axes[0]
    # Benchmark (只画一次)
    first_key = list(all_results.keys())[0]
    first_res = all_results[first_key]
    bh_cum = (1 + first_res['ret_bench1']).cumprod() * 100
    dyn_cum = (1 + first_res['ret_bench2']).cumprod() * 100
    ax1.plot(bh_cum.index, bh_cum.values, 'k-', lw=2, label='Buy & Hold')
    ax1.plot(dyn_cum.index, dyn_cum.values, 'k--', lw=1.5, label='Dynamic (252d)')

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    for i, (n_states, res) in enumerate(all_results.items()):
        hmm_cum = (1 + res['ret_hmm']).cumprod() * 100
        ax1.plot(hmm_cum.index, hmm_cum.values,
                 color=colors[i % len(colors)], lw=1.2,
                 label=f'HMM J={n_states}')

    ax1.set_ylabel('Cumulative Wealth (start=100)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_title(f'HMM Regime Allocation — OOS from {oos_start}, γ={gamma}')
    ax1.grid(alpha=0.3)

    # 下图: HMM 权重
    ax2 = axes[1]
    for i, (n_states, res) in enumerate(all_results.items()):
        ax2.plot(res.index, res['w_hmm'].values * 100,
                 color=colors[i % len(colors)], lw=0.9, alpha=0.8,
                 label=f'HMM J={n_states}')

    ax2.axhline(100, color='k', ls=':', lw=0.8)
    ax2.set_ylabel('Risky Asset Weight (%)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_dir = _ROOT / 'output'
    out_dir.mkdir(exist_ok=True)
    path = out_dir / 'hmm_regime_alloc.png'
    plt.savefig(path, dpi=150)
    print(f'  Plot saved: {path.relative_to(_ROOT)}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='HMM Regime-Based Allocation Backtest')
    parser.add_argument('--states', nargs='+', type=int, default=[2, 3, 4, 5, 6, 7],
                        help='Number of HMM states to test (default: 2 3 4 5 6 7)')
    parser.add_argument('--oos-start', type=str, default='2022-01-01',
                        help='OOS start date (default: 2022-01-01)')
    parser.add_argument('--gamma', type=float, default=6.0,
                        help='Risk aversion (default: 6.0)')
    parser.add_argument('--n-init', type=int, default=10,
                        help='HMM random initializations (default: 10)')
    parser.add_argument('--tc', type=float, default=10.0,
                        help='Transaction cost in bps (default: 10)')
    args = parser.parse_args()

    from .backtest import run_backtest
    from .model import save_model

    # ── 加载数据 ──
    print('Loading data...')
    spy_close, tbill_rate = load_data()
    print(f'  SPY: {spy_close.index[0].date()} ~ {spy_close.index[-1].date()} ({len(spy_close)} bars)')
    print(f'  T-Bill: {tbill_rate.index[0].date()} ~ {tbill_rate.index[-1].date()}')

    # ── 逐状态数回测 ──
    all_metrics = {}
    all_results = {}
    all_models = {}

    for n_states in args.states:
        print(f'\n  Fitting HMM J={n_states}...', end=' ', flush=True)
        result = run_backtest(
            spy_prices=spy_close,
            tbill_rates=tbill_rate,
            n_states=n_states,
            oos_start=args.oos_start,
            gamma=args.gamma,
            n_init=args.n_init,
            tc_bps=args.tc,
        )
        sr = result['metrics']['HMM']['Sharpe']
        print(f'Sharpe={sr:.2f}')

        all_metrics[n_states] = result['metrics']
        all_results[n_states] = result['results']
        if result.get('model') is not None:
            all_models[n_states] = (result['model'], result['fit_end'])

    # ── 输出 ──
    print_table(all_metrics, gamma=args.gamma, oos_start=args.oos_start)
    plot_results(all_results, gamma=args.gamma, oos_start=args.oos_start)

    # ── 保存模型 ──
    if all_models:
        model_dir = _ROOT / 'output' / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        for n_states, (model, fit_end) in all_models.items():
            metadata = {
                'n_states': n_states,
                'gamma': args.gamma,
                'oos_start': args.oos_start,
                'fit_end': str(fit_end.date()) if hasattr(fit_end, 'date') else str(fit_end),
                'n_init': args.n_init,
            }
            try:
                train_ret = spy_close.pct_change().dropna()
                train_ret = train_ret[train_ret.index <= fit_end].values
                metadata['score'] = float(model.score(train_ret.reshape(-1, 1)))
            except Exception:
                pass

            path = model_dir / f'hmm_{n_states}states.joblib'
            save_model(model, path, metadata=metadata)

        print(f'  Models saved: {model_dir.relative_to(_ROOT)}/ ({len(all_models)} files)')


if __name__ == '__main__':
    main()
