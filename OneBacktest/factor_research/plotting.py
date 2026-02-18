"""
因子回测可视化

四面板报告:
1. 五分位累计收益
2. Long-Short 累计收益 + 关键指标
3. IC 时序 + 滚动均值
4. 分位年化收益柱状图
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict


def plot_factor_report(
    backtest_result: Dict,
    metrics: Dict,
    factor_name: str = 'Factor',
    save_path: str = None,
):
    """
    四面板因子分析报告。

    Args:
        backtest_result: from run_factor_backtest / build_monthly_rebalance
        metrics: from compute_factor_metrics
        factor_name: 因子名称
        save_path: 保存路径 (None = 显示)
    """
    qret = backtest_result['quantile_returns']
    ls = backtest_result['long_short']
    ic = backtest_result['ic_series']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{factor_name} Factor Analysis', fontsize=14, fontweight='bold')

    n_q = qret.shape[1]
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_q))

    # ── Panel 1: 五分位累计收益 ─────────────────────
    ax = axes[0, 0]
    for q in range(1, n_q + 1):
        cum = (1 + qret[q].fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, label=f'Q{q}', color=colors[q - 1], linewidth=1.5)
    ax.set_title('Quintile Cumulative Returns')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylabel('Cumulative Return')
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Long-Short 累计收益 + 指标汇总 ────
    ax = axes[0, 1]
    ls_cum = (1 + ls.fillna(0)).cumprod()
    ax.plot(ls_cum.index, ls_cum.values, color='navy', linewidth=1.5)
    ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'Long-Short (Q{n_q}-Q1)')
    ax.set_ylabel('Cumulative Return')
    ax.grid(True, alpha=0.3)

    # 在图上显示关键指标
    textstr = (
        f'Return: {metrics["ls_annual_return"]:.1%}\n'
        f'Vol: {metrics["ls_annual_vol"]:.1%}\n'
        f'Sharpe: {metrics["ls_sharpe"]:.2f}\n'
        f'WinRate: {metrics["ls_win_rate"]:.1%}\n'
        f'MaxDD: {metrics["ls_max_drawdown"]:.2%}\n'
        f'RankIC: {metrics["rank_ic"]:.4f}\n'
        f'RankICIR: {metrics["rank_icir"]:.2f}\n'
        f'Mono: {metrics["monotonicity"]:.2f}'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace', bbox=props)

    # ── Panel 3: IC 时序 ───────────────────────────
    ax = axes[1, 0]
    ic_clean = ic.dropna()
    bar_width = max(1, (ic_clean.index[-1] - ic_clean.index[0]).days // len(ic_clean) * 0.8) if len(ic_clean) > 1 else 20
    ax.bar(ic_clean.index, ic_clean.values, width=bar_width, alpha=0.4,
           color='steelblue', label='Rank IC')
    if len(ic_clean) >= 12:
        ic_rolling = ic_clean.rolling(12).mean()
        ax.plot(ic_rolling.index, ic_rolling.values, color='red',
                linewidth=2, label='12-period Rolling')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.set_title(f'Rank IC={metrics["rank_ic"]:.4f}, '
                 f'Rank ICIR={metrics["rank_icir"]:.2f}')
    ax.legend(fontsize=8)
    ax.set_ylabel('Rank IC')
    ax.grid(True, alpha=0.3)

    # ── Panel 4: 分位年化收益柱状图 ─────────────────
    ax = axes[1, 1]
    q_labels = [f'Q{q}' for q in range(1, n_q + 1)]
    q_ann_ret = [metrics['quantile_metrics'][q]['annual_return'] for q in range(1, n_q + 1)]
    bars = ax.bar(q_labels, q_ann_ret, color=colors, edgecolor='gray', alpha=0.8)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.set_title(f'Quintile Annual Returns (Monotonicity={metrics["monotonicity"]:.2f})')
    ax.set_ylabel('Annual Return')
    for bar, val in zip(bars, q_ann_ret):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {save_path}')
    else:
        plt.show()
