"""
因子绩效分析指标

标准指标集:
- 5分位多空收益率、波动率、Sharpe、胜率、最大回撤
- Rank IC, Rank ICIR
- 单调性
"""
import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats


def _infer_ann_factor(index: pd.DatetimeIndex) -> int:
    """从 DatetimeIndex 间距推断年化因子"""
    if len(index) < 2:
        return 12
    median_gap = pd.Series(index).diff().dropna().median().days
    if median_gap <= 2:
        return 252   # daily
    elif median_gap <= 8:
        return 52    # weekly
    else:
        return 12    # monthly


def compute_factor_metrics(backtest_result: Dict) -> Dict:
    """
    从回测结果计算全部绩效指标。

    Returns:
        dict with keys:
        - ls_annual_return, ls_annual_vol, ls_sharpe, ls_win_rate, ls_max_drawdown
        - rank_ic, rank_icir
        - monotonicity
        - quantile_metrics: {q: {annual_return, sharpe}}
        - turnover_mean
        - ann_factor (for reference)
    """
    qret = backtest_result['quantile_returns']
    ls = backtest_result['long_short']
    ic = backtest_result['ic_series']
    quantiles = backtest_result['quantiles']

    ann_factor = _infer_ann_factor(qret.index)
    metrics = {'ann_factor': ann_factor}

    # ── Long-Short 指标 ─────────────────────────────
    ls_clean = ls.dropna()

    ls_ann_ret = ls_clean.mean() * ann_factor
    ls_ann_vol = ls_clean.std() * np.sqrt(ann_factor)
    ls_sharpe = ls_ann_ret / ls_ann_vol if ls_ann_vol > 0 else 0.0
    ls_win_rate = (ls_clean > 0).mean() if len(ls_clean) > 0 else 0.0

    cum = (1 + ls_clean).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min() if len(drawdown) > 0 else 0.0

    metrics['ls_annual_return'] = ls_ann_ret
    metrics['ls_annual_vol'] = ls_ann_vol
    metrics['ls_sharpe'] = ls_sharpe
    metrics['ls_win_rate'] = ls_win_rate
    metrics['ls_max_drawdown'] = max_dd

    # ── Rank IC / Rank ICIR ─────────────────────────
    ic_clean = ic.dropna()
    metrics['rank_ic'] = ic_clean.mean() if len(ic_clean) > 0 else np.nan
    ic_std = ic_clean.std()
    metrics['rank_icir'] = (ic_clean.mean() / ic_std) if (ic_std > 0) else 0.0
    metrics['ic_count'] = len(ic_clean)

    # ── 单调性 ──────────────────────────────────────
    n_q = qret.shape[1]
    q_mean_ret = [qret[q].mean() for q in range(1, n_q + 1)]
    q_nums = list(range(1, n_q + 1))
    if len(q_mean_ret) >= 3:
        mono_corr, _ = stats.spearmanr(q_nums, q_mean_ret)
        metrics['monotonicity'] = mono_corr
    else:
        metrics['monotonicity'] = np.nan

    # ── 分位收益指标 ────────────────────────────────
    quantile_metrics = {}
    for q in range(1, n_q + 1):
        qr = qret[q].dropna()
        ann_ret = qr.mean() * ann_factor
        ann_vol = qr.std() * np.sqrt(ann_factor)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        quantile_metrics[q] = {
            'annual_return': ann_ret,
            'annual_vol': ann_vol,
            'sharpe': sharpe,
        }
    metrics['quantile_metrics'] = quantile_metrics

    # ── 换手率 ──────────────────────────────────────
    q5_mask = (quantiles == n_q).astype(float)
    turnover = q5_mask.diff().abs().sum(axis=1) / 2
    holdings_count = q5_mask.sum(axis=1).replace(0, np.nan)
    turnover_rate = (turnover / holdings_count).dropna()
    metrics['turnover_mean'] = turnover_rate.mean()

    return metrics


def format_metrics(metrics: Dict, factor_name: str = '') -> str:
    """格式化输出指标"""
    lines = []
    if factor_name:
        lines.append(f'=== {factor_name} ===')

    freq_label = {252: 'Daily', 52: 'Weekly', 12: 'Monthly'}.get(
        metrics['ann_factor'], f'{metrics["ann_factor"]}x')

    lines.append(f'L/S ({freq_label}): '
                 f'Return={metrics["ls_annual_return"]:.2%}, '
                 f'Vol={metrics["ls_annual_vol"]:.2%}, '
                 f'Sharpe={metrics["ls_sharpe"]:.2f}, '
                 f'WinRate={metrics["ls_win_rate"]:.1%}, '
                 f'MaxDD={metrics["ls_max_drawdown"]:.2%}')

    lines.append(f'Rank IC={metrics["rank_ic"]:.4f}, '
                 f'Rank ICIR={metrics["rank_icir"]:.2f} '
                 f'(n={metrics["ic_count"]})')

    lines.append(f'Monotonicity={metrics["monotonicity"]:.2f}, '
                 f'Turnover={metrics["turnover_mean"]:.2%}')

    lines.append('Quantile annual returns:')
    for q, m in metrics['quantile_metrics'].items():
        lines.append(f'  Q{q}: {m["annual_return"]:+.2%} '
                     f'(Sharpe {m["sharpe"]:.2f})')

    return '\n'.join(lines)
