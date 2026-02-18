"""
回测绩效分析
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from event import FillEvent, OrderSide


def calc_metrics(equity_curve: pd.DataFrame, trade_log: List[FillEvent],
                 initial_capital: float = 100000.0,
                 risk_free_rate: float = 0.0,
                 periods_per_year: int = 252) -> Dict:
    """
    计算回测绩效指标

    Args:
        equity_curve: DataFrame, index=timestamp, 需含 'total' 列
        trade_log: FillEvent 列表
        initial_capital: 初始资金
        risk_free_rate: 无风险年利率
        periods_per_year: 每年交易日数

    Returns:
        dict of metrics
    """
    total = equity_curve['total']
    returns = total.pct_change().dropna()

    # ---- 收益 ----
    final_value = total.iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital
    n_periods = len(total)
    years = n_periods / periods_per_year
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    # ---- 风险 ----
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - daily_rf
    sharpe = np.sqrt(periods_per_year) * excess.mean() / excess.std() if excess.std() > 0 else 0.0

    downside = returns[returns < daily_rf] - daily_rf
    sortino = np.sqrt(periods_per_year) * excess.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0.0

    # ---- 回撤 ----
    cummax = total.cummax()
    drawdown = (total - cummax) / cummax
    max_dd = drawdown.min()

    # 最大回撤持续期
    in_dd = drawdown < 0
    dd_groups = (~in_dd).cumsum()
    if in_dd.any():
        dd_durations = in_dd.groupby(dd_groups).sum()
        max_dd_duration = int(dd_durations.max())
    else:
        max_dd_duration = 0

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # ---- 交易统计 ----
    n_trades = len(trade_log)
    trade_pnls = _calc_trade_pnls(trade_log)
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf') if wins else 0.0

    return {
        'total_return': total_return,
        'annual_return': ann_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'max_dd_duration': max_dd_duration,
        'calmar_ratio': calmar,
        'total_trades': n_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'final_value': final_value,
        'initial_capital': initial_capital,
    }


def _calc_trade_pnls(trade_log: List[FillEvent]) -> List[float]:
    """
    简单配对法计算每笔交易 P&L（FIFO）

    买入记录开仓成本，卖出时计算差价。
    """
    # {symbol: [(price, qty), ...]}
    open_positions: Dict[str, list] = {}
    pnls = []

    for fill in trade_log:
        s = fill.symbol
        if s not in open_positions:
            open_positions[s] = []

        if fill.side == OrderSide.BUY:
            open_positions[s].append((fill.fill_price, fill.fill_quantity))
        elif fill.side == OrderSide.SELL:
            qty_to_close = fill.fill_quantity
            sell_price = fill.fill_price
            while qty_to_close > 0 and open_positions[s]:
                entry_price, entry_qty = open_positions[s][0]
                closed = min(qty_to_close, entry_qty)
                pnl = (sell_price - entry_price) * closed
                pnls.append(pnl)
                qty_to_close -= closed
                if closed == entry_qty:
                    open_positions[s].pop(0)
                else:
                    open_positions[s][0] = (entry_price, entry_qty - closed)

    return pnls


def print_report(metrics: Dict, name: str = "Strategy"):
    """打印绩效报告"""
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Initial Capital:    ${metrics['initial_capital']:>12,.2f}")
    print(f"  Final Value:        ${metrics['final_value']:>12,.2f}")
    print(f"  Total Return:        {metrics['total_return']:>11.2%}")
    print(f"  Annual Return:       {metrics['annual_return']:>11.2%}")
    print(f"  {'-' * 46}")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>11.2f}")
    print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>11.2f}")
    print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>11.2f}")
    print(f"  Max Drawdown:        {metrics['max_drawdown']:>11.2%}")
    print(f"  Max DD Duration:     {metrics['max_dd_duration']:>8d} days")
    print(f"  {'-' * 46}")
    print(f"  Total Trades:        {metrics['total_trades']:>11d}")
    print(f"  Win Rate:            {metrics['win_rate']:>11.2%}")
    print(f"  Avg Win:            ${metrics['avg_win']:>12,.2f}")
    print(f"  Avg Loss:           ${metrics['avg_loss']:>12,.2f}")
    print(f"  Profit Factor:       {metrics['profit_factor']:>11.2f}")
    print(f"{'=' * 50}")
