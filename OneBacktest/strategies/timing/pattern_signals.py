"""
技术形态事件信号 — 每日信号生成器

独立于多因子选股系统, 基于 Lo-Mamaysky-Wang (2000) 技术形态识别.
事件驱动: 检测形态 → 检查 regime → 入场 → 持有 20 天 → 平仓.

研究结论:
  - 6 个有效形态 (Phase 1 survivor, |excess| > 20 bps, |t| > 2 at 20d)
  - HighVol regime 下信号最强 (Sharpe 1.12, Calmar 1.44)
  - 与动量/反转/波动正交 (Phase 3 double sort 确认)
  - TTOP 是核心 alpha 来源 (+95 bps at 20d, t=10.7)

用法:
    cd OneBacktest
    python -m strategies.cross_section.pattern_signals              # 今日信号
    python -m strategies.cross_section.pattern_signals --all-regime # 不限 regime
    python -m strategies.cross_section.pattern_signals --history    # 显示持仓历史
    python -m strategies.cross_section.pattern_signals --no-state   # 不保存状态
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.prices import load_index_symbols, load_price_panel
from strategies.timing.patterns import detect_patterns_panel, PatternType

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

# Phase 1 survivors: direction from excess return sign
# (注意: direction 由数据驱动, 不是论文的 bullish/bearish 分类)
SURVIVOR_SIGNALS = {
    'triangle_top':       +1,   # excess=+95 bps, t=+10.7 → LONG
    'rectangle_bottom':   -1,   # excess=-50 bps, t=-9.3  → SHORT
    'broadening_bottom':  +1,   # excess=+32 bps, t=+3.9  → LONG
    'double_bottom':      -1,   # excess=-130 bps, t=-3.2 → SHORT
    'head_shoulders':     -1,   # excess=-27 bps, t=-2.1  → SHORT
    'triangle_bottom':    +1,   # excess=+20 bps, t=+2.9  → LONG
}

HOLDING_PERIOD = 20  # trading days

STATE_FILE = Path(__file__).resolve().parent.parent.parent / 'output' / 'pattern_positions.json'


# ═══════════════════════════════════════════════════════════════
# Regime Detection
# ═══════════════════════════════════════════════════════════════

def check_high_vol_regime(spy_close: pd.Series) -> bool:
    """
    判断当前是否处于 HighVol regime.
    RVol_60d > expanding median → HighVol.
    """
    ret = spy_close.pct_change()
    rvol = ret.rolling(60, min_periods=30).std() * np.sqrt(252)
    median = rvol.expanding(60).median()

    latest_rvol = rvol.iloc[-1]
    latest_med = median.iloc[-1]

    if pd.isna(latest_rvol) or pd.isna(latest_med):
        return False
    return latest_rvol > latest_med


def get_regime_info(spy_close: pd.Series) -> dict:
    """返回当前 regime 状态."""
    ret = spy_close.pct_change()
    rvol = ret.rolling(60, min_periods=30).std() * np.sqrt(252)
    median = rvol.expanding(60).median()
    ma200 = spy_close.rolling(200, min_periods=100).mean()

    latest = spy_close.iloc[-1]
    latest_ma200 = ma200.iloc[-1]
    latest_rvol = rvol.iloc[-1]
    latest_med = median.iloc[-1]

    return {
        'spy_price': float(latest),
        'spy_ma200': float(latest_ma200) if not pd.isna(latest_ma200) else None,
        'trend': 'Bull' if latest > latest_ma200 else 'Bear',
        'rvol_60d': float(latest_rvol) if not pd.isna(latest_rvol) else None,
        'rvol_median': float(latest_med) if not pd.isna(latest_med) else None,
        'volatility': 'HighVol' if latest_rvol > latest_med else 'LowVol',
    }


# ═══════════════════════════════════════════════════════════════
# Pattern Detection (latest day only)
# ═══════════════════════════════════════════════════════════════

def detect_today_signals(
    close: pd.DataFrame,
    window: int = 35,
    lag: int = 3,
    bandwidth: float = 3.0,
) -> Dict[str, List[str]]:
    """
    检测最新交易日的形态信号.

    Returns:
        {pattern_name: [list of symbols with detection]}
    """
    raw_panels = detect_patterns_panel(
        close, window=window, lag=lag, bandwidth=bandwidth,
    )

    today = close.index[-1]
    signals = {}

    for pname, direction in SURVIVOR_SIGNALS.items():
        if pname not in raw_panels:
            continue
        panel = raw_panels[pname]
        hits = panel.loc[today]
        symbols = hits[hits == 1].index.tolist()
        if symbols:
            signals[pname] = symbols

    return signals


# ═══════════════════════════════════════════════════════════════
# Position State Management
# ═══════════════════════════════════════════════════════════════

def load_state(path: Path) -> List[dict]:
    """加载持仓状态."""
    if not path.exists():
        return []
    with open(path, 'r') as f:
        return json.load(f)


def save_state(positions: List[dict], path: Path):
    """保存持仓状态."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(positions, f, indent=2, default=str)


def update_positions(
    positions: List[dict],
    new_entries: List[dict],
    today: str,
    close: pd.DataFrame,
) -> tuple:
    """
    更新持仓状态.

    Returns:
        (active_positions, exits_today, new_entries_added)
    """
    today_dt = pd.Timestamp(today)
    today_close = close.iloc[-1]

    # Check exits
    exits = []
    active = []
    for pos in positions:
        exit_date = pd.Timestamp(pos['exit_date'])
        if today_dt >= exit_date:
            # Mark exit
            exit_price = today_close.get(pos['symbol'], np.nan)
            pos['actual_exit_date'] = today
            pos['exit_price'] = float(exit_price) if not pd.isna(exit_price) else None
            if pos.get('entry_price') and pos.get('exit_price'):
                raw_ret = pos['exit_price'] / pos['entry_price'] - 1
                pos['return'] = raw_ret * pos['direction']
            exits.append(pos)
        else:
            # Update current price
            cur_price = today_close.get(pos['symbol'], np.nan)
            pos['current_price'] = float(cur_price) if not pd.isna(cur_price) else None
            active.append(pos)

    # Add new entries (skip duplicates)
    held_symbols = {p['symbol'] for p in active}
    added = []
    for entry in new_entries:
        if entry['symbol'] not in held_symbols:
            active.append(entry)
            added.append(entry)
            held_symbols.add(entry['symbol'])

    return active, exits, added


# ═══════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════

def print_regime(regime: dict):
    """显示当前 regime."""
    print(f'\n  Regime Status:')
    print(f'    SPY:       ${regime["spy_price"]:.2f}  '
          f'(MA200: ${regime["spy_ma200"]:.2f})' if regime['spy_ma200'] else '')
    print(f'    Trend:     {regime["trend"]}')
    if regime['rvol_60d']:
        print(f'    RVol 60d:  {regime["rvol_60d"]:.1%}  '
              f'(median: {regime["rvol_median"]:.1%})')
    print(f'    Volatility: {regime["volatility"]}')


def print_new_signals(signals: Dict[str, List[str]], regime_ok: bool):
    """显示新检测到的信号."""
    if not signals:
        print('\n  New Signals: None detected today')
        return

    total = sum(len(v) for v in signals.values())
    status = 'ACTIVE' if regime_ok else 'BLOCKED (LowVol regime)'
    print(f'\n  New Signals: {total} detections  [{status}]')

    for pname, syms in sorted(signals.items()):
        direction = SURVIVOR_SIGNALS[pname]
        d_label = 'LONG' if direction > 0 else 'SHORT'
        print(f'    {pname:<25s} [{d_label}]  {len(syms)} stocks: '
              f'{", ".join(syms[:10])}{"..." if len(syms) > 10 else ""}')


def print_positions(active: List[dict], exits: List[dict]):
    """显示持仓状态."""
    if not active and not exits:
        print('\n  Open Positions: None')
        return

    if active:
        long_pos = [p for p in active if p['direction'] > 0]
        short_pos = [p for p in active if p['direction'] < 0]
        print(f'\n  Open Positions: {len(active)} '
              f'({len(long_pos)} long, {len(short_pos)} short)')
        print(f'    {"Symbol":<8s} {"Dir":>5s} {"Signal":<22s} '
              f'{"Entry":>8s} {"Current":>8s} {"P&L":>7s} {"Exit Due":>10s}')
        print(f'    {"-"*72}')

        for p in sorted(active, key=lambda x: x['exit_date']):
            ep = p.get('entry_price', 0) or 0
            cp = p.get('current_price', 0) or 0
            pnl = (cp / ep - 1) * p['direction'] if ep > 0 and cp > 0 else 0
            d_label = 'LONG' if p['direction'] > 0 else 'SHORT'
            print(f'    {p["symbol"]:<8s} {d_label:>5s} {p["signal"]:<22s} '
                  f'${ep:>7.2f} ${cp:>7.2f} {pnl:>+6.1%} {p["exit_date"]:>10s}')

    if exits:
        print(f'\n  Exits Today: {len(exits)}')
        for p in exits:
            ret = p.get('return', 0) or 0
            d_label = 'LONG' if p['direction'] > 0 else 'SHORT'
            print(f'    {p["symbol"]:<8s} {d_label:>5s} {p["signal"]:<22s} '
                  f'return: {ret:>+6.1%}')


def print_summary(active: List[dict], exits_history: List[dict]):
    """显示历史统计."""
    if not exits_history:
        return
    rets = [e.get('return', 0) for e in exits_history if e.get('return') is not None]
    if not rets:
        return
    rets = np.array(rets)
    print(f'\n  Historical Performance ({len(rets)} closed trades):')
    print(f'    Win Rate:    {(rets > 0).mean():.1%}')
    print(f'    Avg Return:  {rets.mean():+.2%}')
    print(f'    Avg Win:     {rets[rets > 0].mean():+.2%}' if (rets > 0).any() else '')
    print(f'    Avg Loss:    {rets[rets <= 0].mean():+.2%}' if (rets <= 0).any() else '')
    print(f'    Total P&L:   {rets.sum():+.2%}')


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Pattern event signals (Lo-Mamaysky-Wang 2000)')
    parser.add_argument('--all-regime', action='store_true',
                        help='Trade in all regimes (default: HighVol only)')
    parser.add_argument('--no-state', action='store_true',
                        help='Do not save/load position state')
    parser.add_argument('--history', action='store_true',
                        help='Show closed trade history')
    parser.add_argument('--holding', type=int, default=HOLDING_PERIOD,
                        help=f'Holding period in days (default: {HOLDING_PERIOD})')
    parser.add_argument('--lookback', type=str, default='2024-06-01',
                        help='Price data start (default: 2024-06-01)')
    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')

    print()
    print('=' * 64)
    print(f'  Pattern Event Signals — {today_str}')
    print('=' * 64)

    # ── 1. Load data ──
    print('\n[1/3] Loading data ...')
    t0 = time.time()
    symbols = load_index_symbols()

    panels = load_price_panel(symbols, start=args.lookback)
    close = panels['close']
    latest_date = close.index[-1].strftime('%Y-%m-%d')
    print(f'  {close.shape[1]} symbols x {close.shape[0]} days  '
          f'(latest: {latest_date}, {time.time()-t0:.1f}s)')

    spy_panels = load_price_panel(['SPY'], start='2023-01-01')
    spy_close = spy_panels['close']['SPY']

    # ── 2. Regime check ──
    regime = get_regime_info(spy_close)
    regime_ok = args.all_regime or regime['volatility'] == 'HighVol'
    print_regime(regime)

    if not regime_ok:
        print(f'\n  ** LowVol regime — signals will be shown but NOT entered **')
        print(f'  ** Use --all-regime to override **')

    # ── 3. Detect patterns ──
    print(f'\n[2/3] Detecting patterns (window=35, lag=3) ...')
    t0 = time.time()
    signals = detect_today_signals(close)
    print(f'  Done in {time.time()-t0:.1f}s')

    print_new_signals(signals, regime_ok)

    # ── 4. Position management ──
    print(f'\n[3/3] Position management ...')

    if args.no_state:
        positions = []
    else:
        positions = load_state(STATE_FILE)

    # Build new entries
    new_entries = []
    if regime_ok and signals:
        # Compute exit date (holding period trading days from latest_date)
        bdays = pd.bdate_range(
            start=pd.Timestamp(latest_date) + timedelta(days=1),
            periods=args.holding,
        )
        exit_date = bdays[-1].strftime('%Y-%m-%d')

        next_bday = bdays[0].strftime('%Y-%m-%d')
        entry_close = close.iloc[-1]

        for pname, syms in signals.items():
            direction = SURVIVOR_SIGNALS[pname]
            for sym in syms:
                ep = entry_close.get(sym, np.nan)
                new_entries.append({
                    'symbol': sym,
                    'signal': pname,
                    'direction': direction,
                    'entry_date': latest_date,
                    'entry_price': float(ep) if not pd.isna(ep) else None,
                    'exit_date': exit_date,
                    'current_price': float(ep) if not pd.isna(ep) else None,
                })

    active, exits, added = update_positions(
        positions, new_entries, latest_date, close)

    print_positions(active, exits)

    if added:
        print(f'\n  New Entries: {len(added)} positions opened')

    # Save state
    if not args.no_state:
        # Keep exits in a separate history
        history_file = STATE_FILE.with_name('pattern_history.json')
        history = load_state(history_file) if history_file.exists() else []
        history.extend(exits)

        save_state(active, STATE_FILE)
        if history:
            save_state(history, history_file)

        print(f'\n  State saved to {STATE_FILE.name}')

    # Show history
    if args.history or exits:
        history_file = STATE_FILE.with_name('pattern_history.json')
        history = load_state(history_file) if history_file.exists() else []
        all_exits = history  # already includes today's exits
        print_summary(active, all_exits)

    # Action summary
    print()
    print('=' * 64)
    n_long = sum(1 for p in active if p['direction'] > 0)
    n_short = sum(1 for p in active if p['direction'] < 0)
    print(f'  Portfolio: {len(active)} positions '
          f'({n_long} long, {n_short} short)')
    if exits:
        print(f'  Exits:     {len(exits)} positions closed today')
    if added:
        print(f'  New:       {len(added)} positions opened today')
    if not regime_ok:
        print(f'  Regime:    LowVol — waiting for HighVol to enter')
    print('=' * 64)
    print()


if __name__ == '__main__':
    main()
