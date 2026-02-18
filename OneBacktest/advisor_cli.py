"""
IBKR Signal Advisor — CLI 入口

用法:
    python advisor_cli.py                              # 连接 IBKR paper (7497)
    python advisor_cli.py --port 7496                  # 连接 IBKR live
    python advisor_cli.py --offline                    # 不连 IBKR，只看信号
    python advisor_cli.py --positions AAPL:100,NVDA:50 # 手动输入持仓
    python advisor_cli.py --symbols AAPL,NVDA,GOOG     # 指定标的
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from data.storage.parquet import ParquetStorage
from advisor.signal_engine import compute_all_signals
from advisor.comparator import generate_recommendations
from advisor.ibkr_reader import IBKRReader, IBAccount, HAS_IB_INSYNC, PORTS


DEFAULT_SYMBOLS = ['AVGO', 'AAPL', 'NBIS', 'TSM', 'GOOG', 'COHR', 'NVDA']
DEFAULT_DATA_DIR = r'D:\04_Project\quant-lab\data\processed\bars_1d'


def parse_positions(pos_str: str) -> dict:
    """解析 'AAPL:100,NVDA:50' 格式的持仓字符串"""
    positions = {}
    if not pos_str:
        return positions
    for item in pos_str.split(','):
        item = item.strip()
        if ':' in item:
            sym, qty = item.split(':', 1)
            positions[sym.strip().upper()] = float(qty.strip())
    return positions


def print_table(recommendations, account=None, offline=False):
    """打印建议表"""
    print()
    print("=" * 60)
    print("  Signal Advisor")
    print("=" * 60)

    if account:
        print(f"  Account: {account.account_id}"
              f"  |  Net Liq: ${account.net_liquidation:,.0f}"
              f"  |  Cash: ${account.total_cash:,.0f}")

    if recommendations and recommendations[0].data_date:
        print(f"  Data: {recommendations[0].data_date}")

    if offline:
        print("  Mode: OFFLINE (signals only)")

    print()

    if not offline:
        # 完整模式：持仓 + 信号 + 建议
        header = (f"{'Symbol':<7} {'Qty':>6} {'AvgCost':>9} {'P&L':>9}"
                  f" | {'HHT':^6} {'QRS':^14}"
                  f" | {'Action':<15} {'Reason'}")
        print(header)
        print("-" * len(header))

        for r in recommendations:
            qty_str = f"{r.current_qty:,.0f}" if r.current_qty != 0 else "--"
            cost_str = f"{r.avg_cost:,.1f}" if r.avg_cost is not None else "--"
            pnl_str = f"{r.unrealized_pnl:+,.0f}" if r.unrealized_pnl is not None else "--"

            qrs_detail = r.qrs_label
            if r.qrs_value is not None:
                qrs_detail = f"{r.qrs_label} ({r.qrs_value:+.2f})"

            print(f"{r.symbol:<7} {qty_str:>6} {cost_str:>9} {pnl_str:>9}"
                  f" | {r.hht_label:^6} {qrs_detail:^14}"
                  f" | {r.action:<15} {r.reason}")
    else:
        # 离线模式：只显示信号
        header = (f"{'Symbol':<7} | {'HHT':^8} {'Phase':>8}"
                  f" | {'QRS':^8} {'Value':>8}"
                  f" | {'Action':<15} {'Reason'}")
        print(header)
        print("-" * len(header))

        for r in recommendations:
            phase_str = f"{r.hht_value:.3f}" if r.hht_value is not None else "N/A"
            qrs_val_str = f"{r.qrs_value:+.3f}" if r.qrs_value is not None else "N/A"

            print(f"{r.symbol:<7} | {r.hht_label:^8} {phase_str:>8}"
                  f" | {r.qrs_label:^8} {qrs_val_str:>8}"
                  f" | {r.action:<15} {r.reason}")

    print()


def main():
    parser = argparse.ArgumentParser(description='IBKR Signal Advisor')
    parser.add_argument('--host', default='127.0.0.1',
                        help='IBKR host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=None,
                        help='IBKR port (default: auto from --mode)')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                        help='Trading mode (default: paper)')
    parser.add_argument('--client-id', type=int, default=10,
                        help='IBKR client ID (default: 10)')
    parser.add_argument('--offline', action='store_true',
                        help='Skip IBKR connection, signals only')
    parser.add_argument('--positions', type=str, default=None,
                        help='Manual positions: AAPL:100,NVDA:50')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated symbol list')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR,
                        help='Parquet data directory')

    args = parser.parse_args()

    # 解析 symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = list(DEFAULT_SYMBOLS)

    # 解析端口
    if args.port is None:
        args.port = PORTS['tws_paper'] if args.mode == 'paper' else PORTS['tws_live']

    # 获取持仓
    positions = {}
    position_details = {}
    account = None
    offline = args.offline
    has_positions = False

    if args.positions:
        # 手动输入持仓
        positions = parse_positions(args.positions)
        has_positions = True
        # 把持仓中的 symbol 也加入观测列表
        for sym in positions:
            if sym not in symbols:
                symbols.append(sym)

    elif not args.offline:
        # 尝试连接 IBKR
        if not HAS_IB_INSYNC:
            print("WARNING: ib_insync not installed. Falling back to offline mode.")
            print("         Install with: pip install ib_insync")
            print("         Or use: --offline / --positions")
            offline = True
        else:
            reader = IBKRReader(args.host, args.port, args.client_id)
            if reader.connect():
                ib_positions = reader.get_positions()
                account = reader.get_account()
                for p in ib_positions:
                    positions[p.symbol] = p.quantity
                    position_details[p.symbol] = p
                    if p.symbol not in symbols:
                        symbols.append(p.symbol)
                reader.disconnect()
            else:
                print("\nWARNING: Could not connect to IBKR. Running in offline mode.")
                print("         Use --offline to skip, or --positions for manual input.\n")
                offline = True

    # 计算信号
    storage = ParquetStorage(args.data_dir)
    print(f"Computing signals for {len(symbols)} symbols...", flush=True)
    signals = compute_all_signals(symbols, storage)

    # 生成建议
    show_positions = has_positions or (account is not None)
    if offline and not positions:
        # 纯离线模式：无持仓信息，positions 全为 0
        positions = {s: 0 for s in symbols}

    recommendations = generate_recommendations(
        positions, signals, position_details or None)

    # 输出
    print_table(recommendations, account, not show_positions)


if __name__ == '__main__':
    main()
