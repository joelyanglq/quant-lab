"""
IBKR 只读连接 — 读取持仓和账户信息

依赖: pip install ib_insync
不安装时 offline 模式仍可用（import guard）
"""
from dataclasses import dataclass
from typing import List, Optional

try:
    from ib_insync import IB, Stock
    HAS_IB_INSYNC = True
except ImportError:
    HAS_IB_INSYNC = False


# ── 端口约定 ────────────────────────────────────────────────

PORTS = {
    'tws_paper': 7497,
    'tws_live': 7496,
    'gw_paper': 4002,
    'gw_live': 4001,
}


# ── 数据结构 ────────────────────────────────────────────────

@dataclass
class IBPosition:
    """IBKR 持仓"""
    symbol: str
    quantity: float
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float


@dataclass
class IBAccount:
    """IBKR 账户摘要"""
    account_id: str
    net_liquidation: float
    total_cash: float


# ── 连接类 ──────────────────────────────────────────────────

class IBKRReader:
    """
    IBKR 只读连接

    只负责读取持仓和账户信息，不下单。

    Usage:
        reader = IBKRReader(port=7497)
        if reader.connect():
            positions = reader.get_positions()
            account = reader.get_account()
            reader.disconnect()
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 7497,
                 client_id: int = 10):
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib: Optional['IB'] = None

    def connect(self, timeout: int = 5) -> bool:
        """
        连接到 TWS/Gateway

        Returns:
            True 连接成功, False 连接失败
        """
        if not HAS_IB_INSYNC:
            print("ERROR: ib_insync not installed. Run: pip install ib_insync")
            return False

        self._ib = IB()
        try:
            self._ib.connect(
                self.host, self.port, clientId=self.client_id,
                timeout=timeout, readonly=True,
            )
            return True
        except Exception as e:
            print(f"ERROR: Cannot connect to IBKR at {self.host}:{self.port}")
            print(f"       {e}")
            print(f"       Make sure TWS or IB Gateway is running.")
            self._ib = None
            return False

    def disconnect(self):
        """断开连接"""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
        self._ib = None

    @property
    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()

    def get_positions(self) -> List[IBPosition]:
        """
        获取所有股票持仓

        只返回 STK（股票）类型，忽略期权/期货等。
        """
        if not self.is_connected:
            return []

        positions = []
        for item in self._ib.positions():
            contract = item.contract
            # 只看股票
            if contract.secType != 'STK':
                continue

            pos = item.position
            avg_cost = item.avgCost

            # 获取市场价（通过 portfolio items）
            market_price = 0.0
            market_value = 0.0
            unrealized_pnl = 0.0

            positions.append(IBPosition(
                symbol=contract.symbol,
                quantity=float(pos),
                avg_cost=float(avg_cost),
                market_price=market_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
            ))

        # 尝试从 portfolio 获取市场价和 P&L
        try:
            portfolio_items = self._ib.portfolio()
            pf_map = {}
            for pf in portfolio_items:
                if pf.contract.secType == 'STK':
                    pf_map[pf.contract.symbol] = pf

            for p in positions:
                if p.symbol in pf_map:
                    pf = pf_map[p.symbol]
                    p.market_price = float(pf.marketPrice)
                    p.market_value = float(pf.marketValue)
                    p.unrealized_pnl = float(pf.unrealizedPNL)
        except Exception:
            pass

        return positions

    def get_account(self) -> Optional[IBAccount]:
        """获取账户摘要"""
        if not self.is_connected:
            return None

        try:
            values = self._ib.accountSummary()
            data = {}
            for item in values:
                data[item.tag] = item.value

            return IBAccount(
                account_id=data.get('AccountCode', values[0].account if values else ''),
                net_liquidation=float(data.get('NetLiquidation', 0)),
                total_cash=float(data.get('TotalCashValue', 0)),
            )
        except Exception:
            # fallback: 从 accountValues 获取
            try:
                account_id = self._ib.managedAccounts()[0] if self._ib.managedAccounts() else ''
                return IBAccount(
                    account_id=account_id,
                    net_liquidation=0.0,
                    total_cash=0.0,
                )
            except Exception:
                return None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
