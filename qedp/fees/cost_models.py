"""Slippage and cost models."""
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class SlippageModel(ABC):
    """Abstract base class for slippage models."""
    
    @abstractmethod
    def calculate_slippage(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        market_data: Dict[str, Any]
    ) -> float:
        """
        Calculate slippage for an order.
        
        Returns:
            Slippage in price units (positive = worse execution)
        """
        pass


class LinearImpactSlippage(SlippageModel):
    """
    Linear market impact model with random component.
    
    slip = k * (qty / ADV) * price + random_component
    """
    
    def __init__(self, k: float = 0.15, random_sigma_bps: float = 1.5, seed: int | None = None):
        self.k = k
        self.random_sigma_bps = random_sigma_bps
        self.rng = np.random.default_rng(seed)
    
    def calculate_slippage(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate slippage with linear impact + random component."""
        # Get average daily volume (ADV)
        adv = market_data.get("adv", market_data.get("volume", 1_000_000))
        
        # Linear impact component
        impact_ratio = qty / adv if adv > 0 else 0
        impact_slip = self.k * impact_ratio * price
        
        # Random component (basis points)
        random_bps = self.rng.normal(0, self.random_sigma_bps)
        random_slip = price * (random_bps / 10000)
        
        # Total slippage (always positive for cost)
        total_slip = abs(impact_slip) + abs(random_slip)
        
        return total_slip


class FixedBpsSlippage(SlippageModel):
    """Fixed basis points slippage."""
    
    def __init__(self, bps: float = 5.0):
        self.bps = bps
    
    def calculate_slippage(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate fixed bps slippage."""
        return price * (self.bps / 10000)


class FeeModel:
    """
    Transaction fee model.
    
    Includes commission, SEC fees, exchange fees, etc.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.commission_bps = config.get("commission_bps", 0.5)
        self.sec_fee_bps = config.get("sec_fee_bps", 0.02)
        self.min_commission = config.get("min_commission", 0.0)
    
    def calculate_fees(self, symbol: str, side: str, qty: int, price: float) -> float:
        """
        Calculate total fees for a trade.
        
        Returns:
            Total fees in dollars
        """
        notional = qty * price
        
        # Commission
        commission = max(
            notional * (self.commission_bps / 10000),
            self.min_commission
        )
        
        # SEC fee (sells only in US equities)
        sec_fee = 0.0
        if side == "SELL":
            sec_fee = notional * (self.sec_fee_bps / 10000)
        
        return commission + sec_fee
