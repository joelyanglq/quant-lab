"""
IBKR Signal Advisor

连接 IBKR 持仓 + HHT/QRS 策略信号 → 输出操作建议
"""
from advisor.signal_engine import SignalResult, compute_all_signals
from advisor.comparator import Recommendation, generate_recommendations

__all__ = [
    'SignalResult',
    'compute_all_signals',
    'Recommendation',
    'generate_recommendations',
]
