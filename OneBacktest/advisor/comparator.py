"""
持仓 vs 信号对比 → 生成操作建议

决策矩阵:
    HHT=BUY  + QRS=BUY  + 无仓 → BUY
    HHT=BUY  + QRS=BUY  + 有仓 → HOLD
    HHT=BUY  + QRS=HOLD + 无仓 → CONSIDER BUY
    HHT=BUY  + QRS=HOLD + 有仓 → HOLD
    HHT=BUY  + QRS=SELL        → REVIEW (冲突)
    HHT=FLAT + QRS=BUY  + 无仓 → CONSIDER BUY
    HHT=FLAT + QRS=BUY  + 有仓 → REVIEW (冲突)
    HHT=FLAT + QRS=HOLD + 无仓 → —
    HHT=FLAT + QRS=HOLD + 有仓 → CONSIDER SELL
    HHT=FLAT + QRS=SELL + 无仓 → —
    HHT=FLAT + QRS=SELL + 有仓 → SELL
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

from advisor.signal_engine import SignalResult


@dataclass
class Recommendation:
    """单个标的的操作建议"""
    symbol: str
    # 持仓
    current_qty: float
    avg_cost: Optional[float]
    unrealized_pnl: Optional[float]
    # 信号
    hht_label: str      # "BUY" / "FLAT" / "N/A"
    hht_value: Optional[float]
    qrs_label: str      # "BUY" / "SELL" / "HOLD" / "N/A"
    qrs_value: Optional[float]
    # 建议
    action: str
    reason: str
    data_date: Optional[str]


def _decide(hht: str, qrs: str, has_position: bool) -> tuple:
    """
    根据 HHT/QRS 信号和持仓状态返回 (action, reason)
    """
    if hht == 'N/A' and qrs == 'N/A':
        return ('N/A', 'insufficient data')

    # HHT = BUY
    if hht == 'BUY':
        if qrs == 'BUY':
            if has_position:
                return ('HOLD', 'both bullish, keep position')
            else:
                return ('BUY', 'both bullish')
        elif qrs == 'HOLD' or qrs == 'N/A':
            if has_position:
                return ('HOLD', 'HHT bullish')
            else:
                return ('CONSIDER BUY', 'HHT bullish')
        elif qrs == 'SELL':
            if has_position:
                return ('REVIEW', 'conflicting signals')
            else:
                return ('REVIEW', 'conflicting signals')

    # HHT = FLAT
    elif hht == 'FLAT':
        if qrs == 'BUY':
            if has_position:
                return ('REVIEW', 'conflicting signals')
            else:
                return ('CONSIDER BUY', 'QRS bullish')
        elif qrs == 'HOLD' or qrs == 'N/A':
            if has_position:
                return ('CONSIDER SELL', 'HHT bearish')
            else:
                return ('-', 'no buy signal')
        elif qrs == 'SELL':
            if has_position:
                return ('SELL', 'both bearish')
            else:
                return ('-', 'no position, no signal')

    # HHT = N/A
    elif hht == 'N/A':
        if qrs == 'BUY':
            if has_position:
                return ('HOLD', 'QRS bullish')
            else:
                return ('CONSIDER BUY', 'QRS bullish')
        elif qrs == 'SELL':
            if has_position:
                return ('CONSIDER SELL', 'QRS bearish')
            else:
                return ('-', 'no position')
        else:
            return ('-', 'HHT insufficient data')

    return ('-', '')


def generate_recommendations(
    positions: Dict[str, float],
    signals: Dict[str, Dict[str, SignalResult]],
    position_details: Optional[Dict[str, object]] = None,
) -> List[Recommendation]:
    """
    生成操作建议

    Args:
        positions: {symbol: quantity} 当前持仓
        signals: compute_all_signals() 的输出
        position_details: {symbol: IBPosition} 可选的详细持仓信息

    Returns:
        按 symbol 排序的建议列表
    """
    # 合并所有 symbol（持仓 + 信号）
    all_symbols = sorted(set(list(positions.keys()) + list(signals.keys())))

    recommendations = []
    for sym in all_symbols:
        qty = positions.get(sym, 0)
        has_pos = qty != 0

        # 持仓详情
        avg_cost = None
        pnl = None
        if position_details and sym in position_details:
            detail = position_details[sym]
            avg_cost = getattr(detail, 'avg_cost', None)
            pnl = getattr(detail, 'unrealized_pnl', None)

        # 信号
        sym_signals = signals.get(sym, {})
        hht = sym_signals.get('HHT')
        qrs = sym_signals.get('QRS')

        hht_label = hht.label if hht else 'N/A'
        hht_value = hht.raw_value if hht else None
        qrs_label = qrs.label if qrs else 'N/A'
        qrs_value = qrs.raw_value if qrs else None

        data_date = None
        if hht and hht.data_date:
            data_date = str(hht.data_date.date())
        elif qrs and qrs.data_date:
            data_date = str(qrs.data_date.date())

        action, reason = _decide(hht_label, qrs_label, has_pos)

        recommendations.append(Recommendation(
            symbol=sym,
            current_qty=qty,
            avg_cost=avg_cost,
            unrealized_pnl=pnl,
            hht_label=hht_label,
            hht_value=hht_value,
            qrs_label=qrs_label,
            qrs_value=qrs_value,
            action=action,
            reason=reason,
            data_date=data_date,
        ))

    return recommendations
