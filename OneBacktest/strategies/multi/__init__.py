"""
multi — 多策略组合

三层乘性架构:
    CrossSectionAlpha    × HMMRegimeStrategy × IndexTimingStrategy
    (选什么股)             (仓位缩放)           (何时入场)
"""
from .cross_section_alpha import CrossSectionAlpha
from .index_timing_strategy import IndexTimingStrategy
from strategies.regime_alloc.hmm.strategy import HMMRegimeStrategy

__all__ = [
    'CrossSectionAlpha',
    'HMMRegimeStrategy',
    'IndexTimingStrategy',
]
