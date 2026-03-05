"""
Lo-Mamaysky-Wang (2000) 技术形态识别

"Foundations of Technical Analysis" 算法实现:
  核回归平滑 → 局部极值检测 → 10 种经典形态匹配 → 因子化

Usage:
    from strategies.timing.patterns import compute_pattern_factors
    factors = compute_pattern_factors(close_panel)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# Layer 1: 核回归平滑
# ═══════════════════════════════════════════════════════════════

def _gaussian_kernel(u: np.ndarray, h: float) -> np.ndarray:
    """Gaussian kernel K_h(u) = exp(-u²/(2h²))."""
    return np.exp(-u ** 2 / (2.0 * h ** 2))


def _nadaraya_watson(prices: np.ndarray, h: float) -> np.ndarray:
    """
    Nadaraya-Watson 核回归估计.

    m̂_h(τ) = Σ K_h(τ - s) · P_s / Σ K_h(τ - s)
    """
    n = len(prices)
    tau = np.arange(n, dtype=float)
    # (n, n) 距离矩阵: tau[i] - tau[j]
    diff = tau[:, None] - tau[None, :]
    weights = _gaussian_kernel(diff, h)
    # 每行归一化
    w_sum = weights.sum(axis=1)
    w_sum[w_sum == 0] = 1.0
    smoothed = (weights @ prices) / w_sum
    return smoothed


def _cross_validation_bandwidth(prices: np.ndarray) -> float:
    """
    LOO 交叉验证选带宽.

    CV(h) = (1/n) Σ (P_i - m̂_{-i,h}(i))²
    Returns h* = argmin CV(h)
    """
    n = len(prices)
    tau = np.arange(n, dtype=float)
    diff = tau[:, None] - tau[None, :]  # (n, n)

    h_candidates = np.linspace(0.3, 8.0, 40)
    best_h, best_cv = h_candidates[0], np.inf

    for h in h_candidates:
        weights = _gaussian_kernel(diff, h)
        # LOO: 把对角线设为 0
        np.fill_diagonal(weights, 0.0)
        w_sum = weights.sum(axis=1)
        w_sum[w_sum == 0] = 1.0
        fitted = (weights @ prices) / w_sum
        cv = np.mean((prices - fitted) ** 2)
        if cv < best_cv:
            best_cv = cv
            best_h = h

    return best_h


def kernel_smooth(
    prices: np.ndarray,
    bandwidth: Union[float, str] = 'cv',
    bandwidth_factor: float = 0.3,
) -> np.ndarray:
    """
    论文完整流程: CV 选带宽 → × bandwidth_factor → Nadaraya-Watson 平滑.

    Args:
        prices: 1D price array
        bandwidth: 'cv' (交叉验证) 或固定数值 (跳过 CV, 更快)
        bandwidth_factor: 缩放因子, 论文推荐 0.3
    """
    if isinstance(bandwidth, str) and bandwidth == 'cv':
        h_star = _cross_validation_bandwidth(prices)
        h = h_star * bandwidth_factor
    else:
        h = float(bandwidth) * bandwidth_factor

    # 最小带宽防止退化
    h = max(h, 0.1)
    return _nadaraya_watson(prices, h)


# ═══════════════════════════════════════════════════════════════
# Layer 2: 极值检测
# ═══════════════════════════════════════════════════════════════

@dataclass
class Extremum:
    index: int        # 在窗口内的位置
    value: float      # 原始价格值
    is_max: bool      # True=极大, False=极小


def find_extrema(
    prices: np.ndarray,
    smoothed: np.ndarray,
    search_radius: int = 2,
) -> List[Extremum]:
    """
    在平滑曲线上找局部极值, 在原始价格上定位真实值.

    1. 计算 smoothed 的一阶差分 (近似导数)
    2. 找符号变化点 → 极大 (+ → -) 或极小 (- → +)
    3. 在 prices 的 [位置-radius, 位置+radius] 区间找真实极值
    4. 保证交替: max, min, max, min, ...
    """
    n = len(smoothed)
    if n < 3:
        return []

    # 一阶差分
    deriv = np.diff(smoothed)
    signs = np.sign(deriv)
    # 去掉 0 (用前一个符号填充)
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]

    raw_extrema: List[Extremum] = []
    for i in range(len(signs) - 1):
        if signs[i] > 0 and signs[i + 1] < 0:
            # 极大值: 位置 i+1 (在 smoothed 上)
            loc = i + 1
            lo = max(0, loc - search_radius)
            hi = min(n - 1, loc + search_radius)
            real_loc = lo + int(np.argmax(prices[lo:hi + 1]))
            raw_extrema.append(Extremum(real_loc, float(prices[real_loc]), True))
        elif signs[i] < 0 and signs[i + 1] > 0:
            # 极小值
            loc = i + 1
            lo = max(0, loc - search_radius)
            hi = min(n - 1, loc + search_radius)
            real_loc = lo + int(np.argmin(prices[lo:hi + 1]))
            raw_extrema.append(Extremum(real_loc, float(prices[real_loc]), False))

    if not raw_extrema:
        return []

    # 保证交替: 连续同类型只保留最极端的
    result: List[Extremum] = [raw_extrema[0]]
    for ext in raw_extrema[1:]:
        if ext.is_max == result[-1].is_max:
            # 同类: 保留更极端的
            if ext.is_max and ext.value > result[-1].value:
                result[-1] = ext
            elif not ext.is_max and ext.value < result[-1].value:
                result[-1] = ext
        else:
            result.append(ext)

    return result


# ═══════════════════════════════════════════════════════════════
# Layer 3: 形态匹配
# ═══════════════════════════════════════════════════════════════

class PatternType(Enum):
    HS = 'head_shoulders'
    IHS = 'inv_head_shoulders'
    BTOP = 'broadening_top'
    BBOT = 'broadening_bottom'
    TTOP = 'triangle_top'
    TBOT = 'triangle_bottom'
    RTOP = 'rectangle_top'
    RBOT = 'rectangle_bottom'
    DTOP = 'double_top'
    DBOT = 'double_bottom'


# 看涨 / 看跌分类
BULLISH_PATTERNS = {
    PatternType.IHS, PatternType.BBOT, PatternType.TBOT,
    PatternType.RBOT, PatternType.DBOT,
}
BEARISH_PATTERNS = {
    PatternType.HS, PatternType.BTOP, PatternType.TTOP,
    PatternType.RTOP, PatternType.DTOP,
}


@dataclass
class PatternMatch:
    pattern: PatternType
    extrema: List[Extremum] = field(default_factory=list)
    start_idx: int = 0
    end_idx: int = 0
    confidence: float = 1.0


def _within_pct(a: float, b: float, tol: float) -> Tuple[bool, float]:
    """检查 a 和 b 是否在均值 tol% 内. 返回 (pass, deviation)."""
    avg = (a + b) / 2.0
    if avg == 0:
        return a == b, 0.0
    dev = abs(a - b) / abs(avg)
    return dev <= tol, dev


def _check_hs(exts: List[Extremum], tol: float) -> Optional[PatternMatch]:
    """
    头肩顶: E1=max, E3>E1 且 E3>E5, E1/E5 在均值 tol 内, E2/E4 在均值 tol 内.
    """
    if len(exts) < 5:
        return None
    e1, e2, e3, e4, e5 = exts[-5:]
    if not e1.is_max:
        return None
    # E3 (头) > 两肩
    if not (e3.value > e1.value and e3.value > e5.value):
        return None
    # 两肩对称
    ok1, d1 = _within_pct(e1.value, e5.value, tol)
    if not ok1:
        return None
    # 两颈对称
    ok2, d2 = _within_pct(e2.value, e4.value, tol)
    if not ok2:
        return None
    conf = 1.0 - max(d1, d2) / tol
    return PatternMatch(PatternType.HS, exts[-5:], e1.index, e5.index, conf)


def _check_ihs(exts: List[Extremum], tol: float) -> Optional[PatternMatch]:
    """头肩底: 镜像 HS."""
    if len(exts) < 5:
        return None
    e1, e2, e3, e4, e5 = exts[-5:]
    if e1.is_max:
        return None
    if not (e3.value < e1.value and e3.value < e5.value):
        return None
    ok1, d1 = _within_pct(e1.value, e5.value, tol)
    if not ok1:
        return None
    ok2, d2 = _within_pct(e2.value, e4.value, tol)
    if not ok2:
        return None
    conf = 1.0 - max(d1, d2) / tol
    return PatternMatch(PatternType.IHS, exts[-5:], e1.index, e5.index, conf)


def _check_btop(exts: List[Extremum]) -> Optional[PatternMatch]:
    """扩散顶: E1=max, E1<E3<E5, E2>E4."""
    if len(exts) < 5:
        return None
    e1, e2, e3, e4, e5 = exts[-5:]
    if not e1.is_max:
        return None
    if not (e1.value < e3.value < e5.value):
        return None
    if not (e2.value > e4.value):
        return None
    return PatternMatch(PatternType.BTOP, exts[-5:], e1.index, e5.index)


def _check_bbot(exts: List[Extremum]) -> Optional[PatternMatch]:
    """扩散底: E1=min, E1>E3>E5, E2<E4."""
    if len(exts) < 5:
        return None
    e1, e2, e3, e4, e5 = exts[-5:]
    if e1.is_max:
        return None
    if not (e1.value > e3.value > e5.value):
        return None
    if not (e2.value < e4.value):
        return None
    return PatternMatch(PatternType.BBOT, exts[-5:], e1.index, e5.index)


def _check_ttop(exts: List[Extremum]) -> Optional[PatternMatch]:
    """三角顶: E1=max, E1>E3>E5, E2<E4."""
    if len(exts) < 5:
        return None
    e1, e2, e3, e4, e5 = exts[-5:]
    if not e1.is_max:
        return None
    if not (e1.value > e3.value > e5.value):
        return None
    if not (e2.value < e4.value):
        return None
    return PatternMatch(PatternType.TTOP, exts[-5:], e1.index, e5.index)


def _check_tbot(exts: List[Extremum]) -> Optional[PatternMatch]:
    """三角底: E1=min, E1<E3<E5, E2>E4."""
    if len(exts) < 5:
        return None
    e1, e2, e3, e4, e5 = exts[-5:]
    if e1.is_max:
        return None
    if not (e1.value < e3.value < e5.value):
        return None
    if not (e2.value > e4.value):
        return None
    return PatternMatch(PatternType.TBOT, exts[-5:], e1.index, e5.index)


def _check_rtop(exts: List[Extremum], tol: float) -> Optional[PatternMatch]:
    """
    矩形顶: E1=max, tops 在均值 tol 内, bottoms 在均值 tol 内,
    lowest top > highest bottom.
    """
    if len(exts) < 5:
        return None
    e1, e2, e3, e4, e5 = exts[-5:]
    if not e1.is_max:
        return None
    tops = [e1.value, e3.value, e5.value]
    bots = [e2.value, e4.value]
    top_avg = np.mean(tops)
    bot_avg = np.mean(bots)
    if top_avg == 0 or bot_avg == 0:
        return None
    # 每个 top/bot 在均值 tol 内
    max_dev = 0.0
    for t in tops:
        dev = abs(t - top_avg) / abs(top_avg)
        if dev > tol:
            return None
        max_dev = max(max_dev, dev)
    for b in bots:
        dev = abs(b - bot_avg) / abs(bot_avg)
        if dev > tol:
            return None
        max_dev = max(max_dev, dev)
    # lowest top > highest bottom
    if min(tops) <= max(bots):
        return None
    conf = 1.0 - max_dev / tol
    return PatternMatch(PatternType.RTOP, exts[-5:], e1.index, e5.index, conf)


def _check_rbot(exts: List[Extremum], tol: float) -> Optional[PatternMatch]:
    """矩形底: 镜像 RTOP. E1=min."""
    if len(exts) < 5:
        return None
    e1, e2, e3, e4, e5 = exts[-5:]
    if e1.is_max:
        return None
    bots = [e1.value, e3.value, e5.value]
    tops = [e2.value, e4.value]
    top_avg = np.mean(tops)
    bot_avg = np.mean(bots)
    if top_avg == 0 or bot_avg == 0:
        return None
    max_dev = 0.0
    for t in tops:
        dev = abs(t - top_avg) / abs(top_avg)
        if dev > tol:
            return None
        max_dev = max(max_dev, dev)
    for b in bots:
        dev = abs(b - bot_avg) / abs(bot_avg)
        if dev > tol:
            return None
        max_dev = max(max_dev, dev)
    if min(tops) <= max(bots):
        return None
    conf = 1.0 - max_dev / tol
    return PatternMatch(PatternType.RBOT, exts[-5:], e1.index, e5.index, conf)


def _check_dtop(
    exts: List[Extremum], tol: float, min_gap: int,
) -> Optional[PatternMatch]:
    """
    双顶: E1=max, 在后续极大值中找 Ea 使得 E1/Ea 在均值 tol 内,
    且间隔 ≥ min_gap.
    """
    if len(exts) < 3:
        return None
    # 找所有极大值
    maxima = [e for e in exts if e.is_max]
    if len(maxima) < 2:
        return None
    # 取最后两个极大值
    ea, eb = maxima[-2], maxima[-1]
    if abs(eb.index - ea.index) < min_gap:
        return None
    ok, dev = _within_pct(ea.value, eb.value, tol)
    if not ok:
        return None
    conf = 1.0 - dev / tol
    used = [e for e in exts if ea.index <= e.index <= eb.index]
    return PatternMatch(PatternType.DTOP, used, ea.index, eb.index, conf)


def _check_dbot(
    exts: List[Extremum], tol: float, min_gap: int,
) -> Optional[PatternMatch]:
    """双底: 镜像 DTOP."""
    if len(exts) < 3:
        return None
    minima = [e for e in exts if not e.is_max]
    if len(minima) < 2:
        return None
    ea, eb = minima[-2], minima[-1]
    if abs(eb.index - ea.index) < min_gap:
        return None
    ok, dev = _within_pct(ea.value, eb.value, tol)
    if not ok:
        return None
    conf = 1.0 - dev / tol
    used = [e for e in exts if ea.index <= e.index <= eb.index]
    return PatternMatch(PatternType.DBOT, used, ea.index, eb.index, conf)


def match_patterns(
    extrema: List[Extremum],
    tolerance_pct: float = 0.015,
    rect_tolerance_pct: float = 0.0075,
    min_dtop_gap: int = 22,
) -> List[PatternMatch]:
    """
    遍历极值序列, 检查所有 10 种形态定义.

    Args:
        extrema: 交替排列的极值列表
        tolerance_pct: HS/IHS/DTOP/DBOT 的对称容差 (1.5%)
        rect_tolerance_pct: RTOP/RBOT 的容差 (0.75%)
        min_dtop_gap: 双顶/双底最小间隔 (天)

    Returns:
        匹配到的形态列表 (可能多个).
    """
    matches: List[PatternMatch] = []
    if len(extrema) < 3:
        return matches

    checkers_5 = [
        lambda e: _check_hs(e, tolerance_pct),
        lambda e: _check_ihs(e, tolerance_pct),
        lambda e: _check_btop(e),
        lambda e: _check_bbot(e),
        lambda e: _check_ttop(e),
        lambda e: _check_tbot(e),
        lambda e: _check_rtop(e, rect_tolerance_pct),
        lambda e: _check_rbot(e, rect_tolerance_pct),
    ]

    # 5-极值形态
    if len(extrema) >= 5:
        for checker in checkers_5:
            m = checker(extrema)
            if m is not None:
                matches.append(m)

    # 双顶/双底 (不需要严格 5 极值)
    m = _check_dtop(extrema, tolerance_pct, min_dtop_gap)
    if m is not None:
        matches.append(m)
    m = _check_dbot(extrema, tolerance_pct, min_dtop_gap)
    if m is not None:
        matches.append(m)

    return matches


# ═══════════════════════════════════════════════════════════════
# Layer 4: 滚动检测
# ═══════════════════════════════════════════════════════════════

def detect_patterns_single(
    prices: np.ndarray,
    window: int = 35,
    lag: int = 3,
    bandwidth: Union[float, str] = 3.0,
    bandwidth_factor: float = 0.3,
) -> List[Tuple[int, PatternMatch]]:
    """
    对单只股票的价格序列做滚动形态检测.

    Args:
        prices: 1D 价格数组
        window: 形态窗口长度 (论文: 35)
        lag: 检测延迟 (论文: 3)
        bandwidth: 'cv' 或固定数值 (默认 3.0, 更快)
        bandwidth_factor: 带宽缩放因子

    Returns:
        [(detection_date_index, PatternMatch), ...]
    """
    total = window + lag
    n = len(prices)
    if n < total:
        return []

    results: List[Tuple[int, PatternMatch]] = []

    for t in range(n - total + 1):
        segment = prices[t: t + total]
        smoothed = kernel_smooth(segment, bandwidth=bandwidth,
                                 bandwidth_factor=bandwidth_factor)
        extrema = find_extrema(segment, smoothed)
        if len(extrema) < 3:
            continue
        matches = match_patterns(extrema)
        detection_idx = t + total - 1
        for m in matches:
            # 调整极值索引到全局坐标
            shifted = PatternMatch(
                pattern=m.pattern,
                extrema=[Extremum(e.index + t, e.value, e.is_max) for e in m.extrema],
                start_idx=m.start_idx + t,
                end_idx=m.end_idx + t,
                confidence=m.confidence,
            )
            results.append((detection_idx, shifted))

    return results


def detect_patterns_panel(
    close: pd.DataFrame,
    window: int = 35,
    lag: int = 3,
    bandwidth: Union[float, str] = 3.0,
    bandwidth_factor: float = 0.3,
) -> Dict[str, pd.DataFrame]:
    """
    批量检测: 对 close 中每只股票做滚动检测.

    Returns:
        {PatternType.value: dates × symbols (1=检测到, 0=未检测)}
    """
    dates = close.index
    symbols = close.columns
    n_dates = len(dates)

    # 初始化: 每种形态一个全 0 面板
    panels: Dict[str, np.ndarray] = {}
    for pt in PatternType:
        panels[pt.value] = np.zeros((n_dates, len(symbols)), dtype=np.float32)

    for j, sym in enumerate(symbols):
        col = close[sym].values
        # 跳过全 NaN
        valid = ~np.isnan(col)
        if valid.sum() < window + lag:
            continue
        # 用 ffill + bfill 填充 NaN (价格序列中的缺失)
        filled = pd.Series(col).ffill().bfill().values
        detections = detect_patterns_single(
            filled, window=window, lag=lag,
            bandwidth=bandwidth, bandwidth_factor=bandwidth_factor,
        )
        for det_idx, pm in detections:
            if 0 <= det_idx < n_dates:
                panels[pm.pattern.value][det_idx, j] = 1.0

    result: Dict[str, pd.DataFrame] = {}
    for pt in PatternType:
        result[pt.value] = pd.DataFrame(
            panels[pt.value], index=dates, columns=symbols,
        )
    return result


# ═══════════════════════════════════════════════════════════════
# Layer 5: 因子化
# ═══════════════════════════════════════════════════════════════

def compute_pattern_factors(
    close: pd.DataFrame,
    window: int = 35,
    lag: int = 3,
    bandwidth: Union[float, str] = 3.0,
    bandwidth_factor: float = 0.3,
    decay_halflife: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    将形态检测结果转化为连续截面因子.

    对每种形态 p, 每只股票 i, 每天 t:
      raw[i,t] = 1 if 检测到 p at day t, else 0
      factor[i,t] = Σ_{s≤t} raw[i,s] × exp(-(t-s) × ln2 / halflife)

    合成:
      pattern_bullish = Σ (看涨形态因子)
      pattern_bearish = Σ (看跌形态因子)
      pattern_net     = bullish - bearish

    Args:
        close: dates × symbols 收盘价面板
        window: 形态窗口
        lag: 检测延迟
        bandwidth: 带宽 ('cv' 或固定值)
        bandwidth_factor: 带宽缩放
        decay_halflife: 指数衰减半衰期 (天)

    Returns:
        {'pattern_bullish': ..., 'pattern_bearish': ..., 'pattern_net': ...}
        每个为 dates × symbols DataFrame
    """
    raw_panels = detect_patterns_panel(
        close, window=window, lag=lag,
        bandwidth=bandwidth, bandwidth_factor=bandwidth_factor,
    )

    decay = np.log(2) / max(decay_halflife, 1)
    n_dates = len(close.index)

    def _apply_decay(raw: np.ndarray) -> np.ndarray:
        """EWM-like 指数衰减累积."""
        out = np.zeros_like(raw, dtype=np.float64)
        out[0] = raw[0]
        alpha = np.exp(-decay)
        for t in range(1, n_dates):
            out[t] = out[t - 1] * alpha + raw[t]
        return out

    bullish_acc = np.zeros((n_dates, len(close.columns)), dtype=np.float64)
    bearish_acc = np.zeros((n_dates, len(close.columns)), dtype=np.float64)

    for pt in PatternType:
        raw = raw_panels[pt.value].values
        decayed = _apply_decay(raw)
        if pt in BULLISH_PATTERNS:
            bullish_acc += decayed
        else:
            bearish_acc += decayed

    idx, cols = close.index, close.columns
    return {
        'pattern_bullish': pd.DataFrame(bullish_acc, index=idx, columns=cols),
        'pattern_bearish': pd.DataFrame(bearish_acc, index=idx, columns=cols),
        'pattern_net': pd.DataFrame(bullish_acc - bearish_acc, index=idx, columns=cols),
    }
