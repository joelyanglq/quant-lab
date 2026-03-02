"""
HHT / QRS 信号计算引擎

从 check_signals.py 提取的独立信号计算模块，供 advisor 和其他模块复用。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import hilbert


@dataclass
class SignalResult:
    """单个策略对单个标的的信号结果"""
    symbol: str
    strategy: str       # "HHT" / "QRS"
    signal: int         # HHT: 1=BUY, 0=FLAT, -1=N/A | QRS: 1=BUY, -1=SELL, 0=HOLD
    label: str          # "BUY" / "FLAT" / "SELL" / "HOLD" / "N/A"
    raw_value: Optional[float]    # HHT: phase | QRS: signal_val (zscore * R^2)
    data_date: Optional[pd.Timestamp]


# ── HHT ─────────────────────────────────────────────────────

def compute_hht(
    closes: np.ndarray,
    ma_period: int = 60,
    ht_period: int = 30,
) -> Tuple[int, Optional[float]]:
    """
    计算 HHT 二值信号

    Returns:
        (signal, phase)
        signal: 1=BUY, 0=FLAT, -1=数据不足
        phase: 当前 Hilbert 相位 (radians), None if 数据不足
    """
    if len(closes) < ma_period + ht_period:
        return -1, None

    ma = np.convolve(closes, np.ones(ma_period) / ma_period, mode='valid')
    diff = np.diff(ma)
    if len(diff) < ht_period:
        return -1, None

    window = diff[-ht_period:]
    analytic = hilbert(window)
    phase = np.angle(analytic)
    current_phase = float(phase[-1])

    if -np.pi / 2 <= current_phase <= np.pi / 2:
        return 1, current_phase   # 上升周期 → 做多
    else:
        return 0, current_phase   # 下降周期 → 空仓


# ── QRS ─────────────────────────────────────────────────────

def compute_qrs(
    highs: np.ndarray,
    lows: np.ndarray,
    regression_window: int = 18,
    zscore_window: int = 250,
    upper: float = 0.7,
    lower: float = -0.7,
) -> Tuple[int, Optional[float], Optional[float]]:
    """
    计算 QRS 信号

    Returns:
        (signal_code, signal_value, zscore)
        signal_code: 1=BUY, -1=SELL, 0=HOLD, -99=数据不足
        signal_value: zscore * R^2
        zscore: 原始 z-score
    """
    n = len(highs)
    if n < regression_window + zscore_window:
        return -99, None, None

    # 计算全部 beta 序列
    betas = []
    for i in range(regression_window - 1, n):
        h = highs[i - regression_window + 1:i + 1]
        l = lows[i - regression_window + 1:i + 1]
        std_h = np.std(h)
        std_l = np.std(l)
        if std_l == 0:
            betas.append(np.nan)
            continue
        corr = np.corrcoef(l, h)[0, 1]
        betas.append(std_h / std_l * corr)

    betas = np.array(betas)
    if len(betas) < zscore_window:
        return -99, None, None

    # z-score
    window = betas[-zscore_window:]
    mean = np.nanmean(window)
    std = np.nanstd(window)
    if std == 0:
        return -99, None, None
    zscore = float((window[-1] - mean) / std)

    # R^2
    h_last = highs[-regression_window:]
    l_last = lows[-regression_window:]
    corr = np.corrcoef(l_last, h_last)[0, 1]
    r2 = corr ** 2

    signal_val = float(zscore * r2)

    if signal_val > upper:
        return 1, signal_val, zscore    # BUY
    elif signal_val < lower:
        return -1, signal_val, zscore   # SELL
    else:
        return 0, signal_val, zscore    # HOLD


# ── 批量计算 ────────────────────────────────────────────────

def compute_all_signals(
    symbols: List[str],
    storage,  # ParquetStorage
    hht_params: Optional[dict] = None,
    qrs_params: Optional[dict] = None,
) -> Dict[str, Dict[str, SignalResult]]:
    """
    批量计算全部标的的 HHT + QRS 信号

    Args:
        symbols: 标的列表
        storage: ParquetStorage 实例
        hht_params: HHT 参数 (ma_period, ht_period)
        qrs_params: QRS 参数 (regression_window, zscore_window, upper, lower)

    Returns:
        {symbol: {"HHT": SignalResult, "QRS": SignalResult}}
    """
    hht_p = {'ma_period': 60, 'ht_period': 30}
    if hht_params:
        hht_p.update(hht_params)

    qrs_p = {'regression_window': 18, 'zscore_window': 250,
             'upper': 0.7, 'lower': -0.7}
    if qrs_params:
        qrs_p.update(qrs_params)

    # 加载足够长的历史数据（2 年）
    start = pd.Timestamp.now() - pd.DateOffset(years=2)
    end = pd.Timestamp('2099-12-31')

    results: Dict[str, Dict[str, SignalResult]] = {}

    for sym in symbols:
        sym_results = {}
        try:
            df = storage.load([sym], start, end)
            df = df.sort_index()
            data_date = df.index[-1] if len(df) > 0 else None

            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values

            # HHT
            hht_sig, hht_phase = compute_hht(
                closes, hht_p['ma_period'], hht_p['ht_period'])
            if hht_sig == -1:
                sym_results['HHT'] = SignalResult(
                    sym, 'HHT', -1, 'N/A', None, data_date)
            else:
                label = 'BUY' if hht_sig == 1 else 'FLAT'
                sym_results['HHT'] = SignalResult(
                    sym, 'HHT', hht_sig, label, hht_phase, data_date)

            # QRS
            qrs_code, qrs_val, qrs_z = compute_qrs(
                highs, lows,
                qrs_p['regression_window'], qrs_p['zscore_window'],
                qrs_p['upper'], qrs_p['lower'])
            if qrs_code == -99:
                sym_results['QRS'] = SignalResult(
                    sym, 'QRS', 0, 'N/A', None, data_date)
            else:
                label_map = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}
                sym_results['QRS'] = SignalResult(
                    sym, 'QRS', qrs_code, label_map[qrs_code],
                    qrs_val, data_date)

        except Exception:
            sym_results['HHT'] = SignalResult(sym, 'HHT', -1, 'N/A', None, None)
            sym_results['QRS'] = SignalResult(sym, 'QRS', 0, 'N/A', None, None)

        results[sym] = sym_results

    return results
