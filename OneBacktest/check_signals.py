"""
检查 HHT / QRS 策略最新信号
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from collections import deque
import numpy as np
from scipy.signal import hilbert
import pandas as pd
from data.storage.parquet import ParquetStorage

DATA_DIR = r'D:\04_Project\quant-lab\data\processed\bars_1d'
SYMBOLS = ['SPY', 'QQQ']

# ── HHT 信号计算 ────────────────────────────────────────────
def hht_signal_series(closes: np.ndarray, ma_period=60, ht_period=30):
    """返回最近几天的 HHT 信号 (1=多, 0=空)"""
    if len(closes) < ma_period + ht_period:
        return None, None
    ma = np.convolve(closes, np.ones(ma_period) / ma_period, mode='valid')
    diff = np.diff(ma)
    if len(diff) < ht_period:
        return None, None
    window = diff[-ht_period:]
    analytic = hilbert(window)
    phase = np.angle(analytic)
    current_phase = phase[-1]
    signal = 1 if -np.pi/2 <= current_phase <= np.pi/2 else 0
    return signal, current_phase

# ── QRS 信号计算 ────────────────────────────────────────────
def qrs_signal_series(highs: np.ndarray, lows: np.ndarray,
                      regression_window=18, zscore_window=250):
    """返回最新 QRS 信号值"""
    n = len(highs)
    if n < regression_window + zscore_window:
        return None, None

    # 计算所有 beta
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
        return None, None

    # 最后 zscore_window 个 beta 做 z-score
    window = betas[-zscore_window:]
    mean = np.nanmean(window)
    std = np.nanstd(window)
    if std == 0:
        return None, None
    zscore = (window[-1] - mean) / std

    # R^2
    h_last = highs[-regression_window:]
    l_last = lows[-regression_window:]
    corr = np.corrcoef(l_last, h_last)[0, 1]
    r2 = corr ** 2

    signal_val = zscore * r2
    return signal_val, zscore


def main():
    storage = ParquetStorage(DATA_DIR)
    # 加载足够长的历史 (2年多)
    start = pd.Timestamp('2024-01-01')
    end = pd.Timestamp('2026-12-31')

    print(f"{'Symbol':<8} | {'HHT Signal':^14} {'Phase':>8} | {'QRS Signal':^14} {'ZScore':>8} | {'QRS Action':^12} | {'Last Date'}")
    print("-" * 85)

    for sym in SYMBOLS:
        try:
            df = storage.load([sym], start, end)
            df = df.sort_index()
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            last_date = df.index[-1]
            last_close = closes[-1]

            # HHT
            hht_sig, hht_phase = hht_signal_series(closes, ma_period=60, ht_period=30)
            if hht_sig is not None:
                hht_label = "BUY (多)" if hht_sig == 1 else "FLAT (空)"
                hht_phase_str = f"{hht_phase:.3f}"
            else:
                hht_label = "N/A"
                hht_phase_str = "N/A"

            # QRS
            qrs_sig, qrs_z = qrs_signal_series(highs, lows,
                                                 regression_window=18,
                                                 zscore_window=250)
            if qrs_sig is not None:
                upper, lower = 0.7, -0.7
                if qrs_sig > upper:
                    qrs_action = "BUY (多)"
                elif qrs_sig < lower:
                    qrs_action = "SELL (空)"
                else:
                    qrs_action = "HOLD"
                qrs_sig_str = f"{qrs_sig:+.3f}"
                qrs_z_str = f"{qrs_z:+.3f}"
            else:
                qrs_action = "N/A"
                qrs_sig_str = "N/A"
                qrs_z_str = "N/A"

            print(f"{sym:<8} | {hht_label:^14} {hht_phase_str:>8} | {qrs_sig_str:^14} {qrs_z_str:>8} | {qrs_action:^12} | {last_date}")

        except Exception as e:
            print(f"{sym:<8} | ERROR: {e}")

    # 显示数据截止日期
    df_check = storage.load(SYMBOLS[:1], start, end)
    print(f"\nData up to: {df_check.index.max().date()}")


if __name__ == '__main__':
    main()
