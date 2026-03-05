"""
事件信号研究管线 (Event Signal Research Pipeline)

通用四阶段流程, 从任意事件信号出发, 系统性挖掘可交易 alpha:

  Phase 1: 信号扫描 — 多变体 × 多持有期 → 热力图, 识别有效组合
  Phase 2: 条件分解 — regime / 行业 / 市值 → 一致性检验, 定位有效区间
  Phase 3: 正交性检验 — double sort 控制已知因子 → 确认独立性
  Phase 4: 信号构建 — 综合 Phase 1-3 → 条件过滤 → 可交易信号

Usage:
    from strategies.timing.event_research import run_full_pipeline
    report = run_full_pipeline(events, close, spy_close, ...)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ═══════════════════════════════════════════════════════════════
# Phase 1: 信号扫描
# ═══════════════════════════════════════════════════════════════

@dataclass
class ScanResult:
    """单个事件变体 × 持有期的统计."""
    name: str
    holding: int
    n_events: int
    cond_mean: float     # 条件收益
    uncond_mean: float   # 无条件收益
    excess_bps: float    # 超额 (bps)
    t_stat: float
    win_rate: float


def _forward_returns(close: pd.DataFrame, H: int) -> pd.DataFrame:
    """Entry close[t+1], exit close[t+1+H]."""
    entry = close.shift(-1)
    exit_ = close.shift(-(1 + H))
    return exit_ / entry - 1


def _event_stats(
    fwd_ret: pd.DataFrame,
    det: pd.DataFrame,
    uncond_mean: float,
) -> dict:
    """Compute event conditional return stats."""
    flat = fwd_ret[det == 1].values.flatten()
    flat = flat[~np.isnan(flat)]
    n = len(flat)
    if n < 30:
        return {'n': n, 'cond_mean': np.nan, 'excess_bps': np.nan,
                't_stat': np.nan, 'win_rate': np.nan}
    mean_r = np.mean(flat)
    excess = mean_r - uncond_mean
    se = np.std(flat) / np.sqrt(n)
    t = excess / se if se > 0 else 0
    return {
        'n': n, 'cond_mean': mean_r,
        'excess_bps': excess * 10000,
        't_stat': t,
        'win_rate': float(np.mean(flat > 0)),
    }


def phase1_scan(
    events: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    holdings: List[int] = None,
) -> pd.DataFrame:
    """
    Phase 1: 多变体 × 多持有期扫描.

    Args:
        events: {name: dates × symbols binary panel (1=event, 0=no)}
        close: 收盘价面板
        holdings: 持有期列表 (默认 [1,5,10,20])

    Returns:
        DataFrame with columns: name, holding, n_events, excess_bps, t_stat, ...
    """
    if holdings is None:
        holdings = [1, 5, 10, 20]

    fwd_cache = {H: _forward_returns(close, H) for H in holdings}
    uncond_cache = {}
    for H in holdings:
        flat = fwd_cache[H].values.flatten()
        uncond_cache[H] = float(np.nanmean(flat))

    rows = []
    for name, det in events.items():
        for H in holdings:
            s = _event_stats(fwd_cache[H], det, uncond_cache[H])
            rows.append(ScanResult(
                name=name, holding=H, n_events=s['n'],
                cond_mean=s.get('cond_mean', np.nan),
                uncond_mean=uncond_cache[H],
                excess_bps=s.get('excess_bps', np.nan),
                t_stat=s.get('t_stat', np.nan),
                win_rate=s.get('win_rate', np.nan),
            ))

    return pd.DataFrame([r.__dict__ for r in rows])


# ═══════════════════════════════════════════════════════════════
# Phase 2: 条件分解
# ═══════════════════════════════════════════════════════════════

def build_regimes(
    spy_close: pd.Series,
    close: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """构建标准 regime 划分."""
    idx = close.index
    spy = spy_close.reindex(idx).ffill()
    spy_ret = spy.pct_change()
    regimes = {}

    # Trend: SPY vs MA200
    ma200 = spy.rolling(200, min_periods=100).mean()
    trend = pd.Series(index=idx, dtype=str)
    trend[spy > ma200] = 'Bull'
    trend[spy <= ma200] = 'Bear'
    trend[ma200.isna()] = np.nan
    regimes['Trend'] = trend

    # Volatility
    rvol = spy_ret.rolling(60, min_periods=30).std() * np.sqrt(252)
    vol_med = rvol.expanding(60).median()
    vol = pd.Series(index=idx, dtype=str)
    vol[rvol <= vol_med] = 'LowVol'
    vol[rvol > vol_med] = 'HighVol'
    vol[rvol.isna()] = np.nan
    regimes['Volatility'] = vol

    return regimes


def phase2_decompose(
    events: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    H: int = 20,
    regimes: Optional[Dict[str, pd.Series]] = None,
    sector_map: Optional[pd.Series] = None,
    mktcap: Optional[pd.DataFrame] = None,
    n_cap_groups: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Phase 2: 条件分解.

    对每个维度, 返回 events × groups 的超额收益矩阵.

    Returns:
        {dimension_name: DataFrame(index=event_names, columns=groups)}
    """
    fwd_ret = _forward_returns(close, H)
    results = {}

    # Helper
    def _compute_group_excess(det, fwd, group_labels):
        """Compute excess per group."""
        groups = sorted(group_labels.dropna().unique())
        row = {}
        for g in groups:
            mask = group_labels == g
            mask_dates = mask[mask].index
            fwd_g = fwd.loc[fwd.index.isin(mask_dates)]
            det_g = det.loc[det.index.isin(mask_dates)]
            uc_flat = fwd_g.values.flatten()
            uc_flat = uc_flat[~np.isnan(uc_flat)]
            uc_mean = np.mean(uc_flat) if len(uc_flat) > 0 else 0
            s = _event_stats(fwd_g, det_g, uc_mean)
            row[g] = {'excess_bps': s['excess_bps'], 't_stat': s['t_stat'], 'n': s['n']}
        return row

    # Regime decomposition
    if regimes:
        for rname, rlabels in regimes.items():
            rows_ex = {}
            rows_t = {}
            for ename, det in events.items():
                grp = _compute_group_excess(det, fwd_ret, rlabels)
                rows_ex[ename] = {g: v['excess_bps'] for g, v in grp.items()}
                rows_t[ename] = {g: v['t_stat'] for g, v in grp.items()}
            results[f'{rname}_excess'] = pd.DataFrame(rows_ex).T
            results[f'{rname}_tstat'] = pd.DataFrame(rows_t).T

    # Sector decomposition
    if sector_map is not None:
        rows_ex = {}
        rows_t = {}
        for ename, det in events.items():
            grp_data = {}
            for sec in sorted(sector_map.unique()):
                sec_syms = sector_map[sector_map == sec].index.tolist()
                sec_syms = [s for s in sec_syms if s in close.columns]
                if len(sec_syms) < 10:
                    continue
                fwd_sec = fwd_ret[sec_syms]
                det_sec = det[sec_syms] if all(s in det.columns for s in sec_syms) else det.reindex(columns=sec_syms, fill_value=0)
                uc = np.nanmean(fwd_sec.values.flatten())
                s = _event_stats(fwd_sec, det_sec, uc)
                grp_data[sec] = s
            rows_ex[ename] = {g: v['excess_bps'] for g, v in grp_data.items()}
            rows_t[ename] = {g: v['t_stat'] for g, v in grp_data.items()}
        results['Sector_excess'] = pd.DataFrame(rows_ex).T
        results['Sector_tstat'] = pd.DataFrame(rows_t).T

    # Market cap quintile
    if mktcap is not None:
        cap_q = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        for dt in close.index:
            mc = mktcap.loc[dt].dropna()
            if len(mc) < 50:
                continue
            try:
                q = pd.qcut(mc, n_cap_groups, labels=range(1, n_cap_groups + 1), duplicates='drop')
                cap_q.loc[dt, q.index] = q.astype(float)
            except ValueError:
                continue

        rows_ex = {}
        rows_t = {}
        for ename, det in events.items():
            grp_data = {}
            for qi in range(1, n_cap_groups + 1):
                mask = (cap_q == qi)
                fwd_q = fwd_ret[mask]
                det_q = det[mask].fillna(0)
                uc = np.nanmean(fwd_q.values.flatten())
                s = _event_stats(fwd_q, det_q, uc)
                label = f'Q{qi}'
                grp_data[label] = s
            rows_ex[ename] = {g: v['excess_bps'] for g, v in grp_data.items()}
            rows_t[ename] = {g: v['t_stat'] for g, v in grp_data.items()}
        results['MktCap_excess'] = pd.DataFrame(rows_ex).T
        results['MktCap_tstat'] = pd.DataFrame(rows_t).T

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 3: 正交性检验 (Double Sort)
# ═══════════════════════════════════════════════════════════════

def build_control_factors(close: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """构建标准控制因子."""
    ret = close.pct_change()
    return {
        'Mom_12M': close.shift(21) / close.shift(252) - 1,
        'Rev_5d': ret.rolling(5).sum(),
        'Rev_21d': ret.rolling(21).sum(),
        'Vol_60d': ret.rolling(60, min_periods=30).std(),
    }


def phase3_double_sort(
    events: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    H: int = 20,
    controls: Optional[Dict[str, pd.DataFrame]] = None,
    n_quintiles: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Phase 3: 对每个事件 × 每个控制因子, 做 double sort.

    Returns:
        {event_name: DataFrame(index=control_names, columns=Q1..Q5, values=excess_bps)}
    """
    if controls is None:
        controls = build_control_factors(close)

    fwd_ret = _forward_returns(close, H)
    results = {}

    for ename, det in events.items():
        rows_ex = {}
        rows_t = {}
        for cname, cpanel in controls.items():
            q_excess = {}
            q_tstat = {}
            for q in range(1, n_quintiles + 1):
                cond_rets = []
                uncond_rets = []
                for dt in close.index[::1]:  # every day
                    ctrl = cpanel.loc[dt].dropna()
                    if len(ctrl) < 50:
                        continue
                    try:
                        quintiles = pd.qcut(ctrl, n_quintiles,
                                            labels=range(1, n_quintiles + 1),
                                            duplicates='drop')
                    except ValueError:
                        continue
                    q_syms = quintiles[quintiles == q].index
                    uc = fwd_ret.loc[dt, q_syms].dropna()
                    uncond_rets.extend(uc.values.tolist())
                    det_syms = det.loc[dt, q_syms]
                    hit = det_syms[det_syms == 1].index
                    if len(hit) > 0:
                        cr = fwd_ret.loc[dt, hit].dropna()
                        cond_rets.extend(cr.values.tolist())

                uc_arr = np.array(uncond_rets)
                cd_arr = np.array(cond_rets)
                uc_mean = np.mean(uc_arr) if len(uc_arr) > 0 else 0
                ql = f'Q{q}'
                if len(cd_arr) >= 30:
                    ex = (np.mean(cd_arr) - uc_mean) * 10000
                    se = np.std(cd_arr) / np.sqrt(len(cd_arr))
                    t = (np.mean(cd_arr) - uc_mean) / se if se > 0 else 0
                    q_excess[ql] = ex
                    q_tstat[ql] = t
                else:
                    q_excess[ql] = np.nan
                    q_tstat[ql] = np.nan

            rows_ex[cname] = q_excess
            rows_t[cname] = q_tstat

        results[f'{ename}_excess'] = pd.DataFrame(rows_ex).T
        results[f'{ename}_tstat'] = pd.DataFrame(rows_t).T

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 4: 信号构建
# ═══════════════════════════════════════════════════════════════

@dataclass
class TradingSignal:
    """可交易信号."""
    name: str
    direction: int                           # +1 long, -1 short
    holding_period: int                      # days
    entry_panel: pd.DataFrame                # dates × symbols, 1=entry
    conditions: List[str] = field(default_factory=list)


def phase4_construct(
    events: Dict[str, pd.DataFrame],
    directions: Dict[str, int],
    close: pd.DataFrame,
    holding: int = 20,
    regime_filter: Optional[pd.Series] = None,
    regime_values: Optional[List[str]] = None,
) -> List[TradingSignal]:
    """
    Phase 4: 组装可交易信号.

    Args:
        events: {name: binary panel}
        directions: {name: +1 or -1}
        holding: 持有期
        regime_filter: 可选, regime Series
        regime_values: 可选, 只在这些 regime 下触发

    Returns:
        TradingSignal list
    """
    signals = []
    for name, det in events.items():
        direction = directions.get(name, +1)
        entry = det.copy()

        conditions = [f'Event: {name}', f'Direction: {"Long" if direction > 0 else "Short"}']

        if regime_filter is not None and regime_values is not None:
            regime_mask = regime_filter.isin(regime_values)
            for dt in entry.index:
                if dt not in regime_mask.index or not regime_mask.loc[dt]:
                    entry.loc[dt] = 0
            conditions.append(f'Regime filter: {regime_values}')

        signals.append(TradingSignal(
            name=name, direction=direction,
            holding_period=holding, entry_panel=entry,
            conditions=conditions,
        ))

    return signals


def backtest_signals(
    signals: List[TradingSignal],
    close: pd.DataFrame,
    non_overlapping: bool = True,
) -> pd.DataFrame:
    """
    回测信号组合.

    对每个信号:
      entry dates = panel 中 1 的位置
      return = close[t+1+H] / close[t+1] - 1 × direction

    合成: 等权 L/S

    Returns:
        DataFrame with columns: date, long_ret, short_ret, ls_ret, ...
    """
    long_sigs = [s for s in signals if s.direction > 0]
    short_sigs = [s for s in signals if s.direction < 0]

    H = signals[0].holding_period if signals else 20
    fwd_ret = _forward_returns(close, H)

    # Combine entries
    long_det = None
    for s in long_sigs:
        if long_det is None:
            long_det = s.entry_panel.copy()
        else:
            long_det = (long_det + s.entry_panel).clip(upper=1)

    short_det = None
    for s in short_sigs:
        if short_det is None:
            short_det = s.entry_panel.copy()
        else:
            short_det = (short_det + s.entry_panel).clip(upper=1)

    dates = close.index
    step = H if non_overlapping else 1
    records = []

    for i in range(0, len(dates), step):
        dt = dates[i]
        long_r = np.nan
        short_r = np.nan
        long_n = 0
        short_n = 0

        if long_det is not None:
            lr = fwd_ret.loc[dt][long_det.loc[dt] == 1].dropna()
            if len(lr) >= 1:
                long_r = lr.mean()
                long_n = len(lr)

        if short_det is not None:
            sr = fwd_ret.loc[dt][short_det.loc[dt] == 1].dropna()
            if len(sr) >= 1:
                short_r = sr.mean()
                short_n = len(sr)

        ls_r = np.nan
        if not np.isnan(long_r) and not np.isnan(short_r):
            ls_r = long_r - short_r
        elif not np.isnan(long_r):
            ls_r = long_r
        elif not np.isnan(short_r):
            ls_r = -short_r

        if not np.isnan(ls_r):
            records.append({
                'date': dt, 'long_ret': long_r, 'short_ret': short_r,
                'ls_ret': ls_r, 'long_n': long_n, 'short_n': short_n,
            })

    return pd.DataFrame(records).set_index('date') if records else pd.DataFrame()


def compute_signal_metrics(bt: pd.DataFrame, ann_factor: int = 12) -> dict:
    """Compute standard performance metrics from backtest results."""
    if bt.empty:
        return {}
    ls = bt['ls_ret'].dropna()
    if len(ls) < 3:
        return {}

    cum = (1 + ls).cumprod()
    dd = cum / cum.cummax() - 1
    ann_ret = ls.mean() * ann_factor
    ann_vol = ls.std() * np.sqrt(ann_factor)

    return {
        'n_periods': len(ls),
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': ann_ret / ann_vol if ann_vol > 0 else 0,
        'max_dd': float(dd.min()),
        'win_rate': float((ls > 0).mean()),
        'avg_long_n': bt['long_n'].mean(),
        'avg_short_n': bt['short_n'].mean(),
        'calmar': ann_ret / abs(dd.min()) if dd.min() != 0 else 0,
    }
