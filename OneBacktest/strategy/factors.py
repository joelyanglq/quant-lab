"""
因子库 — 46 个因子, 适配 event-driven 策略框架
====================================================

所有 compute_* 函数 **接受面板数据作为参数**, 不做数据加载。
策略在 on_week_end / on_month_end 中从 self.history.panel() 取数后传入。

用法:
    from strategy.factors import compute_technical_factors, cross_sectional_zscore

分类            因子名                        数据源
────────────────────────────────────────────────────────────
技术(5)         RS_12M, RS_6M, Range_52W,     1d
                RSI_28W, RSI_16W
反转(4)         interday/intraday/overnight    1d
                _rev_volflip, team_coin
基本面(8)       ROE, ROIC, EV_EBITDA, ...     fund
微观结构(3)     go_with_flow, lone_goose,     1min+1d
                sailing
博弈(5)         vol_battle_ret/pos/battle,    1min
                amp_battle, bull_bear_battle
激增(3)         weekly_dazzling_vol/ret,      1min
                moderate_risk
回归(4)         morning_mist, noon_shade,     1min
                night_frost, flower_hidden
模糊(3)         fuzzy_corr, fuzzy_amount_     1min
                ratio, fuzzy_volume_ratio
灾后重建(2)     disaster_rebuild,             1min
                peak_climbing
潮汐(4)         full_tidal, strong_half,      1min
                aggressive_weak_half,
                stable_weak_half
跳跃(5)         weekly_jump,                  1min+1d
                modified_amplitude_1/2,
                modified_amplitude,
                moth_to_flame
"""
import datetime as _dt
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

RTH_START = _dt.time(9, 35)
RTH_END = _dt.time(15, 57)


# ═════════════════════════════════════════════════════════════
# 共用工具 (公开 API)
# ═════════════════════════════════════════════════════════════

def cross_sectional_zscore(
    factor: pd.DataFrame,
    winsorize_sigma: float = 3.0,
) -> pd.DataFrame:
    """横截面 z-score: MAD winsorize → (x-mean)/std"""
    def _mad_win(row, n_sigma):
        median = row.median()
        mad = (row - median).abs().median()
        if mad == 0:
            return row
        cutoff = n_sigma * 1.4826 * mad
        return row.clip(median - cutoff, median + cutoff)

    w = factor.apply(_mad_win, axis=1, n_sigma=winsorize_sigma)
    mu = w.mean(axis=1)
    sd = w.std(axis=1)
    return w.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)


def rolling_composite(
    daily_panel: pd.DataFrame,
    window: int = 5,
    negate: bool = False,
) -> pd.DataFrame:
    """rolling mean + std → z-score 等权, 可选取负"""
    min_p = max(window - 1, 3)
    rolling_mean = daily_panel.rolling(window, min_periods=min_p).mean()
    rolling_std = daily_panel.rolling(window, min_periods=min_p).std()

    valid_rows = rolling_mean.notna().any(axis=1)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        z_mean = cross_sectional_zscore(rolling_mean.loc[valid_rows])
        z_std = cross_sectional_zscore(rolling_std.loc[valid_rows])

    composite = (z_mean + z_std) / 2
    result = -composite if negate else composite
    return result.reindex(daily_panel.index)


def rolling_mean_zscore(
    daily_panel: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """rolling mean → z-score (无 std 成分)"""
    min_p = max(window - 1, 3)
    rolling_mean = daily_panel.rolling(window, min_periods=min_p).mean()

    valid_rows = rolling_mean.notna().any(axis=1)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        result = cross_sectional_zscore(rolling_mean.loc[valid_rows])

    return result.reindex(daily_panel.index)


def moderate_rolling(
    daily_panel: pd.DataFrame,
    window: int = 5,
    negate: bool = True,
) -> pd.DataFrame:
    """适度化 + rolling mean+std → z-score 等权"""
    cs_mean = daily_panel.mean(axis=1)
    moderate = daily_panel.sub(cs_mean, axis=0).abs()

    min_periods = max(window - 1, 3)
    r_mean = moderate.rolling(window, min_periods=min_periods).mean()
    r_std = moderate.rolling(window, min_periods=min_periods).std()

    valid_rows = r_mean.notna().any(axis=1)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        z_mean = cross_sectional_zscore(r_mean.loc[valid_rows])
        z_std = cross_sectional_zscore(r_std.loc[valid_rows])

    composite = (z_mean + z_std) / 2
    result = -composite if negate else composite
    return result.reindex(daily_panel.index)


# ═════════════════════════════════════════════════════════════
# 1. 技术因子 (5)  —  RS_12M, RS_6M, Range_52W, RSI_28W, RSI_16W
#    入参: close, high, low (1d 面板)
# ═════════════════════════════════════════════════════════════

def _rsi(close: pd.DataFrame, period: int) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_technical_factors(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    技术因子 (5): RS_12M, RS_6M, Range_52W, RSI_28W, RSI_16W.

    Args:
        close: 日线收盘面板 (dates × symbols)
        high: 日线最高价面板
        low: 日线最低价面板
    """
    rs_12m = close.shift(21) / close.shift(252) - 1
    rs_6m = close.shift(21) / close.shift(126) - 1
    high_252 = high.rolling(252, min_periods=126).max()
    low_252 = low.rolling(252, min_periods=126).min()
    range_52w = (close - low_252) / (high_252 - low_252).replace(0, np.nan)
    rsi_28w = _rsi(close, 140)
    rsi_16w = _rsi(close, 80)

    return {
        'RS_12M': rs_12m, 'RS_6M': rs_6m, 'Range_52W': range_52w,
        'RSI_28W': rsi_28w, 'RSI_16W': rsi_16w,
    }


# ═════════════════════════════════════════════════════════════
# 2. 反转因子 (4)  —  球队硬币
#    入参: close, open_price (1d 面板)
# ═════════════════════════════════════════════════════════════

def _volatility_flip(daily_panel: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    rolling_mean = daily_panel.rolling(window, min_periods=15).mean()
    rolling_std = daily_panel.rolling(window, min_periods=15).std()
    cs_vol_mean = rolling_std.mean(axis=1)
    is_coin = rolling_std.lt(cs_vol_mean, axis=0)
    factor = rolling_mean.copy()
    factor[is_coin] = -factor[is_coin]
    return factor


def compute_reversal_factors(
    close: pd.DataFrame,
    open_price: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    反转因子 (4): interday/intraday/overnight_rev_volflip, team_coin.

    Args:
        close: 日线收盘面板
        open_price: 日线开盘面板
    """
    interday_ret = close / close.shift(1) - 1
    intraday_ret = close / open_price - 1
    overnight_ret = open_price / close.shift(1) - 1
    overnight_dist = overnight_ret.sub(overnight_ret.mean(axis=1), axis=0).abs()

    f1 = _volatility_flip(interday_ret)
    f2 = _volatility_flip(intraday_ret)
    f3 = _volatility_flip(overnight_dist)

    z1 = cross_sectional_zscore(f1)
    z2 = cross_sectional_zscore(f2)
    z3 = cross_sectional_zscore(f3)

    return {
        'interday_rev_volflip': f1,
        'intraday_rev_volflip': f2,
        'overnight_rev_volflip': f3,
        'team_coin': (z1 + z2 + z3) / 3,
    }


# ═════════════════════════════════════════════════════════════
# 3. 基本面因子 (8)  —  ROE, ROIC, EV_EBITDA, FCF_Yield, PS,
#                       FCF_Growth, EPS_Score, Growth_Stability
#    入参: symbols, close (内部仍从磁盘读 quarterly 数据)
# ═════════════════════════════════════════════════════════════

FUNDAMENTAL_FIELDS = [
    'income_statement__net_income_loss',
    'income_statement__operating_income_loss',
    'income_statement__income_tax_expense_benefit',
    'income_statement__income_loss_from_continuing_operations_before_tax',
    'income_statement__revenues',
    'income_statement__diluted_earnings_per_share',
    'income_statement__diluted_average_shares',
    'balance_sheet__equity',
    'balance_sheet__assets',
    'balance_sheet__current_assets',
    'balance_sheet__current_liabilities',
    'balance_sheet__liabilities',
    'cash_flow_statement__net_cash_flow_from_operating_activities',
    'cash_flow_statement__net_cash_flow_from_investing_activities',
]


def _ols_slope_r2(x: np.ndarray, y: np.ndarray):
    mask = ~np.isnan(y)
    if mask.sum() < 4:
        return np.nan, np.nan
    xm, ym = x[mask], y[mask]
    xbar, ybar = xm.mean(), ym.mean()
    ss_xy = ((xm - xbar) * (ym - ybar)).sum()
    ss_xx = ((xm - xbar) ** 2).sum()
    ss_yy = ((ym - ybar) ** 2).sum()
    if ss_xx == 0 or ss_yy == 0:
        return np.nan, np.nan
    return ss_xy / ss_xx, (ss_xy ** 2) / (ss_xx * ss_yy)


def _prepare_quarterly(symbol: str):
    from data.fundamentals import _load_single_fundamental, _fix_filing_dates
    raw = _load_single_fundamental(symbol)
    if raw is None:
        return None
    qtr = raw[raw['timeframe'] == 'quarterly'].copy()
    if len(qtr) < 8:
        return None
    qtr = _fix_filing_dates(qtr)
    qtr['filing_date'] = pd.to_datetime(qtr['filing_date'])
    qtr['end_date'] = pd.to_datetime(qtr['end_date'])
    qtr = qtr.sort_values('end_date')
    qtr = qtr.drop_duplicates(subset=['fiscal_year', 'fiscal_period'], keep='last')
    return qtr


def _rolling_ols_to_panel(symbol, field, dates, output, window=8):
    qtr = _prepare_quarterly(symbol)
    if qtr is None:
        return
    qtr = qtr.dropna(subset=[field, 'filing_date'])
    if len(qtr) < window:
        return
    results, filing_dates = [], []
    x = np.arange(window, dtype=float)
    for i in range(window - 1, len(qtr)):
        y = qtr.iloc[i - window + 1: i + 1][field].values.astype(float)
        slope, r2 = _ols_slope_r2(x, y)
        results.append(slope * r2 if not np.isnan(slope) else np.nan)
        filing_dates.append(qtr.iloc[i]['filing_date'])
    if not results:
        return
    ts = pd.Series(results, index=pd.DatetimeIndex(filing_dates), dtype=float)
    ts = ts[~ts.index.duplicated(keep='last')].sort_index()
    output[symbol] = ts.reindex(dates, method='ffill')


def _rolling_yoy_r2_to_panel(symbol, field, dates, output, window=8):
    qtr = _prepare_quarterly(symbol)
    if qtr is None:
        return
    qtr = qtr.dropna(subset=[field, 'filing_date'])
    qtr = qtr.sort_values('end_date')
    vals = qtr[field].values.astype(float)
    yoy = np.full_like(vals, np.nan)
    for i in range(4, len(vals)):
        if vals[i - 4] != 0:
            yoy[i] = (vals[i] - vals[i - 4]) / abs(vals[i - 4])
    qtr = qtr.copy()
    qtr['yoy_growth'] = yoy
    valid = qtr.dropna(subset=['yoy_growth'])
    if len(valid) < window:
        return
    results, filing_dates = [], []
    x = np.arange(window, dtype=float)
    for i in range(window - 1, len(valid)):
        y = valid.iloc[i - window + 1: i + 1]['yoy_growth'].values.astype(float)
        _, r2 = _ols_slope_r2(x, y)
        results.append(r2)
        filing_dates.append(valid.iloc[i]['filing_date'])
    if not results:
        return
    ts = pd.Series(results, index=pd.DatetimeIndex(filing_dates), dtype=float)
    ts = ts[~ts.index.duplicated(keep='last')].sort_index()
    output[symbol] = ts.reindex(dates, method='ffill')


def compute_fundamental_factors(
    symbols: List[str],
    close: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    基本面因子 (8): ROE, ROIC, EV_EBITDA, FCF_Yield, PS,
    FCF_Growth, EPS_Score, Growth_Stability.

    Args:
        symbols: 股票列表
        close: 日线收盘面板 (内部从磁盘读 quarterly 数据)
    """
    from data.fundamentals import build_fundamental_panel

    dates = close.index
    panels = build_fundamental_panel(symbols, FUNDAMENTAL_FIELDS, dates)

    ni = panels['income_statement__net_income_loss']
    equity = panels['balance_sheet__equity']
    op_inc = panels['income_statement__operating_income_loss']
    tax = panels['income_statement__income_tax_expense_benefit']
    pretax = panels['income_statement__income_loss_from_continuing_operations_before_tax']
    assets = panels['balance_sheet__assets']
    cur_assets = panels['balance_sheet__current_assets']
    cur_liab = panels['balance_sheet__current_liabilities']
    liabilities = panels['balance_sheet__liabilities']
    revenue = panels['income_statement__revenues']
    shares = panels['income_statement__diluted_average_shares']
    op_cf = panels['cash_flow_statement__net_cash_flow_from_operating_activities']
    inv_cf = panels['cash_flow_statement__net_cash_flow_from_investing_activities']

    roe = ni / equity.replace(0, np.nan)

    eff_tax = (tax / pretax.replace(0, np.nan)).clip(0, 0.5)
    nopat = op_inc * (1 - eff_tax)
    invested_capital = assets - cur_liab
    roic = nopat / invested_capital.replace(0, np.nan)

    mktcap = close * shares
    ev = mktcap + liabilities - cur_assets
    da_approx = (op_cf - ni).clip(lower=0)
    ebitda = op_inc + da_approx
    ev_ebitda = -(ev / ebitda.replace(0, np.nan))

    fcf = op_cf + inv_cf
    fcf_yield = fcf / mktcap.replace(0, np.nan)

    ps = -(mktcap / revenue.replace(0, np.nan))

    fcf_lag = fcf.shift(252)
    fcf_growth = (fcf - fcf_lag) / fcf_lag.abs().replace(0, np.nan)

    eps_score = pd.DataFrame(np.nan, index=dates, columns=symbols)
    for sym in symbols:
        _rolling_ols_to_panel(sym, 'income_statement__diluted_earnings_per_share',
                              dates, eps_score)

    growth_stab = pd.DataFrame(np.nan, index=dates, columns=symbols)
    for sym in symbols:
        _rolling_yoy_r2_to_panel(sym, 'income_statement__revenues',
                                 dates, growth_stab)

    return {
        'ROE': roe, 'ROIC': roic, 'EV_EBITDA': ev_ebitda,
        'FCF_Yield': fcf_yield, 'PS': ps, 'FCF_Growth': fcf_growth,
        'EPS_Score': eps_score, 'Growth_Stability': growth_stab,
    }


# ═════════════════════════════════════════════════════════════
# 4. 微观结构因子 (3)  —  随波逐流, 孤雁出群, 水中行舟
#    入参: close_1m, volume_1m, close_daily, open_daily
# ═════════════════════════════════════════════════════════════

def _compute_daily_high_low_diff(close_1m, volume_1m, close_daily, open_daily):
    daily_intra_ret = close_daily / open_daily - 1
    reasonable_ret = daily_intra_ret.rolling(20, min_periods=15).mean()
    turnover_1m = close_1m * volume_1m

    dates = sorted(set(close_1m.index.date))
    results = {}

    for i, dt in enumerate(dates):
        day_mask = close_1m.index.date == dt
        day_close = close_1m.loc[day_mask]
        day_turnover = turnover_1m.loc[day_mask]

        dt_ts = pd.Timestamp(dt)
        if dt_ts not in open_daily.index or dt_ts not in reasonable_ret.index:
            continue

        day_open = open_daily.loc[dt_ts]
        reas_ret = reasonable_ret.loc[dt_ts]
        rel_ret = day_close.div(day_open, axis=1) - 1
        is_high = rel_ret.gt(reas_ret, axis=1)

        high_to = day_turnover.where(is_high, 0).sum()
        low_to = day_turnover.where(~is_high, 0).sum()
        total_to = (high_to + low_to).replace(0, np.nan)
        results[dt_ts] = (high_to - low_to) / total_to

    return pd.DataFrame(results).T.sort_index()


def _compute_daily_lone_goose(close_1m, volume_1m):
    turnover_1m = close_1m * volume_1m
    dates = sorted(set(close_1m.index.date))
    results = {}

    for i, dt in enumerate(dates):
        day_mask = close_1m.index.date == dt
        day_close = close_1m.loc[day_mask]
        day_turnover = turnover_1m.loc[day_mask]

        if len(day_close) < 30:
            continue

        minute_ret = (day_close / day_close.shift(1) - 1).iloc[1:]
        divergence = minute_ret.std(axis=1)
        mean_div = divergence.mean()
        non_div_mask = divergence < mean_div

        if non_div_mask.sum() < 10:
            continue

        calm_to = day_turnover.loc[non_div_mask.index[non_div_mask]]
        valid_cols = calm_to.columns[(calm_to != 0).any() & (calm_to.std() > 0)]
        if len(valid_cols) < 20:
            continue
        calm_to = calm_to[valid_cols]

        corr = calm_to.corr(method='pearson').fillna(0)
        n = len(valid_cols)
        mean_abs = (corr.abs().sum() - 1) / (n - 1)
        results[pd.Timestamp(dt)] = mean_abs

    df = pd.DataFrame(results).T.sort_index()
    return df.reindex(columns=close_1m.columns)


def _factor_go_with_flow(high_low_diff, window=20):
    dates = high_low_diff.index
    results = {}

    for i in range(window - 1, len(dates)):
        w = high_low_diff.iloc[i - window + 1: i + 1]
        valid_cols = w.columns[w.notna().sum() >= window // 2]
        if len(valid_cols) < 20:
            continue
        w = w[valid_cols]
        corr = w.corr(method='spearman')
        n = len(valid_cols)
        mean_abs = (corr.abs().sum() - 1) / (n - 1)
        results[dates[i]] = mean_abs

    return pd.DataFrame(results).T.reindex(columns=high_low_diff.columns)


def _factor_lone_goose(daily_lg, window=20):
    rolling_mean = daily_lg.rolling(window, min_periods=15).mean()
    rolling_std = daily_lg.rolling(window, min_periods=15).std()
    z_mean = cross_sectional_zscore(rolling_mean)
    z_std = cross_sectional_zscore(rolling_std)
    return -((z_mean + z_std) / 2)


def compute_microstructure_factors(
    close_1m: pd.DataFrame,
    volume_1m: pd.DataFrame,
    close_daily: pd.DataFrame,
    open_daily: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    微观结构因子 (3): go_with_flow, lone_goose, sailing.

    Args:
        close_1m: 1min 收盘面板 (minutes × symbols)
        volume_1m: 1min 成交量面板
        close_daily: 日线收盘面板
        open_daily: 日线开盘面板
    """
    hld = _compute_daily_high_low_diff(close_1m, volume_1m, close_daily, open_daily)
    dlg = _compute_daily_lone_goose(close_1m, volume_1m)

    gwf = _factor_go_with_flow(hld)
    lg = _factor_lone_goose(dlg)

    common_idx = gwf.index.intersection(lg.index)
    common_cols = gwf.columns.intersection(lg.columns)
    z1 = cross_sectional_zscore(gwf.loc[common_idx, common_cols])
    z2 = cross_sectional_zscore(lg.loc[common_idx, common_cols])
    sail = (z1 + z2) / 2

    return {'go_with_flow': gwf, 'lone_goose': lg, 'sailing': sail}


# ═════════════════════════════════════════════════════════════
# 5. 博弈因子 (5)  —  多空博弈
#    入参: raw_1m (长表, 含 symbol/OHLCV 列, RTH 过滤后)
# ═════════════════════════════════════════════════════════════

def _cumsum_battle(values, sort_key):
    valid = ~(np.isnan(values) | np.isnan(sort_key))
    if valid.sum() < 20:
        return np.nan
    v = values[valid]
    k = sort_key[valid]
    idx = np.argsort(k)
    v_asc = v[idx]
    v_desc = v[idx[::-1]]
    return np.sum(np.cumsum(v_asc) - np.cumsum(v_desc))


def _compute_one_day_battle(close, high, low, volume):
    n = len(close)
    if n < 30:
        return np.nan, np.nan, np.nan

    ret5 = np.full(n, np.nan)
    ret5[5:] = close[5:] / close[:-5] - 1

    running_high = np.maximum.accumulate(close)
    running_low = np.minimum.accumulate(close)
    pct_from_high = (close - running_high) / np.where(
        running_high != 0, running_high, np.nan)
    pct_from_low = (close - running_low) / np.where(
        running_low != 0, running_low, np.nan)
    rel_pos = (pct_from_high + pct_from_low) / 2

    amplitude = (high - low) / np.where(close != 0, close, np.nan)

    mask = ~np.isnan(ret5)
    vol = volume.astype(float)

    return (
        _cumsum_battle(vol[mask], ret5[mask]),
        _cumsum_battle(vol, rel_pos),
        _cumsum_battle(amplitude[mask], ret5[mask]),
    )


def _mean_distance(panel):
    row_mean = panel.mean(axis=1)
    row_std = panel.std(axis=1)
    z = panel.sub(row_mean, axis=0).div(row_std.replace(0, np.nan), axis=0)
    return z.abs()


def _aggregate_to_weekly(daily_panel, window=20):
    md = _mean_distance(daily_panel)
    rolling_mean = md.rolling(window, min_periods=window // 2).mean()
    rolling_std = md.rolling(window, min_periods=window // 2).std()
    combined = (rolling_mean + rolling_std) / 2
    weekly = combined.resample('W-FRI').last()
    return weekly.dropna(how='all')


def compute_battle_factors(
    raw_1m: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    博弈因子 (5): vol_battle_ret, vol_battle_pos, amp_battle,
    vol_battle, bull_bear_battle.

    Args:
        raw_1m: 1min 长表 (RTH 已过滤), 需含 symbol, close, high, low, volume 列.
                index = DatetimeIndex.
    """
    all_results = []
    grouped = raw_1m.groupby(['symbol', raw_1m.index.date])

    for (sym, dt), grp in grouped:
        r1, r2, r3 = _compute_one_day_battle(
            grp['close'].values, grp['high'].values,
            grp['low'].values, grp['volume'].values)
        all_results.append((sym, pd.Timestamp(dt), r1, r2, r3))

    res_df = pd.DataFrame(all_results,
                          columns=['symbol', 'date', 'vb_ret', 'vb_pos', 'ab'])

    vb_ret = res_df.pivot(index='date', columns='symbol', values='vb_ret')
    vb_pos = res_df.pivot(index='date', columns='symbol', values='vb_pos')
    ab_ret = res_df.pivot(index='date', columns='symbol', values='ab')

    daily = {'vol_battle_ret': vb_ret, 'vol_battle_pos': vb_pos, 'amp_battle': ab_ret}

    weekly = {}
    weekly['vol_battle_ret'] = _aggregate_to_weekly(daily['vol_battle_ret'])
    weekly['vol_battle_pos'] = _aggregate_to_weekly(daily['vol_battle_pos'])
    weekly['amp_battle'] = _aggregate_to_weekly(daily['amp_battle'])

    vbr_z = _mean_distance(weekly['vol_battle_ret'])
    vbp_z = _mean_distance(weekly['vol_battle_pos'])
    weekly['vol_battle'] = (vbr_z + vbp_z) / 2

    vb_z = _mean_distance(weekly['vol_battle'])
    ab_z = _mean_distance(weekly['amp_battle'])
    weekly['bull_bear_battle'] = (vb_z + ab_z) / 2

    return weekly


# ═════════════════════════════════════════════════════════════
# 6. 激增因子 (3)  —  耀眼波动率, 耀眼收益率, 适度冒险
#    入参: close_1m, volume_1m
# ═════════════════════════════════════════════════════════════

def _compute_daily_surge_metrics(close_1m, volume_1m):
    dates = sorted(set(close_1m.index.date))
    dv_results, dr_results = {}, {}

    for i, dt in enumerate(dates):
        day_mask = close_1m.index.date == dt
        day_close = close_1m.loc[day_mask]
        day_volume = volume_1m.loc[day_mask]

        if len(day_close) < 30:
            continue

        minute_ret = (day_close / day_close.shift(1) - 1).iloc[1:]
        vol_delta = (day_volume - day_volume.shift(1)).iloc[1:]

        mr_values = minute_ret.values
        vd_values = vol_delta.values
        n_mins = mr_values.shape[0]
        symbols = minute_ret.columns

        day_dv, day_dr = {}, {}

        for j, sym in enumerate(symbols):
            vd = vd_values[:, j]
            mr = mr_values[:, j]

            valid = ~np.isnan(vd)
            if valid.sum() < 10:
                continue

            vd_clean = vd[valid]
            mean_vd = np.mean(vd_clean)
            std_vd = np.std(vd_clean, ddof=1)
            if std_vd == 0:
                continue
            threshold = mean_vd + std_vd

            surge_pos = np.where((vd > threshold) & valid)[0]
            if len(surge_pos) == 0:
                continue

            surge_rets = mr[surge_pos]
            valid_sr = surge_rets[~np.isnan(surge_rets)]
            if len(valid_sr) > 0:
                day_dr[sym] = np.mean(valid_sr)

            window_stds = []
            for pos in surge_pos:
                end = min(pos + 5, n_mins)
                window = mr[pos:end]
                window = window[~np.isnan(window)]
                if len(window) >= 2:
                    window_stds.append(np.std(window, ddof=1))

            if len(window_stds) > 0:
                day_dv[sym] = np.mean(window_stds)

        dv_results[pd.Timestamp(dt)] = pd.Series(day_dv)
        dr_results[pd.Timestamp(dt)] = pd.Series(day_dr)

    cols = close_1m.columns
    daily_dv = pd.DataFrame(dv_results).T.sort_index().reindex(columns=cols)
    daily_dr = pd.DataFrame(dr_results).T.sort_index().reindex(columns=cols)
    return daily_dv, daily_dr


def compute_surge_factors(
    close_1m: pd.DataFrame,
    volume_1m: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    激增因子 (3): weekly_dazzling_vol, weekly_dazzling_ret, moderate_risk.

    Args:
        close_1m: 1min 收盘面板
        volume_1m: 1min 成交量面板
    """
    daily_dv, daily_dr = _compute_daily_surge_metrics(close_1m, volume_1m)

    wdv = moderate_rolling(daily_dv, 5, negate=True)
    wdr = moderate_rolling(daily_dr, 5, negate=False)

    common_idx = wdv.index.intersection(wdr.index)
    common_cols = wdv.columns.intersection(wdr.columns)
    valid_rows = (wdv.loc[common_idx, common_cols].notna().any(axis=1)
                  & wdr.loc[common_idx, common_cols].notna().any(axis=1))
    z1 = cross_sectional_zscore(wdv.loc[common_idx[valid_rows], common_cols])
    z2 = cross_sectional_zscore(wdr.loc[common_idx[valid_rows], common_cols])
    mr = ((z1 + z2) / 2).reindex(index=common_idx, columns=common_cols)

    return {'weekly_dazzling_vol': wdv, 'weekly_dazzling_ret': wdr, 'moderate_risk': mr}


# ═════════════════════════════════════════════════════════════
# 7. 回归因子 (4)  —  朝没晨雾, 午蔽古木, 夜眠霜路, 花隐林间
#    入参: close_1m, volume_1m
# ═════════════════════════════════════════════════════════════

def _run_daily_ols(mr, vd):
    n = len(mr)
    if n < 12:
        return None
    obs = n - 5
    y = mr[5:]
    X = np.column_stack([
        np.ones(obs), vd[5:], vd[4:-1], vd[3:-2],
        vd[2:-3], vd[1:-4], vd[0:-5],
    ])
    valid = ~np.isnan(y)
    for c in range(X.shape[1]):
        valid &= ~np.isnan(X[:, c])
    n_valid = valid.sum()
    k = 7
    if n_valid <= k + 2:
        return None
    y = y[valid]
    X = X[valid]
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ y)
        resid = y - X @ beta
        RSS = resid @ resid
        sigma2 = RSS / (n_valid - k)
        se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 1e-30))
        t_stats = beta / se
        TSS = np.sum((y - np.mean(y)) ** 2)
        ESS = TSS - RSS
        F_all = max((ESS / 6) / max(RSS / (n_valid - k), 1e-30), 0)
        return {
            't_intercept': t_stats[0],
            't1': t_stats[2], 't2': t_stats[3], 't3': t_stats[4],
            't4': t_stats[5], 't5': t_stats[6],
            'F_all': F_all,
        }
    except (np.linalg.LinAlgError, ValueError):
        return None


def _compute_daily_regression(close_1m, volume_1m):
    dates = sorted(set(close_1m.index.date))
    mm_res, at_res, fa_res, ti_res = {}, {}, {}, {}

    for i, dt in enumerate(dates):
        day_mask = close_1m.index.date == dt
        day_close = close_1m.loc[day_mask]
        day_volume = volume_1m.loc[day_mask]

        if len(day_close) < 30:
            continue

        minute_ret = (day_close / day_close.shift(1) - 1).iloc[1:]
        vol_delta = (day_volume - day_volume.shift(1)).iloc[1:]

        mr_vals = minute_ret.values
        vd_vals = vol_delta.values
        symbols = minute_ret.columns

        day_mm, day_at, day_fa, day_ti = {}, {}, {}, {}

        for j, sym in enumerate(symbols):
            result = _run_daily_ols(mr_vals[:, j], vd_vals[:, j])
            if result is None:
                continue
            t_lagged = [result['t1'], result['t2'], result['t3'],
                        result['t4'], result['t5']]
            day_mm[sym] = np.std(t_lagged, ddof=1)
            day_at[sym] = abs(result['t_intercept'])
            day_fa[sym] = result['F_all']
            day_ti[sym] = result['t_intercept']

        mm_res[pd.Timestamp(dt)] = pd.Series(day_mm)
        at_res[pd.Timestamp(dt)] = pd.Series(day_at)
        fa_res[pd.Timestamp(dt)] = pd.Series(day_fa)
        ti_res[pd.Timestamp(dt)] = pd.Series(day_ti)

    cols = close_1m.columns
    return (
        pd.DataFrame(mm_res).T.sort_index().reindex(columns=cols),
        pd.DataFrame(at_res).T.sort_index().reindex(columns=cols),
        pd.DataFrame(fa_res).T.sort_index().reindex(columns=cols),
        pd.DataFrame(ti_res).T.sort_index().reindex(columns=cols),
    )


def _factor_night_frost(daily_t_intercept, window=5):
    dates = daily_t_intercept.index
    results = {}

    for i in range(window - 1, len(dates)):
        w = daily_t_intercept.iloc[i - window + 1: i + 1]
        valid_cols = w.columns[w.notna().sum() >= max(window - 1, 3)]
        if len(valid_cols) < 20:
            continue
        w = w[valid_cols]
        corr = w.corr(method='pearson').fillna(0)
        n = len(valid_cols)
        mean_abs = (corr.abs().sum() - 1) / (n - 1)
        results[dates[i]] = mean_abs

    df = pd.DataFrame(results).T.sort_index()
    return df.reindex(columns=daily_t_intercept.columns)


def _factor_noon_shade(daily_abs_ti, daily_F_all, window=5):
    F_cs_mean = daily_F_all.mean(axis=1)
    is_low_F = daily_F_all.lt(F_cs_mean, axis=0)

    daily_noon = daily_abs_ti.copy()
    daily_noon[is_low_F] = -daily_noon[is_low_F]

    return rolling_mean_zscore(daily_noon, window)


def compute_regression_factors(
    close_1m: pd.DataFrame,
    volume_1m: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    回归因子 (4): morning_mist, noon_shade, night_frost, flower_hidden.

    Args:
        close_1m: 1min 收盘面板
        volume_1m: 1min 成交量面板
    """
    daily_mm, daily_abs_ti, daily_fa, daily_ti = _compute_daily_regression(
        close_1m, volume_1m)

    mm = rolling_mean_zscore(daily_mm)
    ns = _factor_noon_shade(daily_abs_ti, daily_fa)
    nf = _factor_night_frost(daily_ti)

    common_idx = mm.index.intersection(ns.index).intersection(nf.index)
    common_cols = mm.columns.intersection(ns.columns).intersection(nf.columns)
    valid_rows = (mm.loc[common_idx, common_cols].notna().any(axis=1)
                  & ns.loc[common_idx, common_cols].notna().any(axis=1)
                  & nf.loc[common_idx, common_cols].notna().any(axis=1))
    z1 = cross_sectional_zscore(mm.loc[common_idx[valid_rows], common_cols])
    z2 = cross_sectional_zscore(ns.loc[common_idx[valid_rows], common_cols])
    z3 = cross_sectional_zscore(nf.loc[common_idx[valid_rows], common_cols])
    fh = ((z1 + z2 + z3) / 3).reindex(index=common_idx, columns=common_cols)

    return {'morning_mist': mm, 'noon_shade': ns, 'night_frost': nf, 'flower_hidden': fh}


# ═════════════════════════════════════════════════════════════
# 8. 模糊性因子 (3)  —  模糊关联度, 模糊金额比, 模糊数量比
#    入参: close_1m, volume_1m
# ═════════════════════════════════════════════════════════════

def _compute_daily_fuzzy_metrics(close_1m, volume_1m):
    turnover_1m = close_1m * volume_1m
    dates = sorted(set(close_1m.index.date))
    fc_res, far_res, fvr_res = {}, {}, {}

    for i, dt in enumerate(dates):
        day_mask = close_1m.index.date == dt
        day_close = close_1m.loc[day_mask]
        day_volume = volume_1m.loc[day_mask]
        day_turnover = turnover_1m.loc[day_mask]

        if len(day_close) < 30:
            continue

        minute_ret = day_close / day_close.shift(1) - 1
        volatility = minute_ret.rolling(5, min_periods=5).std()
        ambiguity = volatility.rolling(5, min_periods=5).std()

        valid_start = 9
        if len(ambiguity) <= valid_start:
            continue

        amb = ambiguity.iloc[valid_start:]
        vol = day_volume.iloc[valid_start:]
        to = day_turnover.iloc[valid_start:]

        day_fc, day_far, day_fvr = {}, {}, {}

        for sym in day_close.columns:
            a = amb[sym].values
            v = vol[sym].values
            t = to[sym].values

            mask = ~(np.isnan(a) | np.isnan(v) | np.isnan(t))
            if mask.sum() < 20:
                continue

            a_clean, v_clean, t_clean = a[mask], v[mask], t[mask]

            if np.std(a_clean) > 0 and np.std(t_clean) > 0:
                day_fc[sym] = np.corrcoef(a_clean, t_clean)[0, 1]

            amb_mean = np.mean(a_clean)
            foggy = a_clean > amb_mean

            if foggy.sum() >= 5:
                total_to_mean = np.mean(t_clean)
                if total_to_mean > 0:
                    day_far[sym] = np.mean(t_clean[foggy]) / total_to_mean
                total_vol_mean = np.mean(v_clean)
                if total_vol_mean > 0:
                    day_fvr[sym] = np.mean(v_clean[foggy]) / total_vol_mean

        fc_res[pd.Timestamp(dt)] = pd.Series(day_fc)
        far_res[pd.Timestamp(dt)] = pd.Series(day_far)
        fvr_res[pd.Timestamp(dt)] = pd.Series(day_fvr)

    cols = close_1m.columns
    return (
        pd.DataFrame(fc_res).T.sort_index().reindex(columns=cols),
        pd.DataFrame(far_res).T.sort_index().reindex(columns=cols),
        pd.DataFrame(fvr_res).T.sort_index().reindex(columns=cols),
    )


def compute_fuzzy_factors(
    close_1m: pd.DataFrame,
    volume_1m: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    模糊性因子 (3): fuzzy_corr, fuzzy_amount_ratio, fuzzy_volume_ratio.

    Args:
        close_1m: 1min 收盘面板
        volume_1m: 1min 成交量面板
    """
    daily_fc, daily_far, daily_fvr = _compute_daily_fuzzy_metrics(close_1m, volume_1m)

    fc = rolling_composite(daily_fc)
    far = rolling_composite(daily_far)
    fvr = rolling_composite(daily_fvr)

    return {'fuzzy_corr': fc, 'fuzzy_amount_ratio': far, 'fuzzy_volume_ratio': fvr}


# ═════════════════════════════════════════════════════════════
# 9. 灾后重建因子 (2)  —  disaster_rebuild, peak_climbing
#    入参: panels_1m (dict of 5 OHLCV 面板)
# ═════════════════════════════════════════════════════════════

def _compute_daily_rebuild_metrics(panels_1m):
    open_1m = panels_1m['open']
    high_1m = panels_1m['high']
    low_1m = panels_1m['low']
    close_1m = panels_1m['close']

    dates = sorted(set(close_1m.index.date))
    rebuild_res, climb_res = {}, {}

    for i, dt in enumerate(dates):
        day_mask = close_1m.index.date == dt
        d_o = open_1m.loc[day_mask].values
        d_h = high_1m.loc[day_mask].values
        d_l = low_1m.loc[day_mask].values
        d_c = close_1m.loc[day_mask].values
        symbols = close_1m.columns
        n_mins = d_c.shape[0]

        if n_mins < 15:
            continue

        day_rebuild, day_climb = {}, {}

        for j, sym in enumerate(symbols):
            o, h, l, c = d_o[:, j], d_h[:, j], d_l[:, j], d_c[:, j]
            if np.isnan(c).all():
                continue

            ohlc_flat = np.column_stack([o, h, l, c]).flatten()
            if len(ohlc_flat) < 20:
                continue

            windows = np.lib.stride_tricks.sliding_window_view(ohlc_flat, 20)[::4]
            w_mean = np.mean(windows, axis=1)
            w_std = np.std(windows, axis=1, ddof=1)

            better_vol = np.full(n_mins, np.nan)
            valid_bv = w_mean > 0
            bv_vals = np.where(valid_bv, (w_std / w_mean) ** 2, np.nan)
            better_vol[4:] = bv_vals

            mr = np.full(n_mins, np.nan)
            valid_c = (c[:-1] != 0) & ~np.isnan(c[:-1]) & ~np.isnan(c[1:])
            mr[1:] = np.where(valid_c, c[1:] / c[:-1] - 1, np.nan)

            valid_rv = ~np.isnan(mr) & ~np.isnan(better_vol) & (better_vol > 0)
            rvr = np.full(n_mins, np.nan)
            rvr[valid_rv] = mr[valid_rv] / better_vol[valid_rv]

            valid = ~np.isnan(rvr) & ~np.isnan(better_vol)
            if valid.sum() < 20:
                continue

            rvr_v = rvr[valid]
            bv_v = better_vol[valid]

            day_rebuild[sym] = np.cov(rvr_v, bv_v)[0, 1]

            bv_mean = np.mean(bv_v)
            bv_std_val = np.std(bv_v, ddof=1)
            high_mask = bv_v >= bv_mean + bv_std_val
            if high_mask.sum() >= 5:
                day_climb[sym] = np.cov(rvr_v[high_mask], bv_v[high_mask])[0, 1]

        rebuild_res[pd.Timestamp(dt)] = pd.Series(day_rebuild)
        climb_res[pd.Timestamp(dt)] = pd.Series(day_climb)

    cols = close_1m.columns
    return (
        pd.DataFrame(rebuild_res).T.sort_index().reindex(columns=cols),
        pd.DataFrame(climb_res).T.sort_index().reindex(columns=cols),
    )


def compute_rebuild_factors(
    panels_1m: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    灾后重建因子 (2): disaster_rebuild, peak_climbing.

    Args:
        panels_1m: {'open': df, 'high': df, 'low': df, 'close': df, 'volume': df}
                   1min OHLCV 宽表
    """
    daily_rb, daily_cl = _compute_daily_rebuild_metrics(panels_1m)

    rb = rolling_composite(daily_rb, 5, negate=True)
    cl = rolling_composite(daily_cl, 5, negate=False)

    return {'disaster_rebuild': rb, 'peak_climbing': cl}


# ═════════════════════════════════════════════════════════════
# 10. 潮汐因子 (4)  —  全潮汐, 强势半潮汐, 激进弱势, 稳定弱势
#     入参: close_1m, volume_1m
# ═════════════════════════════════════════════════════════════

def _compute_daily_tidal_metrics(close_1m, volume_1m):
    dates = sorted(set(close_1m.index.date))
    full_res, strong_res, weak_res = {}, {}, {}
    kernel = np.ones(9)

    for i, dt in enumerate(dates):
        day_mask = close_1m.index.date == dt
        day_close = close_1m.loc[day_mask].values
        day_volume = volume_1m.loc[day_mask].values
        symbols = close_1m.columns
        n_mins = day_close.shape[0]

        if n_mins < 20:
            continue

        day_full, day_strong, day_weak = {}, {}, {}

        for j, sym in enumerate(symbols):
            c = day_close[:, j]
            v = day_volume[:, j]
            if np.isnan(c).all() or np.isnan(v).all():
                continue

            v_clean = np.where(np.isnan(v), 0, v)
            nbhd = np.convolve(v_clean, kernel, mode='same')
            nbhd[:4] = np.nan
            nbhd[-4:] = np.nan

            valid_start = 4
            valid_end = n_mins - 5
            if valid_end - valid_start < 20:
                continue

            peak_start = valid_start + 10
            peak_end = valid_end - 10
            if peak_end <= peak_start:
                continue
            nbhd_slice = nbhd[peak_start:peak_end + 1]
            if np.all(np.isnan(nbhd_slice)) or np.nanmax(nbhd_slice) == 0:
                continue
            t_peak = peak_start + np.nanargmax(nbhd_slice)

            rising = nbhd[valid_start:t_peak]
            if len(rising) == 0 or np.all(np.isnan(rising)):
                continue
            m = valid_start + np.nanargmin(rising)

            ebbing = nbhd[t_peak + 1:valid_end + 1]
            if len(ebbing) == 0 or np.all(np.isnan(ebbing)):
                continue
            n_idx = t_peak + 1 + np.nanargmin(ebbing)

            Cm, Cn, Ct = c[m], c[n_idx], c[t_peak]
            Vm, Vn = nbhd[m], nbhd[n_idx]

            if any(np.isnan(x) for x in [Cm, Cn, Ct, Vm, Vn]):
                continue
            if Cm == 0 or n_idx == m:
                continue

            day_full[sym] = (Cn - Cm) / Cm / (n_idx - m)

            if Vm < Vn:
                if t_peak > m and Cm != 0:
                    day_strong[sym] = (Ct - Cm) / Cm / (t_peak - m)
                if n_idx > t_peak and Ct != 0:
                    day_weak[sym] = (Cn - Ct) / Ct / (n_idx - t_peak)
            else:
                if n_idx > t_peak and Ct != 0:
                    day_strong[sym] = (Cn - Ct) / Ct / (n_idx - t_peak)
                if t_peak > m and Cm != 0:
                    day_weak[sym] = (Ct - Cm) / Cm / (t_peak - m)

        full_res[pd.Timestamp(dt)] = pd.Series(day_full)
        strong_res[pd.Timestamp(dt)] = pd.Series(day_strong)
        weak_res[pd.Timestamp(dt)] = pd.Series(day_weak)

    cols = close_1m.columns
    return (
        pd.DataFrame(full_res).T.sort_index().reindex(columns=cols),
        pd.DataFrame(strong_res).T.sort_index().reindex(columns=cols),
        pd.DataFrame(weak_res).T.sort_index().reindex(columns=cols),
    )


def compute_tidal_factors(
    close_1m: pd.DataFrame,
    volume_1m: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    潮汐因子 (4): full_tidal, strong_half_tidal, aggressive_weak_half,
    stable_weak_half.

    Args:
        close_1m: 1min 收盘面板
        volume_1m: 1min 成交量面板
    """
    daily_full, daily_strong, daily_weak = _compute_daily_tidal_metrics(
        close_1m, volume_1m)

    ft = rolling_mean_zscore(daily_full)
    sht = rolling_mean_zscore(daily_strong)
    awh = rolling_mean_zscore(daily_weak)

    min_p = max(5 - 1, 3)
    rolling_std = daily_weak.rolling(5, min_periods=min_p).std()
    valid_rows = rolling_std.notna().any(axis=1)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        swh = -cross_sectional_zscore(rolling_std.loc[valid_rows])
    swh = swh.reindex(daily_weak.index)

    return {
        'full_tidal': ft, 'strong_half_tidal': sht,
        'aggressive_weak_half': awh, 'stable_weak_half': swh,
    }


# ═════════════════════════════════════════════════════════════
# 11. 跳跃因子 (5)  —  周跳跃度, 修正振幅1/2, 修正振幅, 飞蛾扑火
#     入参: close_1m, close_daily, high_daily, low_daily
# ═════════════════════════════════════════════════════════════

def _compute_daily_jump(close_1m):
    dates = sorted(set(close_1m.index.date))
    jump_res = {}

    for i, dt in enumerate(dates):
        day_mask = close_1m.index.date == dt
        day_close = close_1m.loc[day_mask]
        if len(day_close) < 20:
            continue
        ratio = day_close / day_close.shift(1)
        ratio = ratio.iloc[1:]
        simple_ret = ratio - 1
        log_ret = np.log(ratio)
        taylor = 2 * (simple_ret - log_ret) - log_ret ** 2
        jump_res[pd.Timestamp(dt)] = taylor.mean()

    result = pd.DataFrame(jump_res).T.sort_index()
    return result.reindex(columns=close_1m.columns)


def _compute_flipped_amplitude_1(daily_jump, daily_high, daily_low, daily_close):
    prev_close = daily_close.shift(1)
    amplitude = (daily_high - daily_low) / prev_close

    common_idx = daily_jump.index.intersection(amplitude.index)
    common_cols = daily_jump.columns.intersection(amplitude.columns)

    jump = daily_jump.loc[common_idx, common_cols]
    amp = amplitude.loc[common_idx, common_cols]

    cs_mean = jump.mean(axis=1)
    is_sun = jump.lt(cs_mean, axis=0)

    flipped = amp.copy()
    flipped[is_sun] = -flipped[is_sun]
    return flipped


def _compute_flipped_amplitude_2(daily_high, daily_low, daily_close):
    prev_low = daily_low.shift(1)
    prev_close = daily_close.shift(1)
    amplitude = (daily_high - daily_low) / prev_close

    ratio = daily_high / prev_low
    simple_ret = ratio - 1
    log_ret = np.log(ratio)
    taylor = 2 * (simple_ret - log_ret) - log_ret ** 2

    cs_mean = taylor.mean(axis=1)
    is_sun = taylor.lt(cs_mean, axis=0)

    flipped = amplitude.copy()
    flipped[is_sun] = -flipped[is_sun]
    return flipped


def compute_jump_factors(
    close_1m: pd.DataFrame,
    close_daily: pd.DataFrame,
    high_daily: pd.DataFrame,
    low_daily: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    跳跃因子 (5): weekly_jump, modified_amplitude_1, modified_amplitude_2,
    modified_amplitude, moth_to_flame.

    Args:
        close_1m: 1min 收盘面板
        close_daily: 日线收盘面板
        high_daily: 日线最高价面板
        low_daily: 日线最低价面板
    """
    daily_jump = _compute_daily_jump(close_1m)

    fa1 = _compute_flipped_amplitude_1(daily_jump, high_daily, low_daily, close_daily)
    fa2 = _compute_flipped_amplitude_2(high_daily, low_daily, close_daily)

    wj = rolling_composite(daily_jump)
    ma1 = rolling_mean_zscore(fa1)
    ma2 = rolling_mean_zscore(fa2)

    # modified_amplitude: z(ma1) + z(ma2) 等权
    common_idx = ma1.index.intersection(ma2.index)
    common_cols = ma1.columns.intersection(ma2.columns)
    valid_rows = (ma1.loc[common_idx, common_cols].notna().any(axis=1)
                  & ma2.loc[common_idx, common_cols].notna().any(axis=1))
    z1 = cross_sectional_zscore(ma1.loc[common_idx[valid_rows], common_cols])
    z2 = cross_sectional_zscore(ma2.loc[common_idx[valid_rows], common_cols])
    ma = ((z1 + z2) / 2).reindex(index=common_idx, columns=common_cols)

    # moth_to_flame: z(weekly_jump) + z(modified_amplitude) 等权
    common_idx = wj.index.intersection(ma.index)
    common_cols = wj.columns.intersection(ma.columns)
    valid_rows = (wj.loc[common_idx, common_cols].notna().any(axis=1)
                  & ma.loc[common_idx, common_cols].notna().any(axis=1))
    z1 = cross_sectional_zscore(wj.loc[common_idx[valid_rows], common_cols])
    z2 = cross_sectional_zscore(ma.loc[common_idx[valid_rows], common_cols])
    mtf = ((z1 + z2) / 2).reindex(index=common_idx, columns=common_cols)

    return {
        'weekly_jump': wj,
        'modified_amplitude_1': ma1,
        'modified_amplitude_2': ma2,
        'modified_amplitude': ma,
        'moth_to_flame': mtf,
    }


# ═════════════════════════════════════════════════════════════
# 一键全量: compute_all_factors
# ═════════════════════════════════════════════════════════════

def compute_all_factors(
    history,
    start_1min: Optional[str] = None,
    end_1min: Optional[str] = None,
    periods: int = 504,
) -> Dict[str, pd.DataFrame]:
    """
    一键计算全部因子 (适配 event-driven 策略).

    1d 因子从 history.panel() 获取 OHLCV;
    1min 因子从 history 的 1min 方法加载 (如果配置了 storage_1min);
    基本面因子仍从磁盘读 quarterly data.

    Args:
        history: HistoryManager 实例 (策略中为 self.history)
        start_1min/end_1min: 1min 数据时间范围 (str), 需要时提供
        periods: history.panel 回看天数

    Returns:
        dict: {factor_name: DataFrame (dates × symbols)}
    """
    # ── 1d 面板 ──
    close = history.panel('close', periods)
    high = history.panel('high', periods)
    low = history.panel('low', periods)
    open_p = history.panel('open', periods)
    symbols = list(close.columns)

    all_factors = {}

    # 1. 技术因子
    all_factors.update(compute_technical_factors(close, high, low))

    # 2. 反转因子
    all_factors.update(compute_reversal_factors(close, open_p))

    # 3. 基本面因子
    try:
        all_factors.update(compute_fundamental_factors(symbols, close))
    except Exception:
        pass  # 缺少基本面数据时跳过

    # ── 1min 因子 ──
    if history._storage_1min is not None and start_1min and end_1min:
        try:
            close_1m, volume_1m = history.panel_1min(
                'close', 'volume', start=start_1min, end=end_1min)

            # 4. 微观结构
            all_factors.update(compute_microstructure_factors(
                close_1m, volume_1m, close, open_p))

            # 6. 激增
            all_factors.update(compute_surge_factors(close_1m, volume_1m))

            # 7. 回归
            all_factors.update(compute_regression_factors(close_1m, volume_1m))

            # 8. 模糊
            all_factors.update(compute_fuzzy_factors(close_1m, volume_1m))

            # 10. 潮汐
            all_factors.update(compute_tidal_factors(close_1m, volume_1m))

            # 11. 跳跃
            all_factors.update(compute_jump_factors(
                close_1m, close, high, low))

            # 5. 博弈 (需要长表)
            raw_1m = history.raw_1min(start=start_1min, end=end_1min)
            all_factors.update(compute_battle_factors(raw_1m))

            # 9. 灾后重建 (需要 OHLCV)
            panels_1m = history.ohlcv_1min(start=start_1min, end=end_1min)
            all_factors.update(compute_rebuild_factors(panels_1m))

        except Exception:
            pass  # 缺少 1min 数据时跳过

    return all_factors
