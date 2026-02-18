"""
因子库 — 46 个因子统一定义
=========================================

分类            因子名                        数据源   参考
────────────────────────────────────────────────────────────
技术(5)         RS_12M, RS_6M, Range_52W,     1d      经典动量
                RSI_28W, RSI_16W
反转(4)         interday/intraday/overnight    1d      方正-球队硬币
                _rev_volflip, team_coin
基本面(8)       ROE, ROIC, EV_EBITDA,         fund    Alpha Vantage
                FCF_Yield, PS, FCF_Growth,
                EPS_Score, Growth_Stability
微观结构(3)     go_with_flow, lone_goose,     1min    方正-随波逐流
                sailing
博弈(5)         vol_battle_ret/pos/battle,    1min    方正-多空博弈
                amp_battle, bull_bear_battle
激增(3)         weekly_dazzling_vol/ret,      1min    方正-耀眼
                moderate_risk
回归(4)         morning_mist, noon_shade,     1min    方正-回归分析
                night_frost, flower_hidden
模糊(3)         fuzzy_corr,                   1min    方正-模糊性
                fuzzy_amount_ratio,
                fuzzy_volume_ratio
灾后重建(2)     disaster_rebuild,             1min    方正-灾后重建
                peak_climbing
潮汐(4)         full_tidal, strong_half,      1min    方正-潮汐
                aggressive_weak_half,
                stable_weak_half
跳跃(5)         weekly_jump,                  1min+1d Jiang(2008)
                modified_amplitude_1/2,
                modified_amplitude,
                moth_to_flame

Usage:
    from factor_research.factors import compute_technical_factors, ...
"""
import datetime as _dt
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
BARS_1MIN_DIR = _ROOT / 'data' / 'processed' / 'bars_1min'
BARS_1D_DIR = _ROOT / 'data' / 'processed' / 'bars_1d'

RTH_START = _dt.time(9, 35)
RTH_END = _dt.time(15, 57)


# ═════════════════════════════════════════════════════════════
# 共用工具
# ═════════════════════════════════════════════════════════════

def _cross_sectional_zscore(
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


def _load_1min_panels(
    symbols: List[str],
    start: str = '2025-04-03',
    end: str = '2026-12-31',
):
    """加载 1min close+volume 并 pivot 成宽表 (minutes × symbols)"""
    from data.storage.parquet import ParquetStorage

    storage = ParquetStorage(str(BARS_1MIN_DIR))
    print(f'  Loading 1min for {len(symbols)} symbols...', end='', flush=True)
    raw = storage.load(symbols,
                       pd.Timestamp(start, tz='US/Eastern'),
                       pd.Timestamp(end, tz='US/Eastern'), '1m')
    print(f' {len(raw):,} bars', flush=True)

    t = raw.index.time
    raw = raw[(t >= RTH_START) & (t <= RTH_END)]
    print(f'  After RTH filter: {len(raw):,} bars', flush=True)

    close_1m = raw.pivot_table(index=raw.index, columns='symbol', values='close')
    volume_1m = raw.pivot_table(index=raw.index, columns='symbol', values='volume')
    print(f'  Pivoted: {close_1m.shape[0]:,} minutes × {close_1m.shape[1]} symbols',
          flush=True)
    return close_1m, volume_1m


def _load_1min_ohlcv(
    symbols: List[str],
    start: str = '2025-04-03',
    end: str = '2026-12-31',
) -> Dict[str, pd.DataFrame]:
    """加载 1min OHLCV 并 pivot 成宽表"""
    from data.storage.parquet import ParquetStorage

    storage = ParquetStorage(str(BARS_1MIN_DIR))
    print(f'  Loading 1min OHLCV for {len(symbols)} symbols...', end='', flush=True)
    raw = storage.load(symbols,
                       pd.Timestamp(start, tz='US/Eastern'),
                       pd.Timestamp(end, tz='US/Eastern'), '1m')
    print(f' {len(raw):,} bars', flush=True)

    t = raw.index.time
    raw = raw[(t >= RTH_START) & (t <= RTH_END)]
    print(f'  After RTH filter: {len(raw):,} bars', flush=True)

    panels = {}
    for field in ['open', 'high', 'low', 'close', 'volume']:
        panels[field] = raw.pivot_table(index=raw.index, columns='symbol', values=field)

    n_rows = panels['close'].shape[0]
    n_syms = panels['close'].shape[1]
    print(f'  Pivoted: {n_rows:,} minutes × {n_syms} symbols', flush=True)
    return panels


def _rolling_mean_std_composite(
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
        z_mean = _cross_sectional_zscore(rolling_mean.loc[valid_rows])
        z_std = _cross_sectional_zscore(rolling_std.loc[valid_rows])

    composite = (z_mean + z_std) / 2
    result = -composite if negate else composite
    return result.reindex(daily_panel.index)


def _rolling_mean_zscore(
    daily_panel: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """rolling mean → z-score (无 std 成分)"""
    min_p = max(window - 1, 3)
    rolling_mean = daily_panel.rolling(window, min_periods=min_p).mean()

    valid_rows = rolling_mean.notna().any(axis=1)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        result = _cross_sectional_zscore(rolling_mean.loc[valid_rows])

    return result.reindex(daily_panel.index)


def _moderate_rolling_factor(
    daily_panel: pd.DataFrame,
    window: int = 5,
    negate: bool = True,
) -> pd.DataFrame:
    """适度化 + rolling mean+std → z-score 等权 (for surge factors)"""
    cs_mean = daily_panel.mean(axis=1)
    moderate = daily_panel.sub(cs_mean, axis=0).abs()

    min_periods = max(window - 1, 3)
    rolling_mean = moderate.rolling(window, min_periods=min_periods).mean()
    rolling_std = moderate.rolling(window, min_periods=min_periods).std()

    valid_rows = rolling_mean.notna().any(axis=1)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        z_mean = _cross_sectional_zscore(rolling_mean.loc[valid_rows])
        z_std = _cross_sectional_zscore(rolling_std.loc[valid_rows])

    composite = (z_mean + z_std) / 2
    result = -composite if negate else composite
    return result.reindex(daily_panel.index)


# ═════════════════════════════════════════════════════════════
# 1. 技术因子 (5)  —  RS_12M, RS_6M, Range_52W, RSI_28W, RSI_16W
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
    # RS 12M: 252日回报, 跳过最近21日
    rs_12m = close.shift(21) / close.shift(252) - 1
    # RS 6M
    rs_6m = close.shift(21) / close.shift(126) - 1
    # 52W Range: (close - 52w_low) / (52w_high - 52w_low)
    high_252 = high.rolling(252, min_periods=126).max()
    low_252 = low.rolling(252, min_periods=126).min()
    range_52w = (close - low_252) / (high_252 - low_252).replace(0, np.nan)
    # RSI
    rsi_28w = _rsi(close, 140)
    rsi_16w = _rsi(close, 80)

    return {
        'RS_12M': rs_12m, 'RS_6M': rs_6m, 'Range_52W': range_52w,
        'RSI_28W': rsi_28w, 'RSI_16W': rsi_16w,
    }


# ═════════════════════════════════════════════════════════════
# 2. 反转因子 (4)  —  球队硬币
# ═════════════════════════════════════════════════════════════

def _volatility_flip(daily_panel: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """波动翻转: 低波动→取反, 高波动→不变"""
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
    interday_ret = close / close.shift(1) - 1
    intraday_ret = close / open_price - 1
    overnight_ret = open_price / close.shift(1) - 1
    overnight_dist = overnight_ret.sub(overnight_ret.mean(axis=1), axis=0).abs()

    f1 = _volatility_flip(interday_ret)
    f2 = _volatility_flip(intraday_ret)
    f3 = _volatility_flip(overnight_dist)

    z1 = _cross_sectional_zscore(f1)
    z2 = _cross_sectional_zscore(f2)
    z3 = _cross_sectional_zscore(f3)

    daily = {
        'interday_rev_volflip': f1,
        'intraday_rev_volflip': f2,
        'overnight_rev_volflip': f3,
        'team_coin': (z1 + z2 + z3) / 3,
    }
    for name, df in daily.items():
        n_valid = df.iloc[-1].notna().sum() if len(df) > 0 else 0
        print(f'  {name}: {df.shape[0]} days, coverage={n_valid}', flush=True)
    return daily


# ═════════════════════════════════════════════════════════════
# 3. 基本面因子 (8)  —  ROE, ROIC, EV/EBITDA, FCF_Yield, PS,
#                       FCF_Growth, EPS_Score, Growth_Stability
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
    from .data_loader import _load_single_fundamental, _fix_filing_dates
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


def _rolling_ols_to_panel(
    symbol, field, dates, output, mode='slope_r2', window=8,
):
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
    from .data_loader import build_fundamental_panel

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

    # ROE
    roe = ni / equity.replace(0, np.nan)

    # ROIC
    eff_tax = (tax / pretax.replace(0, np.nan)).clip(0, 0.5)
    nopat = op_inc * (1 - eff_tax)
    invested_capital = assets - cur_liab
    roic = nopat / invested_capital.replace(0, np.nan)

    # EV/EBITDA (取负)
    mktcap = close * shares
    ev = mktcap + liabilities - cur_assets
    da_approx = (op_cf - ni).clip(lower=0)
    ebitda = op_inc + da_approx
    ev_ebitda = -(ev / ebitda.replace(0, np.nan))

    # FCF Yield
    fcf = op_cf + inv_cf
    fcf_yield = fcf / mktcap.replace(0, np.nan)

    # P/S (取负)
    ps = -(mktcap / revenue.replace(0, np.nan))

    # FCF Growth
    fcf_lag = fcf.shift(252)
    fcf_growth = (fcf - fcf_lag) / fcf_lag.abs().replace(0, np.nan)

    # EPS Score
    eps_score = pd.DataFrame(np.nan, index=dates, columns=symbols)
    for sym in symbols:
        _rolling_ols_to_panel(sym, 'income_statement__diluted_earnings_per_share',
                              dates, eps_score)

    # Growth Stability
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
# ═════════════════════════════════════════════════════════════

def _compute_daily_high_low_diff(close_1m, volume_1m, close_daily, open_daily):
    """每日"高低额差": 高位成交额 vs 低位成交额的差异"""
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

        if (i + 1) % 20 == 0:
            print(f'    high_low_diff: {i+1}/{len(dates)} days', flush=True)

    print(f'    high_low_diff: {len(dates)}/{len(dates)} days', flush=True)
    return pd.DataFrame(results).T.sort_index()


def _compute_daily_lone_goose(close_1m, volume_1m):
    """每日"孤雁出群": 不分化时刻成交额相关性均值"""
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

        if (i + 1) % 20 == 0:
            print(f'    lone_goose: {i+1}/{len(dates)} days', flush=True)

    print(f'    lone_goose: {len(dates)}/{len(dates)} days', flush=True)
    df = pd.DataFrame(results).T.sort_index()
    return df.reindex(columns=close_1m.columns)


def _factor_go_with_flow(high_low_diff, window=20):
    """随波逐流: rolling pairwise Spearman |corr| 均值"""
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

        done = i - window + 2
        total = len(dates) - window + 1
        if done % 10 == 0:
            print(f'    go_with_flow: {done}/{total}', flush=True)

    total = len(dates) - window + 1
    print(f'    go_with_flow: {total}/{total}', flush=True)
    return pd.DataFrame(results).T.reindex(columns=high_low_diff.columns)


def _factor_lone_goose(daily_lg, window=20):
    """孤雁出群: rolling mean+std → z-score → 取负"""
    rolling_mean = daily_lg.rolling(window, min_periods=15).mean()
    rolling_std = daily_lg.rolling(window, min_periods=15).std()
    z_mean = _cross_sectional_zscore(rolling_mean)
    z_std = _cross_sectional_zscore(rolling_std)
    return -((z_mean + z_std) / 2)


def compute_microstructure_factors(
    symbols: List[str],
    start: str = '2025-10-01',
    end: str = '2026-12-31',
) -> Dict[str, pd.DataFrame]:
    from .data_loader import load_price_panel

    print('Loading 1min data...', flush=True)
    close_1m, volume_1m = _load_1min_panels(symbols, start, end)

    print('\nLoading daily OHLC...', flush=True)
    panels = load_price_panel(symbols, '2025-01-01', end)
    close_daily = panels['close']
    open_daily = panels['open']
    print(f'  Daily: {close_daily.shape[0]} days × {close_daily.shape[1]} symbols')

    print('\nComputing daily high-low diff...', flush=True)
    hld = _compute_daily_high_low_diff(close_1m, volume_1m, close_daily, open_daily)
    print(f'  Result: {hld.shape}')

    print('\nComputing daily lone goose...', flush=True)
    dlg = _compute_daily_lone_goose(close_1m, volume_1m)
    print(f'  Result: {dlg.shape}')

    print('\nComputing go_with_flow (rolling Spearman corr)...', flush=True)
    gwf = _factor_go_with_flow(hld)
    n = gwf.iloc[-1].notna().sum() if len(gwf) > 0 else 0
    print(f'  go_with_flow: {gwf.shape}, coverage={n}')

    print('\nComputing lone_goose (rolling mean+std, negated)...', flush=True)
    lg = _factor_lone_goose(dlg)
    n = lg.iloc[-1].notna().sum() if len(lg) > 0 else 0
    print(f'  lone_goose: {lg.shape}, coverage={n}')

    print('\nComputing sailing (composite)...', flush=True)
    common_idx = gwf.index.intersection(lg.index)
    common_cols = gwf.columns.intersection(lg.columns)
    z1 = _cross_sectional_zscore(gwf.loc[common_idx, common_cols])
    z2 = _cross_sectional_zscore(lg.loc[common_idx, common_cols])
    sail = (z1 + z2) / 2
    n = sail.iloc[-1].notna().sum() if len(sail) > 0 else 0
    print(f'  sailing: {sail.shape}, coverage={n}')

    return {'go_with_flow': gwf, 'lone_goose': lg, 'sailing': sail}


# ═════════════════════════════════════════════════════════════
# 5. 博弈因子 (5)  —  多空博弈
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
    pct_from_high = (close - running_high) / np.where(running_high != 0, running_high, np.nan)
    pct_from_low = (close - running_low) / np.where(running_low != 0, running_low, np.nan)
    rel_pos = (pct_from_high + pct_from_low) / 2

    amplitude = (high - low) / np.where(close != 0, close, np.nan)

    mask = ~np.isnan(ret5)
    vol = volume.astype(float)

    return (
        _cumsum_battle(vol[mask], ret5[mask]),
        _cumsum_battle(vol, rel_pos),
        _cumsum_battle(amplitude[mask], ret5[mask]),
    )


def _process_battle_batch(symbols_batch, start, end):
    from data.storage.parquet import ParquetStorage

    storage = ParquetStorage(str(BARS_1MIN_DIR))
    try:
        raw = storage.load(symbols_batch,
                           pd.Timestamp(start, tz='US/Eastern'),
                           pd.Timestamp(end, tz='US/Eastern'), '1m')
    except FileNotFoundError:
        return []

    t = raw.index.time
    raw = raw[(t >= RTH_START) & (t <= RTH_END)]
    if raw.empty:
        return []

    results = []
    for (sym, dt), grp in raw.groupby(['symbol', raw.index.date]):
        r1, r2, r3 = _compute_one_day_battle(
            grp['close'].values, grp['high'].values,
            grp['low'].values, grp['volume'].values)
        results.append((sym, pd.Timestamp(dt), r1, r2, r3))
    return results


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
    # 保留日频输出; 周频对齐由 prepare_weekly 统一处理
    return combined.dropna(how='all')


def compute_battle_factors(
    symbols: Optional[List[str]] = None,
    start: str = '2025-10-01',
    end: str = '2026-12-31',
) -> Dict[str, pd.DataFrame]:
    batch_size = 2000

    if symbols is None:
        print('Scanning all symbols in 1min data...', flush=True)
        start_year = pd.Timestamp(start).year
        end_year = pd.Timestamp(end).year
        all_syms = set()
        for f in sorted(BARS_1MIN_DIR.glob('*_1m.parquet')):
            year = int(f.stem.split('_')[0])
            if start_year <= year <= end_year:
                df = pd.read_parquet(f, columns=['symbol'])
                all_syms |= set(df['symbol'].unique())
        symbols = sorted(all_syms)
        print(f'  Found {len(symbols)} symbols', flush=True)

    n_batches = (len(symbols) + batch_size - 1) // batch_size
    use_batch = len(symbols) > batch_size
    all_results = []

    if use_batch:
        print(f'Processing in {n_batches} batches of {batch_size}...', flush=True)
        for i in range(n_batches):
            batch = symbols[i * batch_size: (i + 1) * batch_size]
            print(f'  Batch {i+1}/{n_batches}: {len(batch)} symbols...', end='', flush=True)
            results = _process_battle_batch(batch, start, end)
            print(f' {len(results)} groups', flush=True)
            all_results.extend(results)
    else:
        from data.storage.parquet import ParquetStorage
        storage = ParquetStorage(str(BARS_1MIN_DIR))
        print(f'Loading 1min data for {len(symbols)} symbols...', flush=True)
        raw = storage.load(symbols, pd.Timestamp(start, tz='US/Eastern'),
                           pd.Timestamp(end, tz='US/Eastern'), '1m')
        print(f'  Loaded {len(raw):,} bars', flush=True)

        t = raw.index.time
        raw = raw[(t >= RTH_START) & (t <= RTH_END)]
        print(f'  After RTH filter: {len(raw):,} bars', flush=True)

        grouped = raw.groupby(['symbol', raw.index.date])
        total = len(grouped)
        done = 0

        for (sym, dt), grp in grouped:
            r1, r2, r3 = _compute_one_day_battle(
                grp['close'].values, grp['high'].values,
                grp['low'].values, grp['volume'].values)
            all_results.append((sym, pd.Timestamp(dt), r1, r2, r3))
            done += 1
            if done % 50000 == 0:
                print(f'  Processed {done}/{total} groups...', flush=True)

        print(f'  Done: {done} groups', flush=True)
        del raw, grouped

    print(f'Total groups: {len(all_results)}', flush=True)

    res_df = pd.DataFrame(all_results, columns=['symbol', 'date', 'vb_ret', 'vb_pos', 'ab'])
    del all_results

    vb_ret = res_df.pivot(index='date', columns='symbol', values='vb_ret')
    vb_pos = res_df.pivot(index='date', columns='symbol', values='vb_pos')
    ab_ret = res_df.pivot(index='date', columns='symbol', values='ab')
    del res_df

    daily = {'vol_battle_ret': vb_ret, 'vol_battle_pos': vb_pos, 'amp_battle': ab_ret}

    print('\nAggregating to weekly...', flush=True)
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

    for name, df in weekly.items():
        n_valid = df.iloc[-1].notna().sum() if len(df) > 0 else 0
        print(f'  {name}: {df.shape[0]} weeks × {df.shape[1]} symbols, '
              f'latest coverage={n_valid}', flush=True)

    return weekly


# ═════════════════════════════════════════════════════════════
# 6. 激增因子 (3)  —  耀眼波动率, 耀眼收益率, 适度冒险
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

        if (i + 1) % 20 == 0:
            print(f'    surge_metrics: {i+1}/{len(dates)} days', flush=True)

    print(f'    surge_metrics: {len(dates)}/{len(dates)} days', flush=True)

    cols = close_1m.columns
    daily_dv = pd.DataFrame(dv_results).T.sort_index().reindex(columns=cols)
    daily_dr = pd.DataFrame(dr_results).T.sort_index().reindex(columns=cols)
    return daily_dv, daily_dr


def compute_surge_factors(
    symbols: List[str],
    start: str = '2025-10-01',
    end: str = '2026-12-31',
) -> Dict[str, pd.DataFrame]:
    print('Loading 1min data...', flush=True)
    close_1m, volume_1m = _load_1min_panels(symbols, start, end)

    print('\nComputing daily surge metrics...', flush=True)
    daily_dv, daily_dr = _compute_daily_surge_metrics(close_1m, volume_1m)
    print(f'  daily_dazzling_vol: {daily_dv.shape}')
    print(f'  daily_dazzling_ret: {daily_dr.shape}')

    print('\nComputing weekly_dazzling_vol (5d rolling)...', flush=True)
    wdv = _moderate_rolling_factor(daily_dv, 5, negate=True)
    n = wdv.iloc[-1].notna().sum() if len(wdv) > 0 else 0
    print(f'  weekly_dazzling_vol: {wdv.shape}, coverage={n}')

    print('\nComputing weekly_dazzling_ret (5d rolling)...', flush=True)
    wdr = _moderate_rolling_factor(daily_dr, 5, negate=False)
    n = wdr.iloc[-1].notna().sum() if len(wdr) > 0 else 0
    print(f'  weekly_dazzling_ret: {wdr.shape}, coverage={n}')

    print('\nComputing moderate_risk (composite)...', flush=True)
    common_idx = wdv.index.intersection(wdr.index)
    common_cols = wdv.columns.intersection(wdr.columns)
    valid_rows = (wdv.loc[common_idx, common_cols].notna().any(axis=1)
                  & wdr.loc[common_idx, common_cols].notna().any(axis=1))
    z1 = _cross_sectional_zscore(wdv.loc[common_idx[valid_rows], common_cols])
    z2 = _cross_sectional_zscore(wdr.loc[common_idx[valid_rows], common_cols])
    mr = ((z1 + z2) / 2).reindex(index=common_idx, columns=common_cols)
    n = mr.iloc[-1].notna().sum() if len(mr) > 0 else 0
    print(f'  moderate_risk: {mr.shape}, coverage={n}')

    return {'weekly_dazzling_vol': wdv, 'weekly_dazzling_ret': wdr, 'moderate_risk': mr}


# ═════════════════════════════════════════════════════════════
# 7. 回归因子 (4)  —  朝没晨雾, 午蔽古木, 夜眠霜路, 花隐林间
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

        if (i + 1) % 20 == 0:
            print(f'    regression: {i+1}/{len(dates)} days', flush=True)

    print(f'    regression: {len(dates)}/{len(dates)} days', flush=True)

    cols = close_1m.columns
    return (
        pd.DataFrame(mm_res).T.sort_index().reindex(columns=cols),
        pd.DataFrame(at_res).T.sort_index().reindex(columns=cols),
        pd.DataFrame(fa_res).T.sort_index().reindex(columns=cols),
        pd.DataFrame(ti_res).T.sort_index().reindex(columns=cols),
    )


def _factor_night_frost(daily_t_intercept, window=5):
    """夜眠霜路: rolling pairwise |corr| of t_intercept"""
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

        done = i - window + 2
        total = len(dates) - window + 1
        if done % 20 == 0:
            print(f'    night_frost: {done}/{total}', flush=True)

    total = len(dates) - window + 1
    print(f'    night_frost: {total}/{total}', flush=True)
    df = pd.DataFrame(results).T.sort_index()
    return df.reindex(columns=daily_t_intercept.columns)


def _factor_noon_shade(daily_abs_ti, daily_F_all, window=5):
    """午蔽古木: F-flip → rolling mean → z-score"""
    F_cs_mean = daily_F_all.mean(axis=1)
    is_low_F = daily_F_all.lt(F_cs_mean, axis=0)

    daily_noon = daily_abs_ti.copy()
    daily_noon[is_low_F] = -daily_noon[is_low_F]

    return _rolling_mean_zscore(daily_noon, window)


def compute_regression_factors(
    symbols: List[str],
    start: str = '2025-10-01',
    end: str = '2026-12-31',
) -> Dict[str, pd.DataFrame]:
    print('Loading 1min data...', flush=True)
    close_1m, volume_1m = _load_1min_panels(symbols, start, end)

    print('\nRunning daily intraday regressions...', flush=True)
    daily_mm, daily_abs_ti, daily_fa, daily_ti = _compute_daily_regression(
        close_1m, volume_1m)
    print(f'  daily shapes: mm={daily_mm.shape}, abs_ti={daily_abs_ti.shape}, '
          f'F={daily_fa.shape}, ti={daily_ti.shape}')

    print('\nComputing morning_mist (5d mean of std(t1..t5))...', flush=True)
    mm = _rolling_mean_zscore(daily_mm)
    n = mm.iloc[-1].notna().sum() if len(mm) > 0 else 0
    print(f'  morning_mist: {mm.shape}, coverage={n}')

    print('\nComputing noon_shade (F-flip + 5d mean)...', flush=True)
    ns = _factor_noon_shade(daily_abs_ti, daily_fa)
    n = ns.iloc[-1].notna().sum() if len(ns) > 0 else 0
    print(f'  noon_shade: {ns.shape}, coverage={n}')

    print('\nComputing night_frost (pairwise |corr| of t_intercept)...', flush=True)
    nf = _factor_night_frost(daily_ti)
    n = nf.iloc[-1].notna().sum() if len(nf) > 0 else 0
    print(f'  night_frost: {nf.shape}, coverage={n}')

    print('\nComputing flower_hidden (composite)...', flush=True)
    common_idx = mm.index.intersection(ns.index).intersection(nf.index)
    common_cols = mm.columns.intersection(ns.columns).intersection(nf.columns)
    valid_rows = (mm.loc[common_idx, common_cols].notna().any(axis=1)
                  & ns.loc[common_idx, common_cols].notna().any(axis=1)
                  & nf.loc[common_idx, common_cols].notna().any(axis=1))
    z1 = _cross_sectional_zscore(mm.loc[common_idx[valid_rows], common_cols])
    z2 = _cross_sectional_zscore(ns.loc[common_idx[valid_rows], common_cols])
    z3 = _cross_sectional_zscore(nf.loc[common_idx[valid_rows], common_cols])
    fh = ((z1 + z2 + z3) / 3).reindex(index=common_idx, columns=common_cols)
    n = fh.iloc[-1].notna().sum() if len(fh) > 0 else 0
    print(f'  flower_hidden: {fh.shape}, coverage={n}')

    return {'morning_mist': mm, 'noon_shade': ns, 'night_frost': nf, 'flower_hidden': fh}


# ═════════════════════════════════════════════════════════════
# 8. 模糊性因子 (3)  —  模糊关联度, 模糊金额比, 模糊数量比
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

        if (i + 1) % 20 == 0:
            print(f'    fuzzy_metrics: {i+1}/{len(dates)} days', flush=True)

    print(f'    fuzzy_metrics: {len(dates)}/{len(dates)} days', flush=True)

    cols = close_1m.columns
    return (
        pd.DataFrame(fc_res).T.sort_index().reindex(columns=cols),
        pd.DataFrame(far_res).T.sort_index().reindex(columns=cols),
        pd.DataFrame(fvr_res).T.sort_index().reindex(columns=cols),
    )


def compute_fuzzy_factors(
    symbols: List[str],
    start: str = '2025-10-01',
    end: str = '2026-12-31',
) -> Dict[str, pd.DataFrame]:
    print('Loading 1min data...', flush=True)
    close_1m, volume_1m = _load_1min_panels(symbols, start, end)

    print('\nComputing daily fuzzy metrics...', flush=True)
    daily_fc, daily_far, daily_fvr = _compute_daily_fuzzy_metrics(close_1m, volume_1m)
    print(f'  daily_fuzzy_corr: {daily_fc.shape}')
    print(f'  daily_fuzzy_amount_ratio: {daily_far.shape}')
    print(f'  daily_fuzzy_volume_ratio: {daily_fvr.shape}')

    print('\nComputing fuzzy_corr (5d rolling)...', flush=True)
    fc = _rolling_mean_std_composite(daily_fc)
    n = fc.iloc[-1].notna().sum() if len(fc) > 0 else 0
    print(f'  fuzzy_corr: {fc.shape}, coverage={n}')

    print('\nComputing fuzzy_amount_ratio (5d rolling)...', flush=True)
    far = _rolling_mean_std_composite(daily_far)
    n = far.iloc[-1].notna().sum() if len(far) > 0 else 0
    print(f'  fuzzy_amount_ratio: {far.shape}, coverage={n}')

    print('\nComputing fuzzy_volume_ratio (5d rolling)...', flush=True)
    fvr = _rolling_mean_std_composite(daily_fvr)
    n = fvr.iloc[-1].notna().sum() if len(fvr) > 0 else 0
    print(f'  fuzzy_volume_ratio: {fvr.shape}, coverage={n}')

    return {'fuzzy_corr': fc, 'fuzzy_amount_ratio': far, 'fuzzy_volume_ratio': fvr}


# ═════════════════════════════════════════════════════════════
# 9. 灾后重建因子 (2)  —  disaster_rebuild, peak_climbing
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

        if (i + 1) % 20 == 0:
            print(f'    rebuild_metrics: {i+1}/{len(dates)} days', flush=True)

    print(f'    rebuild_metrics: {len(dates)}/{len(dates)} days', flush=True)

    cols = close_1m.columns
    return (
        pd.DataFrame(rebuild_res).T.sort_index().reindex(columns=cols),
        pd.DataFrame(climb_res).T.sort_index().reindex(columns=cols),
    )


def compute_rebuild_factors(
    symbols: List[str],
    start: str = '2025-10-01',
    end: str = '2026-12-31',
) -> Dict[str, pd.DataFrame]:
    print('Loading 1min OHLCV data...', flush=True)
    panels_1m = _load_1min_ohlcv(symbols, start, end)

    print('\nComputing daily rebuild metrics...', flush=True)
    daily_rb, daily_cl = _compute_daily_rebuild_metrics(panels_1m)
    print(f'  daily_rebuild: {daily_rb.shape}')
    print(f'  daily_climb: {daily_cl.shape}')

    print('\nComputing disaster_rebuild (5d rolling, negated)...', flush=True)
    rb = _rolling_mean_std_composite(daily_rb, 5, negate=True)
    n = rb.iloc[-1].notna().sum() if len(rb) > 0 else 0
    print(f'  disaster_rebuild: {rb.shape}, coverage={n}')

    print('\nComputing peak_climbing (5d rolling)...', flush=True)
    cl = _rolling_mean_std_composite(daily_cl, 5, negate=False)
    n = cl.iloc[-1].notna().sum() if len(cl) > 0 else 0
    print(f'  peak_climbing: {cl.shape}, coverage={n}')

    return {'disaster_rebuild': rb, 'peak_climbing': cl}


# ═════════════════════════════════════════════════════════════
# 10. 潮汐因子 (4)  —  全潮汐, 强势半潮汐, 激进弱势, 稳定弱势
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

        if (i + 1) % 20 == 0:
            print(f'    tidal_metrics: {i+1}/{len(dates)} days', flush=True)

    print(f'    tidal_metrics: {len(dates)}/{len(dates)} days', flush=True)

    cols = close_1m.columns
    return (
        pd.DataFrame(full_res).T.sort_index().reindex(columns=cols),
        pd.DataFrame(strong_res).T.sort_index().reindex(columns=cols),
        pd.DataFrame(weak_res).T.sort_index().reindex(columns=cols),
    )


def compute_tidal_factors(
    symbols: List[str],
    start: str = '2025-10-01',
    end: str = '2026-12-31',
) -> Dict[str, pd.DataFrame]:
    print('Loading 1min data...', flush=True)
    close_1m, volume_1m = _load_1min_panels(symbols, start, end)

    print('\nComputing daily tidal metrics...', flush=True)
    daily_full, daily_strong, daily_weak = _compute_daily_tidal_metrics(
        close_1m, volume_1m)
    print(f'  daily_full: {daily_full.shape}')
    print(f'  daily_strong: {daily_strong.shape}')
    print(f'  daily_weak: {daily_weak.shape}')

    print('\nComputing full_tidal (5d rolling mean)...', flush=True)
    ft = _rolling_mean_zscore(daily_full)
    n = ft.iloc[-1].notna().sum() if len(ft) > 0 else 0
    print(f'  full_tidal: {ft.shape}, coverage={n}')

    print('\nComputing strong_half_tidal (5d rolling mean)...', flush=True)
    sht = _rolling_mean_zscore(daily_strong)
    n = sht.iloc[-1].notna().sum() if len(sht) > 0 else 0
    print(f'  strong_half_tidal: {sht.shape}, coverage={n}')

    print('\nComputing aggressive_weak_half (5d rolling mean)...', flush=True)
    awh = _rolling_mean_zscore(daily_weak)
    n = awh.iloc[-1].notna().sum() if len(awh) > 0 else 0
    print(f'  aggressive_weak_half: {awh.shape}, coverage={n}')

    print('\nComputing stable_weak_half (5d rolling std, negated)...', flush=True)
    min_p = max(5 - 1, 3)
    rolling_std = daily_weak.rolling(5, min_periods=min_p).std()
    valid_rows = rolling_std.notna().any(axis=1)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        swh = -_cross_sectional_zscore(rolling_std.loc[valid_rows])
    swh = swh.reindex(daily_weak.index)
    n = swh.iloc[-1].notna().sum() if len(swh) > 0 else 0
    print(f'  stable_weak_half: {swh.shape}, coverage={n}')

    return {
        'full_tidal': ft, 'strong_half_tidal': sht,
        'aggressive_weak_half': awh, 'stable_weak_half': swh,
    }


# ═════════════════════════════════════════════════════════════
# 11. 跳跃因子 (5)  —  周跳跃度, 修正振幅1/2, 修正振幅, 飞蛾扑火
# ═════════════════════════════════════════════════════════════

def _compute_daily_jump(close_1m):
    """1min 泰勒残项 → 日均值"""
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

        if (i + 1) % 20 == 0:
            print(f'    daily_jump: {i+1}/{len(dates)} days', flush=True)

    print(f'    daily_jump: {len(dates)}/{len(dates)} days', flush=True)
    result = pd.DataFrame(jump_res).T.sort_index()
    return result.reindex(columns=close_1m.columns)


def _compute_flipped_amplitude_1(daily_jump, daily_high, daily_low, daily_close):
    """修正振幅1: 跳跃度翻转振幅"""
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
    """修正振幅2: low→high 泰勒残项翻转振幅"""
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
    symbols: List[str],
    start: str = '2025-10-01',
    end: str = '2026-12-31',
) -> Dict[str, pd.DataFrame]:
    from .data_loader import load_price_panel

    print('Loading 1min data...', flush=True)
    close_1m, _ = _load_1min_panels(symbols, start, end)

    print('\nComputing daily jump degree (from 1min Taylor residual)...', flush=True)
    daily_jump = _compute_daily_jump(close_1m)
    print(f'  daily_jump: {daily_jump.shape}')

    print('\nLoading daily OHLC...', flush=True)
    panels_1d = load_price_panel(symbols, start, end)
    daily_high = panels_1d['high']
    daily_low = panels_1d['low']
    daily_close = panels_1d['close']
    print(f'  daily OHLC: {daily_close.shape}')

    print('\nComputing flipped amplitude 1 (jump-based flip)...', flush=True)
    fa1 = _compute_flipped_amplitude_1(daily_jump, daily_high, daily_low, daily_close)
    print(f'  flipped_amp1: {fa1.shape}')

    print('Computing flipped amplitude 2 (low→high Taylor flip)...', flush=True)
    fa2 = _compute_flipped_amplitude_2(daily_high, daily_low, daily_close)
    print(f'  flipped_amp2: {fa2.shape}')

    # weekly_jump: rolling mean+std → z-score
    print('\nComputing weekly_jump (5d rolling mean+std)...', flush=True)
    wj = _rolling_mean_std_composite(daily_jump)
    n = wj.iloc[-1].notna().sum() if len(wj) > 0 else 0
    print(f'  weekly_jump: {wj.shape}, coverage={n}')

    # modified_amplitude_1: rolling mean → z-score
    print('\nComputing modified_amplitude_1 (5d rolling mean)...', flush=True)
    ma1 = _rolling_mean_zscore(fa1)
    n = ma1.iloc[-1].notna().sum() if len(ma1) > 0 else 0
    print(f'  modified_amplitude_1: {ma1.shape}, coverage={n}')

    # modified_amplitude_2: rolling mean → z-score
    print('\nComputing modified_amplitude_2 (5d rolling mean)...', flush=True)
    ma2 = _rolling_mean_zscore(fa2)
    n = ma2.iloc[-1].notna().sum() if len(ma2) > 0 else 0
    print(f'  modified_amplitude_2: {ma2.shape}, coverage={n}')

    # modified_amplitude: z(ma1) + z(ma2) 等权
    print('\nComputing modified_amplitude (composite)...', flush=True)
    common_idx = ma1.index.intersection(ma2.index)
    common_cols = ma1.columns.intersection(ma2.columns)
    valid_rows = (ma1.loc[common_idx, common_cols].notna().any(axis=1)
                  & ma2.loc[common_idx, common_cols].notna().any(axis=1))
    z1 = _cross_sectional_zscore(ma1.loc[common_idx[valid_rows], common_cols])
    z2 = _cross_sectional_zscore(ma2.loc[common_idx[valid_rows], common_cols])
    ma = ((z1 + z2) / 2).reindex(index=common_idx, columns=common_cols)
    n = ma.iloc[-1].notna().sum() if len(ma) > 0 else 0
    print(f'  modified_amplitude: {ma.shape}, coverage={n}')

    # moth_to_flame: z(weekly_jump) + z(modified_amplitude) 等权
    print('\nComputing moth_to_flame (composite)...', flush=True)
    common_idx = wj.index.intersection(ma.index)
    common_cols = wj.columns.intersection(ma.columns)
    valid_rows = (wj.loc[common_idx, common_cols].notna().any(axis=1)
                  & ma.loc[common_idx, common_cols].notna().any(axis=1))
    z1 = _cross_sectional_zscore(wj.loc[common_idx[valid_rows], common_cols])
    z2 = _cross_sectional_zscore(ma.loc[common_idx[valid_rows], common_cols])
    mtf = ((z1 + z2) / 2).reindex(index=common_idx, columns=common_cols)
    n = mtf.iloc[-1].notna().sum() if len(mtf) > 0 else 0
    print(f'  moth_to_flame: {mtf.shape}, coverage={n}')

    return {
        'weekly_jump': wj,
        'modified_amplitude_1': ma1,
        'modified_amplitude_2': ma2,
        'modified_amplitude': ma,
        'moth_to_flame': mtf,
    }
