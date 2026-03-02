"""
HMM Regime-Based Asset Allocation — OOS 回测

论文 setup (Baitinger & Hoch 2024, Section 4):
    - Expanding window: 初始窗口 → 每月扩展 → 重新拟合 HMM
    - 月度调仓: 月末拟合 → 计算最优权重 → 下月持有
    - 最优权重: w* = (1/γ) · (R̂_monthly / σ̂²_monthly), γ=6
    - 约束: w* ∈ [0%, 150%]
    - 基准1: Buy-and-hold (100% SPY)
    - 基准2: Dynamic (滚动 252 日均值/方差 → 同公式)
    - 交易成本: 10bp × |Δw|

日频→月频转换:
    - R̂_monthly = R̂_daily × 21  (约 21 个交易日/月)
    - σ̂²_monthly = σ̂²_daily × 21
"""
import numpy as np
import pandas as pd

from .model import fit_hmm, forecast_return_variance

TRADING_DAYS_PER_MONTH = 21


def _compute_optimal_weight(
    exp_ret_daily: float,
    exp_var_daily: float,
    gamma: float = 6.0,
    w_min: float = 0.0,
    w_max: float = 1.5,
) -> float:
    """
    论文 Eq(9): w* = (1/γ) · (R̂ / σ̂²)

    日频 → 月频: R̂_m = R̂_d × 21, σ̂²_m = σ̂²_d × 21
    代入: w* = (1/γ) · (R̂_d × 21) / (σ̂²_d × 21) = (1/γ) · (R̂_d / σ̂²_d)
    (缩放因子约掉)
    """
    if exp_var_daily <= 0:
        return 0.0

    w = (1.0 / gamma) * (exp_ret_daily / exp_var_daily)
    return float(np.clip(w, w_min, w_max))


def _monthly_return(daily_returns: pd.Series) -> float:
    """日收益率 → 月收益率 (复利)"""
    return float((1 + daily_returns).prod() - 1)


def run_backtest(
    spy_prices: pd.Series,
    tbill_rates: pd.Series,
    n_states: int = 2,
    oos_start: str = '2022-01-01',
    gamma: float = 6.0,
    n_init: int = 10,
    tc_bps: float = 10.0,
) -> dict:
    """
    HMM regime-based allocation backtest.

    Parameters
    ----------
    spy_prices : daily close prices (DatetimeIndex)
    tbill_rates : daily 3M T-Bill rate in % (DatetimeIndex)
    n_states : HMM hidden states
    oos_start : OOS period start date
    gamma : risk aversion
    n_init : HMM random initializations
    tc_bps : transaction cost in bps per unit |Δw|

    Returns
    -------
    dict with:
        results : DataFrame (monthly: date, w_hmm, ret_hmm, ret_bench1, ret_bench2, ...)
        metrics : dict of summary metrics
    """
    # ── 准备数据 ──
    daily_ret = spy_prices.pct_change().dropna()
    daily_ret.name = 'ret'

    # T-Bill: 年化% → 日收益率
    tbill_daily = tbill_rates / 100 / 252
    tbill_daily = tbill_daily.reindex(daily_ret.index, method='ffill').fillna(0)

    # 月末日期列表 (OOS 区间)
    oos_mask = daily_ret.index >= pd.Timestamp(oos_start)
    oos_dates = daily_ret[oos_mask].index
    month_ends = daily_ret.loc[oos_mask].groupby(
        daily_ret.loc[oos_mask].index.to_period('M')
    ).apply(lambda x: x.index[-1])

    # 所有月份 (包含 OOS start 前一个月末作为第一个 fitting point)
    all_month_ends = daily_ret.groupby(
        daily_ret.index.to_period('M')
    ).apply(lambda x: x.index[-1])
    oos_period_start = pd.Timestamp(oos_start).to_period('M')
    # 找到 OOS start 之前的所有月末 + OOS 内的月末
    pre_oos_ends = all_month_ends[all_month_ends.index < oos_period_start]
    oos_month_ends = all_month_ends[all_month_ends.index >= oos_period_start]

    if len(pre_oos_ends) < 12:
        raise ValueError(f'初始窗口太短: 仅 {len(pre_oos_ends)} 个月在 OOS 之前')

    # ── 月度回测循环 ──
    records = []
    prev_w_hmm = 0.0
    prev_w_bench2 = 0.0
    last_model = None
    last_fit_end = None

    for i, (period, month_end) in enumerate(oos_month_ends.items()):
        # 上一个月末 = fitting cutoff
        if i == 0:
            fit_end = pre_oos_ends.iloc[-1]
        else:
            prev_period = list(oos_month_ends.items())[i - 1]
            fit_end = prev_period[1]

        # 本月的日收益率
        month_daily = daily_ret[(daily_ret.index > fit_end) & (daily_ret.index <= month_end)]
        if month_daily.empty:
            continue
        month_tbill = tbill_daily.reindex(month_daily.index).fillna(0)

        # ── HMM 策略 ──
        train_ret = daily_ret[daily_ret.index <= fit_end].values
        try:
            model = fit_hmm(train_ret, n_states=n_states, n_init=n_init)
            last_model = model
            last_fit_end = fit_end
            exp_ret, exp_var = forecast_return_variance(model, train_ret)
            w_hmm = _compute_optimal_weight(exp_ret, exp_var, gamma=gamma)
        except Exception:
            w_hmm = prev_w_hmm  # fallback: 保持上期权重

        # ── Benchmark 2: 滚动 252 日 ──
        lookback = daily_ret[daily_ret.index <= fit_end].iloc[-252:]
        if len(lookback) >= 60:
            mu_d = lookback.mean()
            var_d = lookback.var()
            w_bench2 = _compute_optimal_weight(mu_d, var_d, gamma=gamma)
        else:
            w_bench2 = prev_w_bench2

        # ── 计算月收益 ──
        spy_month_ret = _monthly_return(month_daily)
        rf_month_ret = _monthly_return(month_tbill)

        ret_hmm = w_hmm * spy_month_ret + (1 - w_hmm) * rf_month_ret
        ret_bench1 = spy_month_ret  # buy-and-hold
        ret_bench2 = w_bench2 * spy_month_ret + (1 - w_bench2) * rf_month_ret

        # 交易成本
        tc_hmm = tc_bps / 10000 * abs(w_hmm - prev_w_hmm)
        tc_bench2 = tc_bps / 10000 * abs(w_bench2 - prev_w_bench2)

        records.append({
            'date': month_end,
            'period': str(period),
            'w_hmm': w_hmm,
            'w_bench2': w_bench2,
            'ret_hmm': ret_hmm,
            'ret_hmm_atc': ret_hmm - tc_hmm,
            'ret_bench1': ret_bench1,
            'ret_bench2': ret_bench2,
            'ret_bench2_atc': ret_bench2 - tc_bench2,
            'turnover_hmm': abs(w_hmm - prev_w_hmm),
            'turnover_bench2': abs(w_bench2 - prev_w_bench2),
        })

        prev_w_hmm = w_hmm
        prev_w_bench2 = w_bench2

    results = pd.DataFrame(records).set_index('date')

    # ── 汇总指标 ──
    metrics = {}
    for label, ret_col, ret_atc_col, w_col, tv_col in [
        ('HMM', 'ret_hmm', 'ret_hmm_atc', 'w_hmm', 'turnover_hmm'),
        ('Bench_BH', 'ret_bench1', 'ret_bench1', None, None),
        ('Bench_Dyn', 'ret_bench2', 'ret_bench2_atc', 'w_bench2', 'turnover_bench2'),
    ]:
        rets = results[ret_col]
        rets_atc = results[ret_atc_col]
        n_months = len(rets)

        ann_ret = (1 + rets).prod() ** (12 / n_months) - 1
        ann_vol = rets.std() * np.sqrt(12)
        sr = ann_ret / ann_vol if ann_vol > 0 else 0

        ann_ret_atc = (1 + rets_atc).prod() ** (12 / n_months) - 1
        sr_atc = ann_ret_atc / ann_vol if ann_vol > 0 else 0

        cum = (1 + rets).cumprod()
        drawdown = cum / cum.cummax() - 1
        max_dd = drawdown.min()

        avg_w = results[w_col].mean() * 100 if w_col else 100.0
        avg_tv = results[tv_col].mean() * 100 if tv_col else 0.0

        metrics[label] = {
            'Ann Return (%)': round(ann_ret * 100, 2),
            'Ann Vol (%)': round(ann_vol * 100, 2),
            'Sharpe': round(sr, 2),
            'Sharpe ATC': round(sr_atc, 2),
            'Max DD (%)': round(max_dd * 100, 2),
            'Avg Exposure (%)': round(avg_w, 1),
            'Avg TV/Month (%)': round(avg_tv, 2),
            'End Wealth': round((1 + rets).prod() * 100, 2),
        }

    return {'results': results, 'metrics': metrics, 'model': last_model, 'fit_end': last_fit_end}
