"""
IC 分析模块

计算并打印：mean_IC, sigma_IC, ICIR, P(IC>0), P(IC>0.02),
多周期 rank IC decay, IC 半衰期, |t|均值, |t|>2占比, t均值/t标准差。

图表: rank_ic 时序(+21MA), cumsum_ic, monthly ICIR, monthly IC mean,
      yearly ICIR, yearly IC mean。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


@dataclass
class ICReport:
    """IC 分析结果"""
    # 核心指标 (per horizon)
    summary: pd.DataFrame          # rows=horizons, cols=metrics
    # 最佳 horizon
    best_horizon: int
    half_life: Optional[float]
    # 日频时序 (at best_horizon)
    daily_rank_ic: pd.Series       # daily rank IC
    daily_rank_ic_ma: pd.Series    # 21-day MA
    # 月度/年度聚合
    monthly_ic_mean: pd.DataFrame  # index=month(1-12), cols=year
    monthly_icir: pd.Series        # index=month(1-12)
    yearly_ic_mean: pd.Series      # index=year
    yearly_icir: pd.Series         # index=year
    # IC decay
    decay_ics: Dict[int, float]    # horizon → mean rank IC


def _rank_ic_series(factor: pd.DataFrame, close: pd.DataFrame,
                    horizon: int) -> pd.Series:
    """计算每日 rank IC (Spearman)。"""
    fwd_ret = close.shift(-horizon) / close - 1
    common_dates = factor.index.intersection(fwd_ret.index)

    ic_vals = []
    ic_dates = []
    for dt in common_dates:
        f = factor.loc[dt].dropna()
        r = fwd_ret.loc[dt].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 10:
            continue
        corr = f[common].rank().corr(r[common].rank())
        if not np.isnan(corr):
            ic_vals.append(corr)
            ic_dates.append(dt)

    return pd.Series(ic_vals, index=pd.DatetimeIndex(ic_dates), name=f"rank_ic_{horizon}d")


def _newey_west_tstat(series: pd.Series, max_lag: Optional[int] = None) -> float:
    """Newey-West t-stat for mean != 0。"""
    n = len(series)
    if n < 5:
        return 0.0
    mean = series.mean()
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))

    # HAC variance
    demeaned = series.values - mean
    gamma0 = np.mean(demeaned ** 2)
    nw_var = gamma0
    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)
        gamma_j = np.mean(demeaned[lag:] * demeaned[:-lag])
        nw_var += 2 * weight * gamma_j

    se = np.sqrt(nw_var / n)
    return mean / se if se > 0 else 0.0


def compute_ic_analysis(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizons: List[int] = None,
) -> ICReport:
    """
    完整 IC 分析。

    Args:
        factor: 预处理后的因子面板
        close: 收盘价面板
        horizons: 评估周期列表，默认 [1, 5, 10, 21, 42, 63]

    Returns:
        ICReport
    """
    if horizons is None:
        horizons = [1, 5, 10, 21, 42, 63]

    # ── 1. 每个 horizon 的 IC 指标 ──
    rows = []
    decay_ics = {}
    ic_series_cache = {}

    for h in horizons:
        ic_s = _rank_ic_series(factor, close, h)
        ic_series_cache[h] = ic_s

        if len(ic_s) < 5:
            rows.append({
                "horizon": h, "mean_IC": np.nan, "sigma_IC": np.nan,
                "ICIR": np.nan, "P(IC>0)": np.nan, "P(IC>0.02)": np.nan,
                "|t|_mean": np.nan, "|t|>2_ratio": np.nan, "t_mean/t_std": np.nan,
            })
            decay_ics[h] = np.nan
            continue

        mean_ic = ic_s.mean()
        sigma_ic = ic_s.std()
        icir = mean_ic / sigma_ic if sigma_ic > 0 else 0
        p_pos = (ic_s > 0).mean()
        p_gt002 = (ic_s > 0.02).mean()
        nw_t = _newey_west_tstat(ic_s)

        # |t| 和 t 统计量 — 按月分组计算 t-stat
        monthly_groups = ic_s.groupby(ic_s.index.to_period("M"))
        t_stats = []
        for _, grp in monthly_groups:
            if len(grp) >= 5:
                t = grp.mean() / (grp.std() / np.sqrt(len(grp))) if grp.std() > 0 else 0
                t_stats.append(t)
        t_arr = np.array(t_stats) if t_stats else np.array([0.0])
        abs_t_mean = np.abs(t_arr).mean()
        abs_t_gt2_ratio = (np.abs(t_arr) > 2).mean()
        t_mean_over_std = t_arr.mean() / t_arr.std() if t_arr.std() > 0 else 0

        rows.append({
            "horizon": h,
            "mean_IC": round(mean_ic, 4),
            "sigma_IC": round(sigma_ic, 4),
            "ICIR": round(icir, 4),
            "P(IC>0)": round(p_pos, 4),
            "P(IC>0.02)": round(p_gt002, 4),
            "|t|_mean": round(abs_t_mean, 4),
            "|t|>2_ratio": round(abs_t_gt2_ratio, 4),
            "t_mean/t_std": round(t_mean_over_std, 4),
        })
        decay_ics[h] = mean_ic

    summary = pd.DataFrame(rows).set_index("horizon")

    # ── 2. Best horizon (max |ICIR|) ──
    valid_icir = summary["ICIR"].dropna()
    best_horizon = int(valid_icir.abs().idxmax()) if len(valid_icir) > 0 else horizons[0]

    # ── 3. IC half-life (指数衰减拟合) ──
    half_life = _fit_half_life(decay_ics, horizons)

    # ── 4. Daily rank IC at best_horizon ──
    daily_ic = ic_series_cache.get(best_horizon, pd.Series(dtype=float))
    daily_ic_ma = daily_ic.rolling(21, min_periods=5).mean() if len(daily_ic) > 0 else daily_ic

    # ── 5. Monthly / Yearly 聚合 ──
    monthly_ic_mean, monthly_icir = _monthly_aggregation(daily_ic)
    yearly_ic_mean, yearly_icir = _yearly_aggregation(daily_ic)

    report = ICReport(
        summary=summary,
        best_horizon=best_horizon,
        half_life=half_life,
        daily_rank_ic=daily_ic,
        daily_rank_ic_ma=daily_ic_ma,
        monthly_ic_mean=monthly_ic_mean,
        monthly_icir=monthly_icir,
        yearly_ic_mean=yearly_ic_mean,
        yearly_icir=yearly_icir,
        decay_ics=decay_ics,
    )

    # ── 6. 打印 ──
    print_ic_report(report)
    return report


def _fit_half_life(decay_ics: Dict[int, float], horizons: List[int]) -> Optional[float]:
    """拟合 |IC(h)| = a * exp(-b * h), half_life = ln(2) / b。"""
    hs = np.array([h for h in horizons if not np.isnan(decay_ics.get(h, np.nan))])
    ics = np.array([np.abs(decay_ics[h]) for h in hs])

    if len(hs) < 3 or ics.max() == 0:
        return None

    # log-linear fit: log(|IC|) = log(a) - b*h
    log_ics = np.log(np.clip(ics, 1e-10, None))
    try:
        slope, intercept, r_value, _, _ = sp_stats.linregress(hs, log_ics)
        if r_value ** 2 < 0.5 or slope >= 0:
            return None
        return round(np.log(2) / (-slope), 1)
    except Exception:
        return None


def _monthly_aggregation(ic_series: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """月度 IC 聚合: pivot(month × year) + monthly ICIR。"""
    if ic_series.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    df = ic_series.to_frame("ic")
    df["month"] = df.index.month
    df["year"] = df.index.year

    # pivot: month × year → mean IC
    monthly_mean = df.pivot_table(values="ic", index="month", columns="year", aggfunc="mean")

    # monthly ICIR: IC_mean / IC_std per month
    grp = df.groupby("month")["ic"]
    icir = grp.mean() / grp.std()
    icir = icir.fillna(0)

    return monthly_mean, icir


def _yearly_aggregation(ic_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """年度 IC 聚合。"""
    if ic_series.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    grp = ic_series.groupby(ic_series.index.year)
    yearly_mean = grp.mean()
    yearly_icir = grp.mean() / grp.std()
    yearly_icir = yearly_icir.fillna(0)

    return yearly_mean, yearly_icir


def print_ic_report(report: ICReport):
    """打印格式化 IC 指标汇总表。"""
    print("\n" + "=" * 80)
    print("IC ANALYSIS SUMMARY")
    print("=" * 80)

    # 汇总表
    fmt = report.summary.copy()
    for col in ["P(IC>0)", "P(IC>0.02)", "|t|>2_ratio"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map(lambda x: f"{x:.1%}" if not np.isnan(x) else "N/A")
    for col in ["mean_IC", "sigma_IC", "ICIR", "|t|_mean", "t_mean/t_std"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")

    print(fmt.to_string())

    # IC 半衰期 + Best horizon
    print(f"\nIC Half-life: {report.half_life if report.half_life else 'N/A'} days")
    print(f"Best horizon: {report.best_horizon}d (max |ICIR|)")
    print("=" * 80 + "\n")
