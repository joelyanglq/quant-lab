"""
因子质量诊断 (Factor Quality Diagnostics)

在做 alpha 测试之前, 先确认因子 "看起来像个因子":
- 分布稳定性: 滚动均值/方差/偏度/峰度, 漂移检测
- 截面覆盖率: 每日有效值数量, 市值集中度分析
- 已知因子相关性: 与 size/value/momentum/quality/low-vol 的相关性
- 换手率与可交易性: 信号隐含换手率, 容量估计
- 行业分布: 各行业因子均值, 分位集中度, HHI, 行业中性 IC
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats
from dataclasses import dataclass, field

from .ranking import assign_quantiles


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class DistributionStability:
    """滚动分布统计"""
    rolling_mean: pd.Series       # 滚动截面均值
    rolling_var: pd.Series        # 滚动截面方差
    rolling_skew: pd.Series       # 滚动截面偏度
    rolling_kurt: pd.Series       # 滚动截面峰度
    drift_detected: bool          # 均值是否显著漂移
    drift_pvalue: float           # 趋势检验 p 值


@dataclass
class CoverageReport:
    """截面覆盖率统计"""
    daily_count: pd.Series        # 每日有效值数量
    coverage_pct: pd.Series       # 每日覆盖率 (有效/总数)
    median_coverage: float        # 中位覆盖率
    mktcap_concentration: Optional[pd.Series]  # 大盘股覆盖占比
    small_cap_bias: Optional[float]  # 因子可得性与市值的相关性


@dataclass
class KnownFactorCorrelation:
    """与已知风格因子的相关性"""
    correlations: pd.Series       # {style_name: 平均截面相关}
    rolling_corr: pd.DataFrame    # 滚动相关 (dates × styles)
    max_abs_corr: float           # 最大绝对相关
    dominant_style: str           # 最相关的风格因子


@dataclass
class TurnoverCapacity:
    """换手率与容量诊断"""
    signal_turnover: pd.Series    # 每期信号换手率
    mean_turnover: float          # 平均换手率
    median_adv_participation: Optional[float]  # 中位 ADV 参与率
    capacity_estimate_usd: Optional[float]      # 粗略容量估计


@dataclass
class IndustryDistribution:
    """行业分布诊断"""
    avg_sector_mean: pd.Series          # 各行业时序平均因子均值
    top_q_sector_pct: pd.Series         # top quintile 各行业占比 (时序平均)
    bot_q_sector_pct: pd.Series         # bottom quintile 各行业占比 (时序平均)
    hhi_top: float                       # top quintile HHI (时序平均)
    hhi_bot: float                       # bottom quintile HHI (时序平均)
    neutralized_ic: Optional[float]      # 行业中性 21d IC
    neutralized_icir: Optional[float]    # 行业中性 21d ICIR
    raw_ic_21d: Optional[float]          # 原始 21d IC (用于对比)


@dataclass
class DiagnosticReport:
    """单因子完整诊断报告"""
    factor_name: str
    distribution: DistributionStability
    coverage: CoverageReport
    known_factors: Optional[KnownFactorCorrelation]
    turnover: TurnoverCapacity
    industry: Optional[IndustryDistribution]
    pass_sanity: bool             # 是否通过全部检查
    warnings: List[str]           # 告警列表


# ═══════════════════════════════════════════════════════════════
# 1. 分布稳定性
# ═══════════════════════════════════════════════════════════════

def compute_distribution_stability(
    factor: pd.DataFrame,
    rolling_window: int = 63,
) -> DistributionStability:
    """
    计算滚动截面分布统计, 检测均值漂移.

    每日计算截面 mean/var/skew/kurt, 再滚动平滑.
    用 OLS 线性趋势检验均值是否漂移 (p < 0.01 视为漂移).

    Args:
        factor: 因子面板 (dates × symbols)
        rolling_window: 滚动窗口 (默认 63 ≈ 3个月)
    """
    min_periods = max(rolling_window // 2, 1)

    # 每日截面统计
    cs_mean = factor.mean(axis=1)
    cs_var = factor.var(axis=1)
    cs_skew = factor.skew(axis=1)
    cs_kurt = factor.kurt(axis=1)

    # 滚动平滑
    r_mean = cs_mean.rolling(rolling_window, min_periods=min_periods).mean().dropna()
    r_var = cs_var.rolling(rolling_window, min_periods=min_periods).mean().dropna()
    r_skew = cs_skew.rolling(rolling_window, min_periods=min_periods).mean().dropna()
    r_kurt = cs_kurt.rolling(rolling_window, min_periods=min_periods).mean().dropna()

    # 趋势检测: OLS rolling_mean ~ time
    drift_detected = False
    drift_pvalue = 1.0
    if len(r_mean) >= 20:
        x = np.arange(len(r_mean), dtype=float)
        y = r_mean.values
        mask = ~np.isnan(y)
        if mask.sum() >= 20:
            x_clean, y_clean = x[mask], y[mask]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            drift_pvalue = float(p_value)
            drift_detected = bool(p_value < 0.01)

    return DistributionStability(
        rolling_mean=r_mean,
        rolling_var=r_var,
        rolling_skew=r_skew,
        rolling_kurt=r_kurt,
        drift_detected=drift_detected,
        drift_pvalue=drift_pvalue,
    )


# ═══════════════════════════════════════════════════════════════
# 2. 截面覆盖率
# ═══════════════════════════════════════════════════════════════

def compute_coverage(
    factor: pd.DataFrame,
    mktcap: Optional[pd.DataFrame] = None,
) -> CoverageReport:
    """
    每日有效因子值数量 & 市值集中度分析.

    Args:
        factor: 因子面板 (dates × symbols)
        mktcap: 市值面板 (dates × symbols), 可选.
                若提供, 检查因子覆盖是否偏向大/小盘.
    """
    n_symbols = factor.shape[1]
    daily_count = factor.notna().sum(axis=1)
    coverage_pct = daily_count / n_symbols
    median_coverage = float(coverage_pct.median())

    mktcap_concentration = None
    small_cap_bias = None

    if mktcap is not None:
        # 对齐日期
        common_idx = factor.index.intersection(mktcap.index)
        common_cols = factor.columns.intersection(mktcap.columns)

        if len(common_idx) > 0 and len(common_cols) > 0:
            fac = factor.loc[common_idx, common_cols]
            mc = mktcap.loc[common_idx, common_cols]

            # 大盘股 top decile 的因子覆盖率
            conc_list = []
            bias_corrs = []
            for dt in common_idx:
                mc_row = mc.loc[dt].dropna()
                fac_valid = fac.loc[dt].notna()
                common_syms = mc_row.index.intersection(fac_valid.index)
                if len(common_syms) < 20:
                    continue
                mc_s = mc_row[common_syms]
                fv = fac_valid[common_syms].astype(float)

                # top decile 覆盖率
                top_thresh = mc_s.quantile(0.9)
                top_mask = mc_s >= top_thresh
                if top_mask.sum() > 0:
                    top_cov = fv[top_mask].mean()
                    conc_list.append(top_cov)

                # 相关性: factor availability vs log(mktcap)
                log_mc = np.log(mc_s.clip(lower=1))
                corr = fv.corr(log_mc)
                if not np.isnan(corr):
                    bias_corrs.append(corr)

            if conc_list:
                mktcap_concentration = pd.Series(conc_list, index=common_idx[:len(conc_list)])
            if bias_corrs:
                small_cap_bias = float(np.mean(bias_corrs))

    return CoverageReport(
        daily_count=daily_count,
        coverage_pct=coverage_pct,
        median_coverage=median_coverage,
        mktcap_concentration=mktcap_concentration,
        small_cap_bias=small_cap_bias,
    )


# ═══════════════════════════════════════════════════════════════
# 3. 已知因子相关性
# ═══════════════════════════════════════════════════════════════

def build_style_factor_proxies(
    close: pd.DataFrame,
    shares: Optional[pd.DataFrame] = None,
    roe: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    用现有数据构建 5 个风格因子代理:
    - size: log(close × shares), 若无 shares 则 log(close)
    - value: -close (原始价格的反向代理, 粗略)
    - momentum: 12-1 月动量 = close.shift(21) / close.shift(252) - 1
    - quality: ROE (若提供)
    - low_vol: -rolling_std(returns, 60)
    """
    proxies = {}

    # Size
    if shares is not None:
        common_idx = close.index.intersection(shares.index)
        common_cols = close.columns.intersection(shares.columns)
        mktcap = close.loc[common_idx, common_cols] * shares.loc[common_idx, common_cols]
        proxies['size'] = np.log(mktcap.clip(lower=1))
    else:
        proxies['size'] = np.log(close.clip(lower=1))

    # Value (crude: negative price as proxy)
    proxies['value'] = -close

    # Momentum: RS_12M (skip most recent month)
    proxies['momentum'] = close.shift(21) / close.shift(252) - 1

    # Quality
    if roe is not None:
        proxies['quality'] = roe
    else:
        proxies['quality'] = None

    # Low Vol
    daily_ret = close.pct_change()
    proxies['low_vol'] = -daily_ret.rolling(60, min_periods=30).std()

    return {k: v for k, v in proxies.items() if v is not None}


def compute_known_factor_correlation(
    factor: pd.DataFrame,
    style_factors: Dict[str, pd.DataFrame],
    rolling_window: int = 52,
) -> KnownFactorCorrelation:
    """
    计算目标因子与每个风格因子的截面 Spearman 相关.

    每日: Spearman(factor_rank, style_rank) across symbols.
    然后取时序均值 + 滚动相关.
    """
    corr_series = {}

    for style_name, style_panel in style_factors.items():
        common_idx = factor.index.intersection(style_panel.index)
        common_cols = factor.columns.intersection(style_panel.columns)

        if len(common_idx) == 0 or len(common_cols) == 0:
            continue

        daily_corrs = {}
        for dt in common_idx:
            f = factor.loc[dt, common_cols].dropna()
            s = style_panel.loc[dt, common_cols].reindex(f.index).dropna()
            common = f.index.intersection(s.index)
            if len(common) < 20:
                continue
            corr = f[common].corr(s[common], method='spearman')
            if not np.isnan(corr):
                daily_corrs[dt] = corr

        if daily_corrs:
            corr_series[style_name] = pd.Series(daily_corrs)

    if not corr_series:
        return KnownFactorCorrelation(
            correlations=pd.Series(dtype=float),
            rolling_corr=pd.DataFrame(),
            max_abs_corr=0.0,
            dominant_style='none',
        )

    # 平均相关
    avg_corrs = pd.Series({k: v.mean() for k, v in corr_series.items()})

    # 滚动相关
    all_corrs = pd.DataFrame(corr_series)
    min_p = max(rolling_window // 2, 1)
    rolling_corr = all_corrs.rolling(rolling_window, min_periods=min_p).mean()

    max_abs = float(avg_corrs.abs().max())
    dominant = avg_corrs.abs().idxmax()

    return KnownFactorCorrelation(
        correlations=avg_corrs,
        rolling_corr=rolling_corr,
        max_abs_corr=max_abs,
        dominant_style=dominant,
    )


# ═══════════════════════════════════════════════════════════════
# 4. 换手率与容量
# ═══════════════════════════════════════════════════════════════

def compute_turnover_capacity(
    factor: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    close: Optional[pd.DataFrame] = None,
    top_quantile_pct: float = 0.2,
    participation_rate: float = 0.05,
) -> TurnoverCapacity:
    """
    信号隐含换手率与容量估计.

    换手率: 每期 top quantile 成分变化比例.
    容量: median(top_q 股票 20d ADV) × participation_rate × n_stocks.

    Args:
        factor: 因子面板 (dates × symbols)
        volume: 日成交量面板, 可选
        close: 日收盘价面板, 可选
        top_quantile_pct: top bucket 占比 (默认 20%)
        participation_rate: 参与率上限 (默认 5%)
    """
    # 信号换手率: 每日 top 组成分变化
    n_top = max(1, int(factor.shape[1] * top_quantile_pct))
    turnover_list = []
    prev_top = None

    for dt in factor.index:
        row = factor.loc[dt].dropna()
        if len(row) < n_top:
            continue
        top = set(row.nlargest(n_top).index)
        if prev_top is not None:
            changed = len(top.symmetric_difference(prev_top))
            turnover_list.append((dt, changed / (2 * n_top)))
        prev_top = top

    signal_turnover = pd.Series(
        [t[1] for t in turnover_list],
        index=pd.DatetimeIndex([t[0] for t in turnover_list]),
    ) if turnover_list else pd.Series(dtype=float)

    mean_turnover = float(signal_turnover.mean()) if len(signal_turnover) > 0 else 0.0

    # 容量估计
    median_adv_participation = None
    capacity_estimate_usd = None

    if volume is not None and close is not None:
        common_cols = factor.columns.intersection(volume.columns).intersection(close.columns)
        if len(common_cols) > 0:
            dollar_volume = (close[common_cols] * volume[common_cols])
            adv_20d = dollar_volume.rolling(20, min_periods=10).mean()

            # 最近一行的 top quantile 股票的 ADV
            last_factor = factor[common_cols].iloc[-1].dropna()
            if len(last_factor) >= n_top:
                top_syms = last_factor.nlargest(n_top).index
                last_adv = adv_20d.iloc[-1].reindex(top_syms).dropna()
                if len(last_adv) > 0:
                    med_adv = float(last_adv.median())
                    median_adv_participation = participation_rate
                    capacity_estimate_usd = med_adv * participation_rate * len(last_adv)

    return TurnoverCapacity(
        signal_turnover=signal_turnover,
        mean_turnover=mean_turnover,
        median_adv_participation=median_adv_participation,
        capacity_estimate_usd=capacity_estimate_usd,
    )


# ═══════════════════════════════════════════════════════════════
# 5. 行业分布
# ═══════════════════════════════════════════════════════════════

def compute_industry_distribution(
    factor: pd.DataFrame,
    gics_df: pd.DataFrame,
    close: Optional[pd.DataFrame] = None,
    top_quantile_pct: float = 0.2,
    ic_horizon: int = 21,
) -> IndustryDistribution:
    """
    行业分布诊断: 因子是否只是行业押注?

    计算:
        1. 各行业因子均值 (时序平均) — 因子是否系统性偏向某些行业
        2. Top/Bottom quintile 行业占比 — 多空头是否集中在少数行业
        3. HHI 集中度指数 — 量化行业集中程度 (1/n_sectors = 均匀分布)
        4. 行业中性 IC — 去行业均值后的残差 rank IC (需要 close)

    Args:
        factor: 因子面板 (dates × symbols)
        gics_df: GICS 映射, 至少含 'symbol' 和 'gics_sector' 列
        close: 收盘价面板, 可选 (计算行业中性 IC 用)
        top_quantile_pct: top/bottom bucket 占比 (默认 20%)
        ic_horizon: 行业中性 IC 的前瞻期 (默认 21d)
    """
    # 构建 symbol → sector 映射
    sector_map = gics_df.set_index('symbol')['gics_sector']
    valid_symbols = factor.columns.intersection(sector_map.index)

    if len(valid_symbols) < 20:
        return IndustryDistribution(
            avg_sector_mean=pd.Series(dtype=float),
            top_q_sector_pct=pd.Series(dtype=float),
            bot_q_sector_pct=pd.Series(dtype=float),
            hhi_top=0.0, hhi_bot=0.0,
            neutralized_ic=None, neutralized_icir=None, raw_ic_21d=None,
        )

    factor_valid = factor[valid_symbols]
    sectors = sector_map[valid_symbols]
    all_sectors = sorted(sectors.unique())
    n_top = max(1, int(len(valid_symbols) * top_quantile_pct))

    # ── 1. 各行业因子均值 (每日 → 时序平均) ──────────────
    sector_means_daily = {}
    for dt in factor_valid.index:
        row = factor_valid.loc[dt].dropna()
        if len(row) < 20:
            continue
        for sec in all_sectors:
            syms = row.index.intersection(sectors[sectors == sec].index)
            if len(syms) >= 2:
                sector_means_daily.setdefault(sec, []).append(row[syms].mean())

    avg_sector_mean = pd.Series({
        sec: np.mean(vals) for sec, vals in sector_means_daily.items()
    }).sort_values()

    # ── 2. Top/Bottom quintile 行业占比 ──────────────────
    top_counts = pd.Series(0.0, index=all_sectors)
    bot_counts = pd.Series(0.0, index=all_sectors)
    n_days = 0

    for dt in factor_valid.index:
        row = factor_valid.loc[dt].dropna()
        if len(row) < n_top * 2:
            continue
        n_days += 1
        top_syms = row.nlargest(n_top).index
        bot_syms = row.nsmallest(n_top).index
        for sym in top_syms:
            sec = sectors.get(sym)
            if sec in top_counts.index:
                top_counts[sec] += 1
        for sym in bot_syms:
            sec = sectors.get(sym)
            if sec in bot_counts.index:
                bot_counts[sec] += 1

    if n_days > 0:
        top_q_pct = (top_counts / top_counts.sum()).sort_values(ascending=False)
        bot_q_pct = (bot_counts / bot_counts.sum()).sort_values(ascending=False)
    else:
        top_q_pct = pd.Series(dtype=float)
        bot_q_pct = pd.Series(dtype=float)

    # ── 3. HHI 集中度 ─────────────────────────────────────
    hhi_top = float((top_q_pct ** 2).sum()) if len(top_q_pct) > 0 else 0.0
    hhi_bot = float((bot_q_pct ** 2).sum()) if len(bot_q_pct) > 0 else 0.0

    # ── 4. 行业中性 IC ─────────────────────────────────────
    neutralized_ic = None
    neutralized_icir = None
    raw_ic_21d = None

    if close is not None:
        fwd_ret = close.pct_change(ic_horizon).shift(-ic_horizon)
        common_cols = factor_valid.columns.intersection(fwd_ret.columns)
        common_idx = factor_valid.index.intersection(fwd_ret.index)

        if len(common_cols) >= 20 and len(common_idx) > ic_horizon:
            f_aligned = factor_valid.loc[common_idx, common_cols]
            r_aligned = fwd_ret.loc[common_idx, common_cols]

            raw_ics = []
            neut_ics = []

            for dt in common_idx:
                f_row = f_aligned.loc[dt].dropna()
                r_row = r_aligned.loc[dt].reindex(f_row.index).dropna()
                common = f_row.index.intersection(r_row.index)
                if len(common) < 20:
                    continue

                # Raw IC
                raw_ic = f_row[common].corr(r_row[common], method='spearman')
                if not np.isnan(raw_ic):
                    raw_ics.append(raw_ic)

                # 行业中性: 因子值减去行业均值
                f_sub = f_row[common].copy()
                sec_sub = sectors.reindex(common).dropna()
                valid_mask = f_sub.index.intersection(sec_sub.index)
                if len(valid_mask) < 20:
                    continue

                f_neut = f_sub[valid_mask].copy()
                for sec in sec_sub[valid_mask].unique():
                    sec_syms = sec_sub[sec_sub == sec].index.intersection(f_neut.index)
                    if len(sec_syms) >= 2:
                        f_neut[sec_syms] -= f_neut[sec_syms].mean()

                r_sub = r_row.reindex(f_neut.index).dropna()
                common2 = f_neut.index.intersection(r_sub.index)
                if len(common2) >= 20:
                    neut_ic = f_neut[common2].corr(r_sub[common2], method='spearman')
                    if not np.isnan(neut_ic):
                        neut_ics.append(neut_ic)

            if raw_ics:
                raw_ic_21d = float(np.mean(raw_ics))
            if neut_ics:
                neut_arr = np.array(neut_ics)
                neutralized_ic = float(neut_arr.mean())
                std = neut_arr.std()
                if std > 0 and len(neut_arr) > 1:
                    neutralized_icir = float(
                        neut_arr.mean() / std * np.sqrt(len(neut_arr))
                    )

    return IndustryDistribution(
        avg_sector_mean=avg_sector_mean,
        top_q_sector_pct=top_q_pct,
        bot_q_sector_pct=bot_q_pct,
        hhi_top=hhi_top,
        hhi_bot=hhi_bot,
        neutralized_ic=neutralized_ic,
        neutralized_icir=neutralized_icir,
        raw_ic_21d=raw_ic_21d,
    )


# ═══════════════════════════════════════════════════════════════
# 6. Master 诊断函数
# ═══════════════════════════════════════════════════════════════

def run_diagnostics(
    factor: pd.DataFrame,
    factor_name: str,
    close: Optional[pd.DataFrame] = None,
    volume: Optional[pd.DataFrame] = None,
    shares: Optional[pd.DataFrame] = None,
    roe: Optional[pd.DataFrame] = None,
    gics_df: Optional[pd.DataFrame] = None,
    style_factors: Optional[Dict[str, pd.DataFrame]] = None,
    rolling_window: int = 63,
    min_coverage: float = 0.3,
    max_style_corr: float = 0.7,
    max_drift_pvalue: float = 0.01,
) -> DiagnosticReport:
    """
    单因子完整诊断.

    编排:
        1. compute_distribution_stability
        2. compute_coverage
        3. compute_known_factor_correlation (若有 close)
        4. compute_turnover_capacity
        5. compute_industry_distribution (若有 gics_df)

    告警:
        - 覆盖率 < min_coverage
        - 均值漂移 p < max_drift_pvalue
        - |corr| > max_style_corr (与已知因子高度相关)
        - 平均换手 > 0.8
        - HHI > 0.25 (行业过度集中)
    """
    warnings_list = []

    # 1. 分布稳定性
    dist = compute_distribution_stability(factor, rolling_window)
    if dist.drift_detected:
        warnings_list.append(
            f"分布漂移: 截面均值存在显著线性趋势 (p={dist.drift_pvalue:.4f})"
        )

    # 2. 覆盖率
    mktcap = None
    if close is not None and shares is not None:
        common_idx = close.index.intersection(shares.index)
        common_cols = close.columns.intersection(shares.columns)
        if len(common_idx) > 0 and len(common_cols) > 0:
            mktcap = close.loc[common_idx, common_cols] * shares.loc[common_idx, common_cols]

    cov = compute_coverage(factor, mktcap)
    if cov.median_coverage < min_coverage:
        warnings_list.append(
            f"覆盖率不足: 中位覆盖率 {cov.median_coverage:.1%} < {min_coverage:.0%}"
        )

    # 3. 已知因子相关性
    known = None
    if close is not None:
        if style_factors is None:
            style_factors = build_style_factor_proxies(close, shares, roe)
        known = compute_known_factor_correlation(factor, style_factors)
        if known.max_abs_corr > max_style_corr:
            warnings_list.append(
                f"与已知因子高度相关: {known.dominant_style} "
                f"(|corr|={known.max_abs_corr:.3f} > {max_style_corr})"
            )

    # 4. 换手率
    turn = compute_turnover_capacity(factor, volume, close)
    if turn.mean_turnover > 0.8:
        warnings_list.append(
            f"换手率过高: {turn.mean_turnover:.1%} > 80%"
        )

    # 5. 行业分布
    industry = None
    if gics_df is not None and len(gics_df) > 0:
        industry = compute_industry_distribution(factor, gics_df, close)
        # HHI > 0.25 说明行业过度集中 (均匀分布 11 行业 HHI ≈ 0.09)
        if industry.hhi_top > 0.25:
            top_sec = industry.top_q_sector_pct.idxmax() if len(industry.top_q_sector_pct) > 0 else '?'
            warnings_list.append(
                f"多头行业集中: HHI={industry.hhi_top:.3f} > 0.25, "
                f"最大行业={top_sec} ({industry.top_q_sector_pct.max():.0%})"
            )

    pass_sanity = len(warnings_list) == 0

    return DiagnosticReport(
        factor_name=factor_name,
        distribution=dist,
        coverage=cov,
        known_factors=known,
        turnover=turn,
        industry=industry,
        pass_sanity=pass_sanity,
        warnings=warnings_list,
    )


def run_batch_diagnostics(
    factors: Dict[str, pd.DataFrame],
    close: Optional[pd.DataFrame] = None,
    volume: Optional[pd.DataFrame] = None,
    shares: Optional[pd.DataFrame] = None,
    roe: Optional[pd.DataFrame] = None,
    gics_df: Optional[pd.DataFrame] = None,
    **kwargs,
) -> Dict[str, DiagnosticReport]:
    """
    批量因子诊断. style_factors 只构建一次.
    """
    style_factors = None
    if close is not None:
        style_factors = build_style_factor_proxies(close, shares, roe)

    results = {}
    for name, panel in factors.items():
        results[name] = run_diagnostics(
            panel, name,
            close=close, volume=volume, shares=shares, roe=roe,
            gics_df=gics_df,
            style_factors=style_factors,
            **kwargs,
        )
    return results


def format_diagnostic_report(report: DiagnosticReport) -> str:
    """格式化诊断报告"""
    lines = [f"=== {report.factor_name} 质量诊断 ==="]

    # 分布稳定性
    lines.append(f"分布漂移: {'是' if report.distribution.drift_detected else '否'} "
                 f"(p={report.distribution.drift_pvalue:.4f})")

    # 覆盖率
    lines.append(f"中位覆盖率: {report.coverage.median_coverage:.1%}")
    if report.coverage.small_cap_bias is not None:
        lines.append(f"小盘偏差: corr(availability, log_mktcap)={report.coverage.small_cap_bias:.3f}")

    # 已知因子相关性
    if report.known_factors is not None:
        lines.append("风格因子相关:")
        for style, corr in report.known_factors.correlations.items():
            flag = " ***" if abs(corr) > 0.7 else ""
            lines.append(f"  {style}: {corr:+.3f}{flag}")

    # 换手率
    lines.append(f"平均换手率: {report.turnover.mean_turnover:.1%}")
    if report.turnover.capacity_estimate_usd is not None:
        lines.append(f"容量估计: ${report.turnover.capacity_estimate_usd:,.0f}")

    # 行业分布
    if report.industry is not None:
        ind = report.industry
        lines.append("行业分布:")
        if len(ind.top_q_sector_pct) > 0:
            top3 = ind.top_q_sector_pct.head(3)
            lines.append(f"  多头 Top3: {', '.join(f'{s} {v:.0%}' for s, v in top3.items())}")
            lines.append(f"  多头 HHI: {ind.hhi_top:.3f}" +
                         (" (集中)" if ind.hhi_top > 0.25 else " (分散)"))
        if len(ind.bot_q_sector_pct) > 0:
            bot3 = ind.bot_q_sector_pct.head(3)
            lines.append(f"  空头 Top3: {', '.join(f'{s} {v:.0%}' for s, v in bot3.items())}")
            lines.append(f"  空头 HHI: {ind.hhi_bot:.3f}" +
                         (" (集中)" if ind.hhi_bot > 0.25 else " (分散)"))
        if ind.neutralized_ic is not None and ind.raw_ic_21d is not None:
            decay = (1 - ind.neutralized_ic / ind.raw_ic_21d) * 100 if ind.raw_ic_21d != 0 else 0
            lines.append(f"  行业中性 IC(21d): {ind.neutralized_ic:+.4f} "
                         f"(原始: {ind.raw_ic_21d:+.4f}, 衰减 {decay:.0f}%)")
            if ind.neutralized_icir is not None:
                lines.append(f"  行业中性 ICIR: {ind.neutralized_icir:.2f}")

    # 结论
    if report.pass_sanity:
        lines.append("结论: PASS")
    else:
        lines.append("结论: WARN")
        for w in report.warnings:
            lines.append(f"  - {w}")

    return '\n'.join(lines)
