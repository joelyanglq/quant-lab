"""
因子预测力评估 (Factor Alpha Test)

三板斧 + 子样本分析:
1. 多周期 IC: 日/周/月 IC, 衰减曲线, Newey-West t-stat
2. 增强分组回测: 扣费 L/S, 分位 MaxDD/Calmar
3. Fama-MacBeth 回归: 控制市值/行业/动量, 看边际贡献
4. 子样本分析: 牛熊/高低波/走样本外
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats
from dataclasses import dataclass, field

from .screening import compute_rank_ic
from .backtest import compute_forward_returns, build_periodic_rebalance
from .analytics import compute_factor_metrics, _infer_ann_factor


# ── 路径 ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DATA_DIR = _ROOT / 'data'


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class MultiHorizonIC:
    """多周期 IC 分析"""
    ic_by_horizon: Dict[int, pd.Series]       # {horizon: IC series}
    mean_ic: Dict[int, float]                  # {horizon: 均值 IC}
    icir: Dict[int, float]                     # {horizon: ICIR}
    tstat_nw: Dict[int, float]                 # {horizon: Newey-West t-stat}
    ic_half_life: Optional[float]              # IC 半衰期 (天)
    rolling_ic: pd.DataFrame                   # 滚动 IC (dates × horizons)
    ic_quantiles: Dict[int, Dict[str, float]]  # {horizon: {q25, q50, q75}}


@dataclass
class EnhancedQuantileMetrics:
    """增强分组回测结果"""
    base_metrics: Dict                          # analytics.compute_factor_metrics 输出
    backtest_result: Dict                       # 原始回测结果
    quantile_max_dd: Dict[int, float]           # {quantile: 最大回撤}
    quantile_calmar: Dict[int, float]           # {quantile: Calmar ratio}
    ls_with_costs: pd.Series                    # 扣费后 L/S 收益
    ls_sharpe_net: float                        # 扣费后 Sharpe
    ls_max_dd: float                            # L/S 最大回撤


@dataclass
class FamaMacBethResult:
    """Fama-MacBeth 回归结果"""
    mean_slope: float              # 因子斜率时序均值
    tstat: float                   # Newey-West t-stat
    mean_alpha: float              # 截距时序均值
    alpha_tstat: float             # 截距 t-stat
    slope_series: pd.Series        # 逐期斜率
    r2_series: pd.Series           # 逐期 R²
    control_slopes: Dict[str, float]  # 控制变量斜率均值


@dataclass
class SubSampleAnalysis:
    """子样本分析"""
    regime_labels: pd.Series       # dates → regime label
    regime_stats: pd.DataFrame     # index=regime, columns=[mean_ic, icir, ls_sharpe, n_periods]
    oos_stats: Optional[Dict]      # 走样本外 IC/ICIR


@dataclass
class AlphaTestReport:
    """单因子完整 Alpha Test 报告"""
    factor_name: str
    multi_horizon_ic: MultiHorizonIC
    enhanced_quantile: EnhancedQuantileMetrics
    fama_macbeth: Optional[FamaMacBethResult]
    sub_sample: Optional[SubSampleAnalysis]


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _newey_west_tstat(
    series: pd.Series,
    max_lag: Optional[int] = None,
) -> float:
    """
    Newey-West HAC 调整的 t 统计量.

    Bartlett kernel: w_j = 1 - j/(L+1)
    NW方差 = γ₀ + 2·Σ(j=1..L) w_j · γ_j
    t = mean / sqrt(NW_var / T)
    """
    x = series.dropna().values
    T = len(x)
    if T < 3:
        return 0.0

    if max_lag is None:
        max_lag = int(np.floor(4 * (T / 100) ** (2 / 9)))
    max_lag = max(max_lag, 1)

    mean = x.mean()
    demeaned = x - mean

    # γ₀
    gamma_0 = float(np.dot(demeaned, demeaned) / T)

    # Σ w_j · γ_j
    nw_sum = 0.0
    for j in range(1, max_lag + 1):
        gamma_j = float(np.dot(demeaned[j:], demeaned[:-j]) / T)
        w_j = 1.0 - j / (max_lag + 1)
        nw_sum += w_j * gamma_j

    nw_var = gamma_0 + 2 * nw_sum
    if nw_var <= 0:
        return 0.0

    se = np.sqrt(nw_var / T)
    return mean / se if se > 0 else 0.0


def _estimate_ic_half_life(
    mean_ics: Dict[int, float],
) -> Optional[float]:
    """
    从多周期均值 IC 拟合指数衰减, 估计半衰期.

    模型: ln|IC(h)| = a - b·h
    半衰期 = ln(2) / b

    仅当 R² > 0.5 且 b > 0 时返回有效值.
    """
    horizons = sorted(mean_ics.keys())
    if len(horizons) < 3:
        return None

    h = np.array(horizons, dtype=float)
    ic_abs = np.array([abs(mean_ics[hi]) for hi in horizons])

    # 过滤掉 IC = 0 的点 (无法取 log)
    mask = ic_abs > 1e-10
    if mask.sum() < 3:
        return None

    h_clean = h[mask]
    log_ic = np.log(ic_abs[mask])

    slope, intercept, r_value, _, _ = stats.linregress(h_clean, log_ic)

    if r_value ** 2 < 0.5 or slope >= 0:
        return None

    return np.log(2) / (-slope)


# ═══════════════════════════════════════════════════════════════
# 1. 多周期 IC 分析
# ═══════════════════════════════════════════════════════════════

def compute_multi_horizon_ic(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizons: List[int] = None,
    rolling_window: int = 52,
    min_obs: int = 20,
) -> MultiHorizonIC:
    """
    多周期 IC + 衰减估计 + Newey-West t-stat.

    对每个 horizon:
        1. compute_forward_returns(close, horizon)
        2. compute_rank_ic(factor, fwd_ret)
        3. 均值/ICIR/NW t-stat
        4. 滚动 IC

    半衰期: 拟合 |IC(h)| ~ exp(-b·h).
    """
    if horizons is None:
        horizons = [1, 5, 21, 63]

    ic_by_horizon = {}
    mean_ic = {}
    icir = {}
    tstat_nw = {}
    ic_quantiles = {}
    rolling_parts = {}

    for h in horizons:
        fwd_ret = compute_forward_returns(close, periods=h)
        ic_series = compute_rank_ic(factor, fwd_ret, min_obs=min_obs)

        if len(ic_series) == 0:
            continue

        ic_by_horizon[h] = ic_series
        mean_ic[h] = float(ic_series.mean())
        ic_std = ic_series.std()
        icir[h] = float(ic_series.mean() / ic_std) if ic_std > 0 else 0.0
        tstat_nw[h] = _newey_west_tstat(ic_series)

        # 分位数
        ic_quantiles[h] = {
            'q25': float(ic_series.quantile(0.25)),
            'q50': float(ic_series.quantile(0.50)),
            'q75': float(ic_series.quantile(0.75)),
        }

        # 滚动 IC
        min_p = max(rolling_window // 2, 1)
        rolling_parts[h] = ic_series.rolling(rolling_window, min_periods=min_p).mean()

    rolling_ic = pd.DataFrame(rolling_parts)
    ic_half_life = _estimate_ic_half_life(mean_ic) if len(mean_ic) >= 3 else None

    return MultiHorizonIC(
        ic_by_horizon=ic_by_horizon,
        mean_ic=mean_ic,
        icir=icir,
        tstat_nw=tstat_nw,
        ic_half_life=ic_half_life,
        rolling_ic=rolling_ic,
        ic_quantiles=ic_quantiles,
    )


# ═══════════════════════════════════════════════════════════════
# 2. 增强分组回测
# ═══════════════════════════════════════════════════════════════

def compute_enhanced_quantile_metrics(
    backtest_result: Dict,
    transaction_cost_bps: float = 10.0,
) -> EnhancedQuantileMetrics:
    """
    扩展回测指标: 分位 MaxDD/Calmar + 扣费 L/S.

    交易成本: 每次调仓扣除 cost_bps × turnover × 2 (双边).
    """
    base_metrics = compute_factor_metrics(backtest_result)
    qret = backtest_result['quantile_returns']
    ls = backtest_result['long_short']
    quantiles = backtest_result['quantiles']
    n_q = qret.shape[1]
    ann_factor = base_metrics['ann_factor']

    # 分位 MaxDD & Calmar
    quantile_max_dd = {}
    quantile_calmar = {}
    for q in range(1, n_q + 1):
        qr = qret[q].dropna()
        if len(qr) == 0:
            quantile_max_dd[q] = 0.0
            quantile_calmar[q] = 0.0
            continue
        cum = (1 + qr).cumprod()
        dd = cum / cum.cummax() - 1
        max_dd = float(dd.min())
        quantile_max_dd[q] = max_dd
        ann_ret = qr.mean() * ann_factor
        quantile_calmar[q] = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    # 扣费 L/S
    # 换手率: Q_top 和 Q_bottom 的成分变化
    q_top = (quantiles == n_q).astype(float)
    q_bot = (quantiles == 1).astype(float)
    top_turn = q_top.diff().abs().sum(axis=1) / 2
    bot_turn = q_bot.diff().abs().sum(axis=1) / 2
    top_count = q_top.sum(axis=1).replace(0, np.nan)
    bot_count = q_bot.sum(axis=1).replace(0, np.nan)
    turnover_rate = ((top_turn / top_count) + (bot_turn / bot_count)).fillna(0) / 2

    cost_per_period = turnover_rate * transaction_cost_bps / 10000 * 2  # 双边
    ls_net = ls - cost_per_period.reindex(ls.index, fill_value=0)

    ls_net_clean = ls_net.dropna()
    ls_ann_ret = ls_net_clean.mean() * ann_factor
    ls_ann_vol = ls_net_clean.std() * np.sqrt(ann_factor)
    ls_sharpe_net = ls_ann_ret / ls_ann_vol if ls_ann_vol > 0 else 0.0

    cum_ls = (1 + ls.dropna()).cumprod()
    dd_ls = cum_ls / cum_ls.cummax() - 1
    ls_max_dd = float(dd_ls.min()) if len(dd_ls) > 0 else 0.0

    return EnhancedQuantileMetrics(
        base_metrics=base_metrics,
        backtest_result=backtest_result,
        quantile_max_dd=quantile_max_dd,
        quantile_calmar=quantile_calmar,
        ls_with_costs=ls_net,
        ls_sharpe_net=ls_sharpe_net,
        ls_max_dd=ls_max_dd,
    )


# ═══════════════════════════════════════════════════════════════
# 3. Fama-MacBeth 回归
# ═══════════════════════════════════════════════════════════════

def load_gics_sectors(
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    加载 GICS 行业分类.

    Returns:
        DataFrame, 至少含 'symbol' 和 'gics_sector' 列.
    """
    fp = _DATA_DIR / '_gics_sectors.parquet'
    if not fp.exists():
        return pd.DataFrame(columns=['symbol', 'gics_sector'])
    df = pd.read_parquet(fp)
    # 统一列名
    col_map = {}
    for c in df.columns:
        cl = c.lower().replace(' ', '_')
        if 'symbol' in cl or 'ticker' in cl:
            col_map[c] = 'symbol'
        elif 'sector' in cl:
            col_map[c] = 'gics_sector'
    if col_map:
        df = df.rename(columns=col_map)
    if symbols is not None and 'symbol' in df.columns:
        df = df[df['symbol'].isin(symbols)]
    return df


def build_sector_dummies(
    symbols: List[str],
    gics_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    构建行业 dummy 矩阵 (drop_first 避免共线性).

    Returns:
        DataFrame(index=symbols, columns=sector_names), 值为 0/1.
    """
    if gics_df.empty or 'symbol' not in gics_df.columns or 'gics_sector' not in gics_df.columns:
        return pd.DataFrame(index=symbols)

    mapping = gics_df.set_index('symbol')['gics_sector'].to_dict()
    sectors = pd.Series({s: mapping.get(s, 'Unknown') for s in symbols})
    dummies = pd.get_dummies(sectors, drop_first=True, dtype=float)
    dummies.index = symbols
    return dummies


def fama_macbeth_regression(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    controls: Optional[Dict[str, pd.DataFrame]] = None,
    sector_dummies: Optional[pd.DataFrame] = None,
    min_obs: int = 30,
) -> FamaMacBethResult:
    """
    Fama-MacBeth 两步回归.

    Pass 1 (每期截面回归):
        ret_i(t) = α(t) + β(t)·factor_i(t) + Σγ_j(t)·control_j_i(t) + δ·sector + ε

    Pass 2 (时序):
        mean(β(t)) → Newey-West t-stat
    """
    common_idx = factor.index.intersection(forward_returns.index)

    slopes = []
    alphas = []
    r2s = []
    control_slope_acc = {}
    dates = []

    for dt in common_idx:
        # 准备 y (收益)
        y = forward_returns.loc[dt].dropna()
        if len(y) < min_obs:
            continue

        # 准备 X: [constant, factor, controls..., sector_dummies...]
        f = factor.loc[dt].reindex(y.index).dropna()
        common_syms = y.index.intersection(f.index)
        if len(common_syms) < min_obs:
            continue

        y_vals = y[common_syms]
        x_parts = [pd.Series(1.0, index=common_syms, name='const'), f[common_syms]]
        col_names = ['const', 'factor']

        # 控制变量
        if controls:
            for ctrl_name, ctrl_panel in controls.items():
                if dt in ctrl_panel.index:
                    ctrl = ctrl_panel.loc[dt].reindex(common_syms)
                    x_parts.append(ctrl)
                    col_names.append(ctrl_name)

        # 行业 dummies
        if sector_dummies is not None and not sector_dummies.empty:
            for scol in sector_dummies.columns:
                dummy_col = sector_dummies[scol].reindex(common_syms).fillna(0)
                x_parts.append(dummy_col)
                col_names.append(scol)

        X = pd.concat(x_parts, axis=1).dropna()
        X.columns = col_names
        y_aligned = y_vals.reindex(X.index).dropna()
        X = X.loc[y_aligned.index]

        if len(y_aligned) < max(min_obs, X.shape[1] + 2):
            continue

        # OLS
        X_mat = X.values.astype(float)
        y_vec = y_aligned.values.astype(float)
        try:
            beta, residuals, rank, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # 提取系数
        alphas.append(beta[0])
        slopes.append(beta[1])
        dates.append(dt)

        # R²
        y_hat = X_mat @ beta
        ss_res = np.sum((y_vec - y_hat) ** 2)
        ss_tot = np.sum((y_vec - y_vec.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2s.append(r2)

        # 控制变量斜率
        for i, name in enumerate(col_names):
            if name in ('const', 'factor'):
                continue
            if name not in control_slope_acc:
                control_slope_acc[name] = []
            control_slope_acc[name].append(beta[i])

    if len(slopes) == 0:
        return FamaMacBethResult(
            mean_slope=0.0, tstat=0.0,
            mean_alpha=0.0, alpha_tstat=0.0,
            slope_series=pd.Series(dtype=float),
            r2_series=pd.Series(dtype=float),
            control_slopes={},
        )

    slope_s = pd.Series(slopes, index=pd.DatetimeIndex(dates))
    alpha_s = pd.Series(alphas, index=pd.DatetimeIndex(dates))
    r2_s = pd.Series(r2s, index=pd.DatetimeIndex(dates))

    ctrl_avg = {k: np.mean(v) for k, v in control_slope_acc.items()}

    return FamaMacBethResult(
        mean_slope=float(slope_s.mean()),
        tstat=_newey_west_tstat(slope_s),
        mean_alpha=float(alpha_s.mean()),
        alpha_tstat=_newey_west_tstat(alpha_s),
        slope_series=slope_s,
        r2_series=r2_s,
        control_slopes=ctrl_avg,
    )


# ═══════════════════════════════════════════════════════════════
# 4. 子样本分析
# ═══════════════════════════════════════════════════════════════

def classify_market_regime(
    spy_close: pd.Series,
    method: str = 'ma200',
) -> pd.Series:
    """
    市场 regime 分类.

    方法:
        'ma200': SPY > 200日均线 → 'bull', 否则 'bear'
        'drawdown': 回撤 < 10% → 'bull', > 20% → 'bear', 中间 → 'neutral'
        'vol': 60日波动率 > 中位数 → 'high_vol', 否则 'low_vol'
    """
    if method == 'ma200':
        ma200 = spy_close.rolling(200, min_periods=100).mean()
        labels = pd.Series('bear', index=spy_close.index)
        labels[spy_close > ma200] = 'bull'
        # 前 200 天没有 MA, 标记为 NaN
        labels[ma200.isna()] = np.nan
        return labels

    elif method == 'drawdown':
        cum_max = spy_close.cummax()
        dd = spy_close / cum_max - 1
        labels = pd.Series('neutral', index=spy_close.index)
        labels[dd > -0.10] = 'bull'
        labels[dd < -0.20] = 'bear'
        return labels

    elif method == 'vol':
        daily_ret = spy_close.pct_change()
        rvol = daily_ret.rolling(60, min_periods=30).std() * np.sqrt(252)
        median_vol = rvol.median()
        labels = pd.Series('low_vol', index=spy_close.index)
        labels[rvol > median_vol] = 'high_vol'
        labels[rvol.isna()] = np.nan
        return labels

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_sub_sample_stats(
    ic_series: pd.Series,
    ls_returns: pd.Series,
    regime_labels: pd.Series,
    ann_factor: int = 12,
) -> pd.DataFrame:
    """
    按 regime 分组计算 IC/ICIR/L-S Sharpe.
    """
    # 对齐到 IC 的日期
    aligned_labels = regime_labels.reindex(ic_series.index, method='ffill').dropna()
    common = ic_series.index.intersection(aligned_labels.index)

    if len(common) == 0:
        return pd.DataFrame(columns=['mean_ic', 'icir', 'ls_sharpe', 'n_periods'])

    ic_aligned = ic_series.loc[common]
    labels = aligned_labels.loc[common]
    ls_aligned = ls_returns.reindex(common)

    records = []
    for regime in labels.unique():
        if pd.isna(regime):
            continue
        mask = labels == regime
        ic_sub = ic_aligned[mask]
        ls_sub = ls_aligned[mask].dropna()
        n = len(ic_sub)
        if n < 3:
            continue

        m_ic = float(ic_sub.mean())
        ic_std = ic_sub.std()
        m_icir = m_ic / ic_std if ic_std > 0 else 0.0

        ls_ann = ls_sub.mean() * ann_factor
        ls_vol = ls_sub.std() * np.sqrt(ann_factor)
        ls_sharpe = ls_ann / ls_vol if ls_vol > 0 else 0.0

        records.append({
            'regime': regime,
            'mean_ic': m_ic,
            'icir': m_icir,
            'ls_sharpe': ls_sharpe,
            'n_periods': n,
        })

    if not records:
        return pd.DataFrame(columns=['mean_ic', 'icir', 'ls_sharpe', 'n_periods'])

    return pd.DataFrame(records).set_index('regime')


def walk_forward_split(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    train_pct: float = 0.7,
    horizons: List[int] = None,
    min_obs: int = 20,
) -> Dict:
    """
    简单走样本外: 前 train_pct 为 IS, 后面为 OOS.
    分别计算各 horizon 的 IC/ICIR.
    """
    if horizons is None:
        horizons = [21]

    dates = factor.index
    split_idx = int(len(dates) * train_pct)
    split_date = dates[split_idx]

    result = {'split_date': split_date, 'in_sample': {}, 'out_of_sample': {}}

    for h in horizons:
        fwd_ret = compute_forward_returns(close, periods=h)

        for label, start, end in [
            ('in_sample', dates[0], split_date),
            ('out_of_sample', split_date, dates[-1]),
        ]:
            fac_sub = factor.loc[start:end]
            fwd_sub = fwd_ret.loc[start:end]
            ic = compute_rank_ic(fac_sub, fwd_sub, min_obs=min_obs)

            if len(ic) >= 3:
                ic_std = ic.std()
                result[label][h] = {
                    'mean_ic': float(ic.mean()),
                    'icir': float(ic.mean() / ic_std) if ic_std > 0 else 0.0,
                    'n': len(ic),
                }
            else:
                result[label][h] = {'mean_ic': np.nan, 'icir': np.nan, 'n': len(ic)}

    return result


def run_sub_sample_analysis(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    backtest_result: Dict,
    spy_close: Optional[pd.Series] = None,
    regime_method: str = 'ma200',
    train_pct: float = 0.7,
) -> SubSampleAnalysis:
    """
    完整子样本分析: regime 分解 + 走样本外.
    """
    ic_series = backtest_result['ic_series']
    ls_returns = backtest_result['long_short']
    ann_factor = _infer_ann_factor(ic_series.index)

    # Regime 分类
    if spy_close is None:
        if 'SPY' in close.columns:
            spy_close = close['SPY']

    regime_labels = pd.Series(dtype=str)
    regime_stats = pd.DataFrame(columns=['mean_ic', 'icir', 'ls_sharpe', 'n_periods'])

    if spy_close is not None and len(spy_close) > 0:
        regime_labels = classify_market_regime(spy_close, method=regime_method)
        regime_stats = compute_sub_sample_stats(
            ic_series, ls_returns, regime_labels, ann_factor,
        )

    # 走样本外
    oos_stats = walk_forward_split(factor, close, train_pct)

    return SubSampleAnalysis(
        regime_labels=regime_labels,
        regime_stats=regime_stats,
        oos_stats=oos_stats,
    )


# ═══════════════════════════════════════════════════════════════
# 5. Master Alpha Test
# ═══════════════════════════════════════════════════════════════

def run_alpha_test(
    factor: pd.DataFrame,
    factor_name: str,
    close: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    shares: Optional[pd.DataFrame] = None,
    gics_df: Optional[pd.DataFrame] = None,
    spy_close: Optional[pd.Series] = None,
    rebalance_freq: str = 'M',
    horizons: List[int] = None,
    transaction_cost_bps: float = 10.0,
    do_fama_macbeth: bool = True,
    do_sub_sample: bool = True,
    min_obs: int = 20,
) -> AlphaTestReport:
    """
    单因子完整 Alpha Test.

    1. 多周期 IC (always)
    2. 增强分组回测 (always)
    3. Fama-MacBeth 回归 (optional)
    4. 子样本分析 (optional)
    """
    if horizons is None:
        horizons = [1, 5, 21, 63]

    # 1. 多周期 IC
    multi_ic = compute_multi_horizon_ic(factor, close, horizons, min_obs=min_obs)

    # 2. 分组回测
    bt_result = build_periodic_rebalance(
        factor, close, rebalance_freq=rebalance_freq, min_stocks=min_obs,
    )
    enhanced_q = compute_enhanced_quantile_metrics(bt_result, transaction_cost_bps)

    # 3. Fama-MacBeth
    fm_result = None
    if do_fama_macbeth:
        # 构建控制变量
        controls = {}
        # momentum
        controls['momentum'] = close.shift(21) / close.shift(252) - 1

        # log_mktcap
        if shares is not None:
            common_idx = close.index.intersection(shares.index)
            common_cols = close.columns.intersection(shares.columns)
            if len(common_idx) > 0:
                mktcap = close.loc[common_idx, common_cols] * shares.loc[common_idx, common_cols]
                controls['log_mktcap'] = np.log(mktcap.clip(lower=1))

        # sector dummies
        sector_dummies = None
        if gics_df is not None and not gics_df.empty:
            sector_dummies = build_sector_dummies(
                factor.columns.tolist(), gics_df,
            )

        # 需要 forward returns (月频)
        fwd_ret = compute_forward_returns(close, periods=21)

        fm_result = fama_macbeth_regression(
            factor, fwd_ret,
            controls=controls,
            sector_dummies=sector_dummies,
            min_obs=min_obs,
        )

    # 4. 子样本分析
    sub_sample = None
    if do_sub_sample:
        sub_sample = run_sub_sample_analysis(
            factor, close, bt_result, spy_close,
        )

    return AlphaTestReport(
        factor_name=factor_name,
        multi_horizon_ic=multi_ic,
        enhanced_quantile=enhanced_q,
        fama_macbeth=fm_result,
        sub_sample=sub_sample,
    )


def run_batch_alpha_test(
    factors: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    **kwargs,
) -> Dict[str, AlphaTestReport]:
    """
    批量 Alpha Test. GICS 和 SPY 只加载一次.
    """
    # 预加载公共数据
    if 'gics_df' not in kwargs or kwargs.get('gics_df') is None:
        kwargs['gics_df'] = load_gics_sectors()

    if 'spy_close' not in kwargs or kwargs.get('spy_close') is None:
        if 'SPY' in close.columns:
            kwargs['spy_close'] = close['SPY']

    results = {}
    for name, panel in factors.items():
        results[name] = run_alpha_test(panel, name, close, **kwargs)
    return results


def format_alpha_test_report(report: AlphaTestReport) -> str:
    """格式化 Alpha Test 报告"""
    lines = [f"=== {report.factor_name} Alpha Test ==="]

    # 多周期 IC
    lines.append("\n多周期 IC:")
    lines.append(f"{'Horizon':>8s} {'MeanIC':>8s} {'ICIR':>8s} {'NW t':>8s} {'IC_q25':>8s} {'IC_q75':>8s}")
    for h in sorted(report.multi_horizon_ic.mean_ic.keys()):
        m = report.multi_horizon_ic.mean_ic[h]
        ir = report.multi_horizon_ic.icir[h]
        t = report.multi_horizon_ic.tstat_nw[h]
        q = report.multi_horizon_ic.ic_quantiles.get(h, {})
        q25 = q.get('q25', np.nan)
        q75 = q.get('q75', np.nan)
        lines.append(f"{h:>8d} {m:>+8.4f} {ir:>8.2f} {t:>8.2f} {q25:>+8.4f} {q75:>+8.4f}")

    if report.multi_horizon_ic.ic_half_life is not None:
        lines.append(f"IC 半衰期: {report.multi_horizon_ic.ic_half_life:.1f} 天")

    # 分组回测
    eq = report.enhanced_quantile
    bm = eq.base_metrics
    lines.append(f"\n分组回测 (freq={bm['ann_factor']}x):")
    lines.append(f"  L/S Gross: Return={bm['ls_annual_return']:.2%}, "
                 f"Sharpe={bm['ls_sharpe']:.2f}, MaxDD={eq.ls_max_dd:.2%}")
    lines.append(f"  L/S Net:   Sharpe={eq.ls_sharpe_net:.2f}")
    lines.append(f"  Monotonicity: {bm['monotonicity']:.2f}")

    # Fama-MacBeth
    if report.fama_macbeth is not None:
        fm = report.fama_macbeth
        sig = "***" if abs(fm.tstat) > 2.58 else "**" if abs(fm.tstat) > 1.96 else ""
        lines.append(f"\nFama-MacBeth:")
        lines.append(f"  Factor slope: {fm.mean_slope:.6f} (t={fm.tstat:.2f}) {sig}")
        lines.append(f"  Alpha: {fm.mean_alpha:.6f} (t={fm.alpha_tstat:.2f})")
        lines.append(f"  Avg R²: {fm.r2_series.mean():.4f}")

    # 子样本
    if report.sub_sample is not None and not report.sub_sample.regime_stats.empty:
        lines.append(f"\n子样本分析:")
        rs = report.sub_sample.regime_stats
        for regime in rs.index:
            row = rs.loc[regime]
            lines.append(f"  {regime}: IC={row['mean_ic']:+.4f}, "
                         f"ICIR={row['icir']:.2f}, "
                         f"L/S Sharpe={row['ls_sharpe']:.2f} "
                         f"(n={int(row['n_periods'])})")

        # 走样本外
        oos = report.sub_sample.oos_stats
        if oos and 'in_sample' in oos and 'out_of_sample' in oos:
            lines.append("  走样本外 (IS / OOS):")
            for h in sorted(set(list(oos['in_sample'].keys()) + list(oos['out_of_sample'].keys()))):
                is_stats = oos['in_sample'].get(h, {})
                oos_stats = oos['out_of_sample'].get(h, {})
                lines.append(
                    f"    {h}d: IS IC={is_stats.get('mean_ic', np.nan):+.4f} "
                    f"ICIR={is_stats.get('icir', np.nan):.2f} | "
                    f"OOS IC={oos_stats.get('mean_ic', np.nan):+.4f} "
                    f"ICIR={oos_stats.get('icir', np.nan):.2f}"
                )

    return '\n'.join(lines)
