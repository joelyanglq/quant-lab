"""
因子研究可视化

诊断 + Alpha Test 图表:
- 分布稳定性 (2×2)
- 覆盖率时序
- 风格相关柱状图
- IC 衰减曲线
- 滚动 IC
- Fama-MacBeth 斜率
- 子样本分解
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path

from .diagnostics import DiagnosticReport
from .alpha_test import AlphaTestReport


# ═══════════════════════════════════════════════════════════════
# 诊断可视化
# ═══════════════════════════════════════════════════════════════

def plot_diagnostic_summary(
    report: DiagnosticReport,
    save_path: Optional[str] = None,
):
    """
    2×2 诊断总览:
    ① 分布稳定性 (rolling mean + var)
    ② 覆盖率时序
    ③ 风格因子相关柱状图
    ④ 换手率时序
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = f'{report.factor_name} Diagnostic Summary'
    if not report.pass_sanity:
        title += '  [WARN]'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ── Panel 1: 分布稳定性 ──────────────────────
    ax = axes[0, 0]
    d = report.distribution
    ax.plot(d.rolling_mean.index, d.rolling_mean.values,
            color='steelblue', label='Rolling Mean', linewidth=1.5)
    ax2 = ax.twinx()
    ax2.plot(d.rolling_var.index, d.rolling_var.values,
             color='coral', label='Rolling Var', linewidth=1.5, alpha=0.7)
    ax.set_title('Distribution Stability')
    ax.set_ylabel('Mean', color='steelblue')
    ax2.set_ylabel('Variance', color='coral')
    drift_label = f"Drift: {'YES' if d.drift_detected else 'NO'} (p={d.drift_pvalue:.4f})"
    ax.text(0.03, 0.97, drift_label, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.grid(True, alpha=0.3)

    # ── Panel 2: 覆盖率 ─────────────────────────
    ax = axes[0, 1]
    c = report.coverage
    ax.plot(c.coverage_pct.index, c.coverage_pct.values,
            color='teal', linewidth=1.5)
    ax.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='30% threshold')
    ax.set_title(f'Coverage (median={c.median_coverage:.1%})')
    ax.set_ylabel('Coverage %')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: 风格因子相关 ────────────────────
    ax = axes[1, 0]
    if report.known_factors is not None and len(report.known_factors.correlations) > 0:
        corrs = report.known_factors.correlations
        colors = ['green' if abs(v) < 0.3 else 'orange' if abs(v) < 0.7 else 'red'
                  for v in corrs.values]
        bars = ax.bar(corrs.index, corrs.values, color=colors, edgecolor='gray', alpha=0.8)
        ax.axhline(0.7, color='red', linestyle='--', alpha=0.3)
        ax.axhline(-0.7, color='red', linestyle='--', alpha=0.3)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        for bar, val in zip(bars, corrs.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:+.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
        ax.set_title('Style Factor Correlation')
        ax.set_ylabel('Spearman Corr')
    else:
        ax.text(0.5, 0.5, 'No style data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Style Factor Correlation')
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel 4: 换手率 ─────────────────────────
    ax = axes[1, 1]
    t = report.turnover
    if len(t.signal_turnover) > 0:
        ax.plot(t.signal_turnover.index, t.signal_turnover.values,
                color='purple', linewidth=1, alpha=0.6)
        if len(t.signal_turnover) >= 20:
            rolling_turn = t.signal_turnover.rolling(20).mean()
            ax.plot(rolling_turn.index, rolling_turn.values,
                    color='darkred', linewidth=2, label='20d rolling')
        ax.axhline(0.8, color='red', linestyle='--', alpha=0.3, label='80% threshold')
        ax.set_title(f'Signal Turnover (mean={t.mean_turnover:.1%})')
        ax.set_ylabel('Turnover Rate')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No turnover data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Signal Turnover')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════
# Alpha Test 可视化
# ═══════════════════════════════════════════════════════════════

def plot_ic_decay_curve(
    report: AlphaTestReport,
    save_path: Optional[str] = None,
):
    """IC 衰减曲线: 柱状 + 拟合指数衰减"""
    mic = report.multi_horizon_ic
    if not mic.mean_ic:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    horizons = sorted(mic.mean_ic.keys())
    ic_vals = [mic.mean_ic[h] for h in horizons]
    colors = ['green' if abs(v) > 0.02 else 'orange' if abs(v) > 0.005 else 'gray'
              for v in ic_vals]

    bars = ax.bar([str(h) + 'd' for h in horizons], ic_vals,
                  color=colors, edgecolor='gray', alpha=0.8)

    for bar, val, h in zip(bars, ic_vals, horizons):
        t = mic.tstat_nw.get(h, 0)
        sig = '***' if abs(t) > 2.58 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.64 else ''
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:+.4f}\n(t={t:.1f}){sig}',
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

    # 半衰期标注
    if mic.ic_half_life is not None:
        ax.text(0.97, 0.97, f'Half-life: {mic.ic_half_life:.1f}d',
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.set_title(f'{report.factor_name} IC Decay Curve')
    ax.set_xlabel('Horizon')
    ax.set_ylabel('Mean Rank IC')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_fama_macbeth(
    report: AlphaTestReport,
    save_path: Optional[str] = None,
):
    """FM 斜率时序 + R² 时序"""
    fm = report.fama_macbeth
    if fm is None or len(fm.slope_series) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f'{report.factor_name} Fama-MacBeth Regression', fontsize=14, fontweight='bold')

    # 上: 斜率时序
    s = fm.slope_series
    ax1.plot(s.index, s.values, color='steelblue', linewidth=1, alpha=0.6)
    if len(s) >= 12:
        rolling_s = s.rolling(12).mean()
        ax1.plot(rolling_s.index, rolling_s.values, color='navy', linewidth=2, label='12-period rolling')
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax1.axhline(fm.mean_slope, color='red', linestyle='--', alpha=0.5, label=f'mean={fm.mean_slope:.6f}')

    sig = '***' if abs(fm.tstat) > 2.58 else '**' if abs(fm.tstat) > 1.96 else ''
    ax1.set_title(f'Factor Slope (mean={fm.mean_slope:.6f}, t={fm.tstat:.2f}) {sig}')
    ax1.set_ylabel('Slope')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 下: R²
    ax2.plot(fm.r2_series.index, fm.r2_series.values, color='teal', linewidth=1, alpha=0.6)
    if len(fm.r2_series) >= 12:
        rolling_r2 = fm.r2_series.rolling(12).mean()
        ax2.plot(rolling_r2.index, rolling_r2.values, color='darkgreen', linewidth=2, label='12-period rolling')
    ax2.set_title(f'Cross-sectional R² (mean={fm.r2_series.mean():.4f})')
    ax2.set_ylabel('R²')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_sub_sample_decomposition(
    report: AlphaTestReport,
    save_path: Optional[str] = None,
):
    """子样本分解: 逐年 IC 柱状图 + regime 柱状图"""
    ss = report.sub_sample
    if ss is None:
        return

    has_yearly = ss.yearly_stats is not None and not ss.yearly_stats.empty
    has_regime = not ss.regime_stats.empty

    if not has_yearly and not has_regime:
        return

    n_rows = (1 if has_yearly else 0) + (1 if has_regime else 0)
    fig, all_axes = plt.subplots(n_rows, 3, figsize=(14, 5 * n_rows))
    fig.suptitle(f'{report.factor_name} Sub-sample Analysis', fontsize=14, fontweight='bold')

    if n_rows == 1:
        all_axes = all_axes.reshape(1, -1)

    row_idx = 0

    # ── 逐年分解 ──
    if has_yearly:
        ys = ss.yearly_stats
        years = [str(y) for y in ys.index]
        n_yr = len(years)
        colors_yr = ['steelblue'] * n_yr

        for ax, (col, label) in zip(all_axes[row_idx], [
            ('mean_ic', 'Mean IC by Year'),
            ('icir', 'ICIR by Year'),
            ('ls_sharpe', 'L/S Sharpe by Year'),
        ]):
            vals = ys[col].values
            bars = ax.bar(years, vals, color=colors_yr, edgecolor='gray', alpha=0.8)
            ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
            # highlight negative bars
            for bar, val in zip(bars, vals):
                if val < 0:
                    bar.set_color('salmon')
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.3f}', ha='center',
                        va='bottom' if val >= 0 else 'top', fontsize=8)
            ax.set_title(label)
            ax.grid(True, alpha=0.3, axis='y')
        row_idx += 1

    # ── Regime 分解 ──
    if has_regime:
        rs = ss.regime_stats
        regimes = rs.index.tolist()
        n = len(regimes)
        palette = plt.cm.Set2(np.linspace(0, 1, max(n, 2)))

        for ax, (col, label) in zip(all_axes[row_idx], [
            ('mean_ic', 'Mean IC by Regime'),
            ('icir', 'ICIR by Regime'),
            ('ls_sharpe', 'L/S Sharpe by Regime'),
        ]):
            vals = [rs.loc[r, col] for r in regimes]
            bars = ax.bar(regimes, vals, color=palette[:n], edgecolor='gray', alpha=0.8)
            ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.3f}', ha='center',
                        va='bottom' if val >= 0 else 'top', fontsize=9)
            ax.set_title(label)
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_alpha_test_summary(
    report: AlphaTestReport,
    save_path: Optional[str] = None,
):
    """
    3×2 Alpha Test 总览:
    ① IC 衰减曲线
    ② 滚动 IC (21d horizon)
    ③ 五分位累计收益
    ④ L/S 毛 vs 净
    ⑤ FM 斜率时序
    ⑥ 子样本柱状图
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle(f'{report.factor_name} Alpha Test Summary', fontsize=14, fontweight='bold')

    mic = report.multi_horizon_ic
    eq = report.enhanced_quantile
    bm = eq.base_metrics
    bt = eq.backtest_result

    # ── ① IC 衰减曲线 ────────────────────────────
    ax = axes[0, 0]
    if mic.mean_ic:
        horizons = sorted(mic.mean_ic.keys())
        ic_vals = [mic.mean_ic[h] for h in horizons]
        ax.bar([str(h) + 'd' for h in horizons], ic_vals,
               color='steelblue', edgecolor='gray', alpha=0.8)
        for i, (h, v) in enumerate(zip(horizons, ic_vals)):
            t = mic.tstat_nw.get(h, 0)
            sig = '***' if abs(t) > 2.58 else '**' if abs(t) > 1.96 else ''
            ax.text(i, v, f'{v:+.4f}{sig}', ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=8)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    hl_str = f' HL={mic.ic_half_life:.0f}d' if mic.ic_half_life else ''
    ax.set_title(f'IC Decay{hl_str}')
    ax.grid(True, alpha=0.3, axis='y')

    # ── ② 滚动 IC ────────────────────────────────
    ax = axes[0, 1]
    if not mic.rolling_ic.empty:
        for col in mic.rolling_ic.columns:
            s = mic.rolling_ic[col].dropna()
            if len(s) > 0:
                ax.plot(s.index, s.values, label=f'{col}d', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.set_title('Rolling IC (52-period)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── ③ 五分位累计收益 ─────────────────────────
    ax = axes[1, 0]
    qret = bt['quantile_returns']
    n_q = qret.shape[1]
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_q))
    for q in range(1, n_q + 1):
        cum = (1 + qret[q].fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, label=f'Q{q}', color=colors[q - 1], linewidth=1.5)
    ax.set_title(f'Quintile Returns (Mono={bm["monotonicity"]:.2f})')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    # ── ④ L/S 毛 vs 净 ──────────────────────────
    ax = axes[1, 1]
    ls_gross = (1 + bt['long_short'].fillna(0)).cumprod()
    ls_net = (1 + eq.ls_with_costs.fillna(0)).cumprod()
    ax.plot(ls_gross.index, ls_gross.values, color='navy', label='Gross', linewidth=1.5)
    ax.plot(ls_net.index, ls_net.values, color='coral', label='Net', linewidth=1.5, linestyle='--')
    ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'L/S: Gross Sharpe={bm["ls_sharpe"]:.2f}, Net Sharpe={eq.ls_sharpe_net:.2f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── ⑤ FM 斜率 ────────────────────────────────
    ax = axes[2, 0]
    fm = report.fama_macbeth
    if fm is not None and len(fm.slope_series) > 0:
        s = fm.slope_series
        ax.plot(s.index, s.values, color='steelblue', linewidth=1, alpha=0.6)
        if len(s) >= 12:
            rolling_s = s.rolling(12).mean()
            ax.plot(rolling_s.index, rolling_s.values, color='navy', linewidth=2)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        sig = '***' if abs(fm.tstat) > 2.58 else '**' if abs(fm.tstat) > 1.96 else ''
        ax.set_title(f'FM Slope (t={fm.tstat:.2f}) {sig}')
    else:
        ax.text(0.5, 0.5, 'No FM data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Fama-MacBeth')
    ax.grid(True, alpha=0.3)

    # ── ⑥ 子样本 ─────────────────────────────────
    ax = axes[2, 1]
    ss = report.sub_sample
    if ss is not None and not ss.regime_stats.empty:
        rs = ss.regime_stats
        regimes = rs.index.tolist()
        x = np.arange(len(regimes))
        width = 0.25
        ax.bar(x - width, rs['mean_ic'], width, label='IC', alpha=0.8)
        ax.bar(x, rs['icir'], width, label='ICIR', alpha=0.8)
        ax.bar(x + width, rs['ls_sharpe'], width, label='L/S Sharpe', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(regimes)
        ax.legend(fontsize=8)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax.set_title('Sub-sample Decomposition')
    else:
        ax.text(0.5, 0.5, 'No sub-sample data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Sub-sample')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════
# 行业分布可视化
# ═══════════════════════════════════════════════════════════════

def plot_industry_distribution(
    report: DiagnosticReport,
    save_path: Optional[str] = None,
):
    """
    2×2 行业分布诊断:
    ① 各行业平均因子值 (水平 bar)
    ② Top/Bottom quintile 行业占比 (并排 bar)
    ③ HHI 集中度 vs 均匀分布基准
    ④ 行业中性 IC vs 原始 IC
    """
    ind = report.industry
    if ind is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{report.factor_name} Industry Distribution',
                 fontsize=14, fontweight='bold')

    # ── Panel 1: 各行业平均因子值 ──────────────────────
    ax = axes[0, 0]
    if len(ind.avg_sector_mean) > 0:
        sm = ind.avg_sector_mean.sort_values()
        colors = ['coral' if v > 0 else 'steelblue' for v in sm.values]
        ax.barh(sm.index, sm.values, color=colors, edgecolor='gray', alpha=0.8)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
        for i, (sec, val) in enumerate(sm.items()):
            ax.text(val, i, f' {val:+.3f}', va='center', fontsize=8,
                    ha='left' if val >= 0 else 'right')
    ax.set_title('Avg Factor Value by Sector')
    ax.grid(True, alpha=0.3, axis='x')

    # ── Panel 2: Top/Bottom quintile 行业占比 ─────────
    ax = axes[0, 1]
    if len(ind.top_q_sector_pct) > 0:
        all_secs = sorted(set(ind.top_q_sector_pct.index) | set(ind.bot_q_sector_pct.index))
        x = np.arange(len(all_secs))
        width = 0.35
        top_vals = [ind.top_q_sector_pct.get(s, 0) for s in all_secs]
        bot_vals = [ind.bot_q_sector_pct.get(s, 0) for s in all_secs]
        ax.bar(x - width / 2, top_vals, width, label='Long (Top Q)', color='green', alpha=0.7)
        ax.bar(x + width / 2, bot_vals, width, label='Short (Bot Q)', color='red', alpha=0.7)
        uniform = 1.0 / len(all_secs) if len(all_secs) > 0 else 0
        ax.axhline(uniform, color='gray', linestyle='--', alpha=0.5,
                   label=f'Uniform ({uniform:.1%})')
        ax.set_xticks(x)
        ax.set_xticklabels([s[:12] for s in all_secs], rotation=45, ha='right', fontsize=7)
        ax.legend(fontsize=8)
    ax.set_title('Sector Composition: Long vs Short')
    ax.set_ylabel('Weight')
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel 3: HHI 集中度 ───────────────────────────
    ax = axes[1, 0]
    n_sectors = max(len(ind.top_q_sector_pct), 1)
    uniform_hhi = 1.0 / n_sectors
    bars = ax.bar(['Long HHI', 'Short HHI', 'Uniform'],
                  [ind.hhi_top, ind.hhi_bot, uniform_hhi],
                  color=['green', 'red', 'gray'], alpha=0.7, edgecolor='gray')
    ax.axhline(0.25, color='orange', linestyle='--', alpha=0.5, label='Concentrated (0.25)')
    for bar, val in zip(bars, [ind.hhi_top, ind.hhi_bot, uniform_hhi]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_title('HHI Concentration Index')
    ax.set_ylabel('HHI')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel 4: 行业中性 IC vs 原始 IC ────────────────
    ax = axes[1, 1]
    if ind.raw_ic_21d is not None and ind.neutralized_ic is not None:
        labels = ['Raw IC (21d)', 'Neutralized IC']
        vals = [ind.raw_ic_21d, ind.neutralized_ic]
        colors = ['steelblue', 'teal']
        bars = ax.bar(labels, vals, color=colors, edgecolor='gray', alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:+.4f}', ha='center',
                    va='bottom' if val >= 0 else 'top', fontsize=10)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        if ind.raw_ic_21d != 0:
            decay = (1 - ind.neutralized_ic / ind.raw_ic_21d) * 100
            ax.text(0.97, 0.97, f'IC decay: {decay:.0f}%',
                    transform=ax.transAxes, fontsize=10, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.set_title('Industry-Neutralized IC')
    else:
        ax.text(0.5, 0.5, 'No IC data\n(close not provided)',
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('Industry-Neutralized IC')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════
# 完整报告生成
# ═══════════════════════════════════════════════════════════════

def plot_full_factor_report(
    diag_report: DiagnosticReport,
    alpha_report: AlphaTestReport,
    save_dir: Optional[str] = None,
):
    """
    生成单因子的全部图表, 保存到目录.
    """
    name = alpha_report.factor_name

    if save_dir:
        d = Path(save_dir) / name
        d.mkdir(parents=True, exist_ok=True)

        plot_diagnostic_summary(diag_report, str(d / 'diagnostic_summary.png'))
        if diag_report.industry is not None:
            plot_industry_distribution(diag_report, str(d / 'industry_distribution.png'))
        plot_alpha_test_summary(alpha_report, str(d / 'alpha_test_summary.png'))
        plot_ic_decay_curve(alpha_report, str(d / 'ic_decay.png'))
        plot_fama_macbeth(alpha_report, str(d / 'fama_macbeth.png'))
        plot_sub_sample_decomposition(alpha_report, str(d / 'sub_sample.png'))
        print(f'Saved {name} report to {d}')
    else:
        plot_diagnostic_summary(diag_report)
        if diag_report.industry is not None:
            plot_industry_distribution(diag_report)
        plot_alpha_test_summary(alpha_report)
        plot_ic_decay_curve(alpha_report)
        plot_fama_macbeth(alpha_report)
        plot_sub_sample_decomposition(alpha_report)
