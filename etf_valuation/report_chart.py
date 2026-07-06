"""Generate valuation dashboard chart from latest snapshots."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from etf_valuation.config import load_config, METRICS

plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d', 'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9', 'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
    'grid.color': '#21262d', 'font.size': 9,
    'font.family': 'Microsoft YaHei',
})

config = load_config()
snapshots_dir = config.get_storage_path("snapshots")

TICKERS = [
    'SPY',
    'XLK','XLF','XLV','XLY','XLP','XLE','XLI','XLU','XLB','XLRE','XLC',
    'SMH','IBB','XBI','ICLN','KRE','XOP','IYR','XHB','ITB','HACK','BOTZ','ARKK','TAN',
]
METRICS_LIST = ['pe_ttm','pb_lf','ps_ttm','div_yield','ev_ebitda','erp']
METRIC_LABELS = ['PE TTM','P/B','P/S','Div Yield','EV/EBITDA','ERP']


def pct_to_signal(p):
    if p <= 0.10: return 'STRONG_BUY'
    if p <= 0.25: return 'BUY'
    if p <= 0.40: return 'LEAN_BUY'
    if p <= 0.60: return 'HOLD'
    if p <= 0.75: return 'LEAN_SELL'
    if p <= 0.90: return 'SELL'
    return 'STRONG_SELL'


def signal_color(p):
    if p <= 0.10: return '#238636'
    if p <= 0.25: return '#3fb950'
    if p <= 0.40: return '#56d364'
    if p <= 0.60: return '#d29922'
    if p <= 0.75: return '#db6d28'
    if p <= 0.90: return '#f85149'
    return '#da3633'


def fmt_val(metric, val):
    if np.isnan(val):
        return ''
    if metric in ('div_yield', 'erp'):
        return f'{val*100:.1f}%'
    return f'{val:.1f}'


def load_data():
    rows = []
    for ticker in TICKERS:
        f = snapshots_dir / f'{ticker}_history.parquet'
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        if len(df) < 4:
            continue
        etf_def = config.etfs.get(ticker)
        if not etf_def:
            continue

        row = {'ticker': ticker, 'name': etf_def.name, 'tier': etf_def.tier, 'primary': etf_def.primary}
        for metric in METRICS_LIST:
            if metric not in df.columns:
                continue
            series = df[metric].dropna()
            if len(series) < 3:
                continue
            current = series.iloc[-1]
            pct = float(np.mean(series.values <= current))
            meta = METRICS.get(metric, {})
            direction = meta.get('direction', '')
            signal_pct = (1.0 - pct) if direction == 'higher_cheap' else pct
            row[f'{metric}_val'] = current
            row[f'{metric}_pct'] = pct
            row[f'{metric}_signal_pct'] = signal_pct
        rows.append(row)
    return pd.DataFrame(rows)


def draw_dashboard(data, out_path):
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(f'US ETF Valuation Dashboard — {date.today().isoformat()}',
                 fontsize=18, fontweight='bold', color='#58a6ff', y=0.98)

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.7], width_ratios=[1, 1.3],
                          hspace=0.25, wspace=0.15)

    # ── Panel 1: Primary Metric Signal Bars ──
    ax1 = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(len(data))
    signal_pcts = []
    colors = []
    for _, r in data.iterrows():
        pri = r['primary']
        sp = r.get(f'{pri}_signal_pct', 0.5)
        signal_pcts.append(sp)
        colors.append(signal_color(sp))

    ax1.barh(y_pos, [p * 100 for p in signal_pcts], color=colors, height=0.7,
             edgecolor='#30363d', linewidth=0.5)

    for thresh in [10, 25, 50, 75, 90]:
        ax1.axvline(thresh, color='#484f58', linewidth=0.8, linestyle=':', alpha=0.6)
    ax1.axvspan(0, 25, alpha=0.05, color='#3fb950')
    ax1.axvspan(75, 100, alpha=0.05, color='#f85149')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([r['ticker'] for _, r in data.iterrows()], fontsize=9, fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('Signal Percentile (0%=Cheapest, 100%=Most Expensive)', fontsize=9)
    ax1.set_title('Primary Metric Signal', color='#f0f6fc', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()

    for i, (sp, (_, r)) in enumerate(zip(signal_pcts, data.iterrows())):
        pri = r['primary']
        sig = pct_to_signal(sp)
        val = r.get(f'{pri}_val', np.nan)
        val_s = fmt_val(pri, val)
        x_text = sp * 100 + 1.5
        ha = 'left'
        if x_text > 80:
            x_text = sp * 100 - 1.5
            ha = 'right'
        ax1.text(x_text, i, f'{val_s}  {sig}', va='center', ha=ha, fontsize=7.5,
                 color='#f0f6fc', fontweight='bold')

    ax1.grid(True, axis='x', alpha=0.15)

    # ── Panel 2: Multi-Metric Percentile Heatmap ──
    ax2 = fig.add_subplot(gs[0, 1])
    heatmap_data = np.full((len(data), len(METRICS_LIST)), np.nan)
    for i, (_, r) in enumerate(data.iterrows()):
        for j, metric in enumerate(METRICS_LIST):
            heatmap_data[i, j] = r.get(f'{metric}_signal_pct', np.nan)

    cmap_custom = LinearSegmentedColormap.from_list('valuation',
        ['#238636', '#3fb950', '#56d364', '#d29922', '#db6d28', '#f85149', '#da3633'])

    im = ax2.imshow(heatmap_data, cmap=cmap_custom, aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(METRICS_LIST)))
    ax2.set_xticklabels(METRIC_LABELS, fontsize=9, fontweight='bold')
    ax2.set_yticks(range(len(data)))
    ax2.set_yticklabels([r['ticker'] for _, r in data.iterrows()], fontsize=9, fontweight='bold')

    for i, (_, r) in enumerate(data.iterrows()):
        for j, metric in enumerate(METRICS_LIST):
            val = r.get(f'{metric}_val', np.nan)
            sp = r.get(f'{metric}_signal_pct', np.nan)
            if np.isnan(val):
                continue
            txt = fmt_val(metric, val)
            text_color = '#0d1117' if 0.3 < sp < 0.7 else '#f0f6fc'
            fw = 'bold' if metric == r['primary'] else 'normal'
            ax2.text(j, i, txt, ha='center', va='center', fontsize=7, color=text_color, fontweight=fw)
            if metric == r['primary']:
                ax2.plot([j-0.4, j+0.4], [i+0.35, i+0.35], color='#f0f6fc', linewidth=1.5)

    ax2.set_title('All Metrics Signal Percentile (bold underline = primary)',
                  color='#f0f6fc', fontsize=12, fontweight='bold')

    # Tier separators
    prev_tier = None
    for i, (_, r) in enumerate(data.iterrows()):
        if prev_tier and r['tier'] != prev_tier:
            ax2.axhline(i - 0.5, color='#58a6ff', linewidth=1.5)
            ax1.axhline(i - 0.5, color='#58a6ff', linewidth=1.5)
        prev_tier = r['tier']

    cbar = fig.colorbar(im, ax=ax2, shrink=0.5, pad=0.02)
    cbar.set_label('Signal Pct (0=Cheap, 1=Expensive)', color='#c9d1d9', fontsize=8)
    cbar.set_ticks([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    cbar.set_ticklabels(['0%', '10%', '25%', '50%', '75%', '90%', '100%'])
    cbar.ax.tick_params(colors='#8b949e', labelsize=7)

    # ── Panel 3: Top Buy / Hold / Sell ──
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_xlim(0, 20)
    ax3.set_ylim(0, 10)
    ax3.axis('off')

    buy_signals = []
    sell_signals = []
    hold_signals = []
    for _, r in data.iterrows():
        pri = r['primary']
        sp = r.get(f'{pri}_signal_pct', 0.5)
        val = r.get(f'{pri}_val', np.nan)
        pct_raw = r.get(f'{pri}_pct', 0.5)
        entry = {
            'ticker': r['ticker'], 'name': r['name'], 'metric': pri,
            'val': val, 'pct': pct_raw, 'signal_pct': sp, 'signal': pct_to_signal(sp),
        }
        if sp <= 0.25:
            buy_signals.append(entry)
        elif sp >= 0.75:
            sell_signals.append(entry)
        else:
            hold_signals.append(entry)

    buy_signals.sort(key=lambda x: x['signal_pct'])
    sell_signals.sort(key=lambda x: -x['signal_pct'])

    # Buy column
    ax3.text(2.5, 9.5, 'BUY Signals', fontsize=13, fontweight='bold', color='#3fb950', ha='center', va='top')
    ax3.plot([0.3, 4.7], [9.1, 9.1], color='#3fb950', linewidth=2)
    for i, e in enumerate(buy_signals[:8]):
        y = 8.5 - i * 1.0
        val_s = fmt_val(e['metric'], e['val'])
        ax3.scatter([0.5], [y], s=120, color=signal_color(e['signal_pct']),
                    zorder=5, edgecolors='white', linewidth=1)
        ax3.text(0.9, y, e['ticker'], fontsize=11, fontweight='bold', va='center', color='#f0f6fc')
        ax3.text(2.2, y, e['name'][:12], fontsize=8, va='center', color='#8b949e')
        ax3.text(3.8, y, f"{e['metric'].upper()} {val_s}", fontsize=8, va='center',
                 color='#3fb950', fontweight='bold')

    # Divider
    ax3.plot([5.2, 5.2], [0.5, 9.5], color='#30363d', linewidth=1)

    # Hold column
    ax3.text(7.5, 9.5, 'HOLD / NEUTRAL', fontsize=13, fontweight='bold', color='#d29922', ha='center', va='top')
    ax3.plot([5.5, 9.5], [9.1, 9.1], color='#d29922', linewidth=2)
    for i, e in enumerate(hold_signals[:8]):
        y = 8.5 - i * 1.0
        val_s = fmt_val(e['metric'], e['val'])
        ax3.scatter([5.7], [y], s=120, color=signal_color(e['signal_pct']),
                    zorder=5, edgecolors='white', linewidth=1)
        ax3.text(6.1, y, e['ticker'], fontsize=11, fontweight='bold', va='center', color='#f0f6fc')
        ax3.text(7.4, y, e['name'][:12], fontsize=8, va='center', color='#8b949e')
        ax3.text(8.8, y, f"{e['metric'].upper()} {val_s}", fontsize=8, va='center',
                 color='#d29922', fontweight='bold')

    # Divider
    ax3.plot([10.0, 10.0], [0.5, 9.5], color='#30363d', linewidth=1)

    # Sell column
    ax3.text(14, 9.5, 'SELL Signals', fontsize=13, fontweight='bold', color='#f85149', ha='center', va='top')
    ax3.plot([10.3, 17.7], [9.1, 9.1], color='#f85149', linewidth=2)
    for i, e in enumerate(sell_signals[:8]):
        y = 8.5 - i * 1.0
        val_s = fmt_val(e['metric'], e['val'])
        ax3.scatter([10.5], [y], s=120, color=signal_color(e['signal_pct']),
                    zorder=5, edgecolors='white', linewidth=1)
        ax3.text(10.9, y, e['ticker'], fontsize=11, fontweight='bold', va='center', color='#f0f6fc')
        ax3.text(12.2, y, e['name'][:12], fontsize=8, va='center', color='#8b949e')
        ax3.text(14.0, y, f"{e['metric'].upper()} {val_s}", fontsize=8, va='center',
                 color='#f85149', fontweight='bold')
        ax3.text(16.0, y, f"Pct {e['pct']*100:.0f}%", fontsize=8, va='center', color='#ff7b72')

    # Footer
    ax3.text(10, 0.1,
             f'Data: FMP ETF Holdings + Ratios TTM | 25 ETFs | {date.today()} | History: 28Q (2019Q3→2026Q2)',
             fontsize=8, ha='center', va='bottom', color='#484f58')

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    data = load_data()
    print(f'Loaded {len(data)} ETFs')
    out = Path('reports') / f'valuation_dashboard_{date.today()}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    draw_dashboard(data, out)
