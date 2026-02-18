"""
完整流程: 周频单因子 IC → 研报分组选优 → 筛选 → 选股
"""
import sys, time, warnings, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# ── 研报分组: 一份研报对应一组因子 ──
REPORT_GROUPS = {
    '方正-球队硬币': ['interday_rev_volflip', 'intraday_rev_volflip',
                      'overnight_rev_volflip', 'team_coin'],
    '方正-随波逐流': ['go_with_flow', 'lone_goose', 'sailing'],
    '方正-多空博弈': ['vol_battle_ret', 'vol_battle_pos', 'amp_battle',
                      'vol_battle', 'bull_bear_battle'],
    '方正-耀眼':     ['weekly_dazzling_vol', 'weekly_dazzling_ret', 'moderate_risk'],
    '方正-回归分析': ['morning_mist', 'noon_shade', 'night_frost', 'flower_hidden'],
    '方正-模糊性':   ['fuzzy_corr', 'fuzzy_amount_ratio', 'fuzzy_volume_ratio'],
    '方正-灾后重建': ['disaster_rebuild', 'peak_climbing'],
    '方正-潮汐':     ['full_tidal', 'strong_half_tidal', 'aggressive_weak_half',
                      'stable_weak_half'],
    'Jiang-跳跃':    ['weekly_jump', 'modified_amplitude_1', 'modified_amplitude_2',
                      'modified_amplitude', 'moth_to_flame'],
}
# 不属于研报的因子 (标准因子, 全部保留)
STANDARD_FACTORS = [
    'RS_12M', 'RS_6M', 'Range_52W', 'RSI_28W', 'RSI_16W',
    'ROE', 'ROIC', 'EV_EBITDA', 'FCF_Yield', 'PS',
    'FCF_Growth', 'EPS_Score', 'Growth_Stability',
]

# ═══════════════════════════════════════════════════════════════
print('=' * 70)
print('STEP 1: COMPUTE ALL 46 FACTORS')
print('=' * 70)

from factor_research.weekly_pipeline import compute_all_factors, prepare_weekly
from factor_research.data_loader import load_index_symbols, load_price_panel
from factor_research.screening import compute_ic_summary, screen_factors

symbols = load_index_symbols()
print(f'Universe: {len(symbols)} symbols')

t_total = time.time()
all_factors = compute_all_factors(symbols)
print(f'\nTotal factors computed: {len(all_factors)}')
print(f'Factor names: {sorted(all_factors.keys())}')

# ═══════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('STEP 2: WEEKLY IC TEST (ALL 46 FACTORS)')
print('=' * 70)

panels_1d = load_price_panel(symbols, '2025-01-01', '2026-12-31')
close = panels_1d['close']
factors_w, fwd_ret_w = prepare_weekly(all_factors, close)

ic_df = compute_ic_summary(factors_w, fwd_ret_w)
ic_df = ic_df.sort_values('icir', ascending=False)

print(f'\n{"Factor":<30s} {"mean_IC":>8s} {"IC_std":>8s} {"ICIR":>8s} {"n_wk":>6s}')
print('-' * 62)
for name, row in ic_df.iterrows():
    print(f'{name:<30s} {row["mean_ic"]:>+8.4f} {row["ic_std"]:>8.4f} '
          f'{row["icir"]:>+8.2f} {int(row["n_periods"]):>6d}')

# 保存完整 IC 结果
ic_df.to_csv(r'D:\04_Project\quant-lab\output\ic_summary_all46.csv')
print(f'\nSaved to output/ic_summary_all46.csv')

# ═══════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('STEP 3: KEEP BEST FACTOR PER RESEARCH REPORT')
print('=' * 70)

best_per_report = {}
for report, members in REPORT_GROUPS.items():
    available = [m for m in members if m in ic_df.index]
    if not available:
        print(f'  {report}: NO FACTORS AVAILABLE')
        continue
    sub = ic_df.loc[available].copy()
    sub['abs_icir'] = sub['icir'].abs()
    best_name = sub['abs_icir'].idxmax()
    best_icir = ic_df.loc[best_name, 'icir']
    print(f'  {report:<16s} → {best_name:<30s} ICIR={best_icir:+.2f}  '
          f'(from {len(available)} candidates)')
    for m in available:
        tag = ' ★' if m == best_name else '  '
        print(f'    {tag} {m:<28s} ICIR={ic_df.loc[m, "icir"]:+.2f}')
    best_per_report[report] = best_name

# 构建保留因子列表
kept_factors = list(STANDARD_FACTORS)  # 13 standard
kept_factors += list(best_per_report.values())  # 9 best per report
# 去重 (以防重叠)
kept_factors = list(dict.fromkeys(kept_factors))
# 只保留实际计算出的
kept_factors = [f for f in kept_factors if f in factors_w]

print(f'\nKept factors: {len(kept_factors)}')
for f in kept_factors:
    icir = ic_df.loc[f, 'icir'] if f in ic_df.index else float('nan')
    src = 'standard'
    for rep, best in best_per_report.items():
        if f == best:
            src = rep
            break
    print(f'  {f:<30s} ICIR={icir:+.2f}  ({src})')

# ═══════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('STEP 4: FACTOR SCREENING (IC + CORRELATION DEDUP)')
print('=' * 70)

kept_factors_w = {k: factors_w[k] for k in kept_factors if k in factors_w}
selected, ic_screened = screen_factors(kept_factors_w, fwd_ret_w,
                                        min_abs_ic=0.005, max_corr=0.7)

print(f'\nScreening result: {len(selected)}/{len(kept_factors)} factors survived')
print(f'\nSelected factors:')
for i, f in enumerate(selected, 1):
    icir = ic_screened.loc[f, 'icir'] if f in ic_screened.index else 0
    ic = ic_screened.loc[f, 'mean_ic'] if f in ic_screened.index else 0
    print(f'  {i:>2d}. {f:<30s} IC={ic:+.4f}  ICIR={icir:+.2f}')

# 被筛掉的
removed = set(kept_factors) - set(selected)
if removed:
    print(f'\nRemoved ({len(removed)}):')
    for f in removed:
        reason = 'low IC' if f in ic_screened.index and abs(ic_screened.loc[f, 'mean_ic']) < 0.005 else 'corr dedup'
        print(f'  - {f:<30s} ({reason})')

# ═══════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('STEP 5: STOCK SELECTION WITH LATEST DATA')
print('=' * 70)

# 取每个因子最新值, 按 IC 方向调整, z-score 等权打分
latest = {}
ic_direction = {}
for name in selected:
    panel = all_factors[name]
    row = panel.iloc[-1].dropna()
    if len(row) > 0:
        latest[name] = row
    # IC 方向
    ic_val = ic_screened.loc[name, 'mean_ic'] if name in ic_screened.index else 0
    ic_direction[name] = +1 if ic_val >= 0 else -1

df = pd.DataFrame(latest)

# MAD winsorize + z-score + IC 方向
for col in df.columns:
    direction = ic_direction.get(col, +1)
    s = df[col]
    median = s.median()
    mad = (s - median).abs().median()
    if mad > 0:
        cutoff = 3 * 1.4826 * mad
        s = s.clip(median - cutoff, median + cutoff)
    mu, sd = s.mean(), s.std()
    if sd > 0:
        df[col] = direction * (s - mu) / sd
    else:
        df[col] = 0.0

df['score'] = df.mean(axis=1)
scored = df.sort_values('score', ascending=False)

today = pd.Timestamp.now().strftime('%Y-%m-%d')
n_universe = len(scored)

print(f'\n  Date: {today}')
print(f'  Universe: {n_universe} symbols')
print(f'  Factors: {len(selected)} screened')
print()

# 展示 top 5 因子列
show_cols = selected[:5]
header = f'  {"Rank":>4}  {"Symbol":<6}  {"Score":>7}'
for f in show_cols:
    header += f'  {f[:12]:>12}'
print(header)
print('  ' + '-' * (len(header) - 2))

for i, (sym, row) in enumerate(scored.head(20).iterrows(), 1):
    line = f'  {i:>4}  {sym:<6}  {row["score"]:>+7.3f}'
    for f in show_cols:
        val = row.get(f, np.nan)
        if pd.isna(val):
            line += f'  {"N/A":>12}'
        else:
            line += f'  {val:>+12.3f}'
    print(line)

# 保存结果
scored.to_csv(r'D:\04_Project\quant-lab\output\stock_selection.csv')

print(f'\nTotal time: {time.time()-t_total:.0f}s')
print(f'Results saved to output/stock_selection.csv')

# 保存筛选后的因子列表 + IC方向到 JSON
result_meta = {
    'selected_factors': selected,
    'ic_direction': ic_direction,
    'ic_stats': {f: {
        'mean_ic': float(ic_screened.loc[f, 'mean_ic']),
        'icir': float(ic_screened.loc[f, 'icir']),
    } for f in selected if f in ic_screened.index},
}
with open(r'D:\04_Project\quant-lab\output\screening_result.json', 'w') as fh:
    json.dump(result_meta, fh, indent=2)
print('Screening metadata saved to output/screening_result.json')
