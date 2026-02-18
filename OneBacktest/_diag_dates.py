"""Quick diagnostic: check date types and overlap for each factor."""
import sys, warnings, pandas as pd
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
from factor_research.data_loader import load_index_symbols, load_price_panel
from factor_research.weekly_pipeline import compute_all_factors

symbols = load_index_symbols()
factors = compute_all_factors(symbols)
panels = load_price_panel(symbols, '2025-01-01', '2026-12-31')
close = panels['close']
daily_dates = close.index

selected = ['noon_shade','vol_battle_pos','full_tidal','RS_12M']

print(f'\nclose idx type: {type(close.index[0])}')
print(f'close tz: {getattr(close.index, "tz", None)}')
print(f'close[0:3]: {list(close.index[:3])}')
print()

for name in selected:
    raw = factors[name]
    idx = raw.index
    overlap = len(idx.intersection(daily_dates))
    print(f'--- {name} ---')
    print(f'  idx type: {type(idx[0])}')
    print(f'  tz: {getattr(idx, "tz", None)}')
    print(f'  first 3: {list(idx[:3])}')
    print(f'  last 3: {list(idx[-3:])}')
    print(f'  n={len(idx)}, overlap_w_close={overlap}')
    # Try union
    combined = idx.union(daily_dates).sort_values()
    reindexed = raw.reindex(index=combined).ffill().reindex(index=daily_dates)
    valid = reindexed.notna().any(axis=1).sum()
    print(f'  After reindex+ffill: {valid} valid daily dates')
    print()
