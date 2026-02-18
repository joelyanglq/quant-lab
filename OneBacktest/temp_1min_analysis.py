"""
Investigate why some symbols in 1min data have fewer than 390 bars per day.
A regular US equity trading day runs 09:30-16:00 ET = 390 minutes.

NOTE: The 2025 file is ~1.1GB / 88M rows. Use 2026 for speed, or sample from 2025.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time as dtime

DATA_DIR = Path(r"D:\04_Project\quant-lab\data\processed\bars_1min")

# ── 1. List available files ──────────────────────────────────────
print("=" * 70)
print("AVAILABLE FILES")
print("=" * 70)
for f in sorted(DATA_DIR.iterdir()):
    if f.suffix == ".parquet":
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:30s}  {size_mb:8.1f} MB")

# ── 2. Load data ─────────────────────────────────────────────────
# Use 2026 (smaller), then fall back to 2025
for fname in ["2026_1m.parquet", "2025_1m.parquet"]:
    fpath = DATA_DIR / fname
    if fpath.exists():
        print(f"\n{'=' * 70}")
        print(f"LOADING: {fname}")
        print(f"{'=' * 70}")
        df = pd.read_parquet(fpath)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Index type: {type(df.index).__name__}")
        print(f"dtypes:\n{df.dtypes}")
        print(f"\nFirst 3 rows:\n{df.head(3)}")
        print(f"\nLast 3 rows:\n{df.tail(3)}")
        break

# ── Normalize timestamp ──────────────────────────────────────────
if isinstance(df.index, pd.DatetimeIndex):
    ts_col = 'timestamp'
    df = df.reset_index(names=ts_col)
elif 'timestamp' in df.columns:
    ts_col = 'timestamp'

df[ts_col] = pd.to_datetime(df[ts_col])

# Strip timezone for easier handling
if df[ts_col].dt.tz is not None:
    df[ts_col] = df[ts_col].dt.tz_localize(None)

df['date'] = df[ts_col].dt.date
df['time'] = df[ts_col].dt.time

print(f"\nTimestamp range: {df[ts_col].min()} to {df[ts_col].max()}")
print(f"Unique symbols: {df['symbol'].nunique()}")
print(f"Unique dates: {df['date'].nunique()}")

# ── 3. Pre/Post market check ────────────────────────────────────
print(f"\n{'=' * 70}")
print("PRE/POST MARKET BAR CHECK")
print(f"{'=' * 70}")
pre_market = df[df['time'] < dtime(9, 30)]
post_market = df[df['time'] >= dtime(16, 0)]
regular = df[(df['time'] >= dtime(9, 30)) & (df['time'] < dtime(16, 0))]

print(f"  Pre-market bars  (before 09:30): {len(pre_market):>12,} ({100*len(pre_market)/len(df):.2f}%)")
print(f"  Regular session  (09:30-15:59):  {len(regular):>12,} ({100*len(regular)/len(df):.2f}%)")
print(f"  Post-market bars (16:00+):       {len(post_market):>12,} ({100*len(post_market)/len(df):.2f}%)")

if len(pre_market) > 0:
    print(f"  Pre-market time range: {pre_market['time'].min()} to {pre_market['time'].max()}")
if len(post_market) > 0:
    print(f"  Post-market time range: {post_market['time'].min()} to {post_market['time'].max()}")

# ── 4. Count ALL bars per (symbol, day) - including extended hours ──
print(f"\n{'=' * 70}")
print("ALL-HOURS BAR COUNTS PER (SYMBOL, DAY)")
print(f"{'=' * 70}")
all_counts = df.groupby(['symbol', 'date']).size().reset_index(name='n_bars')
print(f"Total (symbol, day) pairs: {len(all_counts):,}")
print(f"Bar count stats:\n{all_counts['n_bars'].describe()}")

# ── 5. Count REGULAR-SESSION bars per (symbol, day) ─────────────
print(f"\n{'=' * 70}")
print("REGULAR SESSION (09:30-15:59) BAR COUNTS PER (SYMBOL, DAY)")
print(f"{'=' * 70}")
reg_counts = regular.groupby(['symbol', 'date']).size().reset_index(name='n_bars')
total_pairs = len(reg_counts)
print(f"Total (symbol, day) pairs: {total_pairs:,}")
print(f"Bar count stats:\n{reg_counts['n_bars'].describe()}")

# ── 6. Distribution ─────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("DISTRIBUTION OF REGULAR-SESSION BAR COUNTS")
print(f"{'=' * 70}")

# Exact key values
for threshold in [390, 389, 388, 387, 386, 385, 380, 370, 360, 350,
                  300, 250, 210, 200, 100, 50, 1]:
    exact = (reg_counts['n_bars'] == threshold).sum()
    if exact > 0:
        print(f"  Exactly {threshold:>3d} bars: {exact:>6,} pairs ({100*exact/total_pairs:.2f}%)")

above = (reg_counts['n_bars'] > 390).sum()
if above > 0:
    print(f"  > 390 bars:     {above:>6,} pairs ({100*above/total_pairs:.2f}%)")

# Ranges
print(f"\n  Bar count ranges:")
ranges = [
    (390, 999, "Exactly 390 or more"),
    (385, 389, "385-389 (nearly full)"),
    (380, 384, "380-384"),
    (350, 379, "350-379"),
    (300, 349, "300-349"),
    (200, 299, "200-299 (half-day range)"),
    (100, 199, "100-199"),
    (1, 99, "1-99 (very sparse)"),
]
for lo, hi, label in ranges:
    count = ((reg_counts['n_bars'] >= lo) & (reg_counts['n_bars'] <= hi)).sum()
    if count > 0:
        print(f"    {label:30s}: {count:>6,} ({100*count/total_pairs:.2f}%)")

below_390 = (reg_counts['n_bars'] < 390).sum()
print(f"\n  SUMMARY: {below_390:,} of {total_pairs:,} pairs "
      f"({100*below_390/total_pairs:.1f}%) have < 390 regular-session bars")

# ── 7. Half-day analysis ────────────────────────────────────────
print(f"\n{'=' * 70}")
print("HALF-DAY ANALYSIS (known early close days)")
print(f"{'=' * 70}")

# Check what the max time is per date
max_time_by_date = regular.groupby('date')[ts_col].max().reset_index()
max_time_by_date['max_time'] = max_time_by_date[ts_col].dt.strftime('%H:%M')
max_time_by_date['max_hour'] = max_time_by_date[ts_col].dt.hour

early_close = max_time_by_date[max_time_by_date['max_hour'] < 15]
normal_close = max_time_by_date[max_time_by_date['max_hour'] >= 15]

print(f"  Normal close dates (last bar >= 15:00): {len(normal_close)}")
print(f"  Early close dates  (last bar < 15:00):  {len(early_close)}")

if len(early_close) > 0:
    print(f"\n  Early close dates:")
    for _, row in early_close.iterrows():
        d = row['date']
        date_bars = reg_counts[reg_counts['date'] == d]
        mean_b = date_bars['n_bars'].mean()
        wday = pd.Timestamp(d).day_name()
        print(f"    {d}  ({wday})  last bar: {row['max_time']}  "
              f"mean bars/symbol: {mean_b:.0f}")

# ── 8. Separate analysis: half-days vs normal days ──────────────
print(f"\n{'=' * 70}")
print("SEPARATE ANALYSIS: HALF-DAYS vs NORMAL DAYS")
print(f"{'=' * 70}")

half_day_dates = set(early_close['date'].values)
normal_day_dates = set(normal_close['date'].values)

reg_normal = reg_counts[reg_counts['date'].isin(normal_day_dates)]
reg_half = reg_counts[reg_counts['date'].isin(half_day_dates)]

print(f"\n  NORMAL DAYS ({len(normal_day_dates)} dates, {len(reg_normal):,} symbol-day pairs):")
if len(reg_normal) > 0:
    below_n = (reg_normal['n_bars'] < 390).sum()
    exact_n = (reg_normal['n_bars'] == 390).sum()
    print(f"    Exactly 390 bars: {exact_n:,} ({100*exact_n/len(reg_normal):.1f}%)")
    print(f"    Below 390 bars:   {below_n:,} ({100*below_n/len(reg_normal):.1f}%)")
    print(f"    Stats:\n{reg_normal['n_bars'].describe()}")

if len(reg_half) > 0:
    print(f"\n  HALF-DAYS ({len(half_day_dates)} dates, {len(reg_half):,} symbol-day pairs):")
    below_h = (reg_half['n_bars'] < 390).sum()
    print(f"    Below 390 bars: {below_h:,} ({100*below_h/len(reg_half):.1f}%)")
    print(f"    Stats:\n{reg_half['n_bars'].describe()}")

# ── 9. Symbols most affected (normal days only) ─────────────────
print(f"\n{'=' * 70}")
print("SYMBOLS MOST AFFECTED (normal days only, highest % < 390 bars)")
print(f"{'=' * 70}")

sym_stats = reg_normal.groupby('symbol').agg(
    total_days=('n_bars', 'size'),
    days_below_390=('n_bars', lambda x: (x < 390).sum()),
    mean_bars=('n_bars', 'mean'),
    min_bars=('n_bars', 'min'),
    median_bars=('n_bars', 'median'),
).reset_index()
sym_stats['pct_below'] = 100 * sym_stats['days_below_390'] / sym_stats['total_days']
sym_stats = sym_stats.sort_values('pct_below', ascending=False)

print(f"\n  Top 30 symbols with most incomplete normal days:")
print(sym_stats.head(30).to_string(index=False))

print(f"\n  Bottom 10 (most complete):")
print(sym_stats.tail(10).to_string(index=False))

# ── 10. Dates most affected (normal days only) ──────────────────
print(f"\n{'=' * 70}")
print("DATES MOST AFFECTED (normal days, most symbols < 390)")
print(f"{'=' * 70}")

date_stats = reg_normal.groupby('date').agg(
    n_symbols=('n_bars', 'size'),
    sym_below_390=('n_bars', lambda x: (x < 390).sum()),
    mean_bars=('n_bars', 'mean'),
    min_bars=('n_bars', 'min'),
    max_bars=('n_bars', 'max'),
).reset_index()
date_stats['pct_below'] = 100 * date_stats['sym_below_390'] / date_stats['n_symbols']
date_stats = date_stats.sort_values('pct_below', ascending=False)

print(f"\n  Top 30 dates:")
print(date_stats.head(30).to_string(index=False))

# ── 11. Time-of-day pattern: where are bars missing? ────────────
print(f"\n{'=' * 70}")
print("TIME-OF-DAY ANALYSIS: Where are bars missing? (normal days)")
print(f"{'=' * 70}")

# Get incomplete normal-day pairs
incomplete = reg_normal[reg_normal['n_bars'] < 390][['symbol', 'date']]
sample_size = min(500, len(incomplete))
if sample_size > 0:
    sample = incomplete.sample(n=sample_size, random_state=42)

    # All expected minutes: 09:30 to 15:59
    expected_minutes = set(pd.date_range('09:30', '15:59', freq='1min').time)

    missing_at_open = 0
    missing_at_close = 0
    missing_middle = 0
    first_bars = []
    last_bars = []

    for _, row in sample.iterrows():
        mask = ((regular['symbol'] == row['symbol']) &
                (regular['date'] == row['date']))
        sub = regular.loc[mask]
        times = set(sub['time'].values)

        first_bars.append(sub[ts_col].min())
        last_bars.append(sub[ts_col].max())

        # Near open (09:30-09:35)
        open_times = {dtime(9, m) for m in range(30, 36)}
        if open_times - times:
            missing_at_open += 1

        # Near close (15:55-15:59)
        close_times = {dtime(15, m) for m in range(55, 60)}
        if close_times - times:
            missing_at_close += 1

        # Middle gaps
        actual_sorted = sorted(times)
        if len(actual_sorted) >= 2:
            covered = {t for t in expected_minutes
                       if actual_sorted[0] <= t <= actual_sorted[-1]}
            gaps_inside = covered - times
            if gaps_inside:
                missing_middle += 1

    print(f"\n  Of {sample_size} sampled incomplete (symbol,day) pairs on NORMAL days:")
    print(f"    Missing bars near OPEN  (09:30-09:35): {missing_at_open} ({100*missing_at_open/sample_size:.1f}%)")
    print(f"    Missing bars near CLOSE (15:55-15:59): {missing_at_close} ({100*missing_at_close/sample_size:.1f}%)")
    print(f"    Gaps in MIDDLE of session:             {missing_middle} ({100*missing_middle/sample_size:.1f}%)")

    # First and last bar distributions
    first_series = pd.Series(first_bars).dropna()
    last_series = pd.Series(last_bars).dropna()

    print(f"\n  First bar time distribution (top 15):")
    first_str = first_series.dt.strftime('%H:%M')
    for t, c in first_str.value_counts().head(15).items():
        print(f"    {t}: {c} ({100*c/len(first_str):.1f}%)")

    print(f"\n  Last bar time distribution (top 15):")
    last_str = last_series.dt.strftime('%H:%M')
    for t, c in last_str.value_counts().head(15).items():
        print(f"    {t}: {c} ({100*c/len(last_str):.1f}%)")
else:
    print("  No incomplete pairs on normal days!")

# ── 12. Examples of very low bar counts ──────────────────────────
print(f"\n{'=' * 70}")
print("EXAMPLES: LOWEST BAR COUNTS (normal days)")
print(f"{'=' * 70}")
low = reg_normal[reg_normal['n_bars'] < 390].sort_values('n_bars')
print(f"\n  Bottom 30:")
print(low.head(30).to_string(index=False))

# ── 13. Detailed look: AAPL or top symbol ────────────────────────
print(f"\n{'=' * 70}")
sym_check = 'AAPL'
if sym_check not in df['symbol'].values:
    sym_check = df['symbol'].value_counts().index[0]
print(f"DETAILED: {sym_check} daily bar counts (last 30 trading days)")
print(f"{'=' * 70}")

sym_reg = regular[regular['symbol'] == sym_check]
sym_daily = sym_reg.groupby('date').agg(
    n_bars=(ts_col, 'size'),
    first_bar=(ts_col, 'min'),
    last_bar=(ts_col, 'max'),
).reset_index()
sym_daily['first_time'] = sym_daily['first_bar'].dt.strftime('%H:%M')
sym_daily['last_time'] = sym_daily['last_bar'].dt.strftime('%H:%M')
print(sym_daily.tail(30)[['date', 'n_bars', 'first_time', 'last_time']].to_string(index=False))

# Show days with < 390 for this symbol
missing_days = sym_daily[sym_daily['n_bars'] < 390]
print(f"\n  {sym_check} days with < 390 bars: {len(missing_days)} / {len(sym_daily)}")
if len(missing_days) > 0:
    print(missing_days[['date', 'n_bars', 'first_time', 'last_time']].to_string(index=False))

# ── 14. Check if specific "missing bar" patterns exist ───────────
print(f"\n{'=' * 70}")
print("MINUTE-BY-MINUTE COVERAGE: How many symbols have a bar at each minute?")
print(f"{'=' * 70}")
# Pick a specific normal day with many symbols
if len(normal_day_dates) > 0:
    # Pick the most recent normal day with lots of data
    latest_normal = date_stats.sort_values('date', ascending=False).iloc[0]
    check_date = latest_normal['date']
    day_data = regular[regular['date'] == check_date]
    n_sym = day_data['symbol'].nunique()

    minute_coverage = day_data.groupby('time')['symbol'].nunique().reset_index()
    minute_coverage.columns = ['time', 'n_symbols']
    minute_coverage = minute_coverage.sort_values('time')

    print(f"\n  Date: {check_date}  ({n_sym} symbols)")
    print(f"  Bars at each minute (showing gaps only where < {n_sym} symbols):")
    gaps = minute_coverage[minute_coverage['n_symbols'] < n_sym]
    if len(gaps) == 0:
        print(f"    All 390 minutes have all {n_sym} symbols -- no gaps!")
    else:
        print(f"    {len(gaps)} minutes with fewer than {n_sym} symbols:")
        # Show distribution of gap times
        for _, row in gaps.head(20).iterrows():
            print(f"    {row['time']}  {row['n_symbols']}/{n_sym} symbols")
        if len(gaps) > 20:
            print(f"    ... and {len(gaps) - 20} more")
        # Bin by hour
        gaps['hour'] = pd.Series([t.hour for t in gaps['time'].values])
        print(f"\n  Gap minutes by hour:")
        for hour, group in gaps.groupby('hour'):
            print(f"    {hour:02d}:xx  {len(group)} minutes with gaps")

print(f"\n{'=' * 70}")
print("ANALYSIS COMPLETE")
print(f"{'=' * 70}")
