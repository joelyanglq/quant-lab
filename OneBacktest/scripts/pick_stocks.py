"""
选股脚本 — 基于 17 个筛选因子 + 等权 z-score 打分

流程:
    1. 加载 S&P500 + NDX100 股票池, 读取 1d (+ 可选 1min) 数据
    2. 计算 17 个筛选后因子
    3. 按 IC 方向调整, 横截面 z-score 等权打分
    4. 输出 top N 选股结果

因子筛选结果来自 run_full_pipeline:
    46 因子 → 研报分组选优 (22) → IC filter |IC|>=0.005 + corr dedup <0.7 → 17

用法:
    cd OneBacktest
    python scripts/pick_stocks.py                  # 完整运行
    python scripts/pick_stocks.py --no-1min        # 只用 1d 因子 (9 个)
    python scripts/pick_stocks.py --top-n 20       # top 20
    python scripts/pick_stocks.py --start 2023-01-01
"""
import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ── 确保 OneBacktest 在 sys.path ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.prices import load_index_symbols, load_price_panel
from data.storage.parquet import ParquetStorage
from strategy.factors import (
    compute_technical_factors,
    compute_reversal_factors,
    compute_fundamental_factors,
    compute_surge_factors,
    compute_regression_factors,
    compute_fuzzy_factors,
    compute_tidal_factors,
    compute_jump_factors,
    compute_microstructure_factors,
    compute_battle_factors,
    compute_rebuild_factors,
    RTH_START, RTH_END,
)

BARS_1MIN_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'processed' / 'bars_1min'

# ═══════════════════════════════════════════════════════════════
# 筛选后的 17 因子 (研报分组选优 + IC filter + correlation dedup)
#
# direction: +1 表示 IC 为正 (值越大越好), -1 表示 IC 为负 (值越小越好)
# ═══════════════════════════════════════════════════════════════

SELECTED_FACTORS = {
    # ── 1d 因子 (9) ──
    # Technical (2)
    'RS_12M':                +1,   # IC=+0.038, ICIR=+0.29
    'Range_52W':             +1,   # IC=+0.038, ICIR=+0.27
    # Reversal (1) — 方正-球队硬币 best
    'intraday_rev_volflip':  +1,   # IC=+0.021, ICIR=+0.26
    # Fundamental (6)
    'PS':                    +1,   # IC=+0.026, ICIR=+0.20
    'EV_EBITDA':             +1,   # IC=+0.017, ICIR=+0.14
    'EPS_Score':             +1,   # IC=+0.010, ICIR=+0.10
    'FCF_Growth':            -1,   # IC=-0.006, ICIR=-0.06
    'ROE':                   -1,   # IC=-0.011, ICIR=-0.14
    'ROIC':                  -1,   # IC=-0.017, ICIR=-0.18

    # ── 1min 因子 (8) ──
    # Regression — 方正-回归分析 best
    'noon_shade':            +1,   # IC=+0.028, ICIR=+0.56
    # Microstructure — 方正-随波逐流 best
    'lone_goose':            +1,   # IC=+0.038, ICIR=+0.49
    # Rebuild — 方正-灾后重建 best
    'peak_climbing':         +1,   # IC=+0.033, ICIR=+0.43
    # Jump — Jiang-跳跃 best
    'modified_amplitude_2':  +1,   # IC=+0.040, ICIR=+0.40
    # Surge — 方正-耀眼 best
    'moderate_risk':         +1,   # IC=+0.019, ICIR=+0.23
    # Battle — 方正-多空博弈 best
    'vol_battle_pos':        +1,   # IC=+0.012, ICIR=+0.11
    # Fuzzy — 方正-模糊性 best
    'fuzzy_corr':            +1,   # IC=+0.006, ICIR=+0.08
    # Tidal — 方正-潮汐 best
    'full_tidal':            -1,   # IC=-0.026, ICIR=-0.27
}

# 1d-only 因子 (不需要 1min 数据)
_1D_FACTORS = {
    'RS_12M', 'Range_52W',
    'intraday_rev_volflip',
    'PS', 'EV_EBITDA', 'EPS_Score', 'FCF_Growth', 'ROE', 'ROIC',
}


# ═══════════════════════════════════════════════════════════════
# 1. 因子计算 (只算需要的)
# ═══════════════════════════════════════════════════════════════

def compute_selected_factors(
    symbols: List[str],
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    open_p: pd.DataFrame,
    storage_1min=None,
    start_1min: str = None,
    end_1min: str = None,
    use_1min: bool = True,
) -> Dict[str, pd.DataFrame]:
    """计算筛选后的因子, 返回 {name: DataFrame(dates x symbols)}."""
    all_factors = {}

    # ── 1d 因子 ──
    print('  Computing technical factors ...', end=' ')
    t0 = time.time()
    all_factors.update(compute_technical_factors(close, high, low))
    print(f'{time.time()-t0:.1f}s')

    print('  Computing reversal factors ...', end=' ')
    t0 = time.time()
    all_factors.update(compute_reversal_factors(close, open_p))
    print(f'{time.time()-t0:.1f}s')

    print('  Computing fundamental factors ...', end=' ')
    t0 = time.time()
    try:
        all_factors.update(compute_fundamental_factors(symbols, close))
    except Exception as e:
        print(f'SKIP ({e})')
    else:
        print(f'{time.time()-t0:.1f}s')

    # ── 1min 因子 ──
    if use_1min and storage_1min is not None and start_1min and end_1min:
        print(f'  Loading 1min data [{start_1min} ~ {end_1min}] ...', end=' ')
        t0 = time.time()
        try:
            raw_1m = storage_1min.load(
                symbols,
                pd.Timestamp(start_1min, tz='US/Eastern'),
                pd.Timestamp(end_1min, tz='US/Eastern'), '1m')
            t = raw_1m.index.time
            raw_1m = raw_1m[(t >= RTH_START) & (t <= RTH_END)]
            close_1m = raw_1m.pivot_table(index=raw_1m.index, columns='symbol', values='close')
            vol_1m = raw_1m.pivot_table(index=raw_1m.index, columns='symbol', values='volume')
            print(f'{time.time()-t0:.1f}s  ({len(close_1m)} bars)')
        except Exception as e:
            print(f'SKIP ({e})')
            storage_1min = None

        if storage_1min is not None:
            factor_groups = [
                ('surge', lambda: compute_surge_factors(close_1m, vol_1m)),
                ('regression', lambda: compute_regression_factors(close_1m, vol_1m)),
                ('fuzzy', lambda: compute_fuzzy_factors(close_1m, vol_1m)),
                ('tidal', lambda: compute_tidal_factors(close_1m, vol_1m)),
                ('jump', lambda: compute_jump_factors(close_1m, close, high, low)),
                ('microstructure', lambda: compute_microstructure_factors(close_1m, vol_1m, close, open_p)),
                ('battle', lambda: compute_battle_factors(raw_1m)),
            ]
            for name, fn in factor_groups:
                print(f'  Computing {name} ...', end=' ')
                t0 = time.time()
                try:
                    all_factors.update(fn())
                    print(f'{time.time()-t0:.1f}s')
                except Exception as e:
                    print(f'SKIP ({e})')

            print(f'  Computing rebuild ...', end=' ')
            t0 = time.time()
            try:
                panels_1m = {}
                for f in ['open', 'high', 'low', 'close', 'volume']:
                    panels_1m[f] = raw_1m.pivot_table(
                        index=raw_1m.index, columns='symbol', values=f)
                all_factors.update(compute_rebuild_factors(panels_1m))
                print(f'{time.time()-t0:.1f}s')
            except Exception as e:
                print(f'SKIP ({e})')

    # 只保留筛选后的因子
    selected_names = set(SELECTED_FACTORS.keys())
    if not use_1min:
        selected_names = selected_names & _1D_FACTORS
    computed = {k: v for k, v in all_factors.items() if k in selected_names}
    return computed


# ═══════════════════════════════════════════════════════════════
# 2. 打分 & 排名
# ═══════════════════════════════════════════════════════════════

def score_stocks(
    factors: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    取每个因子最新值, 按 IC 方向调整, MAD winsorize + z-score 等权打分.
    缺失因子的 z-score 视为 0 (截面均值).

    Returns:
        DataFrame: index=symbol, columns=[score, factor1, factor2, ...]
    """
    latest = {}
    for name, panel in factors.items():
        row = panel.iloc[-1].dropna()
        if len(row) > 0:
            latest[name] = row

    if not latest:
        return pd.DataFrame()

    df = pd.DataFrame(latest)

    # MAD winsorize + z-score (横截面), 按 IC 方向调整符号
    for col in df.columns:
        direction = SELECTED_FACTORS.get(col, +1)
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

    # 缺失因子填 0 (z-score 下 0 = 截面均值, 中性)
    df = df.fillna(0.0)
    df['score'] = df.mean(axis=1)
    return df.sort_values('score', ascending=False)


# ═══════════════════════════════════════════════════════════════
# 3. 输出
# ═══════════════════════════════════════════════════════════════

def print_result(
    scored: pd.DataFrame,
    factors_used: List[str],
    top_n: int = 10,
):
    today = pd.Timestamp.now().strftime('%Y-%m-%d')
    n_universe = len(scored)
    n_factors = len(factors_used)

    print()
    print('=' * 64)
    print(f'  Stock Selection - {today}')
    print(f'  Universe: {n_universe} symbols | '
          f'Factors: {n_factors}/17 used')
    print('=' * 64)
    print()

    top = scored.head(top_n)

    # 展示 top 5 因子 (按 |ICIR| 排序)
    icir_order = sorted(factors_used,
                        key=lambda f: abs(SELECTED_FACTORS.get(f, 0)),
                        reverse=True)[:5]
    header = f'  {"Rank":>4}  {"Symbol":<6}  {"Score":>7}'
    for f in icir_order:
        header += f'  {f[:12]:>12}'
    print(header)
    print('  ' + '-' * (len(header) - 2))

    for i, (sym, row) in enumerate(top.iterrows(), 1):
        line = f'  {i:>4}  {sym:<6}  {row["score"]:>+7.3f}'
        for f in icir_order:
            val = row.get(f, np.nan)
            if pd.isna(val):
                line += f'  {"N/A":>12}'
            else:
                line += f'  {val:>+12.3f}'
        print(line)

    # 因子摘要
    print()
    missing = set(SELECTED_FACTORS.keys()) - set(factors_used)
    if missing:
        print(f'  Missing factors ({len(missing)}): {", ".join(sorted(missing))}')
    print()


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Multi-factor stock selection (17 screened factors)')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top picks')
    parser.add_argument('--start', type=str, default='2023-01-01',
                        help='History start date for 1d data')
    parser.add_argument('--no-1min', action='store_true',
                        help='Skip 1min factors (use 9 daily factors only)')
    parser.add_argument('--start-1min', type=str, default=None,
                        help='1min data start (default: 4 months ago)')
    args = parser.parse_args()

    warnings.filterwarnings('ignore', 'Mean of empty slice')
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    use_1min = not args.no_1min

    # ── 1. 加载数据 ──
    print('[1/3] Loading data ...')
    symbols = load_index_symbols()
    print(f'  Universe: {len(symbols)} symbols')

    t0 = time.time()
    prices = load_price_panel(symbols, start=args.start)
    close = prices['close']
    high = prices['high']
    low = prices['low']
    open_p = prices['open']
    print(f'  1d bars: {close.shape[0]} days x {close.shape[1]} symbols  ({time.time()-t0:.1f}s)')

    # 1min storage
    storage_1min = None
    start_1min = args.start_1min
    end_1min = None
    if use_1min and BARS_1MIN_DIR.exists():
        storage_1min = ParquetStorage(str(BARS_1MIN_DIR))
        if start_1min is None:
            # 使用已加载的 1d 数据中最早的时间作为 start_1min
            start_1min = close.index[0].strftime('%Y-%m-%d')
        # 使用已加载的 1d 数据中最晚的时间作为 end_1min
        end_1min = close.index[-1].strftime('%Y-%m-%d')
        print(f'  1min range: {start_1min} ~ {end_1min}')
    elif use_1min:
        print(f'  1min data not found at {BARS_1MIN_DIR}, using 1d factors only')
        use_1min = False

    # ── 2. 计算因子 ──
    print('\n[2/3] Computing factors ...')
    t0 = time.time()
    factors = compute_selected_factors(
        symbols, close, high, low, open_p,
        storage_1min, start_1min, end_1min, use_1min)

    factors_used = list(factors.keys())
    expected = set(SELECTED_FACTORS.keys())
    if not use_1min:
        expected = expected & _1D_FACTORS
    n_expected = len(expected)
    print(f'  Computed: {len(factors_used)}/{n_expected} factors  ({time.time()-t0:.1f}s)')

    if not factors_used:
        print('ERROR: No factors computed. Check data availability.')
        return

    # ── 3. 打分 & 排名 ──
    print('\n[3/3] Scoring ...')
    scored = score_stocks(factors)

    if scored.empty:
        print('ERROR: No valid scores.')
        return

    # 过滤: 过去 5 日平均成交量 < 5M 的剔除
    volume = prices['volume']
    avg_vol_5d = volume.iloc[-5:].mean()
    liquid = avg_vol_5d[avg_vol_5d >= 5_000_000].index
    n_before = len(scored)
    scored = scored.loc[scored.index.intersection(liquid)]
    n_dropped = n_before - len(scored)
    if n_dropped > 0:
        print(f'  Filtered: {n_dropped} symbols with avg 5d volume < 5M')

    print_result(scored, factors_used, args.top_n)


if __name__ == '__main__':
    main()
