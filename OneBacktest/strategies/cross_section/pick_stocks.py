"""
选股脚本 — 基于 factor_registry.csv 中 active 因子 + 等权 z-score 打分

流程:
    1. 从 factor_registry.csv 读取 active 因子列表和方向
    2. 加载 S&P500 + NDX100 股票池, 读取 1d (+ 可选 1min) 数据
    3. 计算 style + alpha101 因子
    4. 按 IC 方向调整, 横截面 z-score 等权打分
    5. 输出 top N 选股结果

用法:
    cd OneBacktest
    python -m strategies.cross_section.pick_stocks                  # 完整运行
    python -m strategies.cross_section.pick_stocks --no-1min        # 只用 1d 因子
    python -m strategies.cross_section.pick_stocks --no-alpha101    # 只用 style 因子
    python -m strategies.cross_section.pick_stocks --top-n 20       # top 20
    python -m strategies.cross_section.pick_stocks --start 2023-01-01
"""
import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# ── 确保 OneBacktest 在 sys.path ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.prices import load_index_symbols, load_price_panel
from data.storage.parquet import ParquetStorage
from strategies.cross_section.factors import (
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
from strategies.cross_section.alpha101 import Alphas

ROOT = Path(__file__).resolve().parent.parent.parent
BARS_1MIN_DIR = ROOT.parent / 'data' / 'processed' / 'bars_1min'
REGISTRY_PATH = ROOT / 'output' / 'factor_registry.csv'


# ═══════════════════════════════════════════════════════════════
# 0. 从 factor_registry.csv 加载 active 因子
# ═══════════════════════════════════════════════════════════════

def load_active_factors(
    use_1min: bool = True,
    use_alpha101: bool = True,
) -> Tuple[Dict[str, int], Set[str], Set[str], Set[str]]:
    """
    读取 factor_registry.csv, 返回 prod 因子信息.

    compute_key 从 formula_code_ref 最后一段提取:
      "factors::compute_technical_factors::RS_12M" → "RS_12M"
      "alpha101::Alphas::alpha_033"                → "alpha_033"

    Returns:
        selected_factors:   {compute_key: direction} 全部 prod 因子
        factors_1d:         1d-only 因子 compute_key 集合 (input_data != 'price_1min')
        factors_alpha101:   alpha101 因子 compute_key 集合
        factors_neutralize: 需要行业中性化的因子 compute_key 集合
    """
    from strategies.cross_section.neutralize import VOLUME_BIASED_FACTORS

    df = pd.read_csv(REGISTRY_PATH, encoding='utf-8-sig')
    prod = df[df['status'] == 'prod'].copy()

    if not use_alpha101:
        prod = prod[~prod['factor_id'].str.startswith('CS_A101_')]
    if not use_1min:
        prod = prod[prod['input_data'] != 'price_1min']

    selected_factors = {}
    factors_1d = set()
    factors_alpha101 = set()
    factors_neutralize = set()

    for _, row in prod.iterrows():
        ref = row['formula_code_ref']
        compute_key = ref.split('::')[-1]
        direction = int(row['direction'])
        selected_factors[compute_key] = direction

        if row['input_data'] != 'price_1min':
            factors_1d.add(compute_key)
        if row['factor_id'].startswith('CS_A101_'):
            factors_alpha101.add(compute_key)
        if row['input_data'] == 'fundamental' or compute_key in VOLUME_BIASED_FACTORS:
            factors_neutralize.add(compute_key)

    return selected_factors, factors_1d, factors_alpha101, factors_neutralize


# ═══════════════════════════════════════════════════════════════
# 1. 因子计算 (只算需要的)
# ═══════════════════════════════════════════════════════════════

def compute_selected_factors(
    symbols: List[str],
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    open_p: pd.DataFrame,
    volume: pd.DataFrame,
    selected_factors: Dict[str, int],
    factors_1d: Set[str],
    factors_alpha101: Set[str],
    storage_1min=None,
    start_1min: str = None,
    end_1min: str = None,
    use_1min: bool = True,
) -> Dict[str, pd.DataFrame]:
    """计算筛选后的因子, 返回 {name: DataFrame(dates x symbols)}."""
    all_factors = {}
    target_names = set(selected_factors.keys())

    # ── Style 1d 因子 ──
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

    # ── Alpha101 因子 ──
    if factors_alpha101:
        print(f'  Computing alpha101 ({len(factors_alpha101)} factors) ...', end=' ')
        t0 = time.time()
        try:
            alphas_obj = Alphas(close, open_p, high, low, volume)
            for fid in factors_alpha101:
                # factor_id 格式: alpha_033 → 方法名: alpha_033
                method = getattr(alphas_obj, fid, None)
                if method is None:
                    print(f'\n    {fid}: method not found', end='')
                    continue
                val = method()
                if val is not None and isinstance(val, pd.DataFrame) and not val.empty:
                    val = val.replace([np.inf, -np.inf], np.nan)
                    all_factors[fid] = val
            print(f'{time.time()-t0:.1f}s')
        except Exception as e:
            print(f'SKIP ({e})')

    # ── Style 1min 因子 ──
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

    # 只保留 registry 中 active 的因子
    computed = {k: v for k, v in all_factors.items() if k in target_names}
    return computed


# ═══════════════════════════════════════════════════════════════
# 2. 打分 & 排名
# ═══════════════════════════════════════════════════════════════

def score_stocks(
    factors: Dict[str, pd.DataFrame],
    selected_factors: Dict[str, int],
    neutralize_factor_names: Set[str] = None,
    neutralize_sectors: bool = True,
) -> pd.DataFrame:
    """
    取每个因子最新值, 按 IC 方向调整, MAD winsorize + z-score 等权打分.
    缺失因子的 z-score 视为 0 (截面均值).

    Args:
        factors:                  {name: DataFrame(dates x symbols)}
        selected_factors:         {compute_key: direction}
        neutralize_factor_names:  需要行业中性化的因子名集合
        neutralize_sectors:       是否做行业中性化

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

    # 行业中性化: z-score 前减行业均值
    if neutralize_sectors and neutralize_factor_names:
        cols = neutralize_factor_names & set(df.columns)
        if cols:
            from strategies.cross_section.neutralize import (
                load_gics_sector_map, neutralize_factors,
            )
            sector_map = load_gics_sector_map()
            if sector_map is not None:
                df = neutralize_factors(df, sector_map, cols)

    # MAD winsorize + z-score (横截面), 按 IC 方向调整符号
    for col in df.columns:
        direction = selected_factors.get(col, +1)
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
    selected_factors: Dict[str, int],
    top_n: int = 10,
):
    today = pd.Timestamp.now().strftime('%Y-%m-%d')
    n_universe = len(scored)
    n_factors = len(factors_used)

    n_total = len(selected_factors)

    print()
    print('=' * 64)
    print(f'  Stock Selection - {today}')
    print(f'  Universe: {n_universe} symbols | '
          f'Factors: {n_factors}/{n_total} used')
    print('=' * 64)
    print()

    top = scored.head(top_n)

    # 展示 top 5 因子 (按 |direction| 排序 — 都是 1, 所以按名称)
    display_cols = factors_used[:5]
    header = f'  {"Rank":>4}  {"Symbol":<6}  {"Score":>7}'
    for f in display_cols:
        header += f'  {f[:12]:>12}'
    print(header)
    print('  ' + '-' * (len(header) - 2))

    for i, (sym, row) in enumerate(top.iterrows(), 1):
        line = f'  {i:>4}  {sym:<6}  {row["score"]:>+7.3f}'
        for f in display_cols:
            val = row.get(f, np.nan)
            if pd.isna(val):
                line += f'  {"N/A":>12}'
            else:
                line += f'  {val:>+12.3f}'
        print(line)

    # 因子摘要
    print()
    missing = set(selected_factors.keys()) - set(factors_used)
    if missing:
        print(f'  Missing factors ({len(missing)}): {", ".join(sorted(missing))}')
    print()


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Multi-factor stock selection (registry-driven)')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top picks')
    parser.add_argument('--start', type=str, default='2023-01-01',
                        help='History start date for 1d data')
    parser.add_argument('--no-1min', action='store_true',
                        help='Skip 1min factors (use daily factors only)')
    parser.add_argument('--no-alpha101', action='store_true',
                        help='Skip alpha101 factors (use style factors only)')
    parser.add_argument('--no-neutralize', action='store_true',
                        help='Skip sector neutralization for fundamental factors')
    parser.add_argument('--start-1min', type=str, default=None,
                        help='1min data start (default: use full 1d range)')
    args = parser.parse_args()

    warnings.filterwarnings('ignore', 'Mean of empty slice')
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    use_1min = not args.no_1min
    use_alpha101 = not args.no_alpha101

    # ── 0. 加载因子注册表 ──
    print('[0/3] Loading factor registry ...')
    if not REGISTRY_PATH.exists():
        print(f'ERROR: {REGISTRY_PATH} not found.')
        return
    selected_factors, factors_1d, factors_alpha101, factors_neutralize = \
        load_active_factors(use_1min=use_1min, use_alpha101=use_alpha101)
    print(f'  Active factors: {len(selected_factors)} '
          f'(style: {len(selected_factors) - len(factors_alpha101)}, '
          f'alpha101: {len(factors_alpha101)}, '
          f'neutralize: {len(factors_neutralize)})')

    # ── 1. 加载数据 ──
    print('\n[1/3] Loading data ...')
    symbols = load_index_symbols()
    print(f'  Universe: {len(symbols)} symbols')

    t0 = time.time()
    prices = load_price_panel(symbols, start=args.start)
    close = prices['close']
    high = prices['high']
    low = prices['low']
    open_p = prices['open']
    volume = prices['volume']
    print(f'  1d bars: {close.shape[0]} days x {close.shape[1]} symbols  ({time.time()-t0:.1f}s)')

    # 1min storage
    storage_1min = None
    start_1min = args.start_1min
    end_1min = None
    if use_1min and BARS_1MIN_DIR.exists():
        storage_1min = ParquetStorage(str(BARS_1MIN_DIR))
        if start_1min is None:
            start_1min = close.index[0].strftime('%Y-%m-%d')
        end_1min = close.index[-1].strftime('%Y-%m-%d')
        print(f'  1min range: {start_1min} ~ {end_1min}')
    elif use_1min:
        print(f'  1min data not found at {BARS_1MIN_DIR}, using 1d factors only')
        use_1min = False

    # ── 2. 计算因子 ──
    print('\n[2/3] Computing factors ...')
    t0 = time.time()
    factors = compute_selected_factors(
        symbols, close, high, low, open_p, volume,
        selected_factors, factors_1d, factors_alpha101,
        storage_1min, start_1min, end_1min, use_1min)

    factors_used = list(factors.keys())
    n_expected = len(selected_factors)
    print(f'  Computed: {len(factors_used)}/{n_expected} factors  ({time.time()-t0:.1f}s)')

    if not factors_used:
        print('ERROR: No factors computed. Check data availability.')
        return

    # ── 3. 打分 & 排名 ──
    neutralize = not args.no_neutralize
    print(f'\n[3/3] Scoring ... (neutralize={neutralize})')
    scored = score_stocks(
        factors, selected_factors,
        neutralize_factor_names=factors_neutralize,
        neutralize_sectors=neutralize,
    )

    if scored.empty:
        print('ERROR: No valid scores.')
        return

    print_result(scored, factors_used, selected_factors, args.top_n)


if __name__ == '__main__':
    main()
