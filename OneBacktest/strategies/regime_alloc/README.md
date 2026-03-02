# HMM/HSMM Regime-Based Asset Allocation

基于 Baitinger & Hoch (2024) 的市场状态识别资产配置策略。

## 核心思路

用 HMM/HSMM 识别市场状态 (bull/bear/neutral)，根据状态概率动态调整
S&P 500 与 T-Bill 之间的配置比例。

## 所需数据

| 数据 | 来源 | 状态 |
|------|------|------|
| S&P 500 日收益率 | `data/processed/bars_1d/` (SPY) | 已有 |
| 3M T-Bill 利率 | `data/processed/rates/tbill_3m.parquet` | 已有 |

## 参考文献

- Baitinger & Hoch (2024) "Simplicity versus Complexity: HMM vs HSMM for Regime-Based Asset Allocation"
- 论文结论：简单 HMM (2-3 states) + 日频数据即可，HSMM 的额外复杂度不产生 OOS 收益提升
