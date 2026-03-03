# 截面多因子框架 (Cross-Section Multi-Factor Framework)

## 核心结论

本框架实现了从 **因子生产 → 因子验证 → 因子选股** 的完整量化选股流水线。覆盖 46 个因子（11 大类），经过 IC 筛选 + 相关性去重后保留 17 个有效因子，用等权 z-score 合成最终选股信号。

框架分为两条独立的工作流：

| 工作流 | 目的 | 频率 | 入口 |
|--------|------|------|------|
| **因子研究** | 新因子入库前的严格验证 | 因子开发时 | `run_factor_research.py` |
| **因子选股** | 用已验证因子生成持仓 | 每日/每周 | `pick_stocks.py` |

---

## 一、因子生产层

### 1.1 数据基础

```
data/
├── prices.py          load_index_symbols(), load_price_panel()
│                      S&P500+NDX100 成分股, 1d OHLCV (Massive API)
├── fundamentals.py    build_fundamental_panel(), build_shares_panel()
│                      季报 → TTM, Point-in-Time 无前视偏差
└── storage/           ParquetStorage 年度切分存储
```

- **1d 数据**: ~500 股 × 5yr+，来自 Massive API
- **1min 数据**: Twelve Data，近 4 个月（用于微观结构因子）
- **基本面**: Alpha Vantage 季报 → TTM，以 filing_date 避免前视偏差

### 1.2 因子库 (`factors.py`, 46 因子)

| 类别 | 数据源 | 因子数 | 代表因子 |
|------|--------|--------|----------|
| **技术面** | 1d | 5 | RS_12M, RSI_28W, Range_52W |
| **反转** | 1d | 4 | interday_rev_volflip, team_coin |
| **基本面** | 季报 | 8 | ROE, ROIC, EV_EBITDA, FCF_Yield |
| **微观结构** | 1min | 3 | go_with_flow, lone_goose |
| **多空博弈** | 1min | 5 | vol_battle_pos, bull_bear_battle |
| **耀眼因子** | 1min | 3 | weekly_dazzling_vol, moderate_risk |
| **回归因子** | 1min | 4 | morning_mist, noon_shade |
| **模糊性** | 1min | 3 | fuzzy_corr |
| **灾后重建** | 1min | 2 | disaster_rebuild, peak_climbing |
| **潮汐因子** | 1min | 4 | full_tidal |
| **跳跃因子** | 1min+1d | 5 | modified_amplitude_2 |

所有因子输出为 `pd.DataFrame(index=dates, columns=symbols)` 标准面板格式。

### 1.3 标准化 (`ranking.py`)

```
原始因子值 → MAD 缩尾 (3σ) → 截面 z-score → 标准化因子
```

- MAD winsorization: `σ_MAD = 1.4826 × median(|x - median(x)|)`
- 每日独立标准化，保证截面可比性

---

## 二、因子研究流水线

**目标**: 新因子入库前的完整验证，回答 "这个因子值得用吗？"

**入口**: `python -m strategies.cross_section.run_factor_research --factor <NAME>`

### 2.1 质量诊断 (`diagnostics.py`)

在做 alpha 测试之前，先确认因子 "看起来像个因子"：

| 检查项 | 方法 | 告警阈值 |
|--------|------|----------|
| **分布稳定性** | 滚动截面 mean/var/skew/kurt + OLS 趋势检验 | p < 0.01 → 漂移告警 |
| **截面覆盖率** | 每日有效值 / 总数 + 大小盘覆盖偏差 | < 30% → 告警 |
| **风格因子相关** | 与 size/value/momentum/quality/low_vol 截面 Spearman | \|corr\| > 0.7 → 告警 |
| **换手率与容量** | 信号 top 20% 每日成分变化 + ADV 参与率 | 换手 > 80% → 告警 |
| **行业分布** | 各行业因子均值 + 分位集中度 HHI + 行业中性 IC | HHI > 0.25 → 告警 |

**输出**:
- `DiagnosticReport` 数据结构（含 PASS/WARN 判定）
- `diagnostic_summary.png` (2×2: 分布/覆盖/风格/换手)
- `industry_distribution.png` (2×2: 行业均值/分位占比/HHI/中性IC)

### 2.2 Alpha 测试 (`alpha_test.py`)

因子通过质量诊断后，评估其预测能力：

| 分析项 | 方法 | 关键指标 |
|--------|------|----------|
| **多周期 IC** | 1d/5d/21d/63d 前瞻收益 Spearman IC | Mean IC, ICIR, NW t-stat |
| **IC 衰减** | ln\|IC(h)\| = a - b·h 指数衰减拟合 | Half-life (天) |
| **分组回测** | 五分位等权组合 + 扣费 L/S | Gross/Net Sharpe, MaxDD, 单调性 |
| **Fama-MacBeth** | 截面回归控制 mktcap + momentum + 行业 | Factor slope t-stat |
| **子样本分析** | 牛熊分解 (SPY vs 200MA) + 走样本外 | 分 regime IC/Sharpe |

**输出**:
- `AlphaTestReport` 数据结构
- `alpha_test_summary.png` (3×2 总览)
- `ic_decay.png`, `fama_macbeth.png`, `sub_sample.png`

### 2.3 完整研究流程

```
加载数据 → 计算因子 → 质量诊断 → Alpha 测试 → 出图 + CSV
   ↓           ↓           ↓            ↓           ↓
 prices    factors.py  diagnostics  alpha_test  output/factor_research/
```

```bash
# 单因子完整研究
python -m strategies.cross_section.run_factor_research --factor RS_12M --save-csv

# 全量因子扫描 (快速模式)
python -m strategies.cross_section.run_factor_research --n-symbols 50 --no-fama-macbeth

# 只做诊断
python -m strategies.cross_section.run_factor_research --stage diagnostics
```

---

## 三、因子筛选流水线

**目标**: 从 46 个因子中筛选出有效且不冗余的子集

**入口**: `python -m strategies.cross_section.run_pipeline`

### 3.1 筛选流程

```
46 因子 (全量)
    ↓ 分组去重 (每组保留 |ICIR| 最高)
22 因子
    ↓ IC 过滤 (|mean_IC| ≥ 0.005)
    ↓ 相关性去重 (max_corr ≤ 0.7, 贪心删低ICIR)
17 因子 (入选)
```

### 3.2 筛选标准

| 步骤 | 标准 | 说明 |
|------|------|------|
| IC 过滤 | \|mean_IC\| ≥ 0.005 | 过滤无预测力的因子 |
| 相关去重 | pairwise corr ≤ 0.7 | 避免信息冗余，贪心保留高 ICIR |
| 分组去重 | 同一研究来源保留最佳 | 9 个研报各取最优代表 |

### 3.3 模块依赖

```
screening.py
├── compute_rank_ic()        # 逐期 Spearman Rank IC
├── ic_filter()              # |IC| 阈值过滤
├── correlation_dedup()      # 贪心相关性去重
└── orthogonalize()          # 截面 OLS 正交化 (备用)
```

---

## 四、因子选股流水线

**目标**: 用已验证因子生成当期持仓推荐

**入口**: `python -m strategies.cross_section.pick_stocks`

### 4.1 选股流程

```
加载最新数据 → 计算 17 因子 → 截面标准化 → IC 方向调整 → 等权合成 → 排序出票
```

### 4.2 入选因子 (17 个)

**日频因子 (9 个)**:

| 因子 | IC | ICIR | 方向 |
|------|-----|------|------|
| RS_12M | +0.038 | +0.29 | +1 |
| Range_52W | +0.038 | +0.27 | +1 |
| intraday_rev_volflip | +0.021 | +0.26 | +1 |
| PS | +0.026 | +0.20 | +1 |
| EV_EBITDA | +0.017 | +0.14 | +1 |
| EPS_Score | +0.010 | +0.10 | +1 |
| FCF_Growth | -0.006 | -0.06 | -1 |
| ROE | -0.011 | -0.14 | -1 |
| ROIC | -0.017 | -0.18 | -1 |

**分钟频因子 (8 个)**:

| 因子 | IC | ICIR | 方向 |
|------|-----|------|------|
| noon_shade | +0.028 | +0.56 | +1 |
| lone_goose | +0.038 | +0.49 | +1 |
| peak_climbing | +0.033 | +0.43 | +1 |
| modified_amplitude_2 | +0.040 | +0.40 | +1 |
| moderate_risk | +0.019 | +0.23 | +1 |
| vol_battle_pos | +0.012 | +0.11 | +1 |
| fuzzy_corr | +0.006 | +0.08 | +1 |
| full_tidal | -0.026 | -0.27 | -1 |

### 4.3 合成方法

```python
# 对每个因子:
z_i = MAD_winsorize(factor_i) → z_score → × IC_direction_i

# 合成:
score = mean(z_1, z_2, ..., z_17)  # 等权, 缺失因子视为 0

# 选股:
top_N = sort_by(score, descending)[:N]
```

---

## 五、回测与打分

**目标**: 配置调仓频率、因子选择、打分方式，运行完整回测

**入口**: `python -m strategies.cross_section.run_backtest`

### 5.1 打分方式 (`scorer.py`)

| 方式 | 函数 | 说明 |
|------|------|------|
| **等权合成** | `score_equal_weight(factors, directions)` | z-score × direction → 等权 mean |
| **ML walk-forward** | `score_ml_walk_forward(factors, fwd_ret, method)` | 滚动训练 → 截面预测 |

ML 模型: `ridge` / `rf` / `xgb` / `lgb`

### 5.2 CLI 选项

```bash
python -m strategies.cross_section.run_backtest \
    --freq M                # D | W-FRI | M | Q
    --factors selected      # selected (17) | auto (IC筛选) | all | RS_12M,ROE,...
    --scorer equal_weight   # equal_weight | ridge | rf | xgb | lgb
    --top-n 30              # long-only top N
    --live                  # 输出最新选股
```

---

## 六、模块依赖关系

```
data/prices.py ─────────────────────────────┐
data/fundamentals.py ───────────────────────┤
                                            ↓
factors.py ←── 46 因子计算 ─────────────────┤
                                            ↓
ranking.py ←── 标准化 (MAD + z-score) ──────┤
                                            ↓
        ┌───────────────────────────────────┴───────────────────────┐
        ↓                                                           ↓
screening.py ←── IC/去重/正交                     diagnostics.py ←── 质量诊断
        ↓                                         alpha_test.py ←── Alpha 测试
backtest.py ←── 分组回测                          diagnostics_plotting.py ←── 可视化
analytics.py ←── 绩效指标                                 ↓
        ↓                                         run_factor_research.py ←── 研究 CLI
scorer.py ←── 等权/ML 打分
        ↓
run_backtest.py ←── 统一回测 CLI
pick_stocks.py ←── 生产选股
```

---

## 七、输出文件总览

### 因子研究输出 (`output/factor_research/`)

```
output/factor_research/
├── summary_diagnostics.csv      # 全因子诊断汇总
├── summary_alpha_test.csv       # 全因子 Alpha 汇总
└── <FACTOR_NAME>/
    ├── diagnostic_summary.png   # 2×2 诊断图
    ├── industry_distribution.png # 2×2 行业分布图
    ├── alpha_test_summary.png   # 3×2 Alpha 总览
    ├── ic_decay.png             # IC 衰减曲线
    ├── fama_macbeth.png         # FM 回归时序
    └── sub_sample.png           # 牛熊分解
```

### 回测输出 (`output/backtest/`)

```
output/backtest/
├── backtest_metrics.csv         # 绩效指标
├── long_short_returns.csv       # L/S 收益序列
├── quantile_returns.csv         # 分位收益
├── factors_used.csv             # 使用的因子 + 方向
├── live_picks.csv               # 最新选股 (--live)
└── backtest_summary.png         # 4 面板总览图
```
