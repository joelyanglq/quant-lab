# 因子研究流水线: "这个因子值得用吗?"

新因子入库前的完整验证流程. 四个阶段, 每阶段有明确的淘汰逻辑, 一个因子必须**连闯四关**才能入库.

代码入口: `strategies/cross_section/run_alpha101.py` (批量) 或 `run_factor_research.py` (单因子)

---

## Phase 1: IC 筛选 — "有预测力吗?"

**模块**: `screening.py` → `compute_rank_ic()`

### 核心计算

对每个截面日期 t, 计算因子截面值与前瞻收益的 Spearman 秩相关:

$$IC(t) = \text{SpearmanCorr}\big( \text{factor}_i(t),\ r_i(t \to t+h) \big) \quad \text{跨全部股票 } i$$

其中前瞻收益:

$$r_i(t \to t+h) = \frac{\text{close}_i(t+h)}{\text{close}_i(t)} - 1$$

h 取 1d / 5d / 21d / 63d 四个周期.

用 Spearman 而非 Pearson, 因为截面因子值可能有极端值, rank 相关对离群值鲁棒.

### 显著性检验: Newey-West HAC t 统计量

`alpha_test.py` → `_newey_west_tstat()`

$$\text{NW方差} = \gamma_0 + 2 \sum_{j=1}^{L} \left(1 - \frac{j}{L+1}\right) \gamma_j$$

其中:

$$\gamma_j = \frac{1}{T} \sum_{t} \big(IC(t) - \bar{IC}\big)\big(IC(t-j) - \bar{IC}\big) \quad \text{(j 阶自协方差)}$$

$$L = \lfloor 4 \cdot (T/100)^{2/9} \rfloor \quad \text{(Bartlett kernel 截断)}$$

$$t = \frac{\bar{IC}}{\sqrt{\text{NW方差} / T}}$$

普通 t 统计量假设 IC 序列独立, 但 IC 有正自相关 (连续几天同一因子有效), Newey-West 修正了这种序列依赖, 是因子研究的**标准做法**. 不做修正会高估显著性.

### IC 半衰期

`alpha_test.py` → `_estimate_ic_half_life()`

用多周期均值 IC 拟合指数衰减:

$$\ln|IC(h)| = a - b \cdot h \quad \text{(OLS)}$$

$$\text{半衰期} = \frac{\ln 2}{b} \quad \text{(仅 } R^2 > 0.5 \text{ 且 } b > 0 \text{ 时有效)}$$

告诉你因子信号"几天内失效". 比如 alpha_054 半衰期 23 天, 意味着 1d IC 最强但 21d 时已衰减过半.

### 淘汰线

|NW t| ≤ 2 的因子在 Phase 1 直接淘汰. Alpha101 中 79/101 通过了 1d horizon 的检验. 取 Top 20 进入 Phase 2.

---

## Phase 2: 诊断 — "是个干净的因子吗?"

**模块**: `diagnostics.py` → `run_diagnostics()`

Phase 1 只检验"有没有 IC", Phase 2 检验因子本身的**健康度** — 一个 IC 很高但不健康的因子不能用.

### 2.1 分布漂移检测

`compute_distribution_stability()`

每天计算因子的截面均值, 用 63 日滚动平滑, 然后做线性趋势回归:

$$\mu(t) = \text{mean}_i\big(\text{factor}_i(t)\big)$$

$$\text{rolling\_mean}(t) = \alpha + \beta \cdot t + \varepsilon \quad \text{(OLS)}$$

$$p\text{-value} < 0.01 \implies \text{漂移}$$

**为什么检查**: 如果因子截面分布随时间系统性地往一个方向移动 (比如全市场的 P/E 持续上升), 过去的分位分组在未来不再可比 — 去年的 Q5 今年可能变成 Q3.

不直接淘汰, 但标记 WARN. Alpha101 中 16/20 触发了这个警告.

### 2.2 覆盖率

`compute_coverage()`

$$\text{coverage}(t) = \frac{\text{count(非空因子值)}}{\text{总股票数}}$$

中位覆盖率 < 30% → WARN. 低覆盖率意味着因子只对部分股票有定义, 截面比较不公平.

### 2.3 已知风格因子相关性

`compute_known_factor_correlation()`

构建 5 个风格因子代理:

| 风格 | 代理 | 代码 |
|------|------|------|
| size | log(close × shares) | `np.log(close * shares)` |
| value | -close (粗略) | `-close` |
| momentum | 12-1 月动量 | `close.shift(21) / close.shift(252) - 1` |
| quality | ROE (若有) | `roe` |
| low_vol | 负波动率 | `-returns.rolling(60).std()` |

然后对每个截面日期计算因子与每个风格的 Spearman 相关:

$$\text{style\_corr}(t) = \text{SpearmanCorr}\big(\text{factor\_rank}(t),\ \text{style\_rank}(t)\big) \quad \text{跨股票}$$

$$\text{max\_abs\_corr} = \max_s \big| \text{mean}_t(\text{style\_corr}_s(t)) \big|$$

**淘汰线**: `max_abs_corr > 0.7` → 这个因子就是已知因子的"换皮", 不提供独立 alpha. Alpha_100 就是这样被发现是 value 因子马甲的 (corr=0.71).

### 2.4 换手率

`compute_turnover_capacity()`

每天取因子值最高的 top 20% 股票构成多头组:

$$\text{turnover}(t) = \frac{|\text{top\_set}(t)\ \triangle\ \text{top\_set}(t-1)|}{2 \times n_{\text{top}}}$$

对称差集 / (2 × 组大小) 就是单边换手率. 平均换手率 > 80% → WARN.

### 2.5 行业集中度

`compute_industry_distribution()`

每天的 top/bottom 五分位按 GICS 行业统计占比, 用 HHI 衡量集中度:

$$\text{HHI} = \sum_k (\text{行业占比}_k)^2$$

- 均匀分布 11 行业: HHI ≈ 1/11 ≈ 0.09
- HHI > 0.25 → 因子本质是行业押注 → WARN

还计算**行业中性 IC**: 因子值在每个行业内去均值后再算 IC. 如果中性后 IC 显著低于原始 IC, 说明因子 alpha 主要来自行业暴露, 不是个股选择.

### Phase 2 结论

PASS (无告警) 或 WARN (有告警). WARN 不直接淘汰但记录在案, 实盘时需要对应监控.

---

## Phase 3: Alpha Test — "能赚钱吗? 在什么环境下赚?"

**模块**: `alpha_test.py` → `run_alpha_test()`

这是最重的一关, 包含四个子测试.

### 3.1 多周期 IC 完整分布

`compute_multi_horizon_ic()`

Phase 1 只看了均值 IC, Phase 3 看完整分布:

```
对每个 horizon h ∈ {1, 5, 21, 63}:

  IC_series(h)  = {IC(t, h) for all t}
  mean_IC(h)    = mean(IC_series)
  ICIR(h)       = mean(IC_series) / std(IC_series)
  NW_t(h)       = Newey-West t-stat
  IC_q25/q50/q75 = 分位数
  rolling_IC(h) = IC_series.rolling(52 周).mean()
```

**分位数的意义**: mean_IC=0.04 但 q25=-0.02, q75=0.08 — 说明 IC 不稳定, 有 25% 的月份 IC 为负. 纯看均值会被少数极端月份拉高.

### 3.2 增强分位回测

`compute_enhanced_quantile_metrics()`

将股票按因子值分为 5 个分位组, 月度调仓, 做多 Q5 做空 Q1:

```
1. 每月调仓日: 按因子值分为 Q1 (最小) ~ Q5 (最大)
2. 各组等权持有至下次调仓
3. L/S = Q5_return - Q1_return
```

**交易成本扣除**:

$$\text{cost}(t) = \text{turnover\_rate}(t) \times 10\text{bps} \times 2\ \text{(双边)}$$

$$\text{turnover\_rate}(t) = \frac{1}{2}\left(\frac{|Q5\ \text{成分变化}|}{n_{Q5}} + \frac{|Q1\ \text{成分变化}|}{n_{Q1}}\right)$$

$$\text{ls\_net}(t) = \text{ls\_gross}(t) - \text{cost}(t)$$

$$\text{Sharpe\_net} = \frac{\text{mean}(\text{ls\_net}) \times f}{\text{std}(\text{ls\_net}) \times \sqrt{f}} \quad (f = \text{年化因子, 月频}=12)$$

$$\text{MaxDD} = \min_t \left(\frac{\text{cum\_ls}(t)}{\max_{s \leq t} \text{cum\_ls}(s)} - 1\right)$$

**单调性 (Monotonicity)**:

$$\text{mono} = \text{SpearmanCorr}\big([Q1\_\text{ret},\ Q2\_\text{ret},\ Q3\_\text{ret},\ Q4\_\text{ret},\ Q5\_\text{ret}],\ [1,2,3,4,5]\big)$$

mono=+1.0 表示完美单调 (Q5 最高 Q1 最低), mono=0 表示无序. 这比单看 L-S 更严格 — L-S 可能由 Q1 的极端负收益驱动, 但中间组是混乱的.

**实际阈值**: Net Sharpe > 0.5 且 MaxDD < 50% 才值得用.

### 3.3 Fama-MacBeth 截面回归

`fama_macbeth_regression()`

这是因子研究的**金标准** — 控制其他因子后, 看你的因子还有没有边际贡献.

**两步回归**:

**Pass 1 — 截面回归** (每月 t 做一次):

$$r_i(t \to t+21) = \alpha(t) + \beta(t) \cdot \text{factor}_i(t) + \gamma_1(t) \cdot \text{momentum}_i(t) + \gamma_2(t) \cdot \ln\text{mktcap}_i(t) + \sum_k \delta_k(t) \cdot \text{sector\_dummy}_k(i) + \varepsilon_i(t)$$

控制变量:

| 变量 | 定义 | 为什么控制 |
|------|------|-----------|
| momentum | close.shift(21) / close.shift(252) - 1 | 12-1 月动量 |
| log_mktcap | ln(close × shares) | 消除市值效应 |
| sector_dummy | GICS 11 行业 one-hot (drop_first) | 消除行业暴露 |

每月 t 得到一个 β(t) — 因子对收益的截面斜率.

**Pass 2 — 时序检验**:

$$\bar{\beta} = \frac{1}{T} \sum_t \beta(t)$$

$$t_{FM} = \text{NW\_t\_stat}\big(\{\beta(t)\}\big)$$

**解读**: β(t) 的 NW t > 2 意味着: **在控制了市值、动量、行业之后**, 因子仍然显著预测收益. 这排除了因子是 size/momentum/industry bet 的可能.

Alpha_054 的 FM t=7.05 非常强; alpha_100 的 FM t=0.97, 说明它的 IC 全来自 value 暴露, 控制后消失.

### 3.4 子样本分析

`run_sub_sample_analysis()`

三层分解, 检验因子不是只在某个特定环境下有效:

#### (a) 逐年分解

`compute_yearly_stats()`

```
对每年 yr:
  ic_yr       = mean(IC(t) where year(t) == yr)
  icir_yr     = ic_yr / std(IC(t) where year(t) == yr)
  ls_sharpe_yr = 该年的年化 L/S Sharpe

ic_positive_years = count(ic_yr > 0) / total_years
```

6/6 年正 IC = 极稳健 (只有 alpha_033 做到了); 3/6 = IC 在衰减.

#### (b) Regime 分解

`classify_market_regime()` + `compute_sub_sample_stats()`

```
趋势 regime: SPY > MA(200) → bull, else → bear
波动 regime: RVol(60d) > expanding_median → high_vol, else → low_vol

对每个 regime R:
  mean_IC_regime = mean(IC(t) where regime(t) == R)
  L/S_Sharpe_regime = 该 regime 内的年化 Sharpe
```

一个只在 bull 市有效的因子, 真正有价值的部分很少 — 牛市你买什么都涨.

#### (c) Walk-forward

`walk_forward_split()`

```
前 70% 数据 = In-Sample (IS)
后 30% 数据 = Out-of-Sample (OOS)

IS_IC  = mean(IC(t) for t in IS)
OOS_IC = mean(IC(t) for t in OOS)
```

如果 OOS IC 远低于 IS IC, 因子可能是过拟合的 (尤其对参数优化过的因子).

---

## Phase 4: 可视化 — "看图确认"

**模块**: `diagnostics_plotting.py`

每个通过 Phase 3 的因子生成 4 张图:

1. **IC 时序图**: 柱状图 + 52 周滚动均线 — 看 IC 是否稳定或有周期
2. **分位净值曲线**: Q1~Q5 累积收益 — 看是否单调扇形展开
3. **因子分布图**: 截面 histogram + 滚动统计 — 看是否有结构性变化
4. **子样本分解**: 逐年 IC 柱状图 + regime 柱状图 — 看稳定性

人眼比数字更擅长发现异常模式 (比如 IC 突然在某个时期跳崖).

---

## 最终判定: 入库评级

基于 Phase 3 的全部指标, 给出 A/B/C/D 评级:

| 评级 | 条件 | 含义 | registry status |
|------|------|------|-----------------|
| **A** | FM t > 2 AND Net Sharpe > 0.5 AND Yr+ >= 5/6 | 可直接上实盘 | `candidate` |
| **B** | 部分指标达 A 线, 但有明显缺陷 | 搭配使用 | `candidate` |
| **C** | IC 显著但 L-S 或稳定性不达标 | 仅作信号参考 | `tested` |
| **D** | 风格马甲 / 方向错误 / 崩溃回撤 | 排除 | `excluded` |

然后写入 `output/factor_registry.csv`, 重建命令:

```bash
python -m strategies.cross_section.build_factor_registry
```

---

## 代码模块索引

| 文件 | 功能 | Phase |
|------|------|-------|
| `screening.py` | Rank IC 计算, IC filter, 相关性去冗余, 正交化 | 1 |
| `diagnostics.py` | 分布漂移, 覆盖率, 风格相关, 换手率, 行业集中度 | 2 |
| `alpha_test.py` | 多周期 IC, 分位回测, Fama-MacBeth, 子样本分析 | 3 |
| `diagnostics_plotting.py` | IC 时序图, 分位净值, 因子分布, 子样本分解 | 4 |
| `backtest.py` | 前瞻收益, 分位分组, 调仓逻辑 | 3 (依赖) |
| `analytics.py` | 年化因子, Sharpe, 单调性等指标计算 | 3 (依赖) |
| `ranking.py` | 截面 z-score, 分位赋值 | 2-3 (依赖) |

## CLI 快速使用

```bash
cd OneBacktest

# 单因子完整研究 (4 phase 全跑)
python -m strategies.cross_section.run_factor_research --factor RS_12M

# Alpha101 批量管线
python -m strategies.cross_section.run_alpha101                # 全量
python -m strategies.cross_section.run_alpha101 --n-symbols 50 # 快速测试
python -m strategies.cross_section.run_alpha101 --no-plot       # 跳过画图
```
