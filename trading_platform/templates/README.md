# trading_platform/templates

可直接复制的策略骨架。每个文件都是**完全可运行**的（含 `if __name__ == "__main__"` 的端到端回测脚本）。

| 模板 | 何时用 |
|---|---|
| `timing_template.py` | 写新的单票择时规则（HHT/QRS/MA/RSI 之外） |
| `cross_section_template.py` | 加新的截面 alpha 因子 |
| `pairs_template.py` | 自定义 pairs 信号（替换默认 z-score 反转） |
| `rotation_template.py` | 自定义板块/风格轮动规则 |
| `multi_strategy_composite.py` | 把多个策略组合成一个 LayeredCombiner |

## 通用流程

1. **复制** — `cp templates/timing_template.py strategies_user/my_strategy.py`
2. **改类名** — `MyTimingAlpha` → `MyMomentumAlpha`
3. **改业务逻辑** — 编辑 `_compute_raw_for_symbol` / `_compute_my_factor` / `forecast`
4. **直接跑** — `python strategies_user/my_strategy.py`

每个模板都附带最小可运行的 `if __name__ == "__main__"` 块，可以立刻看回测结果（含 Carver SR + CI + 偏度报告）。

## forecast 协议提醒

无论改哪类策略，最终输出必须满足：

| 项 | 要求 |
|---|---|
| 取值 | `[-20, +20]` 或 `NaN` |
| 期望幅度 | `E[\|forecast\|] ≈ 10` |
| 量纲 | 无单位 |
| 与 vol/价格/资金 | **完全无关** |

`ScalingMixin._scale_and_cap()` 自动帮你处理 scaling 和 cap，你只要返回**任意尺度**的 raw 值即可。
