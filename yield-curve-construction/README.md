# Yield Curve Construction

美国国债收益率曲线构建工具

## 项目简介

这是一个用于构建美国国债收益率曲线的Python工具包。项目实现了多种插值方法和引导算法，可以处理不同类型的美国国债（短期国债、中期国债、长期国债），并提供债券定价功能。

## 主要功能

### 📊 收益率曲线构建
- **引导算法（Bootstrapping）**：从市场数据中提取零息收益率曲线
- **多种插值方法**：
  - 对数线性插值（Log-Linear）
  - 三次样条插值（Cubic Spline）
  - Nelson-Siegel-Svensson模型

### 📈 金融工具支持
- **短期国债（T-Bills）**：零息债券
- **中期国债（T-Notes）**：付息债券（≤10年）
- **长期国债（T-Bonds）**：付息债券（>10年）

### 💰 定价功能
- 债券现值计算
- Z-Spread计算
- 价格分解分析

## 项目结构

```
yield-curve-construction/
├── curves/                    # 收益率曲线核心模块
│   ├── curve.py              # YieldCurve类
│   ├── bootstrapping/        # 引导算法
│   │   ├── bootstrapper.py   # 引导器实现
│   │   ├── daycount.py       # 天数计算
│   │   └── root_finding.py   # 数值求解
│   ├── instruments/          # 金融工具
│   │   ├── bill.py          # 短期国债
│   │   ├── bond.py          # 长期国债
│   │   ├── note.py          # 中期国债
│   │   ├── cashflow.py      # 现金流处理
│   │   └── factory.py       # 工厂模式
│   └── interpolation/       # 插值方法
│       ├── cubic_spline.py  # 三次样条
│       └── nelson_siegel_svensson.py  # NSS模型
├── pricing/                 # 定价模块
│   ├── bond_pricer.py     # 债券定价器
│   └── z_spread.py        # Z-Spread计算
├── scripts/               # 示例脚本
│   ├── build_curve.py     # 构建收益率曲线
│   ├── price_off_the_run.py  # 为非基准国债定价
│   └── visualize_curve.py    # 可视化曲线
├── data/                  # 数据文件
└── tests/                 # 测试文件
```

## 安装

### 环境要求
- Python 3.8+
- pandas
- numpy
- scipy
- matplotlib (可选，用于可视化)

### 安装依赖
```bash
pip install pandas numpy scipy matplotlib
```

## 快速开始

### 1. 构建收益率曲线

```python
from curves.bootstrapping.bootstrapper import bootstrap_curve_from_dataframe
from curves.instruments.factory import InstrumentFactory

# 从数据文件加载数据
import pandas as pd
df = pd.read_csv('data/treasuries_2018-12-28.parquet')

# 转换为instrument对象
instruments, errors = InstrumentFactory.from_dataframe(df)

# 构建收益率曲线
curve, nodes, interpolator, errors = bootstrap_curve_from_dataframe(
    df, 
    interpolator='loglinear'
)

print(f"曲线估值日: {curve.val_date}")
print(f"节点数量: {len(nodes)}")
```

### 2. 债券定价

```python
from pricing import BondPricer

# 创建定价器
pricer = BondPricer(curve)

# 为债券定价
result = pricer.price(bond_instrument)

print(f"Dirty Price: {result.dirty_price}")
print(f"Clean Price: {result.clean_price}")
print(f"Accrued Interest: {result.accrued_interest}")
```

### 3. 使用示例脚本

```bash
# 构建收益率曲线
python scripts/build_curve.py -i data/treasuries_2018-12-28.parquet -d 2018-12-28

# 为非基准国债定价
python scripts/price_off_the_run.py -i data/treasuries_2018-12-28.parquet -d 2018-12-28

# 可视化收益率曲线
python scripts/visualize_curve.py -i data/treasuries_2018-12-28.parquet -d 2018-12-28
```

## 详细使用

### 创建收益率曲线

```python
from curves.curve import YieldCurve
from curves.interpolation.cubic_spline import CubicSplineInterpolator

# 手动创建节点
nodes = [
    (0.5, 0.98),   # (时间, 折现因子)
    (1.0, 0.95),
    (2.0, 0.90),
    (5.0, 0.80)
]

# 创建曲线
curve = YieldCurve(
    val_date=datetime(2023, 1, 1),
    nodes=nodes
)

# 拟合插值器
interpolator = CubicSplineInterpolator()
curve.fit(interpolator)

# 查询折现因子
df_1y = curve.df(1.0)
df_1_5y = curve.df(1.5)  # 插值计算

# 查询零息收益率
rate_1y = curve.zero_rate_cc(1.0)  # 连续复利
rate_1y_simple = curve.zero_rate_simple(1.0)  # 简单年化
```

### 创建金融工具

```python
from curves.instruments import Bill, Bond, Note

# 创建短期国债
bill = Bill(
    key="T123456",
    cusip="123456789",
    val_date=datetime(2023, 1, 1),
    maturity_date=datetime(2023, 6, 1),
    clean_price=98.5,
    accrued_interest=0.0
)

# 创建长期国债
bond = Bond(
    key="T987654",
    cusip="987654321",
    val_date=datetime(2023, 1, 1),
    dated_date=datetime(2022, 1, 1),
    maturity_date=datetime(2033, 1, 1),
    coupon_rate=0.04,  # 4%
    freq=2,            # 半年付息
    clean_price=102.5,
    accrued_interest=1.2
)

# 获取现金流
cashflows = bond.cashflows()
for cf in cashflows:
    print(f"日期: {cf.pay_date}, 金额: {cf.amount}")
```

### 高级定价功能

```python
from pricing.z_spread import solve_z_spread, price_with_z_spread

# 计算Z-Spread
z_spread = solve_z_spread(
    instrument=bond,
    curve=curve,
    target_dirty_price=bond.dirty_price
)

print(f"Z-Spread: {z_spread * 10000:.2f} bps")

# 使用Z-Spread定价
price_with_spread = price_with_z_spread(
    instrument=bond,
    curve=curve,
    spread=z_spread
)
```

## 数据格式

项目支持以下数据格式：

### CRSP格式
- `KYTREASNO`: 国债编号
- `TCUSIP`: CUSIP代码
- `CALDT`: 交易日期
- `TMATDT`: 到期日期
- `TDATDT`: 发行日期
- `TDNOMPRC`: 清算价格
- `TDACCINT`: 应计利息
- `ITYPE`: 工具类型（1=债券，2=票据，4=短期国债）
- `TNIPPY`: 付息频率
- `TCOUPRT`: 票息率

### Parquet文件
支持直接读取Parquet格式的CRSP数据文件。

## 测试

运行测试：

```bash
cd yield-curve-construction
python tests/test_pricing.py
```

## 可视化

使用`scripts/visualize_curve.py`脚本可以生成收益率曲线图表：

```bash
python scripts/visualize_curve.py -i data/treasuries_2018-12-28.parquet -d 2018-12-28 --output curve_plot.png
```

生成的图表包括：
- 零息收益率曲线
- 即期利率曲线
- 远期利率曲线
- 原始数据点

## 算法说明

### 引导算法（Bootstrapping）
1. 按期限对国债进行排序
2. 从最短期限开始，逐步求解每个节点的折现因子
3. 使用数值方法（如二分法）求解隐含收益率

### 插值方法
- **对数线性插值**：在ln(df)上进行线性插值
- **三次样条插值**：保证一阶和二阶导数连续
- **Nelson-Siegel-Svensson**：参数化模型，适合拟合整体曲线形状

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用MIT许可证。

## 参考资料

- [Investopedia: Yield Curve](https://www.investopedia.com/terms/y/yieldcurve.asp)
- [Federal Reserve: Treasury Yield Curve](https://www.federalreserve.gov/releases/h15/)
- [Nelson-Siegel-Svensson Model](https://www.ssb.se/en/publications/2013/23/the-estimation-of-the-nelson-and-siegel-model)
