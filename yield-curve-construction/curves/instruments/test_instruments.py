#!/usr/bin/env python3
"""
测试金融工具类
"""

from datetime import date
from curves.instruments import TreasuryBill, TreasuryNote, TreasuryBond


def test_treasury_bill():
    """测试短期国债"""
    print("=== 测试短期国债（T-Bills）===")
    
    # 创建一个6个月的短期国债
    bill = TreasuryBill("2025-01-01", "2025-07-01", face_value=100.0)
    
    # 测试价格计算
    yield_rate = 0.05  # 5%
    price = bill.price(yield_rate)
    print(f"收益率 {yield_rate*100}% 时的价格: {price:.4f}")
    
    # 测试收益率计算
    calculated_yield = bill.yield_to_price(price)
    print(f"价格 {price:.4f} 对应的收益率: {calculated_yield*100:.2f}%")
    
    # 测试现金流
    dates, amounts = bill.cashflows()
    print(f"现金流: {list(zip(dates, amounts))}")
    print()


def test_treasury_note():
    """测试中期国债"""
    print("=== 测试中期国债（T-Notes）===")
    
    # 创建一个2年期的中期国债，票息率3%
    note = TreasuryNote("2025-01-01", "2027-01-01", coupon_rate=0.03, face_value=100.0, frequency=2)
    
    # 测试价格计算
    yield_rate = 0.04  # 4%
    price = note.price(yield_rate)
    print(f"收益率 {yield_rate*100}% 时的价格: {price:.4f}")
    
    # 测试收益率计算
    calculated_yield = note.yield_to_price(price)
    print(f"价格 {price:.4f} 对应的收益率: {calculated_yield*100:.2f}%")
    
    # 测试现金流
    dates, amounts = note.cashflows()
    print("现金流:")
    for d, a in zip(dates, amounts):
        print(f"  {d}: ${a:.2f}")
    print()


def test_treasury_bond():
    """测试长期国债"""
    print("=== 测试长期国债（T-Bonds）===")
    
    # 创建一个10年期的长期国债，票息率4%
    bond = TreasuryBond("2025-01-01", "2035-01-01", coupon_rate=0.04, face_value=100.0, frequency=2)
    
    # 测试价格计算
    yield_rate = 0.05  # 5%
    price = bond.price(yield_rate)
    print(f"收益率 {yield_rate*100}% 时的价格: {price:.4f}")
    
    # 测试收益率计算
    calculated_yield = bond.yield_to_price(price)
    print(f"价格 {price:.4f} 对应的收益率: {calculated_yield*100:.2f}%")
    
    # 测试现金流
    dates, amounts = bond.cashflows()
    print(f"总付息次数: {len(dates)}")
    print(f"第一次付息: {dates[0]} - ${amounts[0]:.2f}")
    print(f"最后一次付息: {dates[-1]} - ${amounts[-1]:.2f}")
    print()


def test_present_value():
    """测试现值计算"""
    print("=== 测试现值计算 ===")
    
    # 创建一个简单的折现曲线函数
    def simple_discount_curve(t):
        """简单的指数折现曲线"""
        r = 0.05  # 5%的折现率
        return np.exp(-r * t)
    
    # 创建一个中期国债
    note = TreasuryNote("2025-01-01", "2027-01-01", coupon_rate=0.03, face_value=100.0, frequency=2)
    
    # 计算现值
    pv = note.present_value(simple_discount_curve)
    print(f"使用折现曲线计算的现值: {pv:.4f}")
    print()


if __name__ == "__main__":
    import numpy as np
    
    test_treasury_bill()
    test_treasury_note()
    test_treasury_bond()
    test_present_value()
    
    print("所有测试完成！")
