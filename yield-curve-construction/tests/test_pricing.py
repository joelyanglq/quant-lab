#!/usr/bin/env python3
"""
测试pricing模块

验证BondPricer类的正确性
"""

from datetime import datetime
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from curves.curve import YieldCurve
from curves.instruments import Bill, Bond
from pricing import BondPricer
from curves.bootstrapping.daycount import yearfrac


def test_bill_pricing():
    """测试Bill定价"""
    print("=== 测试Bill定价 ===")
    
    # 创建一个简单的yield curve
    nodes = [
        (0.1, 0.98),
        (0.5, 0.95),
        (1.0, 0.90),
        (2.0, 0.80),
    ]
    curve = YieldCurve(val_date=datetime(2023, 1, 1), nodes=nodes)
    
    # 创建一个Bill
    bill = Bill(
        key="test_bill",
        cusip="123456789",
        val_date=datetime(2023, 1, 1),
        maturity_date=datetime(2023, 4, 1),
        clean_price=98.0,
        accrued_interest=0.0
    )
    
    pricer = BondPricer(curve)
    result = pricer.price(bill)
    
    # 手动计算验证
    manual_dirty = sum(cf.amount * curve.df(yearfrac(bill.val_date, cf.pay_date)) 
                      for cf in bill.cashflows())
    
    print(f"Bill dirty price: {result.dirty_price}")
    print(f"手动计算dirty price: {manual_dirty}")
    
    # 验证
    assert abs(result.dirty_price - manual_dirty) < 1e-10, "Bill dirty price计算错误"
    assert abs(result.clean_price - result.dirty_price) < 1e-10, "Bill clean price计算错误"
    
    print("✅ Bill定价测试通过")
    return True


def test_bond_pricing():
    """测试Bond定价"""
    print("\n=== 测试Bond定价 ===")
    
    # 创建一个简单的yield curve
    nodes = [
        (0.1, 0.98),
        (0.5, 0.95),
        (1.0, 0.90),
        (2.0, 0.80),
    ]
    curve = YieldCurve(val_date=datetime(2023, 1, 1), nodes=nodes)
    
    # 创建一个Bond
    bond = Bond(
        key="test_bond",
        cusip="987654321",
        val_date=datetime(2023, 1, 1),
        dated_date=datetime(2022, 1, 1),
        maturity_date=datetime(2025, 1, 1),
        coupon_rate=0.05,
        freq=2,
        clean_price=100.0,
        accrued_interest=1.0
    )
    
    pricer = BondPricer(curve)
    result = pricer.price(bond)
    
    # 手动计算验证
    manual_dirty = sum(cf.amount * curve.df(yearfrac(bond.val_date, cf.pay_date)) 
                      for cf in bond.cashflows())
    
    print(f"Bond dirty price: {result.dirty_price}")
    print(f"手动计算dirty price: {manual_dirty}")
    print(f"Bond clean price: {result.clean_price}")
    print(f"手动计算clean price: {manual_dirty - bond.accrued_interest}")
    
    # 验证
    assert abs(result.dirty_price - manual_dirty) < 1e-10, "Bond dirty price计算错误"
    assert abs(result.clean_price - (result.dirty_price - bond.accrued_interest)) < 1e-10, "Bond clean price计算错误"
    
    print("✅ Bond定价测试通过")
    return True


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    # 创建一个简单的yield curve
    nodes = [
        (0.1, 0.98),
        (0.5, 0.95),
        (1.0, 0.90),
        (2.0, 0.80),
    ]
    curve = YieldCurve(val_date=datetime(2023, 1, 1), nodes=nodes)
    
    pricer = BondPricer(curve)
    
    # 测试t=0的现金流（估值日等于现金流日期）
    edge_bill = Bill(
        key="edge_bill",
        cusip="111111111",
        val_date=datetime(2023, 4, 1),
        maturity_date=datetime(2023, 4, 1),
        clean_price=99.0,
        accrued_interest=0.0
    )
    
    print(f"边界Bill现金流: {edge_bill.cashflows()}")
    
    # 应该抛出异常，因为没有有效现金流
    try:
        edge_result = pricer.price(edge_bill)
        assert False, "应该抛出ValueError异常"
    except ValueError as e:
        print(f"✅ 正确处理了无现金流情况: {e}")
    
    # 测试未来现金流
    future_bill = Bill(
        key="future_bill",
        cusip="222222222",
        val_date=datetime(2023, 1, 1),
        maturity_date=datetime(2023, 6, 1),
        clean_price=97.0,
        accrued_interest=0.0
    )
    
    future_result = pricer.price(future_bill)
    print(f"未来Bill dirty price: {future_result.dirty_price}")
    assert future_result.dirty_price > 0, "未来现金流应该有正的现值"
    
    print("✅ 边界情况测试通过")
    return True


def main():
    """运行所有测试"""
    print("开始测试pricing模块...")
    
    try:
        test_bill_pricing()
        test_bond_pricing()
        test_edge_cases()
        
        print("\n🎉 所有测试通过！pricing模块实现正确。")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
