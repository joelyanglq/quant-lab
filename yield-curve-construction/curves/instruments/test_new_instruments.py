#!/usr/bin/env python3
"""
测试新的金融工具类实现
"""

from datetime import datetime
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from curves.instruments import Bill, Bond, Note, InstrumentFactory


def test_bill():
    """测试Bill类"""
    print("=== 测试Bill类 ===")
    
    # 创建一个Bill
    bill = Bill(
        key="206728",
        cusip="912796ZP0",
        val_date=datetime(2018, 12, 28),
        maturity_date=datetime(2019, 4, 15),
        clean_price=98.597656,
        accrued_interest=0.0
    )
    
    print(f"Bill: {bill}")
    print(f"Dirty Price: {bill.dirty_price}")
    
    # 测试现金流
    cfs = bill.cashflows()
    print(f"现金流: {cfs}")
    print()


def test_bond():
    """测试Bond类"""
    print("=== 测试Bond类 ===")
    
    # 创建一个Bond
    bond = Bond(
        key="207215",
        cusip="912828FV4",
        val_date=datetime(2018, 12, 28),
        dated_date=datetime(2018, 5, 15),
        maturity_date=datetime(2028, 5, 15),
        coupon_rate=0.025,  # 2.5%
        freq=2,  # 半年付息
        clean_price=101.257812,
        accrued_interest=0.5
    )
    
    print(f"Bond: {bond}")
    print(f"Dirty Price: {bond.dirty_price}")
    
    # 测试现金流
    cfs = bond.cashflows()
    print(f"现金流数量: {len(cfs)}")
    print("前5个现金流:")
    for i, cf in enumerate(cfs[:5]):
        print(f"  {cf.pay_date}: ${cf.amount:.2f}")
    print()


def test_note():
    """测试Note类"""
    print("=== 测试Note类 ===")
    
    # 创建一个Note
    note = Note(
        key="206195",
        cusip="912828EV6",
        val_date=datetime(2018, 12, 28),
        dated_date=datetime(2018, 11, 15),
        maturity_date=datetime(2020, 11, 15),
        coupon_rate=0.03,
        freq=2,
        clean_price=100.140625,
        accrued_interest=0.2
    )
    
    print(f"Note: {note}")
    print(f"Dirty Price: {note.dirty_price}")
    
    # 测试现金流
    cfs = note.cashflows()
    print(f"现金流数量: {len(cfs)}")
    print("现金流:")
    for cf in cfs:
        print(f"  {cf.pay_date}: ${cf.amount:.2f}")
    print()


def test_factory():
    """测试InstrumentFactory"""
    print("=== 测试InstrumentFactory ===")
    
    # 模拟CRSP数据行
    crsp_row_bill = {
        "KYTREASNO": "206728",
        "TCUSIP": "912796ZP0",
        "CALDT": "2018-12-28",
        "TMATDT": "2019-04-15",
        "TDNOMPRC": 98.597656,
        "TDACCINT": 0.0,
        "ITYPE": 4,
        "TNIPPY": 0,
        "TCOUPRT": 0.0
    }
    
    crsp_row_bond = {
        "KYTREASNO": "207215",
        "TCUSIP": "912828FV4",
        "CALDT": "2018-12-28",
        "TMATDT": "2028-05-15",
        "TDATDT": "2018-05-15",
        "TDNOMPRC": 101.257812,
        "TDACCINT": 0.5,
        "ITYPE": 1,
        "TNIPPY": 2,
        "TCOUPRT": 2.5
    }
    
    crsp_row_note = {
        "KYTREASNO": "206195",
        "TCUSIP": "912828EV6",
        "CALDT": "2018-12-28",
        "TMATDT": "2020-11-15",
        "TDATDT": "2018-11-15",
        "TDNOMPRC": 100.140625,
        "TDACCINT": 0.2,
        "ITYPE": 2,
        "TNIPPY": 2,
        "TCOUPRT": 3.0
    }
    
    # 使用factory创建instrument
    bill = InstrumentFactory.from_crsp_row(crsp_row_bill)
    bond = InstrumentFactory.from_crsp_row(crsp_row_bond)
    note = InstrumentFactory.from_crsp_row(crsp_row_note)
    
    print(f"Bill类型: {type(bill).__name__}")
    print(f"Bond类型: {type(bond).__name__}")
    print(f"Note类型: {type(note).__name__}")
    
    print(f"Bill dirty_price: {bill.dirty_price}")
    print(f"Bond dirty_price: {bond.dirty_price}")
    print(f"Note dirty_price: {note.dirty_price}")
    print()


def test_filter_cashflows():
    """测试现金流过滤"""
    print("=== 测试现金流过滤 ===")
    
    from curves.instruments import filter_cashflows_after
    
    # 创建一些现金流
    cfs = [
        Cashflow(datetime(2024, 1, 1), 10.0),
        Cashflow(datetime(2024, 6, 1), 10.0),
        Cashflow(datetime(2025, 1, 1), 110.0)
    ]
    
    val_date = datetime(2024, 3, 1)
    filtered = filter_cashflows_after(val_date, cfs)
    
    print(f"估值日: {val_date}")
    print(f"原始现金流数量: {len(cfs)}")
    print(f"过滤后现金流数量: {len(filtered)}")
    for cf in filtered:
        print(f"  {cf.pay_date}: ${cf.amount:.2f}")
    print()


if __name__ == "__main__":
    from curves.instruments.cashflow import Cashflow
    
    test_bill()
    test_bond()
    test_note()
    test_factory()
    test_filter_cashflows()
    
    print("所有测试完成！")
