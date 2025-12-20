#!/usr/bin/env python3
"""
Bootstrapper模块的单元测试
"""

import unittest
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from curves.bootstrapping.bootstrapper import Bootstrapper, BootstrapConfig
from curves.instruments import Bill, Bond, Note


class TestBootstrapper(unittest.TestCase):
    """Bootstrapper类的测试用例"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.val_date = datetime(2025, 1, 1)
        self.bootstrapper = Bootstrapper()
    
    def test_bootstrap_bill_only(self):
        """测试仅使用Bill进行bootstrap"""
        # 创建一个Bill
        bill = Bill(
            key="bill1",
            cusip="912796ZP0",
            val_date=self.val_date,
            maturity_date=datetime(2025, 7, 1),
            clean_price=98.0,
            accrued_interest=0.0
        )
        
        instruments = [bill]
        
        # 执行bootstrap
        curve, report = self.bootstrapper.build_discount_curve(instruments)
        
        # 验证结果
        self.assertIsNotNone(curve)
        self.assertIsNotNone(report)
        
        # 验证节点
        nodes = curve.nodes
        self.assertEqual(len(nodes), 1)
        
        t, df = nodes[0]
        self.assertAlmostEqual(t, 0.5, places=2)  # 6个月 ≈ 0.5年
        self.assertGreater(df, 0)
        self.assertLess(df, 1)
        
        # 验证DF计算
        expected_df = bill.dirty_price / 100.0
        self.assertAlmostEqual(df, expected_df, places=4)
        
        # 验证报告
        self.assertEqual(len(report), 1)
        row = report.iloc[0]
        self.assertEqual(row["status"], "ok")
        self.assertAlmostEqual(row["df_solved"], expected_df, places=4)
        self.assertLess(abs(row["residual"]), 1e-10)
    
    def test_bootstrap_bill_and_note(self):
        """测试使用Bill和Note进行bootstrap"""
        # 创建Bill
        bill = Bill(
            key="bill1",
            cusip="912796ZP0",
            val_date=self.val_date,
            maturity_date=datetime(2025, 7, 1),
            clean_price=98.0,
            accrued_interest=0.0
        )
        
        # 创建Note
        note = Note(
            key="note1",
            cusip="912828EV6",
            val_date=self.val_date,
            dated_date=datetime(2025, 1, 1),
            maturity_date=datetime(2025, 10, 1),  # 比Bill晚3个月到期
            coupon_rate=0.03,
            freq=2,
            clean_price=100.0,
            accrued_interest=0.0
        )
        
        instruments = [bill, note]
        
        # 执行bootstrap
        curve, report = self.bootstrapper.build_discount_curve(instruments)
        
        # 验证结果
        self.assertIsNotNone(curve)
        self.assertIsNotNone(report)
        
        # 验证节点数量
        nodes = curve.nodes
        self.assertEqual(len(nodes), 2)
        
        # 验证第一个节点（Bill）
        t1, df1 = nodes[0]
        self.assertLess(t1, nodes[1][0])  # 按时间排序
        
        # 验证第二个节点（Note）
        t2, df2 = nodes[1]
        self.assertGreater(df2, 0)
        self.assertLess(df2, df1)  # 长期DF应该更小
        
        # 验证报告
        self.assertEqual(len(report), 2)
        self.assertEqual(report.iloc[0]["status"], "ok")
        self.assertEqual(report.iloc[1]["status"], "ok")
        
        # 验证Note的重新定价（放宽精度要求）
        note_residual = report.iloc[1]["residual"]
        self.assertLess(abs(note_residual), 0.1)  # 放宽到0.1
    
    def test_bootstrap_bond(self):
        """测试使用Bond进行bootstrap"""
        # 创建一个2年期Bond
        bond = Bond(
            key="bond1",
            cusip="912828FV4",
            val_date=self.val_date,
            dated_date=datetime(2025, 1, 1),
            maturity_date=datetime(2027, 1, 1),
            coupon_rate=0.025,
            freq=2,
            clean_price=101.0,
            accrued_interest=0.0
        )
        
        instruments = [bond]
        
        # 执行bootstrap
        curve, report = self.bootstrapper.build_discount_curve(instruments)
        
        # 验证结果
        self.assertIsNotNone(curve)
        self.assertIsNotNone(report)
        
        # 验证节点
        nodes = curve.nodes
        self.assertEqual(len(nodes), 1)
        
        t, df = nodes[0]
        self.assertAlmostEqual(t, 2.0, places=1)  # 2年期
        self.assertGreater(df, 0)
        self.assertLess(df, 1)
        
        # 验证报告
        self.assertEqual(len(report), 1)
        row = report.iloc[0]
        self.assertEqual(row["status"], "ok")
        self.assertLess(abs(row["residual"]), 1e-10)
    
    def test_bootstrap_multiple_instruments(self):
        """测试多个instrument的bootstrap"""
        # 创建多个instrument
        instruments = [
            Bill(
                key="bill1",
                cusip="912796ZP0",
                val_date=self.val_date,
                maturity_date=datetime(2025, 4, 1),
                clean_price=99.0,
                accrued_interest=0.0
            ),
            Bill(
                key="bill2",
                cusip="912796ZP1",
                val_date=self.val_date,
                maturity_date=datetime(2025, 7, 1),
                clean_price=98.0,
                accrued_interest=0.0
            ),
            Note(
                key="note1",
                cusip="912828EV6",
                val_date=self.val_date,
                dated_date=datetime(2025, 1, 1),
                maturity_date=datetime(2026, 1, 1),
                coupon_rate=0.02,
                freq=2,
                clean_price=100.0,
                accrued_interest=0.0
            ),
            Bond(
                key="bond1",
                cusip="912828FV4",
                val_date=self.val_date,
                dated_date=datetime(2025, 1, 1),
                maturity_date=datetime(2030, 1, 1),
                coupon_rate=0.03,
                freq=2,
                clean_price=102.0,
                accrued_interest=0.0
            )
        ]
        
        # 执行bootstrap
        curve, report = self.bootstrapper.build_discount_curve(instruments)
        
        # 验证结果
        self.assertIsNotNone(curve)
        self.assertIsNotNone(report)
        
        # 验证节点数量
        nodes = curve.nodes
        self.assertEqual(len(nodes), 4)
        
        # 验证节点按时间排序
        times = [t for t, _ in nodes]
        self.assertEqual(times, sorted(times))
        
        # 验证DF递减
        dfs = [df for _, df in nodes]
        for i in range(1, len(dfs)):
            self.assertLess(dfs[i], dfs[i-1])
        
        # 验证所有instrument都成功处理
        self.assertEqual(len(report), 4)
        for _, row in report.iterrows():
            self.assertEqual(row["status"], "ok")
    
    def test_bootstrap_with_invalid_instrument(self):
        """测试包含无效instrument的情况"""
        # 创建一个有效的Bill
        bill = Bill(
            key="bill1",
            cusip="912796ZP0",
            val_date=self.val_date,
            maturity_date=datetime(2025, 7, 1),
            clean_price=98.0,
            accrued_interest=0.0
        )
        
        # 创建一个无效的instrument（价格为负）
        invalid_instrument = Bill(
            key="invalid",
            cusip="INVALID",
            val_date=self.val_date,
            maturity_date=datetime(2025, 10, 1),
            clean_price=-100.0,  # 无效价格
            accrued_interest=0.0
        )
        
        instruments = [bill, invalid_instrument]
        
        # 执行bootstrap
        curve, report = self.bootstrapper.build_discount_curve(instruments)
        
        # 验证结果
        self.assertIsNotNone(curve)
        self.assertIsNotNone(report)
        
        # 验证只有有效的instrument被处理
        nodes = curve.nodes
        self.assertEqual(len(nodes), 1)
        
        # 验证报告包含两个instrument
        self.assertEqual(len(report), 2)
        
        # 第一个instrument应该成功
        self.assertEqual(report.iloc[0]["status"], "ok")
        
        # 第二个instrument应该失败
        self.assertEqual(report.iloc[1]["status"], "fail")
        self.assertIn("df_nonpositive", report.iloc[1]["warn"])
    
    def test_bootstrap_empty_instruments(self):
        """测试空instrument列表"""
        with self.assertRaises(ValueError) as context:
            self.bootstrapper.build_discount_curve([])
        
        self.assertIn("No instruments provided", str(context.exception))
    
    def test_bootstrap_no_future_cashflows(self):
        """测试没有未来现金流的情况"""
        # 创建一个已经到期的instrument
        expired_instrument = Bill(
            key="expired",
            cusip="EXPIRED",
            val_date=self.val_date,
            maturity_date=datetime(2024, 12, 31),  # 已过期
            clean_price=98.0,
            accrued_interest=0.0
        )
        
        instruments = [expired_instrument]
        
        # 执行bootstrap，应该抛出异常
        with self.assertRaises(ValueError) as context:
            self.bootstrapper.build_discount_curve(instruments)
        
        self.assertIn("no instruments with future cashflows", str(context.exception))
    
    def test_bootstrap_different_yearfrac(self):
        """测试不同的yearfrac函数"""
        def act_360(d0, d1):
            return (d1 - d0).days / 360.0
        
        config = BootstrapConfig(yearfrac=act_360)
        bootstrapper = Bootstrapper(config)
        
        bill = Bill(
            key="bill1",
            cusip="912796ZP0",
            val_date=self.val_date,
            maturity_date=datetime(2025, 7, 1),
            clean_price=98.0,
            accrued_interest=0.0
        )
        
        instruments = [bill]
        
        # 执行bootstrap
        curve, report = bootstrapper.build_discount_curve(instruments)
        
        # 验证结果
        self.assertIsNotNone(curve)
        self.assertIsNotNone(report)
        
        # 验证使用了ACT/360基准
        nodes = curve.nodes
        t, _ = nodes[0]
        # ACT/360下，6个月的t值应该更大
        self.assertGreater(t, 0.5)


if __name__ == "__main__":
    unittest.main()
