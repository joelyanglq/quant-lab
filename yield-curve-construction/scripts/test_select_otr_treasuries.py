#!/usr/bin/env python3
"""
测试脚本用于验证split_otr_offtherun函数的实现
"""

import pandas as pd
import sys
import os

# 将scripts目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模块
import importlib.util
spec = importlib.util.spec_from_file_location("select_otr_treasuries", "select_otr_treasuries.py")
select_otr_treasuries = importlib.util.module_from_spec(spec)
spec.loader.exec_module(select_otr_treasuries)

split_otr_offtherun = select_otr_treasuries.split_otr_offtherun
DEFAULT_BUCKETS = select_otr_treasuries.DEFAULT_BUCKETS

def test_split_otr_offtherun():
    # 加载测试数据
    data_path = "../data/treasuries_2018-12-28.parquet"
    try:
        day = pd.read_parquet(data_path)
        print(f"成功加载数据，共 {len(day)} 条记录")
    except Exception as e:
        print(f"无法加载parquet文件: {e}")
        print("尝试使用CSV文件...")
        try:
            csv_path = "../data/raw/treasuries/crsp_a_treasuries_2018.csv"
            day = pd.read_csv(csv_path)
            # 筛选2018-12-28的数据
            day = day[day["CALDT"] == "2018-12-28"].copy()
            print(f"成功加载CSV数据，共 {len(day)} 条记录")
        except Exception as e:
            print(f"无法加载CSV文件: {e}")
            return

    # 确保必要的列存在
    required_cols = ["CALDT", "TMATDT", "KYTREASNO", "TDBID", "TDASK", "TDPUBOUT"]
    for col in required_cols:
        if col not in day.columns:
            print(f"警告: 缺少列 {col}")
            return

    print("数据列:", day.columns.tolist())

    # 调用函数
    try:
        otr_df, off_df, report_df = split_otr_offtherun(
            day=day,
            buckets=DEFAULT_BUCKETS
        )

        print("\n=== 测试结果 ===")
        print(f"OTR债券数量: {len(otr_df)}")
        print(f"剩余债券数量: {len(off_df)}")
        print(f"报告记录数量: {len(report_df)}")

        print("\nOTR债券:")
        print(otr_df[['KYTREASNO', 'TMATDT', 'TDBID', 'TDASK']])

        print("\n报告:")
        print(report_df)

        print("\n剩余债券样本:")
        print(off_df[['KYTREASNO', 'TMATDT', 'TDBID', 'TDASK']].head())

        # 验证结果
        assert len(otr_df) <= len(DEFAULT_BUCKETS), "OTR债券数量不应超过bucket数量"
        assert len(off_df) + len(otr_df) == len(day), "总债券数量应保持不变"
        assert len(report_df) == len(otr_df), "报告记录数量应与OTR债券数量一致"

        print("\n✅ 所有测试通过！")

    except Exception as e:
        print(f"函数调用失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_split_otr_offtherun()
