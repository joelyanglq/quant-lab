import pandas as pd
import argparse
import sys
import os

DEFAULT_BUCKETS = {
    "3M":  (0.12, 0.35),
    "6M":  (0.35, 0.75),
    "1Y":  (0.75, 1.25),
    "2Y":  (1.60, 2.40),
    "5Y":  (4.25, 5.75),
    "10Y": (9.00, 11.00),
    "30Y": (25.0, 35.0),
}


def split_otr_offtherun(
    day: pd.DataFrame,
    buckets: dict[str, tuple[float, float]],
    *,
    date_col: str = "CALDT",
    maturity_col: str = "TMATDT",
    dated_col: str = "TDATDT",
    id_col: str = "KYTREASNO",
    bid_col: str = "TDBID",
    ask_col: str = "TDASK",
    out_col: str | None = "TDPUBOUT",
    use_liquidity_tiebreak: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        otr_df: one selected security per bucket (or empty for missing bucket)
        off_df: remaining securities
        report_df: per-bucket diagnostics (selected id, ttm, spread, etc.)
    """
    # 创建副本以避免修改原始数据
    day = day.copy()

    # 计算到期时间（TTM - Time to Maturity）以年为单位
    day['TTM'] = (pd.to_datetime(day[maturity_col]) - pd.to_datetime(day[date_col])).dt.days / 365.25

    # 初始化结果DataFrame
    otr_list = []
    report_list = []

    # 为每个bucket选择最合适的债券
    for bucket_name, (min_ttm, max_ttm) in buckets.items():
        # 筛选出在当前bucket范围内的债券
        bucket_mask = (day['TTM'] >= min_ttm) & (day['TTM'] <= max_ttm)
        bucket_seurities = day[bucket_mask].copy()

        if len(bucket_seurities) == 0:
            # 如果bucket中没有债券，跳过
            continue

        # 计算价差（ask - bid）
        bucket_seurities['spread'] = bucket_seurities[ask_col] - bucket_seurities[bid_col]

        # 选择策略：多级排序
        # 1. dated date 最新（TDATDT 最大）
        # 2. bid-ask 最窄（spread 最小）
        # 3. outstanding 最大（TDPUBOUT 最大）
        # 4. KYTREASNO 最小（保证确定性）
        
        # 准备排序列
        sort_columns = []
        sort_ascending = []
        
        # 添加dated date（需要转换为日期类型）
        if dated_col in bucket_seurities.columns:
            bucket_seurities[dated_col] = pd.to_datetime(bucket_seurities[dated_col], errors='coerce')
            sort_columns.append(dated_col)
            sort_ascending.append(False)  # 最新日期（降序）
        
        # 添加spread（价差）
        sort_columns.append('spread')
        sort_ascending.append(True)  # 最小价差（升序）
        
        # 添加outstanding（流动性）
        if out_col and out_col in bucket_seurities.columns:
            sort_columns.append(out_col)
            sort_ascending.append(False)  # 最大outstanding（降序）
        
        # 添加ID（保证确定性）
        sort_columns.append(id_col)
        sort_ascending.append(True)  # 最小ID（升序）
        
        # 执行排序并选择第一个
        sorted_df = bucket_seurities.sort_values(
            by=sort_columns,
            ascending=sort_ascending,
            na_position='last'  # NaN值放在最后
        )
        
        selected_security = sorted_df.iloc[0]

        # 将选择的债券添加到otr列表
        otr_list.append(selected_security)

        # 创建报告记录
        report_record = {
            'bucket': bucket_name,
            'selected_id': selected_security[id_col],
            'ttm': selected_security['TTM'],
            'spread': selected_security['spread'],
            'bid': selected_security[bid_col],
            'ask': selected_security[ask_col],
            'liquidity': selected_security.get(out_col, None) if out_col else None
        }
        report_list.append(report_record)

    # 创建otr_df（选择的债券）
    if otr_list:
        otr_df = pd.DataFrame(otr_list)
        # 移除TTM和spread列（如果需要保留可以注释掉）
        otr_df = otr_df.drop(columns=['TTM', 'spread'], errors='ignore')
    else:
        otr_df = pd.DataFrame(columns=day.columns)

    # 创建off_df（剩余的债券）
    if otr_list:
        # 获取选择的债券的ID
        selected_ids = otr_df[id_col].tolist()
        # 筛选出未被选择的债券
        off_df = day[~day[id_col].isin(selected_ids)].copy()
    else:
        off_df = day.copy()

    # 创建report_df（诊断报告）
    report_df = pd.DataFrame(report_list)

    return otr_df, off_df, report_df


def load_data(file_path: str, date_filter: str = None) -> pd.DataFrame:
    """
    加载数据文件（支持CSV和Parquet格式）
    
    Args:
        file_path: 文件路径
        date_filter: 可选的日期过滤器，格式为YYYY-MM-DD
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
    elif file_ext == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")
    
    if date_filter:
        df = df[df['CALDT'] == date_filter].copy()
        if len(df) == 0:
            raise ValueError(f"在文件 {file_path} 中未找到日期 {date_filter} 的数据")
    
    return df


def save_results(otr_df: pd.DataFrame, off_df: pd.DataFrame, report_df: pd.DataFrame, 
                 output_prefix: str):
    """
    保存结果到文件
    
    Args:
        otr_df: OTR债券数据
        off_df: 剩余债券数据
        report_df: 报告数据
        output_prefix: 输出文件前缀
    """
    # 保存OTR债券
    otr_path = f"{output_prefix}_otr.csv"
    otr_df.to_csv(otr_path, index=False)
    print(f"OTR债券已保存到: {otr_path}")
    
    # 保存剩余债券
    off_path = f"{output_prefix}_off.csv"
    off_df.to_csv(off_path, index=False)
    print(f"剩余债券已保存到: {off_path}")
    
    # 保存报告
    report_path = f"{output_prefix}_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"报告已保存到: {report_path}")


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(
        description='选择OTR（On-the-Run）债券并生成报告',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python select_otr_treasuries.py -i data.csv -o output
  python select_otr_treasuries.py -i data.parquet -o output -d 2018-12-28
  python select_otr_treasuries.py -i data.csv -o output --no-liquidity
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='输入文件路径（CSV或Parquet格式）')
    parser.add_argument('-o', '--output', required=True,
                       help='输出文件前缀')
    parser.add_argument('-d', '--date', 
                       help='过滤特定日期的数据（格式: YYYY-MM-DD）')
    parser.add_argument('--no-liquidity', action='store_true',
                       help='禁用基于流动性的选择（仅使用价差作为选择标准）')
    parser.add_argument('--date-col', default='CALDT',
                       help='日期列名（默认: CALDT）')
    parser.add_argument('--maturity-col', default='TMATDT',
                       help='到期日期列名（默认: TMATDT）')
    parser.add_argument('--id-col', default='KYTREASNO',
                       help='债券ID列名（默认: KYTREASNO）')
    parser.add_argument('--bid-col', default='TDBID',
                       help='买价列名（默认: TDBID）')
    parser.add_argument('--ask-col', default='TDASK',
                       help='卖价列名（默认: TDASK）')
    parser.add_argument('--out-col', default='TDPUBOUT',
                       help='流动性列名（默认: TDPUBOUT）')
    
    args = parser.parse_args()
    
    try:
        print("正在加载数据...")
        df = load_data(args.input, args.date)
        print(f"成功加载 {len(df)} 条记录")
        
        print("正在选择OTR债券...")
        otr_df, off_df, report_df = split_otr_offtherun(
            day=df,
            buckets=DEFAULT_BUCKETS,
            date_col=args.date_col,
            maturity_col=args.maturity_col,
            id_col=args.id_col,
            bid_col=args.bid_col,
            ask_col=args.ask_col,
            out_col=args.out_col,
            use_liquidity_tiebreak=not args.no_liquidity
        )
        
        print(f"选择结果:")
        print(f"  - OTR债券数量: {len(otr_df)}")
        print(f"  - 剩余债券数量: {len(off_df)}")
        print(f"  - 报告记录数量: {len(report_df)}")
        
        if len(report_df) > 0:
            print("\n选择的债券:")
            print(report_df.to_string(index=False))
        
        print("\n正在保存结果...")
        save_results(otr_df, off_df, report_df, args.output)
        
        print("完成!")
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
