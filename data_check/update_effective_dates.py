#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新有效开始日期的独立脚本
这个脚本只需要在第一次运行时或者需要更新有效开始日期时运行
"""

import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from data_check.data_check import DataChecker

def main():
    """主函数"""
    print("=" * 60)
    print("有效开始日期检测和更新工具")
    print("=" * 60)
    
    # 使用一个较长的日期范围来检测有效开始日期
    start_date = '2015-01-01'  # 从2015年开始检测
    end_date = '2025-12-31'    # 到2025年结束
    
    print(f"检测日期范围: {start_date} 到 {end_date}")
    print("注意：这个过程可能需要较长时间，因为需要运行所有函数...")
    
    try:
        # 创建检查器
        checker = DataChecker(start_date=start_date, end_date=end_date)
        
        # 运行有效开始日期检测
        print("\n开始检测各函数的有效开始日期...")
        effective_dates = checker.run_effective_date_detection_only()
        
        print(f"\n检测完成！共检测了 {len(effective_dates)} 个函数")
        print("配置文件已更新，现在可以正常使用 check_data_prepare_functions() 方法了")
        
        # 显示一些检测结果
        print("\n部分检测结果示例：")
        count = 0
        for func_name, effective_date in effective_dates.items():
            if count < 5:  # 只显示前5个
                print(f"  {func_name}: {effective_date}")
                count += 1
            else:
                print(f"  ... 还有 {len(effective_dates) - 5} 个函数")
                break
        
        print("\n" + "=" * 60)
        print("SUCCESS: 有效开始日期检测和更新完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: 检测过程中出现错误: {str(e)}")
        print("请检查数据源和依赖模块是否正常")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
