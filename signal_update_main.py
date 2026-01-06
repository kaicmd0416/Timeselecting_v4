"""
择时信号生成系统 - 日常更新主程序

本模块是系统的日常更新入口，负责：
1. 自动确定目标日期
2. 检查数据完整性
3. 生成各层级信号
4. 构建投资组合
5. 执行回测分析

作者: TimeSelecting Team
版本: v3.0
"""
import global_setting.global_dic as glv
import pandas as pd
import os
import sys
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
from datetime import date
import datetime
from data_check.data_check import SignalChecker,PortfolioChecker
from running_main.signal_construct_main import signal_constructing_main
from portfolio.portfolio_construction import portfolio_updating
import os
from backtesting.signal_backtesting import signal_backtesting_main

def target_date_decision():
    """
    自动确定目标日期（target_date）
    
    根据当前时间和工作日情况，确定应该处理的目标日期：
    - 如果是工作日且当前时间 >= 20:00，返回下一个工作日
    - 如果是工作日且当前时间 < 20:00，返回当天
    - 如果不是工作日，返回下一个工作日
    
    Returns:
    --------
    str
        目标日期，格式为 'YYYY-MM-DD'
    """
    critical_time = '20:30'
    if gt.is_workday_auto() == True:
        today = date.today()
        next_day = gt.next_workday_calculate(today)
        time_now = datetime.datetime.now().strftime("%H:%M")
        if time_now >= critical_time:
            return next_day
        else:
            today = gt.strdate_transfer(today)
            return today
    else:
        today = date.today()
        next_day = gt.next_workday_calculate(today)
        return next_day
def update_main():
    """
    日常更新主函数
    
    执行完整的日常更新流程：
    1. 确定目标日期
    2. 检查原始数据完整性
    3. 生成L1/L2/L3信号并检查
    4. 生成L0信号并检查
    5. 构建投资组合并检查
    6. 执行信号回测
    
    如果任何一步检查失败，会抛出ValueError异常
    
    Raises:
    ------
    ValueError
        当数据检查、信号生成或组合构建失败时抛出
    """
    target_date=target_date_decision()
    signal_checker=SignalChecker(target_date, target_date, 'prod')
    portfolio_checker=PortfolioChecker(target_date, 'prod')
    scm = signal_constructing_main(None, target_date, target_date, ['L3', 'L2', 'L1'], 'prod', False, False,
                                   '上证50', '中证2000', 0.15, 0.15)
    scm.running_main()
    results2, status2 = signal_checker.check_all_signals()
    if status2 == 'normal':
        scm = signal_constructing_main(None, target_date, target_date, ['L0'], 'prod', False, False,
                                       '上证50', '中证2000', 0.15, 0.15)
        scm.running_main()
        results3, status3 = signal_checker.check_l0_signals()
        if status3 == 'normal':
            pu = portfolio_updating(target_date)
            pu.portfolio_saving_main()
        else:
            print('L0因子更新存在错误')
            raise ValueError
        results4, status4 = portfolio_checker.check_all_portfolios()
        if status4 == 'error':
            print('Portfolio更新存在错误')
            raise ValueError
        else:
            # 计算start_date：target_date往回推5个工作日
            # 从target_date往前推，找到5个工作日前的日期
            current_date = pd.to_datetime(target_date)
            working_days_count = 0
            days_back = 0

            while working_days_count < 5:
                days_back += 1
                check_date = current_date - pd.DateOffset(days=days_back)
                check_date_str = check_date.strftime('%Y-%m-%d')
                if gt.is_workday(check_date_str):
                    working_days_count += 1
                    if working_days_count == 5:
                        start_date = check_date_str
                        break
            end_date = gt.last_workday_calculate(target_date)
            signal_backtesting_main(start_date, end_date, 'prod', big_indexName='上证50', small_indexName='中证2000')
    else:
        print('L1,L2,L3因子更新存在错误')
        raise ValueError

#sqsqssqsqsqsqssqsqs
if __name__ == "__main__":
    config_path = glv.get('config_path')
    gt.table_manager2(config_path,'signal_v3','timeselecting_l2_test')
    # update_main()
    # rdp = raw_data_processing()
    # rdp.rawData_savingMain('2015-07-01', '2025-04-30')
    # scb = signalCombination('2015-06-01', '2025-03-20', 'prod')
    # scb.signalCombination_main()
    # update_main()



