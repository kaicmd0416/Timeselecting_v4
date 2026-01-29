import global_setting.global_dic as glv
import pandas as pd
import os
import sys
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
from datetime import date
import datetime
from data_check.data_check import SignalChecker
from running_main.signal_construct_main import signal_constructing_main
from portfolio.portfolio_construction import portfolio_updating
def history_config_withdraw():
    inputpath = glv.get('signal_parameters_history')
    df = pd.read_excel(inputpath)
    return df
def update_main(signal_name,start_date,end_date,signal_type_list,mode,backtest,big_indexName,small_indexName,big_proportion,small_proportion,portfolio): #触发这个
    # checker = RawDataChecker(start_date,end_date)
    # results, status = checker.check_data_prepare_functions()
    status='normal'
    if status=='normal':
        scm = signal_constructing_main(signal_name, start_date,end_date, signal_type_list, mode, backtest,
                                       big_indexName, small_indexName, big_proportion, small_proportion)
        scm.running_main()
    else:
        raise ValueError
    if portfolio==True:
        working_days_list=gt.working_days_list(start_date,end_date)
        for date in working_days_list:
            pu = portfolio_updating(date)
            pu.portfolio_saving_main()
if __name__ == "__main__":
    #['StockFundamentals', 'StockEmotion']
    signal_name='StockEmotion' #signal_name如果为None默认跑所选的级别因子的全部
    start_date='2015-01-01'
    end_date='2026-01-29'
    signal_type_list=['L3','L2','L1'] #输出需要跑的因子级别
    mode='prod'#除了生产意外其他的都是test
    backtest=True #看要不要回测报告
    big_indexName='上证50' #所选大盘的指数
    small_indexName='中证2000'#所选小盘的指数
    big_proportion=0.15 #这个对应的是自动调参的时候的大盘指数占比如果auto=False则无关系
    small_proportion=0.15 #这个对应的是自动调参的时候的小盘指数占比如果auto=False则无关系
    portfolio=False#这个是对应要不要存储portfolio到portfolio表里面
    update_main(signal_name, start_date, end_date, signal_type_list, mode, backtest,  big_indexName,
                small_indexName, big_proportion, small_proportion,portfolio)