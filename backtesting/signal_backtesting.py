import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import yaml
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
import global_setting.global_dic as glv
from backtesting.factor_backtesting import factor_backtesting
config_path=glv.get('config_path')
def signal_list_withdraw(signal_level,mode):
    inputpath_base = glv.get(str(signal_level)+'_signalData_'+str(mode))
    df=gt.data_getting(inputpath_base,config_path)
    return df
def signal_backtesting_main(start_date, end_date,mode,big_indexName='上证50',small_indexName='中证2000'):
    for signal_level in ['L0','L1','L2']:
        df=signal_list_withdraw(signal_level,mode)
        if signal_level=='L0':
            signal_name_list=['Combine']
        else:
            signal_name_list=df['signal_name'].unique().tolist()
        for signal_name in signal_name_list:
            print(signal_name)
            fb=factor_backtesting(signal_name, start_date, end_date, 0.00006, mode, signal_level, big_indexName, small_indexName, None, x = None)
            fb.backtesting_main_sql()

def get_factor_info(signal_name, signal_type):
        """
        根据信号名称和类型查找对应的因子列表

        Parameters:
        -----------
        signal_name : str or None
            信号名称，如果为None则返回对应级别的所有信号名称
        signal_type : str
            "L0": 获取所有L1因子名称
            "L1": 根据L1因子名称查找对应的所有L2因子，或获取所有L1因子名称
            "L2": 根据L2因子名称查找对应的所有L3因子，或获取所有L2因子名称
            "L3": 获取所有L3因子名称

        Returns:
        --------
        list
            包含所有匹配的因子名称的列表
        """
        try:
            # 获取配置文件路径
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config_project',
                                       'signal_dictionary.yaml')

            # 读取YAML配置文件
            with open(config_path, 'r', encoding='utf-8') as file:
                signal_dict = yaml.safe_load(file)

            result_factors = []

            if signal_type == "L0":
                # 获取所有L1因子名称
                for factor_key, factor_info in signal_dict.items():
                    l1_factor = factor_info.get('L1_factor')
                    if l1_factor and l1_factor not in result_factors:  # 避免重复
                        result_factors.append(l1_factor)

                print(f"找到所有L1因子: {result_factors}")

            elif signal_type == "L1":
                if signal_name is None:
                    # 获取所有L1因子名称
                    for factor_key, factor_info in signal_dict.items():
                        l1_factor = factor_info.get('L1_factor')
                        if l1_factor and l1_factor not in result_factors:  # 避免重复
                            result_factors.append(l1_factor)
                    print(f"找到所有L1因子: {result_factors}")
                else:
                    # 根据L1因子名称查找对应的所有L2因子
                    for factor_key, factor_info in signal_dict.items():
                        if factor_info.get('L1_factor') == signal_name:
                            l2_factor = factor_info.get('L2_factor')
                            if l2_factor and l2_factor not in result_factors:  # 避免重复
                                result_factors.append(l2_factor)
                    print(f"找到L1因子 '{signal_name}' 对应的L2因子: {result_factors}")

            elif signal_type == "L2":
                if signal_name is None:
                    # 获取所有L2因子名称
                    for factor_key, factor_info in signal_dict.items():
                        l2_factor = factor_info.get('L2_factor')
                        if l2_factor and l2_factor not in result_factors:  # 避免重复
                            result_factors.append(l2_factor)
                    print(f"找到所有L2因子: {result_factors}")
                else:
                    # 根据L2因子名称查找对应的所有L3因子
                    for factor_key, factor_info in signal_dict.items():
                        if factor_info.get('L2_factor') == signal_name:
                            l3_factor = factor_info.get('L3_factor')
                            if l3_factor and l3_factor not in result_factors:  # 避免重复
                                result_factors.append(l3_factor)
                    print(f"找到L2因子 '{signal_name}' 对应的L3因子: {result_factors}")

            elif signal_type == "L3":
                # 获取所有L3因子名称
                for factor_key, factor_info in signal_dict.items():
                    l3_factor = factor_info.get('L3_factor')
                    if l3_factor and l3_factor not in result_factors:  # 避免重复
                        result_factors.append(l3_factor)

                print(f"找到所有L3因子: {result_factors}")
            else:
                print(f"不支持的信号类型: {signal_type}，支持的类型为 'L0', 'L1', 'L2', 'L3'")
                return []

            return result_factors

        except FileNotFoundError:
            print(f"配置文件未找到: {config_path}")
            return []
        except yaml.YAMLError as e:
            print(f"YAML文件解析错误: {e}")
            return []
        except Exception as e:
            print(f"查找因子信息时出错: {e}")
            return []
def signalMatrix_saving(start_date, end_date,signal_level,mode):
    outputpath=glv.get('backtest_output')
    outputpath=os.path.join(outputpath,f"{signal_level}_signal_matrix_{mode}.xlsx")
    inputpath=glv.get(str(signal_level)+'_signalBacktest_'+str(mode))
    inputpath=str(inputpath)+f" Where valuation_date between'{start_date}' and '{end_date}'"
    df = gt.data_getting(inputpath, config_path)
    df=df[['valuation_date','signal_name','excess_return']]
    df=gt.sql_to_timeseries(df)
    df.set_index('valuation_date',inplace=True,drop=True)
    df=(1+df).cumprod()
    df.plot()
    plt.show()
    #df.to_excel(outputpath)
def signalMatrix_saving_split(start_date, end_date,signal_name,signal_level,mode):
    outputpath=glv.get('backtest_output')
    signal_list=get_factor_info(signal_name,'L1')
    print(signal_list)
    outputpath=os.path.join(outputpath,f"{signal_level}_signal_matrix_{mode}.xlsx")
    inputpath=glv.get(str(signal_level)+'_signalBacktest_'+str(mode))
    inputpath=str(inputpath)+f" Where valuation_date between'{start_date}' and '{end_date}'"
    df = gt.data_getting(inputpath, config_path)
    df=df[df['signal_name'].isin(signal_list)]
    df=df[['valuation_date','signal_name','excess_return']]
    df=gt.sql_to_timeseries(df)
    df.set_index('valuation_date',inplace=True,drop=True)
    df=(1+df).cumprod()
    df.plot()
    plt.show()
    #df.to_excel(outputpath)
if __name__ == "__main__":
    #signal_backtesting_main('2016-01-01', '2025-12-31','prod', big_indexName='上证50', small_indexName='中证2000')
    signalMatrix_saving('2025-01-01', '2025-12-31','L1','prod')
    #signalMatrix_saving_split('2024-12-31', '2025-11-31','StockEmotion', 'L2', 'prod')
    #signal_backtesting_main('2016-01-01', '2025-12-01','prod')