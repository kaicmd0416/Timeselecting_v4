"""
信号生成主控制器模块 (signal_construct_main)

本模块负责协调各层级信号的生成流程，根据配置自动调用相应的信号生成类。

作者: TimeSelecting Team
版本: v3.0
"""

import os
import sys
import pandas as pd
import yaml
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
from datetime import datetime
import global_setting.global_dic as glv
from running_main.L0_signal_main import L0_signalConstruction
from running_main.L1_signal_main import L1_signalConstruction
from running_main.L2_signal_main import L2_signalConstruction
from running_main.L3_signal_main import L3_signalConstruction
from backtesting.factor_backtesting import factor_backtesting
config_path=glv.get('config_path')

class signal_constructing_main:
    """
    信号生成主控制器类
    
    负责根据信号类型列表，自动调用相应的信号生成类，并支持回测功能。
    
    Attributes:
    -----------
    signal_name : str or None
        信号名称，如果为None则处理该类型的所有信号
    mode : str
        模式，'prod'（生产模式，保存数据）或 'test'（测试模式）
    start_date : str
        开始日期，格式为 'YYYY-MM-DD'
    end_date : str
        结束日期，格式为 'YYYY-MM-DD'
    signal_type_list : list
        信号类型列表，如 ['L3', 'L2', 'L1'] 或 ['L0']
    backtest : bool
        是否执行回测
    auto : bool
        是否自动模式
    big_indexName : str
        大盘指数名称，如 '上证50'
    small_indexName : str
        小盘指数名称，如 '中证2000'
    big_proportion : float
        大盘指数权重比例
    small_proportion : float
        小盘指数权重比例
    """
    
    def __init__(self, signal_name, start_date, end_date, signal_type_list, mode, backtest,
                 big_indexName, small_indexName, big_proportion, small_proportion):
        """
        初始化信号生成主控制器
        
        Parameters:
        -----------
        signal_name : str or None
            信号名称，如果为None则处理该类型的所有信号
        start_date : str
            开始日期，格式为 'YYYY-MM-DD'
        end_date : str
            结束日期，格式为 'YYYY-MM-DD'
        signal_type_list : list
            信号类型列表，如 ['L3', 'L2', 'L1'] 或 ['L0']
        mode : str
            模式，'prod'（生产模式，保存数据）或 'test'（测试模式）
        backtest : bool
            是否执行回测
        auto : bool
            是否自动模式
        big_indexName : str
            大盘指数名称
        small_indexName : str
            小盘指数名称
        big_proportion : float
            大盘指数权重比例
        small_proportion : float
            小盘指数权重比例
        """
        self.signal_name=signal_name
        self.mode=mode
        self.start_date=start_date
        self.end_date=end_date
        self.signal_type_list=signal_type_list
        self.backtest=backtest
        self.big_indexName = big_indexName
        self.small_indexName = small_indexName
        self.big_proportion = big_proportion
        self.small_proportion = small_proportion

    def get_factor_info(self, signal_name, signal_type):
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
    def running_main(self):
        """
        运行信号生成主函数
        
        根据signal_type_list自动调用相应的信号生成类：
        - ['L3']: 生成L3级别信号
        - ['L2']: 生成L2级别信号
        - ['L1']: 生成L1级别信号
        - ['L0']: 生成L0级别信号
        - ['L3', 'L2']: 先生成L3，再生成L2
        - ['L3', 'L2', 'L1']: 依次生成L3、L2、L1
        
        如果backtest=True，会在信号生成后执行回测。
        """
        if self.signal_type_list==['L3']:
            L3S=L3_signalConstruction(self.signal_name,self.mode,self.start_date,self.end_date)
            L3S.signal_main()
        elif self.signal_type_list==['L2']:
            if self.signal_name==None:
                 L2_signal_name_list=self.get_factor_info(self.signal_name, 'L2')
                 for l2_signal_name in L2_signal_name_list:
                     L2S = L2_signalConstruction(l2_signal_name, self.start_date, self.end_date, 0.00006, self.mode,
                                                 self.big_indexName, self.small_indexName, self.big_proportion,
                                                 self.small_proportion)
                     L2S.L2_construction_main()
                     if self.backtest == True:
                         fb = factor_backtesting(self.signal_name, self.start_date, self.end_date, 0.00006, self.mode,
                                                 'L2',
                                                 self.big_indexName, self.small_indexName, None, None)
                         fb.backtesting_main()
        elif self.signal_type_list==['L1']:
            if self.signal_name == None:
                L1_signal_name_list = self.get_factor_info(self.signal_name, 'L1')
                for L1_signal_name in L1_signal_name_list:
                    L1S = L1_signalConstruction(L1_signal_name, self.start_date, self.end_date, self.mode)
                    L1S.L1_construction_main()
                    if self.backtest == True:
                        fb = factor_backtesting(self.signal_name, self.start_date, self.end_date, 0.00006, self.mode,
                                                'L1',
                                                self.big_indexName, self.small_indexName, None, None)
                        fb.backtesting_main()
        elif self.signal_type_list == ['L0']:
            L0S = L0_signalConstruction(self.start_date, self.end_date, self.mode)
            L0S.L0_construction_main()
            if self.backtest==True:
                fb = factor_backtesting(self.signal_name, self.start_date, self.end_date, 0.00006, self.mode, 'L0',
                                        self.big_indexName, self.small_indexName, None,None)
                fb.backtesting_main()
        elif self.signal_type_list==['L3','L2']:
            if self.signal_name==None:
                 L2_signal_name_list=self.get_factor_info(self.signal_name, 'L2')
                 for l2_signal_name in L2_signal_name_list:
                     L3_signal_name_list = self.get_factor_info(l2_signal_name, 'L2')
                     for signal_name in L3_signal_name_list:
                         L3S = L3_signalConstruction(signal_name, self.mode, self.start_date, self.end_date)
                         L3S.signal_main()
                     L2S = L2_signalConstruction(l2_signal_name, self.start_date, self.end_date, 0.00006, self.mode,
                                                 self.big_indexName, self.small_indexName, self.big_proportion,
                                                 self.small_proportion)
                     L2S.L2_construction_main()
                     if self.backtest == True:
                         fb = factor_backtesting(l2_signal_name, self.start_date, self.end_date, 0.00006, self.mode,
                                                 'L2',
                                                 self.big_indexName, self.small_indexName, None,None)
                         fb.backtesting_main()
            else:
                L3_signal_name_list = self.get_factor_info(self.signal_name, 'L2')
                for signal_name in L3_signal_name_list:
                    L3S = L3_signalConstruction(signal_name, self.mode, self.start_date, self.end_date)
                    L3S.signal_main()
                L2S = L2_signalConstruction(self.signal_name, self.start_date, self.end_date, 0.00006, self.mode,
                                            self.big_indexName, self.small_indexName, self.big_proportion,
                                            self.small_proportion)
                L2S.L2_construction_main()
                if self.backtest == True:
                    fb = factor_backtesting(self.signal_name, self.start_date, self.end_date, 0.00006, self.mode, 'L2',
                                            self.big_indexName, self.small_indexName, None)
                    fb.backtesting_main()

        elif self.signal_type_list==['L3','L2','L1']:
            if self.signal_name == None:
                L1_signal_name_list = self.get_factor_info(self.signal_name, 'L1')
                for L1_signal_name in L1_signal_name_list:
                    L2_signal_name_list = self.get_factor_info(L1_signal_name, 'L1')
                    for L2_signal_name in L2_signal_name_list:
                        L3_signal_name_list = self.get_factor_info(L2_signal_name, 'L2')
                        for L3_signal_name in L3_signal_name_list:
                            L3S = L3_signalConstruction(L3_signal_name, self.mode, self.start_date, self.end_date)
                            L3S.signal_main()
                        L2S = L2_signalConstruction(L2_signal_name, self.start_date, self.end_date, 0.00006, self.mode,
                                                    self.big_indexName, self.small_indexName, self.big_proportion,
                                                    self.small_proportion)
                        L2S.L2_construction_main()
                    L1S = L1_signalConstruction(L1_signal_name, self.start_date, self.end_date, self.mode)
                    L1S.L1_construction_main()
                    if self.backtest == True:
                        fb = factor_backtesting(L1_signal_name, self.start_date, self.end_date, 0.00006, self.mode,
                                                'L1',
                                                self.big_indexName, self.small_indexName, None,None)
                        fb.backtesting_main()
            else:
                L2_signal_name_list = self.get_factor_info(self.signal_name, 'L1')
                for L2_signal_name in L2_signal_name_list:
                    L3_signal_name_list = self.get_factor_info(L2_signal_name, 'L2')
                    for L3_signal_name in L3_signal_name_list:
                        L3S = L3_signalConstruction(L3_signal_name, self.mode, self.start_date, self.end_date)
                        L3S.signal_main()
                    L2S = L2_signalConstruction(L2_signal_name, self.start_date, self.end_date, 0.00006, self.mode,
                                                self.big_indexName, self.small_indexName, self.big_proportion,
                                                self.small_proportion)
                    L2S.L2_construction_main()
                L1S = L1_signalConstruction(self.signal_name, self.start_date, self.end_date, self.mode)
                L1S.L1_construction_main()
                if self.backtest == True:
                    fb = factor_backtesting(self.signal_name, self.start_date, self.end_date, 0.00006, self.mode, 'L1',
                                            self.big_indexName, self.small_indexName, None,None)
                    fb.backtesting_main()
        elif self.signal_type_list==['L3','L2','L1','L0']:
            L1_signal_name_list = self.get_factor_info(self.signal_name, 'L0')
            for L1_signal_name in L1_signal_name_list:
                L2_signal_name_list = self.get_factor_info(L1_signal_name, 'L1')
                for L2_signal_name in L2_signal_name_list:
                    L3_signal_name_list = self.get_factor_info(L2_signal_name, 'L2')
                    for L3_signal_name in L3_signal_name_list:
                        L3S = L3_signalConstruction(L3_signal_name, self.mode, self.start_date, self.end_date)
                        L3S.signal_main()
                    L2S = L2_signalConstruction(L2_signal_name, self.start_date, self.end_date, 0.00006, self.mode,
                                                self.big_indexName, self.small_indexName, self.big_proportion,
                                                self.small_proportion)
                    L2S.L2_construction_main()
                L1S = L1_signalConstruction(L1_signal_name, self.start_date, self.end_date, self.mode)
                L1S.L1_construction_main()
            L0S = L0_signalConstruction(self.start_date, self.end_date, self.mode)
            L0S.L0_construction_main()
            if self.backtest==True:
                fb = factor_backtesting(self.signal_name, self.start_date, self.end_date, 0.00006, self.mode, 'L0',
                                        self.big_indexName, self.small_indexName, None,None)
                fb.backtesting_main()
        else:
            raise ValueError
if __name__ == "__main__":
    scm=signal_constructing_main(None,'2025-01-01','2025-10-29',['L3','L2','L1','L0'],'prod',False,False,'沪深300','中证2000',0.15,0.15)
    scm.running_main()





