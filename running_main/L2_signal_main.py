import os
import sys
import pandas as pd
import yaml
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
from datetime import datetime
import global_setting.global_dic as glv
from backtesting.factor_backtesting import L3factor_backtesting,factor_backtesting
config_path=glv.get('config_path')
pd.set_option('display.max_rows', None)
class L2_signalConstruction:
    def __init__(self,signal_name,start_date,end_date,cost,mode,big_indexName,small_indexName,big_proportion,small_proportion):
        """
        初始化L2信号构建类
        
        Parameters:
        -----------
        signal_name : str
            信号名称
        mode : str
            模式（如'test'或'prod'）
        """
        self.start_date=start_date
        self.end_date=end_date
        self.cost=cost
        self.big_indexName=big_indexName
        self.small_indexName=small_indexName
        self.big_proportion=big_proportion
        self.small_proportion = small_proportion
        self.signal_name = signal_name
        self.mode = mode
        if self.mode=='prod':
           self.inputpath_base = glv.get('L3_signalData_prod')
        else:
            self.inputpath_base = glv.get('L3_signalData_test')
    def get_factor_info(self, factor_name, name=True):
        """
        根据因子名称查找相关信息
        
        Parameters:
        -----------
        factor_name : str
            当name=True时，输入L2因子名称（如"Shibor"、"债券收益率因子"等）
            当name=False时，输入L3因子名称（如"Shibor_2W"、"Bond_10Y"等）
        name : bool
            True: 输入L2因子名称，返回对应的L3因子名称列表
            False: 输入L3因子名称，返回对应的x值
            
        Returns:
        --------
        list or float
            当name=True时，返回包含所有匹配的L3因子名称的列表
            当name=False时，返回对应的x值
        """
        try:
            # 获取配置文件路径
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config_project', 'signal_dictionary.yaml')
            
            # 读取YAML配置文件
            with open(config_path, 'r', encoding='utf-8') as file:
                signal_dict = yaml.safe_load(file)
            
            if name:
                # 根据L2因子名称查找对应的L3因子
                l3_factors = []
                for factor_key, factor_info in signal_dict.items():
                    if factor_info.get('L2_factor') == factor_name:
                        l3_factors.append(factor_info.get('L3_factor'))
                
                print(f"找到L2因子 '{factor_name}' 对应的L3因子: {l3_factors}")
                return l3_factors
            else:
                # 根据L3因子名称查找对应的x值
                for factor_key, factor_info in signal_dict.items():
                    if factor_info.get('L3_factor') == factor_name:
                        x_value = factor_info.get('Best_x')
                        print(f"找到L3因子 '{factor_name}' 对应的x值: {x_value}")
                        return x_value
                
                print(f"未找到L3因子 '{factor_name}' 对应的x值")
                return None
            
        except FileNotFoundError:
            print(f"配置文件未找到: {config_path}")
            return [] if name else None
        except yaml.YAMLError as e:
            print(f"YAML文件解析错误: {e}")
            return [] if name else None
        except Exception as e:
            print(f"查找因子信息时出错: {e}")
            return [] if name else None
    def raw_signal_withdraw(self,signal_name,df_x):
        inputpath = self.inputpath_base
        inputpath = str(inputpath)+f" Where signal_name='{signal_name}' And valuation_date between '{self.start_date}' and '{self.end_date}'"
        df=gt.data_getting(inputpath,config_path)
        # 统一x列的类型为字符串，以便合并
        if 'x' in df_x.columns:
            df_x['x'] = df_x['x'].astype(float)
        if 'x' in df.columns:
            df['x'] = df['x'].astype(float)
        df=df_x.merge(df,on=['valuation_date','x'],how='left')
        df=df[['valuation_date','final_signal']]
        df.columns=['valuation_date',signal_name]
        return df
    def L2_construction_main(self):
        inputpath_sql = glv.get('sql_path')
        if self.mode=='prod':
             sm = gt.sqlSaving_main(inputpath_sql, 'L2_signal_prod', delete=True)
             sm2 = gt.sqlSaving_main(inputpath_sql, 'L3_bext_x_prod')
        else:
            sm = gt.sqlSaving_main(inputpath_sql, 'L2_signal_test', delete=True)
            sm2 = gt.sqlSaving_main(inputpath_sql, 'L3_bext_x_test')
        n=1
        df_final=pd.DataFrame()
        factor_name_list = self.get_factor_info(self.signal_name, True)
        for factor_name in factor_name_list:
            L3fb = L3factor_backtesting(factor_name, self.start_date,self.end_date, self.cost, self.mode,
                                        self.big_indexName, self.small_indexName, self.big_proportion,
                                        self.small_proportion)
            df_x = L3fb.backtesting_main()
            df_x_sql=df_x.copy()
            df_x_sql['signal_name']=factor_name
            df_x_sql['update_time']=datetime.now().replace(tzinfo=None)  # 当前时间
            sm2.df_to_sql(df_x_sql)
            df=self.raw_signal_withdraw(factor_name,df_x)
            if n==1:
                df_final=df
                n+=1
            else:
                df_final=df_final.merge(df,on='valuation_date',how='outer')
        df_final.fillna(0.5,inplace=True)
        df_final.set_index('valuation_date',inplace=True,drop=True)
        def x_processing(x):
            if x<0.5:
                return 0
            elif x==0.5:
                return 0.5
            else:
                return 1
        # 只计算信号列的平均值，排除valuation_date索引
        signal_columns = [col for col in df_final.columns if col != 'valuation_date']
        df_final[self.signal_name] = df_final[signal_columns].mean(axis=1)
        df_final[self.signal_name] = df_final[self.signal_name].apply(lambda x: x_processing(x))
        df_final.reset_index(inplace=True)
        df_final=df_final[['valuation_date',self.signal_name]]
        df_final['signal_name']=self.signal_name
        df_final['update_time']=datetime.now().replace(tzinfo=None)  # 当前时间
        df_final.rename(columns={self.signal_name:'final_signal'},inplace=True)
        sm.df_to_sql(df_final,'signal_name',self.signal_name)
    def L2_backtest_main(self):
        self.L2_construction_main()
        fb=factor_backtesting(self.signal_name,self.start_date,self.end_date,0.00006,self.mode,'L2',self.big_indexName,self.small_indexName,None,None)
        fb.backtesting_main()


if __name__ == "__main__":
    #'LHBProportion', 'LargeOrder_difference', 'USDX','IndividualStock_Emotion',USBond,'IndividualStock_Emotion','ETF_Shares'
    #['MacroLiquidity', 'IndexPriceVolume', 'SpecialFactor', 'StockCapital', 'MacroEconomy', 'StockFundamentals', 'StockEmotion']
    signal_name_list =['BMCI', 'Bank_Momentum', 'Bond', 'CPI', 'CopperGold', 'CreditSpread', 'DBI', 'ETF_Shares',
                 'EarningsYield_Reverse', 'Future_difference', 'Future_holding', 'Growth', 'IndividualStock_Emotion',
                 'LHBProportion', 'LargeOrder_difference', 'M1M2', 'Monthly_effect', 'NLBP_difference', 'PCT', 'PMI',
                 'PPI', 'RRScore_difference', 'RelativeIndex_Std', 'Relative_turnover', 'Shibor',
                 'TargetIndex_Fundamentals', 'TargetIndex_Momentum', 'TargetIndex_Technical', 'TermSpread', 'USBond',
                 'USDX']
    mode = "prod"         # 示例模式
    start_date = "2015-01-01"
    end_date = "2026-01-07"
    cost = 0.00006
    big_indexName = "上证50"
    small_indexName = "中证2000"
    big_proportion = 0.15
    small_proportion = 0.15
    for signal_name in signal_name_list:
        signal_constructor = L2_signalConstruction(signal_name, start_date, end_date, cost, mode,
                                                   big_indexName, small_indexName, big_proportion, small_proportion)
        signal_constructor.L2_backtest_main()
    
    # # 示例1：查找L2因子对应的L3因子
    # l2_factor_name = "Shibor"  # 示例L2因子名称
    # l3_factors = signal_constructor.get_factor_info(l2_factor_name, name=True)
    # print(f"L2因子 '{l2_factor_name}' 对应的L3因子: {l3_factors}")
    #
    # # 示例2：查找L3因子对应的x值
    # l3_factor_name = "Shibor_2W"  # 示例L3因子名称
    # x_value = signal_constructor.get_factor_info(l3_factor_name, name=False)
    # print(f"L3因子 '{l3_factor_name}' 对应的x值: {x_value}")
    #
    # 示例3：选择最佳x值（不自动更新YAML）

    # print(f"L2信号 '{signal_name}' 的最佳x值: {best_x}")
    #
    # 示例4：选择最佳x值并自动更新YAML文件
    # best_x = signal_constructor.select_best_x(auto=True)
    # print(f"L2信号 '{signal_name}' 的最佳x值: {best_x}，已自动更新YAML文件")
    # best_x = signal_constructor.L2_construction_main()
    # 运行信号构建
    # signal_constructor.run() 