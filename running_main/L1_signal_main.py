import os
import sys
import pandas as pd
import yaml
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
from datetime import datetime
import global_setting.global_dic as glv
from backtesting.factor_backtesting import factor_backtesting
config_path=glv.get('config_path')
pd.set_option('display.max_rows', None)


class L1_signalConstruction:
    def __init__(self, signal_name, start_date, end_date,mode):
        """
        初始化L2信号构建类

        Parameters:
        -----------
        signal_name : str
            信号名称
        mode : str
            模式（如'test'或'prod'）
        """
        self.start_date = start_date
        self.end_date = end_date
        self.signal_name = signal_name
        self.mode=mode
        if self.mode == 'prod':
            self.inputpath_base = glv.get('L2_signalData_prod')
        else:
            self.inputpath_base = glv.get('L2_signalData_test')

    def get_factor_info(self, factor_name):
        """
        根据L1因子名称查找对应的所有L2因子名称

        Parameters:
        -----------
        factor_name : str
            L1因子名称（如"利率因子"、"技术指标因子"等）

        Returns:
        --------
        list
            包含所有匹配的L2因子名称的列表
        """
        try:
            # 获取配置文件路径
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config_project',
                                       'signal_dictionary.yaml')

            # 读取YAML配置文件
            with open(config_path, 'r', encoding='utf-8') as file:
                signal_dict = yaml.safe_load(file)

            # 根据L1因子名称查找对应的所有L2因子名称
            l2_factors = []
            for factor_key, factor_info in signal_dict.items():
                if factor_info.get('L1_factor') == factor_name:
                    l2_factor = factor_info.get('L2_factor')
                    if l2_factor not in l2_factors:  # 避免重复
                        l2_factors.append(l2_factor)

            print(f"找到L1因子 '{factor_name}' 对应的L2因子: {l2_factors}")
            return l2_factors

        except FileNotFoundError:
            print(f"配置文件未找到: {config_path}")
            return []
        except yaml.YAMLError as e:
            print(f"YAML文件解析错误: {e}")
            return []
        except Exception as e:
            print(f"查找因子信息时出错: {e}")
            return []
    def raw_signal_withdraw(self,signal_name):
        inputpath = self.inputpath_base
        inputpath = str(inputpath)+f" Where signal_name='{signal_name}' And valuation_date between '{self.start_date}' and '{self.end_date}'"
        df=gt.data_getting(inputpath,config_path)
        df=df[['valuation_date','final_signal']]
        df.columns=['valuation_date',signal_name]
        return df
    def L1_construction_main(self):
        inputpath_sql = glv.get('sql_path')
        if self.mode=='prod':
             sm = gt.sqlSaving_main(inputpath_sql, 'L1_signal_prod', delete=True)
        else:
            sm = gt.sqlSaving_main(inputpath_sql, 'L1_signal_test', delete=True)
        n=1
        df_final=pd.DataFrame()
        factor_name_list = self.get_factor_info(self.signal_name)
        for factor_name in factor_name_list:
            df=self.raw_signal_withdraw(factor_name)
            if n==1:
                df_final=df
                n+=1
            else:
                df_final=df_final.merge(df,on='valuation_date',how='outer')
        df_final.set_index('valuation_date',inplace=True,drop=True)
        df_final.fillna(0.5,inplace=True)
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
    def L1_backtest_main(self):
        big_indexName='上证50'
        small_indexName='中证2000'
        self.L1_construction_main()
        fb=factor_backtesting(self.signal_name,self.start_date,self.end_date,0.00006,self.mode,'L1',big_indexName,small_indexName,None,None)
        fb.backtesting_main()

if __name__ == "__main__":
    # 示例使用['MacroLiquidity', 'IndexPriceVolume', 'SpecialFactor', 'StockCapital', 'MacroEconomy', 'StockFundamentals', 'StockEmotion']
    for signal_name in ['Commodity']:
        #signal_name = "IndexPriceVolume"  # 示例L1因子名称
        mode = "test"  # 示例模式
        start_date = "2015-01-01"
        end_date = "2026-01-21"
        signal_constructor = L1_signalConstruction(signal_name, start_date, end_date, mode)
        signal_constructor.L1_backtest_main()
