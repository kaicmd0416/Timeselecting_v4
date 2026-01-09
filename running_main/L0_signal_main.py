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
from dotenv import load_dotenv


class L0_signalConstruction:
    def __init__(self, start_date, end_date, mode):
        """
        初始化L2信号构建类

        Parameters:
        -----------
        signal_name : str
            信号名称
        mode : str
            模式（如'test'或'prod'）
        """
        load_dotenv()
        self.webhook_url = os.getenv('FEISHU_WEBHOOK_URL')
        self.start_date = start_date
        self.end_date = end_date
        self.mode = mode
        if self.mode == 'prod':
            self.inputpath_base = glv.get('L1_signalData_prod')
        else:
            self.inputpath_base = glv.get('L1_signalData_test')

    def get_factor_info(self):
        """
        获取所有L1因子名称的列表

        Returns:
        --------
        list
            包含所有L1因子名称的列表
        """
        try:
            # 获取配置文件路径
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config_project',
                                       'signal_dictionary.yaml')

            # 读取YAML配置文件
            with open(config_path, 'r', encoding='utf-8') as file:
                signal_dict = yaml.safe_load(file)

            # 获取所有L1因子名称
            l1_factors = []
            for factor_key, factor_info in signal_dict.items():
                l1_factor = factor_info.get('L1_factor')
                if l1_factor not in l1_factors:  # 避免重复
                    l1_factors.append(l1_factor)

            print(f"找到所有L1因子: {l1_factors}")
            return l1_factors

        except FileNotFoundError:
            print(f"配置文件未找到: {config_path}")
            return []
        except yaml.YAMLError as e:
            print(f"YAML文件解析错误: {e}")
            return []
        except Exception as e:
            print(f"查找因子信息时出错: {e}")
            return []

    def raw_signal_withdraw(self, signal_name):
        inputpath = self.inputpath_base
        inputpath = str(
            inputpath) + f" Where signal_name='{signal_name}' And valuation_date between '{self.start_date}' and '{self.end_date}'"
        df = gt.data_getting(inputpath, config_path)
        df = df[['valuation_date', 'final_signal']]
        df.columns = ['valuation_date', signal_name]
        return df

    def L0_construction_main(self):
        inputpath_sql = glv.get('sql_path')
        if self.mode == 'prod':
            sm = gt.sqlSaving_main(inputpath_sql, 'L0_signal_prod')
        else:
            sm = gt.sqlSaving_main(inputpath_sql, 'L0_signal_test')
        n = 1
        df_final = pd.DataFrame()
        factor_name_list = self.get_factor_info()
        for factor_name in factor_name_list:
            df = self.raw_signal_withdraw(factor_name)
            if n == 1:
                df_final = df
                n += 1
            else:
                df_final = df_final.merge(df, on='valuation_date', how='outer')
        df_final.set_index('valuation_date', inplace=True, drop=True)
        df_final.fillna(0.5, inplace=True)

        def x_processing(x):
            if x < 0.5:
                return 0
            elif x == 0.5:
                return 0.5
            else:
                return 1

        # 只计算信号列的平均值，排除valuation_date索引
        signal_columns = [col for col in df_final.columns if col != 'valuation_date']
        df_final['final_value'] = df_final[signal_columns].mean(axis=1)
        df_final['final_signal'] = df_final['final_value'].apply(lambda x: x_processing(x))
        df_final.reset_index(inplace=True)
        df_final = df_final[['valuation_date','final_value','final_signal']]
        df_final['update_time'] = datetime.now().replace(tzinfo=None)  # 当前时间
        sender = gt.FeishuBot(self.webhook_url)
        result_mean_str = df_final.to_string(index=False)
        sender.send_message(result_mean_str)
        sm.df_to_sql(df_final)

    def L0_backtest_main(self):
        big_indexName = '沪深300'
        small_indexName = '中证2000'
        self.L0_construction_main()
        fb = factor_backtesting('final_signal', self.start_date, self.end_date, 0.00006, self.mode, 'L0',
                                big_indexName, small_indexName, None,None)
        fb.backtesting_main()


if __name__ == "__main__":
    # 示例使用
    mode = "prod"  # 示例模式
    start_date = "2015-01-01"
    end_date = "2026-01-08"
    signal_constructor = L0_signalConstruction(start_date, end_date, mode)
    signal_constructor.L0_backtest_main()


