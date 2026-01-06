"""
投资组合构建模块 (portfolio_construction)

本模块负责基于L0信号构建投资组合，包括：
- 提取L0信号
- 根据信号构建投资组合
- 保存组合数据到数据库

作者: TimeSelecting Team
版本: v3.0
"""

from matplotlib import pyplot as plt

import global_setting.global_dic as glv
import pandas as pd
import os
import sys
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
import os
import yaml
from datetime import datetime, timedelta
import calendar
import numpy as np
config_path=glv.get('config_path')

class portfolio_updating:
    """
    投资组合更新类
    
    负责基于L0信号构建和更新投资组合。
    
    Attributes:
    -----------
    target_date : str
        目标日期，格式为 'YYYY-MM-DD'
    now : datetime
        当前时间
    """
    
    def __init__(self, target_date):
        """
        初始化投资组合更新类
        
        Parameters:
        -----------
        target_date : str
            目标日期，格式为 'YYYY-MM-DD'
        """
        self.target_date=target_date
        self.available_date=gt.last_workday_calculate(self.target_date)
        self.now = datetime.now().replace(tzinfo=None)  # 当前时间
    def timeselecting_signalWithdraw(self):
        """
        提取L0信号数据，如果target_date的combine_value为0.5，则查找过去最近一天不为0.5的值
        
        Returns:
            float: combine_value值
        """
        inputpath_mean = glv.get('L0_signalData_prod')
        # 提取全部日期的数据
        df = gt.data_getting(inputpath_mean, config_path)
        
        # 确保日期格式正确
        df['valuation_date'] = pd.to_datetime(df['valuation_date'])
        df = df.sort_values(by='valuation_date')
        
        # 找到target_date对应的数据
        target_datetime = pd.to_datetime(self.target_date)
        target_data = df[df['valuation_date'] == target_datetime]
        
        if target_data.empty:
            raise ValueError(f"未找到 {self.target_date} 的数据")
        
        combine_value = target_data['final_value'].iloc[0]
        
        # # 如果combine_value等于0.5，查找过去最近一天不为0.5的值
        # if combine_value == 0.5:
        #     # 获取target_date之前的所有数据，并按日期降序排列
        #     past_data = df[df['valuation_date'] < target_datetime].copy()
        #     past_data = past_data.sort_values(by='valuation_date', ascending=False)
        #
        #     # 找到第一个final_value不为0.5的记录
        #     non_05_data = past_data[past_data['final_value'] != 0.5]
        #
        #     if not non_05_data.empty:
        #         combine_value = non_05_data['final_value'].iloc[0]
        #         found_date = non_05_data['valuation_date'].iloc[0].strftime('%Y-%m-%d')
        #         print(f"目标日期 {self.target_date} 的combine_value为0.5，使用过去最近一天 {found_date} 的值: {combine_value}")
        #     else:
        #         print(f"警告: 目标日期 {self.target_date} 的combine_value为0.5，且未找到过去任何不为0.5的值，使用0.5")
        
        return combine_value

    def timeselecting_signalWithdraw2(self):
        """
        提取L0信号数据，如果target_date的combine_value为0.5，则查找过去最近一天不为0.5的值

        Returns:
            float: combine_value值
        """
        inputpath_mean = glv.get('L0_signalData_prod')
        # 提取全部日期的数据
        df = gt.data_getting(inputpath_mean, config_path)

        # 确保日期格式正确
        df['valuation_date'] = pd.to_datetime(df['valuation_date'])
        df = df.sort_values(by='valuation_date')

        # 找到target_date对应的数据
        target_datetime = pd.to_datetime(self.target_date)
        target_data = df[df['valuation_date'] == target_datetime]

        if target_data.empty:
            raise ValueError(f"未找到 {self.target_date} 的数据")

        combine_value = target_data['final_value'].iloc[0]

        # 如果combine_value等于0.5，查找过去最近一天不为0.5的值
        if combine_value == 0.5:
            # 获取target_date之前的所有数据，并按日期降序排列
            past_data = df[df['valuation_date'] < target_datetime].copy()
            past_data = past_data.sort_values(by='valuation_date', ascending=False)

            # 找到第一个final_value不为0.5的记录
            non_05_data = past_data[past_data['final_value'] != 0.5]

            if not non_05_data.empty:
                combine_value = non_05_data['final_value'].iloc[0]
                found_date = non_05_data['valuation_date'].iloc[0].strftime('%Y-%m-%d')
                print(f"目标日期 {self.target_date} 的combine_value为0.5，使用过去最近一天 {found_date} 的值: {combine_value}")
            else:
                print(f"警告: 目标日期 {self.target_date} 的combine_value为0.5，且未找到过去任何不为0.5的值，使用0.5")

        return combine_value
    def sql_path_withdraw(self):
        """
        获取组合SQL配置文件路径
        
        Returns:
        --------
        str
            portfolio_sql.yaml配置文件的路径
        """
        workspace_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(workspace_path, 'config_project', 'portfolio_sql.yaml')
        return config_path
    def portfoliol_info_withdraw(self, index_abbr):
        """
        Get portfolio information for a specific index
        
        Args:
            index_abbr (str): Index abbreviation (e.g., 'hs300', 'zz500', 'zz1000')
            
        Returns:
            pd.DataFrame: DataFrame with columns 'portfolio_name' and 'weight'
        """
        # Get the absolute path to the workspace
        workspace_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(workspace_path, 'config_project', 'portfolio.yaml')
        
        # Read portfolio configuration from yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            portfolio_config = yaml.safe_load(f)
        
        # Check if the index exists
        if index_abbr not in portfolio_config['indices']:
            raise ValueError(f"Index {index_abbr} not found in configuration")
        
        # Get portfolios for the specified index
        portfolios = portfolio_config['indices'][index_abbr]['portfolios']
        
        # Create DataFrame
        df = pd.DataFrame(portfolios)
        df = df[['name', 'weight']]  # Select only name and weight columns
        df.columns = ['portfolio_name', 'weight']  # Rename columns
        
        return df
    def portfolio_withdraw(self, portfolio_name):
        """
        获取指定投资组合的持仓数据
        
        Parameters:
        -----------
        portfolio_name : str
            投资组合名称
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - code: 股票代码
            - weight: 权重
        """
        inputpath=glv.get('portfolio')
        inputpath=inputpath+f" Where portfolio_name='{portfolio_name}' and valuation_date='{self.target_date}'"
        df=gt.data_getting(inputpath,config_path)
        df=df[['code','weight']]
        return df
    def opt_portfolio_withdraw(self):
        """
        获取优化后的投资组合数据
        
        根据配置获取各指数的投资组合，并按权重合并
        
        Returns:
        --------
        dict
            包含各指数投资组合的字典，键为指数缩写（如'hs300'、'zz500'等），
            值为包含code和weight列的DataFrame
        """
        df_info_hs300=self.portfoliol_info_withdraw('hs300')
        df_info_zz500=self.portfoliol_info_withdraw('zz500')
        df_info_zzA500=self.portfoliol_info_withdraw('zzA500')
        df_info_zz1000 = self.portfoliol_info_withdraw('zz1000')
        df_info_zz2000=self.portfoliol_info_withdraw('zz2000')
        df_final_300=pd.DataFrame()
        df_final_500=pd.DataFrame()
        df_final_A500=pd.DataFrame()
        df_final_1000=pd.DataFrame()
        df_final_2000=pd.DataFrame()
        for i in range(len(df_info_hs300)):
            portfolio_name=df_info_hs300['portfolio_name'][i]
            weight=df_info_hs300['weight'][i]
            df_hs300=self.portfolio_withdraw(portfolio_name)
            
            # Ensure df_hs300 has code and weight columns
            if 'code' not in df_hs300.columns or 'weight' not in df_hs300.columns:
                raise ValueError(f"Portfolio {portfolio_name} must have 'code' and 'weight' columns")
            
            # Multiply the portfolio weight by the individual stock weights
            df_hs300['weight'] = df_hs300['weight'] * weight
            
            # Merge with final DataFrame
            if df_final_300.empty:
                df_final_300 = df_hs300
            else:
                # Group by code and sum the weights
                df_final_300 = pd.concat([df_final_300, df_hs300])
                df_final_300 = df_final_300.groupby('code')['weight'].sum().reset_index()
        for i in range(len(df_info_zzA500)):
            portfolio_name = df_info_zzA500['portfolio_name'][i]
            weight = df_info_zzA500['weight'][i]
            df_zzA500=self.portfolio_withdraw(portfolio_name)

            # Ensure df_zzA500 has code and weight columns
            if 'code' not in df_zzA500.columns or 'weight' not in df_zzA500.columns:
                raise ValueError(f"Portfolio {portfolio_name} must have 'code' and 'weight' columns")

            # Multiply the portfolio weight by the individual stock weights
            df_zzA500['weight'] = df_zzA500['weight'] * weight

            # Merge with final DataFrame
            if df_final_A500.empty:
                df_final_A500 = df_zzA500
            else:
                # Group by code and sum the weights
                df_final_A500 = pd.concat([df_final_A500, df_zzA500])
                df_final_A500 = df_final_A500.groupby('code')['weight'].sum().reset_index()
        
        # Process ZZ500 portfolios
        for i in range(len(df_info_zz500)):
            portfolio_name=df_info_zz500['portfolio_name'][i]
            weight=df_info_zz500['weight'][i]
            df_zz500=self.portfolio_withdraw(portfolio_name)
            
            if 'code' not in df_zz500.columns or 'weight' not in df_zz500.columns:
                raise ValueError(f"Portfolio {portfolio_name} must have 'code' and 'weight' columns")
            
            df_zz500['weight'] = df_zz500['weight'] * weight
            
            if df_final_500.empty:
                df_final_500 = df_zz500
            else:
                df_final_500 = pd.concat([df_final_500, df_zz500])
                df_final_500 = df_final_500.groupby('code')['weight'].sum().reset_index()
        # Process ZZ1000 portfolios
        for i in range(len(df_info_zz1000)):
            portfolio_name = df_info_zz1000['portfolio_name'][i]
            weight = df_info_zz1000['weight'][i]
            df_zz1000=self.portfolio_withdraw(portfolio_name)

            if 'code' not in df_zz1000.columns or 'weight' not in df_zz1000.columns:
                raise ValueError(f"Portfolio {portfolio_name} must have 'code' and 'weight' columns")

            df_zz1000['weight'] = df_zz1000['weight'] * weight

            if df_final_1000.empty:
                df_final_1000 = df_zz1000
            else:
                df_final_1000 = pd.concat([df_final_1000, df_zz1000])
                df_final_1000 = df_final_1000.groupby('code')['weight'].sum().reset_index()
        # Process ZZ1000 portfolios
        for i in range(len(df_info_zz2000)):
            portfolio_name=df_info_zz2000['portfolio_name'][i]
            weight=df_info_zz2000['weight'][i]
            df_zz2000=self.portfolio_withdraw(portfolio_name)
            
            if 'code' not in df_zz2000.columns or 'weight' not in df_zz2000.columns:
                raise ValueError(f"Portfolio {portfolio_name} must have 'code' and 'weight' columns")
            
            df_zz2000['weight'] = df_zz2000['weight'] * weight
            
            if df_final_2000.empty:
                df_final_2000 = df_zz2000
            else:
                df_final_2000 = pd.concat([df_final_2000, df_zz2000])
                df_final_2000 = df_final_2000.groupby('code')['weight'].sum().reset_index()
        
        return df_final_300, df_final_A500,df_final_500, df_final_1000,df_final_2000
    def stock_portfolio_construction(self):
        """
        构建股票投资组合
        
        根据L0信号（combine_value）构建股票投资组合：
        - combine_value > 0.5: 看多沪深300，构建混合组合
        - combine_value < 0.5: 看多中证2000，构建混合组合
        - combine_value == 0.5: 使用原始组合
        
        Returns:
        --------
        dict
            包含各投资组合的字典，键为组合名称，值为包含code和weight列的DataFrame
        """
        # Get base portfolios
        df_300,df_A500, df_500, df_1000,df_2000 = self.opt_portfolio_withdraw()
        combine_value = self.timeselecting_signalWithdraw()
        
        if combine_value > 0.5:
            # Rename weight columns
            df_300 = df_300.rename(columns={'weight': 'weight_hs300'})
            df_A500=df_A500.rename(columns={'weight':'weight_A500'})
            df_500 = df_500.rename(columns={'weight': 'weight_zz500'})
            df_1000=df_1000.rename(columns={'weight': 'weight_zz1000'})
            df_2000 = df_2000.rename(columns={'weight': 'weight_zz2000'})
            
            # Merge all DataFrames
            df_merged = df_300.merge(df_500, on='code', how='outer').merge(df_2000, on='code', how='outer').merge(df_A500, on='code', how='outer').merge(df_1000, on='code', how='outer')
            
            # Fill NaN values with 0
            df_merged = df_merged.fillna(0)
            
            # Calculate new portfolios
            df_timeselecting_zz500_pro = pd.DataFrame()
            df_timeselecting_zz500_pro['code'] = df_merged['code']
            df_timeselecting_zz500_pro['weight'] = (1 - combine_value) * df_merged['weight_zz500'] + combine_value * df_merged['weight_zz2000']

            # Calculate new portfolios
            df_timeselecting_zz500 = pd.DataFrame()
            df_timeselecting_zz500['code'] = df_merged['code']
            df_timeselecting_zz500['weight'] = 0.5* df_merged['weight_zz500'] + 0.2*df_merged['weight_zz2000'] +0.3*df_merged['weight_zz1000']

            df_timeselecting_hs300 = pd.DataFrame()
            df_timeselecting_hs300['code'] = df_merged['code']
            df_timeselecting_hs300['weight'] = 0.8 * df_merged['weight_hs300'] + 0.2 * df_merged['weight_zz2000']

            df_timeselecting_zzA500 = pd.DataFrame()
            df_timeselecting_zzA500['code'] = df_merged['code']
            df_timeselecting_zzA500['weight'] = 0.8 * df_merged['weight_A500'] + 0.2 * df_merged['weight_zz2000']

            df_timeselecting_zzA500_pro = pd.DataFrame()
            df_timeselecting_zzA500_pro['code'] = df_merged['code']
            df_timeselecting_zzA500_pro['weight'] = (1 - combine_value) * df_merged['weight_A500'] + combine_value * \
                                                   df_merged['weight_zz2000']

            df_timeselecting_hs300_pro = pd.DataFrame()
            df_timeselecting_hs300_pro['code'] = df_merged['code']
            df_timeselecting_hs300_pro['weight'] = (1 - combine_value) * df_merged['weight_hs300'] + combine_value * df_merged['weight_zz2000']
            
            # Remove rows with zero weights
            df_timeselecting_zzA500 = df_timeselecting_zzA500[df_timeselecting_zzA500['weight'] != 0]
            df_timeselecting_zzA500_pro = df_timeselecting_zzA500[df_timeselecting_zzA500['weight'] != 0]
            df_timeselecting_zz500 = df_timeselecting_zz500[df_timeselecting_zz500['weight'] != 0]
            df_timeselecting_hs300 = df_timeselecting_hs300[df_timeselecting_hs300['weight'] != 0]
            df_timeselecting_hs300_pro = df_timeselecting_hs300_pro[df_timeselecting_hs300_pro['weight'] != 0]

        elif combine_value == 0.5:
            # When combine_value = 0.5, return original portfolios with consistent column names
            df_timeselecting_zz500 = df_500.copy()
            df_timeselecting_zz500_pro = df_500.copy()
            df_timeselecting_zzA500 = df_A500.copy()
            df_timeselecting_hs300 = df_300.copy()
            df_timeselecting_hs300_pro = df_300.copy()
            df_timeselecting_zzA500_pro = df_A500.copy()
            # Ensure column names are consistent
            df_timeselecting_zzA500_pro = df_timeselecting_zzA500_pro[['code', 'weight']]
            df_timeselecting_zzA500 = df_timeselecting_zzA500[['code', 'weight']]
            df_timeselecting_zz500 = df_timeselecting_zz500[['code', 'weight']]
            df_timeselecting_zz500_pro = df_timeselecting_zz500_pro[['code', 'weight']]
            df_timeselecting_hs300 = df_timeselecting_hs300[['code', 'weight']]
            df_timeselecting_hs300_pro = df_timeselecting_hs300_pro[['code', 'weight']]
        else:
            # When combine_value < 0.5
            # Rename weight columns
            df_300 = df_300.rename(columns={'weight': 'weight_hs300'})
            df_500 = df_500.rename(columns={'weight': 'weight_zz500'})
            df_A500 = df_A500.rename(columns={'weight': 'weight_zzA500'})
            
            # Merge DataFrames
            df_merged = df_300.merge(df_500, on='code', how='outer')
            
            # Fill NaN values with 0
            df_merged = df_merged.fillna(0)
            
            # Calculate new portfolios
            df_timeselecting_zz500_pro = pd.DataFrame()
            df_timeselecting_zz500_pro['code'] = df_merged['code']
            df_timeselecting_zz500_pro['weight'] = (1 - combine_value) * df_merged['weight_hs300'] + combine_value * df_merged['weight_zz500']

            df_timeselecting_zz500 = pd.DataFrame()
            df_timeselecting_zz500['code'] = df_merged['code']
            df_timeselecting_zz500['weight'] =0.3 * df_merged['weight_hs300'] + 0.7*df_merged['weight_zz500']

            # df_timeselecting_hs300 and df_timeselecting_hs300_pro are just df_300
            df_timeselecting_hs300 = df_300.copy()
            df_timeselecting_hs300_pro = df_300.copy()
            df_timeselecting_zzA500=df_A500.copy()
            df_timeselecting_zzA500_pro = df_A500.copy()
            # Ensure column names are consistent
            df_timeselecting_zzA500_pro.columns = ['code', 'weight']
            df_timeselecting_hs300.columns=['code','weight']
            df_timeselecting_hs300_pro.columns=['code','weight']
            df_timeselecting_zzA500.columns=['code','weight']
            # Remove rows with zero weights
            df_timeselecting_zz500 = df_timeselecting_zz500[df_timeselecting_zz500['weight'] != 0]
        df_timeselecting_zz500_pro['portfolio_name'] = 'Timeselecting_zz500_pro'
        df_timeselecting_zz500['portfolio_name'] = 'Timeselecting_zz500'
        df_timeselecting_zzA500_pro['portfolio_name'] = 'Timeselecting_zzA500_pro'
        df_timeselecting_zzA500['portfolio_name'] = 'Timeselecting_zzA500'
        df_timeselecting_hs300['portfolio_name'] = 'Timeselecting_hs300'
        df_timeselecting_hs300_pro['portfolio_name'] = 'Timeselecting_hs300_pro'
        return df_timeselecting_hs300, df_timeselecting_hs300_pro, df_timeselecting_zz500,df_timeselecting_zz500_pro, df_timeselecting_zzA500,df_timeselecting_zzA500_pro
    def etf_portfolio_construction(self):
        combine_value = self.timeselecting_signalWithdraw()
        df_hszz=pd.DataFrame()
        df_szzz=pd.DataFrame()
        if combine_value>0.5:
            if self.target_date<='2023-09-20':
                code_list_hszz = ['159845.SZ']
                code_list_szzz = ['159845.SZ']
            else:
                code_list_hszz = ['159531.SZ']
                code_list_szzz = ['159531.SZ']
        else:
            code_list_hszz=['510300.SH']
            code_list_szzz=['510050.SH']
        df_hszz['code']=code_list_hszz
        df_szzz['code']=code_list_szzz
        df_hszz['weight']=1
        df_szzz['weight']=1
        df_hszz['portfolio_name']='Timeselecting_ETF_hs300'
        df_szzz['portfolio_name']='Timeselecting_ETF_sz50'
        df_hszz['valuation_date']=self.target_date
        df_szzz['valuation_date']=self.target_date
        return df_hszz,df_szzz
    def future_finding(self):
        """
        找到target_date所在月份的第三个星期五，并根据比较结果返回相应的月份代码
        
        Returns:
            str: 如果target_date小于第三个星期五，返回YYMM格式；如果大于，返回下个月MM
        """
        # 将target_date转换为datetime对象
        if isinstance(self.target_date, str):
            target_dt = datetime.strptime(self.target_date, '%Y-%m-%d')
        else:
            target_dt = self.target_date
        
        year = target_dt.year
        month = target_dt.month
        
        # 找到该月第一个星期五
        first_day = datetime(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        
        # 计算第三个星期五
        third_friday = first_friday + timedelta(days=14)
        
        # 检查第三个星期五是否为工作日
        if not gt.is_workday(third_friday.strftime('%Y-%m-%d')):
            # 如果不是工作日，找到下一个工作日
            third_friday = gt.next_workday_calculate(third_friday.strftime('%Y-%m-%d'))
            third_friday = datetime.strptime(third_friday, '%Y-%m-%d')
        
        # 比较target_date和第三个星期五
        if target_dt < third_friday:
            # 返回当前年份的后两位 + 当月的月份 (YYMM)
            return f"{str(year)[-2:]}{month:02d}"
        else:
            # 返回下个月的月份 (MM)
            if month < 12:
                next_month = month + 1
                next_year = year
            else:
                next_month = 1
                next_year = year + 1
            return f"{str(next_year)[-2:]}{next_month:02d}"
    
    def future_portfolio_construction(self):
        combine_value = self.timeselecting_signalWithdraw()
        future_num=self.future_finding()
        df_hszz = pd.DataFrame()
        df_szzz = pd.DataFrame()
        code_list_hszz = ['IM' + str(future_num), 'IF' + str(future_num)]
        code_list_szzz = ['IM' + str(future_num), 'IH' + str(future_num)]
        if combine_value > 0.5:
            weight_list_hszz=[1,-1]
            weight_list_szzz = [1, -1]
        elif combine_value ==0.5:
            weight_list_szzz = [0, 0]
            weight_list_hszz = [0, 0]
        else:
            weight_list_hszz = [-1, 1]
            weight_list_szzz = [-1, 1]
        df_hszz['code'] = code_list_hszz
        df_szzz['code'] = code_list_szzz
        df_hszz['weight'] = weight_list_hszz
        df_szzz['weight'] = weight_list_szzz
        df_hszz['portfolio_name'] = 'Timeselecting_future_hs300'
        df_szzz['portfolio_name'] = 'Timeselecting_future_sz50'
        df_hszz['valuation_date'] = self.target_date
        df_szzz['valuation_date'] = self.target_date
        return df_hszz, df_szzz
    def future_portfolio_construction_zzA500(self):
        combine_value = self.timeselecting_signalWithdraw2()
        future_num=self.future_finding()
        df_hszz = pd.DataFrame()
        code_list_hszz = ['IC' + str(future_num), 'IF' + str(future_num)]
        if combine_value > 0.5:
            weight_list_hszz=[1,0]
        elif combine_value < 0.5:
            weight_list_hszz = [0, 1]
        else:
            raise ValueError
        df_hszz['code'] = code_list_hszz
        df_hszz['weight'] = weight_list_hszz
        df_hszz['portfolio_name'] = 'Timeselecting_future_A500'
        df_hszz['valuation_date'] = self.target_date
        return df_hszz

    def signal_generator(self,index_type):
        available_date = gt.last_workday_calculate(self.target_date)
        df_index = gt.indexData_withdraw(index_type, '2022-07-20', available_date, ['close'], False)
        mean = np.mean(df_index['close'].tolist()[-5:])
        last_index = df_index['close'].tolist()[-1]
        if mean <= last_index:
            return True
        else:
            return False

    def decision_30050(self,rolling_window):
        available_date=gt.last_workday_calculate(self.target_date)
        # 将available_date转换为datetime对象，减去一年，再转换回字符串格式
        available_dt = datetime.strptime(available_date, '%Y-%m-%d')
        start_dt = available_dt - timedelta(days=400)
        start_date = start_dt.strftime('%Y-%m-%d')
        df_index = gt.indexData_withdraw(None,start_date,available_date, ['pct_chg'], False)
        df_index = gt.sql_to_timeseries(df_index)
        df_index = df_index[['valuation_date', '000016.SH', '000300.SH']]
        df_index.set_index('valuation_date', inplace=True, drop=True)
        df_return = df_index.rolling(rolling_window).sum()
        df_return.dropna(inplace=True)
        df_return['difference'] = df_return['000016.SH'] - df_return['000300.SH']
        df_return['quantile_0.1'] = df_return['difference'].rolling(252).quantile(0.1)
        df_return['quantile_0.9'] = df_return['difference'].rolling(252).quantile(0.9)
        df_return.dropna(inplace=True)
        df_return['signal_momentum'] = 0
        df_return.loc[df_return['difference'] >= 0, ['signal_momentum']] = '000016.SH'
        df_return.loc[df_return['difference'] < 0, ['signal_momentum']] = '000300.SH'
        df_return.loc[df_return['difference'] < df_return['quantile_0.1'], ['signal_momentum']] = '000016.SH'
        df_return.loc[df_return['difference'] > df_return['quantile_0.9'], ['signal_momentum']] = '000300.SH'
        df_return.reset_index(inplace=True)
        signal=df_return[df_return['valuation_date']==available_date]['signal_momentum'].tolist()[0]
        return signal
    def future_portfolio_construction_pro(self):
        combine_value = self.timeselecting_signalWithdraw()
        future_num=self.future_finding()
        df_hszz = pd.DataFrame()
        df_szzz = pd.DataFrame()
        code_list_hszz = ['IM' + str(future_num), 'IF' + str(future_num)]
        code_list_szzz = ['IM' + str(future_num), 'IH' + str(future_num)]
        if combine_value > 0.5:
            signal=self.signal_generator('中证1000')
            if signal==True:
                weight_list_hszz = [1, 0]
                weight_list_szzz = [1, 0]
            else:
                weight_list_hszz = [1, -1]
                weight_list_szzz = [1, -1]
        elif combine_value ==0.5:
            signal=self.signal_generator('中证1000')
            signal1 = self.signal_generator('沪深300')
            signal2 = self.signal_generator('上证50')
            if signal==True and signal1==True:
                weight_list_hszz = [0.5,0.5]
            if signal==True and signal2==True:
                weight_list_szzz = [0.5,0.5]
            if signal==True and signal2==False:
                weight_list_szzz = [0.5, 0]
            if signal==True and signal1==False:
                weight_list_hszz = [0.5, 0]
            if signal==False and signal1==True:
                weight_list_hszz = [0, 0.5]
            if signal==False and signal2==True:
                weight_list_szzz = [0, 0.5]
            if signal==False and signal2==False:
                weight_list_szzz = [0, 0]
            if signal == False and signal1 == False:
                weight_list_hszz = [0, 0]
            else:
                print(signal,signal1,signal2)
        else:
            signal1 = self.signal_generator('沪深300')
            signal2=self.signal_generator('上证50')
            if signal1==True:
                weight_list_hszz = [0, 1]
            else:
                weight_list_hszz=[-1,1]
            if signal2==True:
               weight_list_szzz = [0, 1]
            else:
                weight_list_szzz=[-1,1]
        df_hszz['code'] = code_list_hszz
        df_szzz['code'] = code_list_szzz
        df_hszz['weight'] = weight_list_hszz
        df_szzz['weight'] = weight_list_szzz
        df_hszz['portfolio_name'] = 'Timeselecting_future_hs300_pro'
        df_szzz['portfolio_name'] = 'Timeselecting_future_sz50_pro'
        df_hszz['valuation_date'] = self.target_date
        df_szzz['valuation_date'] = self.target_date
        return df_hszz, df_szzz
    def future_portfolio_construction_long(self):
        combine_value = self.timeselecting_signalWithdraw()
        future_num=self.future_finding()
        df_final = pd.DataFrame()
        if combine_value > 0.5:
            code_list = ['IM' + str(future_num), 'IH' + str(future_num)]
            signal=self.signal_generator('中证1000')
            if signal==True:
                weight_list = [1, 0]
            else:
                weight_list = [0, 0]
        elif combine_value ==0.5:
            signal=self.signal_generator('中证1000')
            signal1 = self.signal_generator('上证50')
            code_list = ['IM' + str(future_num), 'IH' + str(future_num)]
            if signal==True and signal1==True:
                weight_list = [0.5,0.5]
            if signal==True and signal1==False:
                weight_list = [0.5, 0]
            if signal==False and signal1==True:
                weight_list = [0, 0.5]
            if signal == False and signal1 == False:
                weight_list = [0, 0]
            else:
                print(signal,signal1)
        else:
            code_list = ['IM' + str(future_num), 'IH' + str(future_num)]
            signal = self.signal_generator('上证50')
            if signal == True:
                weight_list = [0, 1]
            else:
                weight_list = [0, 0]
        df_final['code'] = code_list
        df_final['weight'] = weight_list
        df_final['portfolio_name'] = 'Timeselecting_future'
        df_final['valuation_date'] = self.target_date
        return df_final

    def portfolio_info_saving(self):
        """
        读取portfolio.yaml配置文件中的index_mapping，并转换为DataFrame

        Returns:
            pd.DataFrame: DataFrame with columns 'portfolio_name' and 'index_type'
        """
        # Get the absolute path to the workspace
        workspace_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(workspace_path, 'config_project', 'portfolio.yaml')

        # Read portfolio configuration from yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            portfolio_config = yaml.safe_load(f)

        # Check if index_mapping exists
        if 'index_mapping' not in portfolio_config:
            raise ValueError("index_mapping not found in configuration")

        # Extract index_mapping data
        index_mapping = portfolio_config['index_mapping']

        # Convert to DataFrame
        portfolio_info_list = []
        for index_type, portfolio_info in index_mapping.items():
            portfolio_names = portfolio_info['portfolio_names']
            for portfolio_name in portfolio_names:
                portfolio_info_list.append({
                    'score_name': portfolio_name,
                    'index_type': index_type
                })
        df_portfolio_info = pd.DataFrame(portfolio_info_list)
        df_portfolio_info.replace('中性', None, inplace=True)
        df_portfolio_info['mode_type'] = 'mode_v1'
        df_portfolio_info['base_score'] = None
        df_portfolio_info['valuation_date'] = self.target_date
        df_portfolio_info['update_time']=self.now
        inputpath_sql = self.sql_path_withdraw()
        sm = gt.sqlSaving_main(inputpath_sql, 'portfolio_info')
        sm.df_to_sql(df_portfolio_info)
    def indexClose_withdraw(self):
        df_index = gt.indexData_withdraw('中证1000', self.available_date, self.available_date, ['close'], False)
        index_close=df_index['close'].tolist()[0]
        return index_close
    def portfolio_to_holding(self,df,df_stock, df_hstock, df_etf, df_option, df_future, df_convertible_bond, df_index,index_close):
        df['valuation_date'] = self.available_date
        df_info, df_detail = gt.portfolio_analyse_manual(df, df_stock, df_hstock, df_etf, df_option, df_future, df_convertible_bond, df_index,index_close*200*2, cost_stock=0, cost_etf=0,
                                                         cost_future=0, cost_option=0, cost_convertiblebond=0,
                                                         realtime=False)
        df_detail=df_detail[['valuation_date','code','quantity','portfolio_name']]
        df_detail['valuation_date']=self.target_date
        return df_detail
    def portfolio_saving_main(self):
        df_etfhs,df_etfsz=self.etf_portfolio_construction()
        df_futurehs,df_futuresz=self.future_portfolio_construction()
        df_futureA500 = self.future_portfolio_construction_zzA500()
        df_futurehs_long=self.future_portfolio_construction_long()
        df_futurehs_pro, df_futuresz_pro = self.future_portfolio_construction_pro()
        inputpath_sql=self.sql_path_withdraw()
        sm=gt.sqlSaving_main(inputpath_sql, 'Portfolio', delete=True)
        sm2=gt.sqlSaving_main(inputpath_sql, 'portfolio_holding', delete=True)
        df_stock, df_hstock, df_etf, df_option, df_future, df_convertible_bond, df_index = gt.mktData_withdraw(
            self.available_date, self.available_date, False)
        index_close=self.indexClose_withdraw()
        for df in [df_etfhs,df_etfsz,df_futurehs,df_futuresz,df_futurehs_pro, df_futuresz_pro,df_futurehs_long,df_futureA500]:
            portfolio_name=df['portfolio_name'].tolist()[0]
            df['valuation_date']=self.target_date
            df['update_time']=self.now
            sm.df_to_sql(df,'portfolio_name',portfolio_name)
        for df2 in [df_futurehs,df_futuresz,df_futurehs_pro, df_futuresz_pro,df_futurehs_long,df_futureA500]:
            portfolio_name = df2['portfolio_name'].tolist()[0]
            df2=self.portfolio_to_holding(df2,df_stock, df_hstock, df_etf, df_option, df_future, df_convertible_bond, df_index,index_close)
            df2['update_time']=self.now
            sm2.df_to_sql(df2,'portfolio_name',portfolio_name)
        self.portfolio_info_saving()
    def portfolio_saving_bu(self):
        #df_timeselecting_hs300, df_timeselecting_hs300_pro, df_timeselecting_zz500,df_timeselecting_zz500_pro, df_timeselecting_zzA500,df_timeselecting_zzA500_pro = self.stock_portfolio_construction()
        df_futurehs=self.future_portfolio_construction_mix()
        inputpath_sql=self.sql_path_withdraw()
        sm=gt.sqlSaving_main(inputpath_sql, 'Portfolio', delete=True)
        for df in [df_futurehs]:
            portfolio_name=df['portfolio_name'].tolist()[0]
            df['valuation_date']=self.target_date
            df['update_time']=self.now
            sm.df_to_sql(df,'portfolio_name',portfolio_name)
        self.portfolio_info_saving()

if __name__ == "__main__":

    # df_singal=timeselecting_signalWithdraw()
    # df_return=decision_30050(5)
    # analyse(df_return, df_singal)
    # #'2022-07-25'
    working_days_list=gt.working_days_list('2025-03-10','2025-12-24')
    for date in working_days_list:
         print(date)
         pu=portfolio_updating(date)
         pu.portfolio_saving_main()