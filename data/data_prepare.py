"""
数据准备模块 (data_prepare)

本模块负责从数据库获取各类原始数据，包括：
- 利率数据（Shibor、国债、国开债等）
- 指数数据（成交额、成交量、换手率、收盘价等）
- 股票数据（收盘价、收益率等）
- 宏观数据（CPI、PPI、PMI、M1M2等）
- 资金数据（龙虎榜、融资融券、大单等）
- 其他数据（VIX、RR评分等）

注意：初始化时会自动将start_date向前推3年，以确保有足够的历史数据用于计算

作者: TimeSelecting Team
版本: v3.0
"""

import os
import pandas as pd
from matplotlib import pyplot as plt

import global_setting.global_dic as glv
import sys
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
import yaml
config_path=glv.get('config_path')

class data_prepare:
    """
    数据准备类
    
    负责从数据库获取各类原始数据。初始化时会自动将start_date向前推3年，
    以确保有足够的历史数据用于技术指标计算。
    
    Attributes:
    -----------
    start_date : str
        实际数据开始日期（输入日期向前推3年），格式为 'YYYY-MM-DD'
    end_date : str
        数据结束日期，格式为 'YYYY-MM-DD'
    working_days_list : list
        从start_date到end_date的所有工作日列表
    """
    
    def __init__(self, start_date, end_date):
        """
        初始化数据准备类
        
        Parameters:
        -----------
        start_date : str
            输入的开始日期，格式为 'YYYY-MM-DD'
            实际使用的开始日期会向前推3年
        end_date : str
            结束日期，格式为 'YYYY-MM-DD'
        """
        start_date=pd.to_datetime(start_date) - pd.DateOffset(years=3)
        self.start_date=start_date.strftime('%Y-%m-%d')
        self.end_date=end_date
        self.working_days_list=gt.working_days_list(self.start_date,self.end_date)

    def _check_working_days_completeness(self, df, func_name):
        """
        检查DataFrame是否包含working_days_list中的所有工作日
        
        如果数据不完整，直接抛出ValueError异常
        
        根据config_checking.yaml中的effective_start_date动态调整检查的工作日列表：
        - 如果self.start_date <= effective_start_date，则使用effective_start_date到self.end_date的工作日列表
        - 否则，使用self.working_days_list
        
        Parameters:
        -----------
        df : pd.DataFrame
            要检查的DataFrame
        func_name : str
            函数名称，用于错误提示和查找配置文件中的effective_start_date
        
        Raises:
        -------
        ValueError
            当DataFrame为空、缺少valuation_date列或缺少工作日数据时抛出异常
        
        Note:
        -----
        本方法使用df.copy()创建副本，不会修改原始DataFrame的任何格式
        """
        if df is None or df.empty:
            raise ValueError(f"{func_name}: DataFrame为空")
        
        if 'valuation_date' not in df.columns:
            raise ValueError(f"{func_name}: DataFrame缺少valuation_date列")
        
        # 创建DataFrame副本，避免修改原始数据
        df_copy = df.copy()
        
        # 从配置文件读取effective_start_date
        effective_start_date = self._get_effective_start_date(func_name)
        
        # 根据effective_start_date决定使用哪个工作日列表
        if effective_start_date:
            # 将日期字符串转换为datetime进行比较
            start_datetime = pd.to_datetime(self.start_date)
            effective_datetime = pd.to_datetime(effective_start_date)
            
            if start_datetime <= effective_datetime:
                # 如果self.start_date <= effective_start_date，使用effective_start_date到self.end_date的工作日列表
                working_days_to_check = gt.working_days_list(effective_start_date, self.end_date)
            else:
                # 否则使用self.working_days_list
                working_days_to_check = self.working_days_list
        else:
            # 如果配置文件中没有找到，使用默认的self.working_days_list
            working_days_to_check = self.working_days_list
        
        # 在副本上确保日期格式正确（不修改原始df）
        df_copy['valuation_date'] = pd.to_datetime(df_copy['valuation_date'])
        df_dates = set(df_copy['valuation_date'].dt.strftime('%Y-%m-%d').tolist())
        
        # 检查是否包含所有工作日
        missing_dates = set(working_days_to_check) - df_dates
        
        if missing_dates:
            # 获取最新数据日期
            latest_date = df_copy['valuation_date'].max().strftime('%Y-%m-%d')
            missing_count = len(missing_dates)
            # 如果缺失日期太多，只显示前10个
            if missing_count <= 10:
                missing_str = ', '.join(sorted(list(missing_dates)))
            else:
                missing_str = ', '.join(sorted(list(missing_dates))[:10]) + f' ... (共{missing_count}个)'
            raise ValueError(
                f"{func_name}: 数据不完整，缺少{missing_count}个工作日，最新更新日期为{latest_date}。"
                f"缺失日期: {missing_str}"
            )
    
    def _get_effective_start_date(self, func_name):
        """
        从config_checking.yaml中获取函数的effective_start_date
        
        Parameters:
        -----------
        func_name : str
            函数名称
        
        Returns:
        --------
        str or None
            有效开始日期，格式为'YYYY-MM-DD'，如果未找到则返回None
        """
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_check', 'config_checking.yaml')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'data_prepare_functions' not in config:
                return None
            
            # 先检查parameterized_functions（有参数的函数）
            if 'parameterized_functions' in config['data_prepare_functions']:
                for func_key, func_config in config['data_prepare_functions']['parameterized_functions'].items():
                    if func_key == func_name:
                        return func_config.get('effective_start_date')
            
            # 再检查parameterless_functions（无参数的函数）
            if 'parameterless_functions' in config['data_prepare_functions']:
                for func_config in config['data_prepare_functions']['parameterless_functions']:
                    if isinstance(func_config, dict):
                        if func_config.get('name') == func_name:
                            return func_config.get('effective_start_date')
                    elif func_config == func_name:
                        # 如果配置是字符串格式，返回None（需要配置文件中定义effective_start_date）
                        return None
            
            return None
        except Exception as e:
            # 如果读取配置文件失败，返回None，使用默认的working_days_list
            return None
    
    def _check_primary_key_conflicts(self, df, func_name):
        """
        检查DataFrame中主键（code和valuation_date）的冲突
        
        如果发现主键冲突（重复的code和valuation_date组合），打印出冲突的行
        
        Parameters:
        -----------
        df : pd.DataFrame
            要检查的DataFrame
        func_name : str
            函数名称，用于错误提示
        
        Note:
        -----
        本方法不会修改原始DataFrame，只是检查和打印冲突信息
        """
        if df is None or df.empty:
            return
        
        # 检查是否包含主键列
        if 'code' not in df.columns or 'valuation_date' not in df.columns:
            return
        
        # 检查重复的主键组合
        duplicate_mask = df.duplicated(subset=['code', 'valuation_date'], keep=False)
        
        if duplicate_mask.any():
            duplicate_df = df[duplicate_mask].copy()
            duplicate_df = duplicate_df.sort_values(['code', 'valuation_date'])
            print(f"\n{'='*80}")
            print(f"{func_name}: 发现主键冲突（code和valuation_date重复）")
            print(f"冲突行数: {len(duplicate_df)}")
            print(f"{'='*80}")
            print(duplicate_df.to_string())
            print(f"{'='*80}\n")
    
    def rename_code_by_folder_wind(self, new_name):
        """
        根据rename_mapping.yaml中的配置，通过new_name查找对应的original_name

        Args:
            new_name (str): 新的名称
        Returns:
            str: 对应的原始名称(original_name)
        """
        # 读取rename_mapping.yaml文件
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config_project','rename_mapping_wind.yaml')
        with open(yaml_path, 'r', encoding='utf-8') as f:
            mapping_config = yaml.safe_load(f)

        # 根据new_name查找对应的original_name
        for mapping in mapping_config['rename_mapping']:
            if mapping['new_name'] == new_name:
                return mapping['original_name']

        # 如果没有找到匹配的映射，返回警告并返回None
        print(f"Warning: No mapping found for new_name: {new_name}")
        return None
    def raw_shibor(self, period):
        """
        获取Shibor（上海银行间同业拆放利率）数据
        
        Parameters:
        -----------
        period : str
            期限，可选值：'2W'（2周）或 '9M'（9个月）
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - Shibor_{period}: Shibor利率值
        
        Raises:
        -------
        ValueError
            当period不是'2W'或'9M'时抛出异常
        """
        inputpath = glv.get('raw_Shibor')
        signal_name='Shibor_'+str(period)
        ori_name=self.rename_code_by_folder_wind(signal_name)
        inputpath = str(
            inputpath) + f" Where code='{ori_name}' And trade_date between '{self.start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1 = df1[['trade_date', 'close']]
        df1.columns = ['valuation_date', signal_name]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1=df1[df1['valuation_date'].isin(self.working_days_list)]
        if signal_name not in df1.columns:
            raise ValueError('period must be 2W or 9M')
        df1 = df1[['valuation_date', signal_name]]
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_shibor')
        return df1
    def raw_bond(self, period):
        """
        获取国债收益率数据
        
        Parameters:
        -----------
        period : str
            期限，可选值：'3Y'（3年）或 '10Y'（10年）
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - CGB_{period}: 国债收益率值
        
        Raises:
        -------
        ValueError
            当period不是'3Y'或'10Y'时抛出异常
        """
        inputpath = glv.get('raw_Bond')
        signal_name = 'CGB_' + str(period)
        ori_name = self.rename_code_by_folder_wind(signal_name)
        inputpath = str(inputpath) + f" Where code='{ori_name}' And trade_date between '{self.start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1 = df1[['trade_date', 'close']]
        df1.columns = ['valuation_date', signal_name]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        if signal_name not in df1.columns:
            raise ValueError('period must be 3Y or 10Y')
        df1=df1[['valuation_date',signal_name]]
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_bond')
        return df1
    def raw_ZZGK(self, period):
        """
        获取国开债（CDBB）收益率数据
        
        Parameters:
        -----------
        period : str
            期限，可选值：'3M'、'9M'、'1Y'、'5Y'、'10Y'
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - CDBB_{period}: 国开债收益率值
        
        Raises:
        -------
        ValueError
            当period不在允许范围内时抛出异常
        """
        inputpath_ZZGK = glv.get('raw_ZZGK')
        signal_name = 'CDBB_' + str(period)
        ori_name = self.rename_code_by_folder_wind(signal_name)
        inputpath = str(inputpath_ZZGK) + f" Where code='{ori_name}' And trade_date between '{self.start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1 = df1[['trade_date', 'close']]
        df1.columns = ['valuation_date', signal_name]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        if signal_name not in df1.columns:
            raise ValueError('period must be 3M or9M or 1Y or 5Y or 10Y')
        df1=df1[['valuation_date',signal_name]]
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_ZZGK')
        return df1
    def raw_ZZZD(self, period):
        """
        获取中债中短（CMTN）收益率数据
        
        Parameters:
        -----------
        period : str
            期限，可选值：'3M'、'9M'、'5Y'
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - CMTN_{period}: 中债中短收益率值
        
        Raises:
        -------
        ValueError
            当period不在允许范围内时抛出异常
        """
        inputpath_ZZZD= glv.get('raw_ZZZD')
        signal_name = 'CMTN_' + str(period)
        ori_name = self.rename_code_by_folder_wind(signal_name)
        inputpath = str(
            inputpath_ZZZD) + f" Where code='{ori_name}' And trade_date between '{self.start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1 = df1[['trade_date', 'close']]
        df1.columns = ['valuation_date', signal_name]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        if signal_name not in df1.columns:
            raise ValueError('period must be 9M or 3M or 5Y ')
        df1=df1[['valuation_date',signal_name]]
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_ZZZD')
        return df1
    def raw_M1M2(self, signal_name):
        """
        获取货币供应量数据（M1或M2）
        
        Parameters:
        -----------
        signal_name : str
            货币类型，可选值：'M1' 或 'M2'
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - {signal_name}: 货币供应量值
        
        Raises:
        -------
        ValueError
            当signal_name不是'M1'或'M2'时抛出异常
        """
        inputpath = glv.get('raw_M1M2')
        ori_name = self.rename_code_by_folder_wind(signal_name)
        inputpath = str(
            inputpath) + f" Where code='{ori_name}' And trade_date between '{self.start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1 = df1[['trade_date', 'close']]
        df1.columns = ['valuation_date', signal_name]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        if signal_name not in df1.columns:
            raise ValueError('type must be M1 or M2')
        df1=df1[['valuation_date',signal_name]]
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_M1M2')
        return df1
    def raw_usdx(self):
        """
        获取美元指数（USDX）数据
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - USDX: 美元指数值
        """
        inputpath1 = glv.get('raw_USDX')
        signal_name='USDX'
        ori_name = self.rename_code_by_folder_wind(signal_name)
        inputpath = str(
            inputpath1) + f" Where code='{ori_name}' And trade_date between '{self.start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1 = df1[['trade_date', 'close']]
        df1.columns = ['valuation_date', signal_name]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_usdx')
        return df1

    def raw_index_earningsyield(self):
        """
        获取指数盈利收益率（Earnings Yield）数据
        
        计算沪深300与国证2000的盈利收益率差值
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - difference_earningsyield: 盈利收益率差值（沪深300 - 国证2000）
        """
        inputpath = glv.get('raw_indexFactor')
        inputpath = str(
            inputpath) + f" Where type='earningsyield' And valuation_date between '{self.start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1 = df1[['valuation_date', 'organization', 'value']]
        df = gt.sql_to_timeseries(df1)
        df['valuation_date'] = pd.to_datetime(df['valuation_date'])
        df['valuation_date'] = df['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df = df[['valuation_date', 'hs300', 'gz2000', 'zz1000']]
        df_final = df.dropna()
        df_final['difference_earningsyield'] = df_final['hs300'] - df_final['gz2000']
        df_final = df_final[['valuation_date', 'difference_earningsyield']]
        # 检查工作日完整性
        self._check_working_days_completeness(df_final, 'raw_index_earningsyield')
        return df_final
    def raw_index_growth(self):
        """
        获取指数成长性（Growth）数据
        
        计算沪深300与国证2000、中证1000的成长性差值
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - difference_Growth: 成长性差值（沪深300 - 国证2000 - 中证1000）
        """
        inputpath = glv.get('raw_indexFactor')
        inputpath = str(
            inputpath) + f" Where type='growth' And valuation_date between '{self.start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1 = df1[['valuation_date', 'organization', 'value']]
        df = gt.sql_to_timeseries(df1)
        df['valuation_date'] = pd.to_datetime(df['valuation_date'])
        df['valuation_date'] = df['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df = df[['valuation_date', 'hs300', 'gz2000', 'zz1000']]
        df_final = df.dropna()
        df_final['difference_Growth'] = df_final['hs300'] - df_final['gz2000'] - df_final['zz1000']
        df_final = df_final[['valuation_date', 'difference_Growth']]
        # 检查工作日完整性
        self._check_working_days_completeness(df_final, 'raw_index_growth')
        return df_final
    def raw_CPI_withdraw(self):
        """
        获取CPI（居民消费价格指数）数据
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - CPI: CPI指数值
        """
        inputpath = glv.get('raw_CPI')
        signal_name = 'CPI'
        ori_name = self.rename_code_by_folder_wind(signal_name)
        inputpath = str(
            inputpath) + f" Where code='{ori_name}' And trade_date between '{self.start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1 = df1[['trade_date', 'close']]
        df1.columns = ['valuation_date', signal_name]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_CPI_withdraw')
        return df1

    def raw_PPI_withdraw(self):
        """
        获取PPI（工业生产者出厂价格指数）数据
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - PPI: PPI指数值
        """
        inputpath = glv.get('raw_PPI')
        signal_name = 'PPI'
        ori_name = self.rename_code_by_folder_wind(signal_name)
        inputpath = str(
            inputpath) + f" Where code='{ori_name}' And trade_date between '{self.start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1 = df1[['trade_date', 'close']]
        df1.columns = ['valuation_date', signal_name]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_PPI_withdraw')
        return df1
    def raw_PMI_withdraw(self):
        """
        获取PMI（采购经理人指数）数据
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - PMI: PMI指数值
        """
        inputpath = glv.get('raw_PMI')
        signal_name = 'PMI'
        ori_name = self.rename_code_by_folder_wind(signal_name)
        inputpath = str(
            inputpath) + f" Where code='{ori_name}' And trade_date between '{self.start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1 = df1[['trade_date', 'close']]
        df1.columns = ['valuation_date', signal_name]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_PMI_withdraw')
        return df1

    def raw_PS_withdraw(self):
        """
        获取PMI（采购经理人指数）数据

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - PMI: PMI指数值
        """
        inputpath = 'D:\Data_prepared_new\PS.xlsx'
        df1 = pd.read_excel(inputpath)
        df1.rename(columns={'日期':'valuation_date'},inplace=True)
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        df1['value']=df1['上证50']-df1['中证1000']
        df1=df1[['valuation_date','value']]
        # 检查工作日完整性
        #self._check_working_days_completeness(df1, 'raw_PMI_withdraw')
        return df1
    def raw_PCF_withdraw(self):
        """
        获取PMI（采购经理人指数）数据

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - PMI: PMI指数值
        """
        inputpath = 'D:\Data_prepared_new\PCF.xlsx'
        df1 = pd.read_excel(inputpath)
        df1.rename(columns={'日期':'valuation_date'},inplace=True)
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        # 如果中证2000列存在，用中证1000填充其None值
        if '中证2000' in df1.columns and '中证1000' in df1.columns:
            df1['中证2000'] = df1['中证2000'].fillna(df1['中证1000'])
        df1['value']=df1['上证50']-df1['中证2000']
        df1=df1[['valuation_date','value']]
        # 检查工作日完整性
        #self._check_working_days_completeness(df1, 'raw_PMI_withdraw')
        return df1
    def raw_Earning_withdraw(self):
        """
        获取PMI（采购经理人指数）数据

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - PMI: PMI指数值
        """
        inputpath = 'D:\Data_prepared_new\\Earning.xlsx'
        df1 = pd.read_excel(inputpath)
        df1.rename(columns={'日期':'valuation_date'},inplace=True)
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        # 如果中证2000列存在，用中证1000填充其None值
        if '中证2000' in df1.columns and '中证1000' in df1.columns:
            df1['中证2000'] = df1['中证2000'].fillna(df1['中证1000'])
        df1['big']=(df1['上证50']+df1['沪深300'])/2
        df1['small']=(df1['中证1000']+df1['中证2000'])/2
        df1['value']=df1['big']-df1['small']
        df1=df1[['valuation_date','value']]
        # 检查工作日完整性
        #self._check_working_days_completeness(df1, 'raw_PMI_withdraw')
        return df1
    def raw_NetProfit_withdraw(self):
        """
        获取PMI（采购经理人指数）数据

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - PMI: PMI指数值
        """
        inputpath = 'D:\Data_prepared_new\\NetProfit.xlsx'
        df1 = pd.read_excel(inputpath)
        df1.rename(columns={'日期':'valuation_date'},inplace=True)
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        # 如果中证2000列存在，用中证1000填充其None值
        if '中证2000' in df1.columns and '中证1000' in df1.columns:
            df1['中证2000'] = df1['中证2000'].fillna(df1['中证1000'])
        df1['big']=(df1['上证50']+df1['沪深300'])/2
        df1['small']=(df1['中证1000']+df1['中证2000'])/2
        df1['value']=df1['big']-df1['small']
        df1=df1[['valuation_date','value']]
        # 检查工作日完整性
        #self._check_working_days_completeness(df1, 'raw_PMI_withdraw')
        return df1
    def raw_ROE_withdraw(self):
        """
        获取PMI（采购经理人指数）数据

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - PMI: PMI指数值
        """
        inputpath = 'D:\Data_prepared_new\\ROE.xlsx'
        df1 = pd.read_excel(inputpath)
        df1.rename(columns={'日期':'valuation_date'},inplace=True)
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        # 如果中证2000列存在，用中证1000填充其None值
        if '中证2000' in df1.columns and '中证1000' in df1.columns:
            df1['中证2000'] = df1['中证2000'].fillna(df1['中证1000'])
        df1['big']=(df1['上证50']+df1['沪深300'])/2
        df1['small']=(df1['中证1000']+df1['中证2000'])/2
        df1['value']=df1['big']-df1['small']
        df1=df1[['valuation_date','value']]
        # 检查工作日完整性
        #self._check_working_days_completeness(df1, 'raw_PMI_withdraw')
        return df1
    def raw_socialfinance(self):
        """
        获取CPI（居民消费价格指数）数据

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - CPI: CPI指数值
        """
        inputpath = glv.get('raw_socialfinance')
        signal_name = 'SocialFinance'
        ori_name = self.rename_code_by_folder_wind(signal_name)
        inputpath = str(
            inputpath) + f" Where code='{ori_name}' And trade_date between '{self.start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1 = df1[['trade_date', 'close']]
        df1.columns = ['valuation_date', signal_name]
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        # 检查工作日完整性
        #self._check_working_days_completeness(df1, 'raw_socialfinance')
        return df1
    def raw_CopperGold(self):
        inputpath = glv.get('raw_CopperGold')
        signal_name = 'Copper'
        ori_name = self.rename_code_by_folder_wind(signal_name)
        inputpath1 = str(
            inputpath) + f" Where code='{ori_name}' And trade_date between '{self.start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath1, config_path)
        df1.fillna(method='ffill', inplace=True)
        signal_name2 = 'Gold'
        ori_name2 = self.rename_code_by_folder_wind(signal_name2)
        inputpath2 = str(
            inputpath) + f" Where code='{ori_name2}' And trade_date between '{self.start_date}' and '{self.end_date}'"
        df2 = gt.data_getting(inputpath2, config_path)
        df1 = df1[['trade_date', 'close']]
        df1.columns = ['valuation_date', signal_name]
        df2 = df2[['trade_date', 'close']]
        df2.columns = ['valuation_date', signal_name2]
        df2.fillna(method='ffill',inplace=True)
        df_final=df1.merge(df2,on='valuation_date',how='left')
        df_final['value']=df_final['Copper']/df_final['Gold']
        df_final= df_final[['valuation_date', 'value']]
        df_final.sort_values('valuation_date', ascending=True, inplace=True)
        df_final['valuation_date'] = pd.to_datetime(df_final['valuation_date'])
        df_final['valuation_date'] = df_final['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df_final = df_final[df_final['valuation_date'].isin(self.working_days_list)]
        df_final.fillna(method='ffill',inplace=True)
        df_final.reset_index(inplace=True, drop=True)
        return df_final
    def raw_BMCI(self):
        start_date=pd.to_datetime(self.start_date)- pd.DateOffset(months=2)
        start_date=gt.strdate_transfer(start_date)
        inputpath = glv.get('raw_BMCI')
        signal_name = 'BMCI'
        ori_name = self.rename_code_by_folder_wind(signal_name)
        inputpath = str(
            inputpath) + f" Where code='{ori_name}' And trade_date between '{start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1 = df1[['trade_date', 'close']]
        df1.fillna(method='ffill', inplace=True)
        df1.columns = ['valuation_date', 'value']
        df1.sort_values('valuation_date', ascending=True, inplace=True)
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1['value'] = df1['value'].rolling(28).mean()
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        df1.dropna(inplace=True)
        df1.reset_index(inplace=True, drop=True)
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_BMCI')
        return df1
    def raw_usrmb(self):
        inputpath="D:\Data_prepared_new\基准汇率_美元兑人民币.csv"
        df1=gt.readcsv(inputpath)
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1.sort_values('valuation_date', ascending=True, inplace=True)
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        df1.fillna('ffill',inplace=True)
        return df1
    def raw_usbond(self,period):
        inputpath=glv.get('raw_usbond')
        signal_name = 'usbond_' + str(period)
        ori_name = self.rename_code_by_folder_wind(signal_name)
        start_date=gt.last_workday_calculate(self.start_date)
        inputpath = str(
            inputpath) + f" Where code='{ori_name}' And trade_date between '{start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1.rename(columns={'trade_date':'valuation_date','close':signal_name},inplace=True)
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1.sort_values('valuation_date', ascending=True, inplace=True)
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        df1=df1[['valuation_date',signal_name]]
        working_days_list = df1['valuation_date'].unique().tolist()
        if self.end_date not in working_days_list:
            available_date = gt.last_workday_calculate(self.end_date)
            slice_df=df1[df1['valuation_date']==available_date]
            if len(slice_df)==0:
                print(f"usbond更新有误,没有{available_date}的数据")
                raise ValueError
            else:
                slice_df['valuation_date']=self.end_date
                df1=pd.concat([df1,slice_df])
        df1[signal_name]=df1[signal_name].shift(1)
        df1.fillna(method='bfill', inplace=True)
        df = self.raw_bond(period)
        df = df.merge(df1, on='valuation_date', how='left')
        df.fillna(method='ffill', inplace=True)
        df.dropna(inplace=True)
        df['value']=df['CGB_'+str(period)]-df[signal_name]
        df=df[['valuation_date','value']]
        self._check_working_days_completeness(df, 'raw_usbond')
        return df
    def raw_fund(self):
        hs300_sz50_etf_codes_full =  [
    # 沪深300（规模/流动性双第一梯队）
    '510300.SH', '510310.SH',
    # 上证50（规模/流动性双第一梯队）
    '510050.SH', '510850.SH'
]
        zz1000_zz2000_etf_codes_full = [
    # 中证1000（规模/流动性双第一梯队）
    '159845.SZ', '512100.SH',
    # 中证2000（规模/流动性双第一梯队）
    '563300.SH', '159531.SZ'
]
        zz500_zz1000_etf_codes_full = [
            # 中证1000（规模/流动性双第一梯队）
            '159845.SZ', '512100.SH',
            # 中证500（规模/流动性双第一梯队）
            '510500.SH', '159922.SZ'
        ]
        # 合并所有需要的ETF列表（包括历史数据需要的zz500_zz1000）
        etf_list_total=hs300_sz50_etf_codes_full+zz1000_zz2000_etf_codes_full+zz500_zz1000_etf_codes_full
        # 去重
        etf_list_total = list(set(etf_list_total))
        inputpath = glv.get('raw_fundshare')
        df_etf=gt.etfData_withdraw(self.start_date,self.end_date,['close'],False)
        inputpath = str(
            inputpath) + f" Where trade_date between '{self.start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1=df1[['ts_code','trade_date','fd_share']]
        df1.columns=['code','valuation_date','fd_share']
        df_etf=df_etf[df_etf['code'].isin(etf_list_total)]
        df1=df1[df1['code'].isin(etf_list_total)]
        df1['valuation_date']=pd.to_datetime(df1['valuation_date'])
        df1['valuation_date']=df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df_etf=df_etf.merge(df1,on=['valuation_date','code'],how='left')
        df_etf['mkt_value']=df_etf['close']*df_etf['fd_share']
        # 添加fd_share_yes列：每个code的前一天的fd_share值
        df_etf = df_etf.sort_values(['code', 'valuation_date'])
        # 按code分组，获取前一天的fd_share值
        df_etf['fd_share_yes'] = df_etf.groupby('code')['fd_share'].shift(1)
        df_etf['mkt_value_yes'] = df_etf.groupby('code')['mkt_value'].shift(1)
        df_etf.dropna(inplace=True)
        df_etf['money_flow']=df_etf['fd_share']-df_etf['fd_share_yes']
        # 分别筛选两个列表的数据
        df_big = df_etf[df_etf['code'].isin(hs300_sz50_etf_codes_full)].copy()
        
        # 根据日期选择不同的small列表
        # 2023-10-01之前使用zz500_zz1000_etf_codes_full，之后使用zz1000_zz2000_etf_codes_full
        cutoff_date = '2023-10-01'
        cutoff_date_dt = pd.to_datetime(cutoff_date)
        # 将valuation_date转换为datetime用于比较（不修改原始df_etf）
        df_etf_temp = df_etf.copy()
        df_etf_temp['valuation_date_dt'] = pd.to_datetime(df_etf_temp['valuation_date'])
        df_small_before = df_etf_temp[(df_etf_temp['code'].isin(zz500_zz1000_etf_codes_full)) & 
                                      (df_etf_temp['valuation_date_dt'] < cutoff_date_dt)].copy()
        df_small_after = df_etf_temp[(df_etf_temp['code'].isin(zz1000_zz2000_etf_codes_full)) & 
                                     (df_etf_temp['valuation_date_dt'] >= cutoff_date_dt)].copy()
        # 合并并删除临时列
        df_small = pd.concat([df_small_before, df_small_after], ignore_index=True)
        df_small.drop(columns=['valuation_date_dt'], inplace=True)
        
        # 按日期分组计算平均值
        df_big_avg = df_big.groupby('valuation_date')['money_flow'].mean().reset_index()
        df_big_avg.columns = ['valuation_date', 'money_flow_big']
        
        df_small_avg = df_small.groupby('valuation_date')['money_flow'].mean().reset_index()
        df_small_avg.columns = ['valuation_date', 'money_flow_small']
        
        # 合并生成df_final
        df_final = df_big_avg.merge(df_small_avg, on='valuation_date', how='outer')
        df_final = df_final.sort_values('valuation_date')
        df_final.dropna(inplace=True)
        df_final.reset_index(drop=True, inplace=True)
        df_final['value']=df_final['money_flow_big']-df_final['money_flow_small']
        df_final=df_final[['valuation_date','value']]
        df_final['value']=df_final['value'].rolling(10).sum()
        df_final.dropna(inplace=True)
        # # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_fund')
        return df_final
    def raw_DBI(self):
        start_date = pd.to_datetime(self.start_date) - pd.DateOffset(months=2)
        start_date = gt.strdate_transfer(start_date)
        inputpath=glv.get('raw_DBI')
        signal_name = 'DBI'
        ori_name = self.rename_code_by_folder_wind(signal_name)
        inputpath = str(
            inputpath) + f" Where code='{ori_name}' And trade_date between '{start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1 = df1[['trade_date', 'close']]
        df1.fillna(method='ffill', inplace=True)
        df1.columns = ['valuation_date', 'value']
        df1.sort_values('valuation_date',ascending=True,inplace=True)
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1['value']=df1['value'].rolling(28).mean()
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        df1.dropna(inplace=True)
        df1.reset_index(inplace=True, drop=True)
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_DBI')
        return df1
    def raw_PCT(self):
        start_date = pd.to_datetime(self.start_date) - pd.DateOffset(months=2)
        start_date = gt.strdate_transfer(start_date)
        inputpath = glv.get('raw_PCT')
        signal_name = 'PCT'
        ori_name = self.rename_code_by_folder_wind(signal_name)
        inputpath = str(
            inputpath) + f" Where code='{ori_name}' And trade_date between '{start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1 = df1[['trade_date', 'close']]
        df1.fillna(method='ffill', inplace=True)
        df1.columns = ['valuation_date', 'value']
        df1.sort_values('valuation_date', ascending=True, inplace=True)
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1['value'] = df1['value'].rolling(28).mean()
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        df1.dropna(inplace=True)
        df1.reset_index(inplace=True, drop=True)
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_PCT')
        return df1
    def raw_PTA(self):
        inputpath='D:\选用的经济增长指标\\PTA.csv'
        df1=pd.read_csv(inputpath,encoding='gbk')
        df1.sort_values('valuation_date',ascending=True,inplace=True)
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        df1.set_index('valuation_date',inplace=True,drop=True)
        df1['value']=df1.mean(axis=1)
        df1.reset_index(inplace=True)
        df1=df1[['valuation_date','value']]
        df1['value']=df1['value'].rolling(28).mean()
        df1.dropna(inplace=True)
        df1.reset_index(inplace=True, drop=True)
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_PTA')
        return df1
    def raw_LHBProportion_withdraw(self):
        """
        获取龙虎榜占比数据
        
        Returns:
        --------
        pd.DataFrame
            包含龙虎榜相关数据的DataFrame，列包括：
            - valuation_date: 日期
            - ts_code: 股票代码
            - amount: 成交金额
            等其他龙虎榜相关字段
        """
        inputpath=glv.get('raw_LHBProportion')
        inputpath = str(inputpath) + f" Where trade_date between '{self.start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1.rename(columns={'trade_date': 'valuation_date'}, inplace=True)
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_LHBProportion_withdraw')
        return df1
    def raw_NetLeverageBuying_withdraw(self):
        """
        获取净杠杆买入数据（融资融券数据）
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - code: 股票代码
            - mrg_long_amt: 融资买入金额
            - mrg_long_repay: 融资偿还金额
        """
        # inputpath='D:\Timeselecting_v3\data\\NetLeverageBuying.csv'
        # df1=pd.read_csv(inputpath)
        inputpath=glv.get('raw_NLBPDifference')
        inputpath = str(inputpath) + f" Where trade_date between '{self.start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1=df1[['trade_date','code','mrg_long_amt','mrg_long_repay']]
        df1.rename(columns={'trade_date':'valuation_date'},inplace=True)
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        working_days_list=df1['valuation_date'].unique().tolist()
        if self.end_date not in working_days_list:
            available_date=gt.last_workday_calculate(self.end_date)
            print(f"Leverage_data缺少{self.end_date}数据，现按照{available_date}的数据补充")
            slice_df=df1[df1['valuation_date']==available_date]
            slice_df['valuation_date']=self.end_date
            df1=pd.concat([df1,slice_df])
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_NetLeverageBuying_withdraw')
        return df1
    def raw_LargeOrder_withdraw(self):
        """
        获取大单资金流入数据
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - organization: 指数名称（如'000300.SH'、'000852.SH'等）
            - value: 大单资金流入值
        """
        inputpath=glv.get('raw_LargeOrder')
        inputpath = str(
                inputpath) + f" Where trade_date between '{self.start_date}' and '{self.end_date}'"
        df1 = gt.data_getting(inputpath, config_path)
        df1=df1[['trade_date','code','mfd_inflow_m']]
        df1.columns=['valuation_date','organization','value']
        df1=gt.sql_to_timeseries(df1)
        df1['valuation_date']=pd.to_datetime(df1['valuation_date'])
        df1['valuation_date']=df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_LargeOrder_withdraw')
        return df1
    def raw_stockClose_withdraw(self):
        """
        获取股票收盘价数据
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - {股票代码}: 每只股票的收盘价（列为股票代码）
        """
        df1=gt.stockData_withdraw(start_date=self.start_date,end_date=self.end_date,columns=['close'])
        df1=gt.sql_to_timeseries(df1)
        # inputpath='D:\TimeSelecting_v2\\stock_close.csv'
        # df1=pd.read_csv(inputpath)
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_stockClose_withdraw')
        return df1
    def raw_stockPct_withdraw(self):
        """
        获取股票涨跌幅数据
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - {股票代码}: 每只股票的涨跌幅（列为股票代码）
        """
        df1=gt.stockData_withdraw(start_date=self.start_date,end_date=self.end_date,columns=['pct_chg'])
        df1=gt.sql_to_timeseries(df1)
        # df1.to_csv('stock_return.csv',index=False)
        # inputpath='D:\TimeSelecting_v2\\stock_return.csv'
        # df1=pd.read_csv(inputpath)
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df1 = df1[df1['valuation_date'].isin(self.working_days_list)]
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_stockPct_withdraw')
        return df1
    def raw_index_volume(self, index_name):
        """
        获取指数成交量数据
        
        Parameters:
        -----------
        index_name : str or None
            指数名称，如'沪深300'、'中证1000'、'国证2000'等
            如果为None，返回所有指数的成交量数据
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - {index_name}: 指数成交量值（如果index_name不为None）
            或包含所有指数代码的列（如果index_name为None）
        """
        df1=gt.indexData_withdraw(index_name,start_date=self.start_date,end_date=self.end_date,columns=['volume'])
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        if index_name!=None:
            df1 = df1[['valuation_date', 'volume']]
            df1.columns = ['valuation_date', index_name]
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_index_volume')
        return df1
    def raw_internationalIndex(self):
        df_final=pd.DataFrame()
        df_final['valuation_date']=self.working_days_list
        inputpath=glv.get('raw_intindex')
        inputpath=str(
                inputpath) + f" Where trade_date between '{self.start_date}' and '{self.end_date}'"
        df=gt.data_getting(inputpath,config_path)
        df=df[df['ts_code'].isin(['DJI','RUT'])]
        df=df[['trade_date','ts_code','pct_chg']]
        df=gt.sql_to_timeseries(df)
        df.set_index('valuation_date',inplace=True,drop=True)
        df=df/100
        df=(1+df).cumprod()
        #df['value']=df['DJI']/df['SPX']
        #df['value'] = df['RUT']
        df.reset_index(inplace=True)
        #df=df[['valuation_date','value']]
        df['valuation_date'] = pd.to_datetime(df['valuation_date'])
        df['valuation_date'] = df['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df_final=df_final.merge(df,on='valuation_date',how='left')
        df_final.fillna(method='ffill',inplace=True)
        return df_final
    def raw_index_weight(self, index_name):
        """
        获取指数成分股权重数据
        
        Parameters:
        -----------
        index_name : str
            指数名称，如'沪深300'、'中证1000'、'国证2000'等
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期
            - code: 股票代码
            - weight: 权重（如果存在）
            等其他成分股相关字段
        """
        inputpath=glv.get('raw_indexweight')
        short_name = gt.index_mapping(index_name, 'shortname')
        inputpath_index = inputpath + f" WHERE valuation_date between '{self.start_date}' and '{self.end_date}' AND organization='{short_name}'"
        df=gt.data_getting(inputpath_index,config_path)
        # 检查工作日完整性
        self._check_working_days_completeness(df, 'raw_index_weight')
        return df
    def raw_index_turnover(self, index_name):
        """
        获取指数换手率数据
        
        Parameters:
        -----------
        index_name : str or None
            指数名称，如'沪深300'、'中证1000'、'国证2000'等
            如果为None，返回所有指数的换手率数据
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - {index_name}: 指数换手率值（如果index_name不为None）
        """
        df1 = gt.indexData_withdraw(index_name, start_date=self.start_date, end_date=self.end_date, columns=['turn_over'])
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        if index_name!=None:
            df1 = df1[['valuation_date', 'turn_over']]
            df1.columns = ['valuation_date', index_name]
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_index_turnover')
        return df1
    def raw_index_close(self, index_name):
        """
        获取指数收盘价数据
        
        Parameters:
        -----------
        index_name : str or None
            指数名称，如'沪深300'、'中证1000'、'国证2000'等
            如果为None，返回所有指数的收盘价数据
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - {index_name}: 指数收盘价值（如果index_name不为None）
            或包含所有指数代码的列（如果index_name为None）
        """
        df1 = gt.indexData_withdraw(index_name, start_date=self.start_date, end_date=self.end_date,
                                    columns=['close'])
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        if index_name != None:
            df1 = df1[['valuation_date', 'close']]
            df1.columns = ['valuation_date', index_name]
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_index_close')
        return df1
    def raw_index_amt(self, index_name):
        """
        获取指数成交额数据
        
        Parameters:
        -----------
        index_name : str or None
            指数名称，如'沪深300'、'中证1000'、'国证2000'等
            如果为None，返回所有指数的成交额数据
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - {index_name}: 指数成交额值（如果index_name不为None）
            或包含所有指数代码的列（如果index_name为None）
        """
        df1 = gt.indexData_withdraw(index_name, start_date=self.start_date, end_date=self.end_date,
                                    columns=['amt'])
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        if index_name!=None:
            df1 = df1[['valuation_date', 'amt']]
            df1.columns = ['valuation_date', index_name]
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'raw_index_amt')
        return df1
    def index_return_withdraw(self):
        """
        获取指数收益率数据
        
        从数据库获取所有指数的收益率数据，并转换为时间序列格式
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - {指数中文名}: 各指数的收益率（列为指数中文名称）
            
        Note:
        -----
        如果中证2000的数据为None值，会自动用国证2000的数据填充
        """
        df1 = gt.indexData_withdraw(index_type=None, start_date=self.start_date, end_date=self.end_date,
                                    columns=['pct_chg'])
        inputpath_info=glv.get('index_info')
        df_info=gt.data_getting(inputpath_info,config_path)
        df_info.rename(columns={'index_code':'code'},inplace=True)
        df1=df1.merge(df_info,on='code',how='left')
        df1=df1[['valuation_date','chi_name','pct_chg']]
        df1.rename(columns={'chi_name': 'code'}, inplace=True)
        df1 = gt.sql_to_timeseries(df1)
        df1['valuation_date'] = pd.to_datetime(df1['valuation_date'])
        df1['valuation_date'] = df1['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        
        # 如果中证2000的数据为None值，用国证2000的数据填充
        if '中证2000' in df1.columns and '国证2000' in df1.columns:
            df1['中证2000'] = df1['中证2000'].fillna(df1['国证2000'])
        
        # 检查工作日完整性
        self._check_working_days_completeness(df1, 'index_return_withdraw')
        return df1
    def index_return_withdraw2(self):
        """
        获取特定指数的收益率数据（沪深300和国证2000）
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - 沪深300: 沪深300指数收益率
            - 国证2000: 国证2000指数收益率
        """
        df_return = self.index_return_withdraw()
        df_return = df_return[['valuation_date', '上证50', '中证2000']]
        df_return[['上证50', '中证2000']] = df_return[['上证50', '中证2000']].astype(float)
        # 检查工作日完整性
        self._check_working_days_completeness(df_return, 'index_return_withdraw2')
        return df_return
    def BankMomentum_withdraw(self):
        """
        获取银行动量数据
        
        计算金融等权指数与国证2000的收益率差值
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - difference: 收益率差值（金融等权 - 国证2000）
        """
        df_return = self.index_return_withdraw()
        df_return['difference']=df_return['金融等权']-df_return['国证2000']
        df_return.reset_index(inplace=True)
        df_return=df_return[['valuation_date','difference']]
        # 检查工作日完整性
        self._check_working_days_completeness(df_return, 'BankMomentum_withdraw')
        return df_return
    #target_index
    def raw_indexBasic(self):
        """
        获取指数基本面数据（PB、PE等）
        
        Returns:
        --------
        pd.DataFrame
            包含指数基本面数据的DataFrame，列包括：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - ts_code: 指数代码
            - pb: 市净率
            - pe_ttm: 市盈率（TTM）
            等其他基本面指标
        """
        inputpath=glv.get('raw_indexbasic')
        inputpath=inputpath+f" WHERE trade_date between '{self.start_date}' and '{self.end_date}'"
        df=gt.data_getting(inputpath,config_path)
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        df['trade_date'] = df['trade_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df.rename(columns={'trade_date': 'valuation_date'}, inplace=True)
        # 检查工作日完整性
        self._check_working_days_completeness(df, 'raw_indexBasic')
        return df
    def target_index(self):
        df_return = self.index_return_withdraw()
        df_return = df_return[['valuation_date', '上证50', '中证2000']]
        df_return.set_index('valuation_date', inplace=True)
        df_return = df_return.astype(float)
        df_return = (1 + df_return).cumprod()
        df_return['target_index'] = df_return['上证50'] / df_return['中证2000']
        df_return.reset_index(inplace=True)
        df_return = df_return[['valuation_date', 'target_index', '上证50', '中证2000']]
        # 检查工作日完整性
        self._check_working_days_completeness(df_return, 'target_index')
        return df_return

    def target_index_candle(self):
        """
        计算目标指数的K线数据（基于沪深300和中证2000）
        
        计算沪深300与中证2000的差值K线（high、close、low）
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - high: 最高价差值
            - close: 收盘价差值（沪深300收盘价 - 中证2000收盘价）
            - low: 最低价差值
        """
        df_index = gt.indexData_withdraw(index_type=None, start_date='2010-12-31', end_date=self.end_date,
                                         columns=['close', 'high', 'low'])
        df_close = df_index[['valuation_date', 'code', 'close']]
        df_high = df_index[['valuation_date', 'code', 'high']]
        df_low = df_index[['valuation_date', 'code', 'low']]
        df_close = gt.sql_to_timeseries(df_close)
        df_high = gt.sql_to_timeseries(df_high)
        df_low = gt.sql_to_timeseries(df_low)
        df_close = df_close[['valuation_date', '000016.SH', '399303.SZ']]
        df_high = df_high[['valuation_date', '000016.SH', '399303.SZ']]
        df_low = df_low[['valuation_date', '000016.SH', '399303.SZ']]
        df_close.columns = ['valuation_date', '000016.SH_close', '399303.SZ_close']
        df_high.columns = ['valuation_date', '000016.SH_high', '399303.SZ_high']
        df_low.columns = ['valuation_date', '000016.SH_low', '399303.SZ_low']
        df_hl = df_high.merge(df_low, on='valuation_date', how='left')
        df_final = df_close.merge(df_hl, on='valuation_date', how='left')
        df_final['close'] = df_final['000016.SH_close'] - df_final['399303.SZ_close']
        df_final['high'] = df_final['000016.SH_high'] - df_final['399303.SZ_low']
        df_final['low'] = df_final['000016.SH_low'] - df_final['399303.SZ_high']
        df_final = df_final[['valuation_date', 'high', 'close', 'low']]
        # 检查工作日完整性
        self._check_working_days_completeness(df_final, 'target_index_candle')
        return df_final

    def target_index_candle2(self):
        """
        计算目标指数的K线数据（基于沪深300和中证2000，使用不同的指数代码）
        
        计算沪深300与932000.CSI的差值K线（high、close、low）
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - high: 最高价差值
            - close: 收盘价差值（沪深300收盘价 - 932000.CSI收盘价）
            - low: 最低价差值
        """
        df_index = gt.indexData_withdraw(index_type=None, start_date=self.start_date, end_date=self.end_date,
                                         columns=['close', 'high', 'low'])
        df_close = df_index[['valuation_date', 'code', 'close']]
        df_high = df_index[['valuation_date', 'code', 'high']]
        df_low = df_index[['valuation_date', 'code', 'low']]
        df_close = gt.sql_to_timeseries(df_close)
        df_high = gt.sql_to_timeseries(df_high)
        df_low = gt.sql_to_timeseries(df_low)
        df_close = df_close[['valuation_date', '000016.SH', '932000.CSI']]
        df_high = df_high[['valuation_date', '000016.SH', '932000.CSI']]
        df_low = df_low[['valuation_date', '000016.SH', '932000.CSI']]
        df_close.columns = ['valuation_date', '000016.SH_close', '932000.CSI_close']
        df_high.columns = ['valuation_date', '000016.SH_high', '932000.CSI_high']
        df_low.columns = ['valuation_date', '000016.SH_low', '932000.CSI_low']
        df_hl = df_high.merge(df_low, on='valuation_date', how='left')
        df_final = df_close.merge(df_hl, on='valuation_date', how='left')
        df_final['close'] = df_final['000016.SH_close'] - df_final['932000.CSI_close']
        df_final['high'] = df_final['000016.SH_high'] - df_final['932000.CSI_low']
        df_final['low'] = df_final['000016.SH_low'] - df_final['932000.CSI_high']
        df_final = df_final[['valuation_date', 'high', 'close', 'low']]
        # 检查工作日完整性
        self._check_working_days_completeness(df_final, 'target_index_candle2')
        return df_final
    def future_difference_withdraw(self):
        """
        获取期货价差数据
        
        Returns:
        --------
        pd.DataFrame
            包含期货相关数据的DataFrame，列包括：
            - valuation_date: 日期
            - code: 期货代码
            - close: 收盘价
            等其他期货相关字段
        """
        df=gt.futureData_withdraw(self.start_date,self.end_date,['close'],False)
        # 检查工作日完整性
        self._check_working_days_completeness(df, 'future_difference_withdraw')
        return df
    def raw_futureHolding(self):
        inputpath=glv.get('raw_futureHolding')
        inputpath = inputpath + f" WHERE trade_date between '{self.start_date}' and '{self.end_date}'"
        df=gt.data_getting(inputpath,config_path)
        df.drop(columns=['vol','vol_chg','exchange'],inplace=True)
        df['trade_date']=pd.to_datetime(df['trade_date'].astype(str))
        df['trade_date']=df['trade_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df['new_symbol']=df['symbol'].apply(lambda x: str(x)[:2])
        df=df[df['new_symbol'].isin(['IH','IM','IC','IF'])]
        df.rename(columns={'trade_date':'valuation_date'},inplace=True)
        # 检查工作日完整性
        self._check_working_days_completeness(df, 'raw_futureHolding')
        return df
    def raw_vix_withdraw(self):
        """
        获取VIX（波动率指数）数据
        
        获取时间加权VIX数据，并进行前向填充处理
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - hs300: 沪深300的VIX值
            - zz1000: 中证1000的VIX值
        """
        inputpath=glv.get('raw_vix')
        inputpath = inputpath + f" Where vix_type='TimeWeighted' And valuation_date between '{self.start_date}' and '{self.end_date}'"
        df = gt.data_getting(inputpath, config_path)
        df=df[df['vix_type']=='TimeWeighted']
        df=df[['valuation_date','organization','ch_vix']]
        df=gt.sql_to_timeseries(df)
        df=df[['valuation_date','hs300','zz1000']]
        df.fillna(method='ffill',inplace=True)
        # 检查工作日完整性
        self._check_working_days_completeness(df, 'raw_vix_withdraw')
        return df
    def raw_rrscore_info(self):
        """
        获取RR评分信息数据
        
        从投资组合信息中提取RR评分相关的base_score信息
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - base_score: RR评分名称（如'rr_1'、'rr_2'等）
            
        Note:
        -----
        对于每个日期，选择数字最大的base_score（如rr_10 > rr_1）
        """
        working_days_list=gt.working_days_list(self.start_date,self.end_date)
        df_final=pd.DataFrame()
        df_final['valuation_date']=working_days_list
        inputpath=glv.get('portfolio_info')
        inputpath=inputpath + f" Where valuation_date between '{self.start_date}' and '{self.end_date}'"
        df = gt.data_getting(inputpath, config_path)
        df.dropna(inplace=True)
        df['base_score2']=df['base_score'].apply(lambda x: str(x)[:2])
        df=df[df['base_score2']=='rr']
        # 提取 base_score 中 rr_ 后面的数字
        df['score_num'] = df['base_score'].str.extract(r'rr_(\d+)').astype(float)
        # 按日期分组，选择数字最大的 base_score
        df_result = df.loc[df.groupby('valuation_date')['score_num'].idxmax()]
        # 只保留需要的两列
        df_result = df_result[['valuation_date', 'base_score']].copy()
        df_result.reset_index(drop=True, inplace=True)
        df_final=df_final.merge(df_result,on='valuation_date',how='left')
        df_final.fillna(method='bfill',inplace=True)
        # 检查工作日完整性
        self._check_working_days_completeness(df_final, 'raw_rrscore_info')
        return df_final
    def raw_rrscore_withdraw(self):
        """
        获取RR评分数据
        
        根据raw_rrscore_info获取的base_score列表，从数据库获取对应的RR评分数据
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期
            - code: 股票代码
            - score_name: 评分名称
            - final_score: 最终评分值
            等其他RR评分相关字段
        """
        inputpath=glv.get('raw_rrscore')
        df_scoreinfo=self.raw_rrscore_info()
        score_list=df_scoreinfo['base_score'].unique().tolist()
        # 构建 SQL IN 子句
        if score_list:
            score_list_str = "', '".join(score_list)
            inputpath = inputpath + f" Where valuation_date between '{self.start_date}' and '{self.end_date}' and score_name IN ('{score_list_str}')"
        else:
            inputpath = inputpath + f" Where valuation_date between '{self.start_date}' and '{self.end_date}' and 1=0"
        df = gt.data_getting(inputpath, config_path)
        # 将 df_scoreinfo 的 base_score 列重命名为 score_name，以便与 df 进行匹配
        df_scoreinfo_merge = df_scoreinfo.rename(columns={'base_score': 'score_name'})
        # 同时匹配 valuation_date 和 score_name
        df = df.merge(df_scoreinfo_merge, on=['valuation_date', 'score_name'], how='inner')
        # 检查工作日完整性
        self._check_working_days_completeness(df, 'raw_rrscore_withdraw')
        return df
if __name__ == "__main__":
    dp=data_prepare('2015-01-03','2026-01-05')
    df=dp.raw_Earning_withdraw()
    df2=dp.target_index()
    df2=df2[['valuation_date','target_index']]
    df=df.merge(df2,on='valuation_date',how='left')
    #df=df[['valuation_date','target_index']]
    df.set_index('valuation_date',inplace=True,drop=True)
    df=(df-df.min())/(df.max()-df.min())
    df.plot()
    plt.show()