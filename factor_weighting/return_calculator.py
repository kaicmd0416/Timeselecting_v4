"""
收益率计算器模块 (return_calculator)

提供收益率计算功能，用于IC加权模块。

作者: TimeSelecting Team
版本: v1.0
"""

import os
import sys
import pandas as pd

# 添加全局工具函数路径
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_setting.global_dic as glv

# 添加项目路径
project_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_path)
from data.data_prepare import data_prepare


class ReturnCalculator:
    """
    收益率计算器

    计算大小盘相对收益率，用于IC计算。

    收益率定义:
        return = 小盘收益率 - 大盘收益率

    这与信号方向一致：signal=1表示买小盘，当小盘跑赢大盘时收益为正。

    Attributes:
    -----------
    start_date : str
        开始日期，格式 'YYYY-MM-DD'
    end_date : str
        结束日期，格式 'YYYY-MM-DD'
    big_index : str
        大盘指数名称，如 '上证50'、'沪深300'
    small_index : str
        小盘指数名称，如 '中证2000'
    """

    # 默认指数配置
    DEFAULT_CONFIG = {
        'L0': {'big': '沪深300', 'small': '中证2000'},
        'L1': {'big': '上证50', 'small': '中证2000'},
        'L2': {'big': '上证50', 'small': '中证2000'},
    }

    def __init__(self, start_date: str, end_date: str,
                 big_index: str = None, small_index: str = None,
                 level: str = 'L1'):
        """
        初始化收益率计算器

        Parameters:
        -----------
        start_date : str
            开始日期，格式 'YYYY-MM-DD'
        end_date : str
            结束日期，格式 'YYYY-MM-DD'
        big_index : str, optional
            大盘指数名称，如果不指定则根据level使用默认值
        small_index : str, optional
            小盘指数名称，如果不指定则根据level使用默认值
        level : str
            信号层级 ('L0', 'L1', 'L2')，用于选择默认指数配置
        """
        self.start_date = start_date
        self.end_date = end_date

        # 如果未指定指数，使用默认配置
        if big_index is None or small_index is None:
            config = self.DEFAULT_CONFIG.get(level, self.DEFAULT_CONFIG['L1'])
            self.big_index = big_index or config['big']
            self.small_index = small_index or config['small']
        else:
            self.big_index = big_index
            self.small_index = small_index

        self._df_returns = None

    def get_index_returns(self) -> pd.DataFrame:
        """
        获取指数收益率数据

        Returns:
        --------
        pd.DataFrame
            包含 valuation_date、大盘收益、小盘收益的DataFrame

        Note:
            由于计算T日权重只需要T-1及以前的数据，这里获取的数据可能不包含end_date当天
        """
        # 尝试获取数据，如果end_date没有数据则自动使用可用的最新数据
        try:
            dp = data_prepare(self.start_date, self.end_date)
            df_return = dp.index_return_withdraw()
        except Exception as e:
            # 如果报错（比如end_date没有数据），尝试用更早的结束日期
            # 获取前一个工作日
            import global_tools as gt
            adjusted_end_date = gt.last_workday_calculate(self.end_date)
            print(f"[ReturnCalculator] 调整end_date: {self.end_date} -> {adjusted_end_date}")
            dp = data_prepare(self.start_date, adjusted_end_date)
            df_return = dp.index_return_withdraw()

        # 提取需要的指数收益率
        columns_needed = ['valuation_date', self.big_index, self.small_index]

        # 检查列是否存在
        missing_cols = [col for col in columns_needed if col not in df_return.columns]
        if missing_cols:
            raise ValueError(f"指数收益率数据中缺少列: {missing_cols}")

        df_return = df_return[columns_needed]
        df_return[[self.big_index, self.small_index]] = df_return[[self.big_index, self.small_index]].astype(float)

        return df_return

    def get_relative_returns(self) -> pd.DataFrame:
        """
        计算相对收益率

        Returns:
        --------
        pd.DataFrame
            包含以下列：
            - valuation_date: 日期
            - return: 相对收益率 (小盘 - 大盘)
        """
        if self._df_returns is not None:
            return self._df_returns

        df_return = self.get_index_returns()

        # 计算相对收益：小盘 - 大盘
        df_return['return'] = df_return[self.small_index] - df_return[self.big_index]
        df_return = df_return[['valuation_date', 'return']]

        # 缓存结果
        self._df_returns = df_return

        return df_return

    def get_forward_returns(self, n_days: int = 1) -> pd.DataFrame:
        """
        计算前瞻N日收益率

        将收益率向前移动N日，用于计算 signal_t 与 return_{t+N} 的相关性

        Parameters:
        -----------
        n_days : int
            前瞻天数，默认1

        Returns:
        --------
        pd.DataFrame
            包含以下列：
            - valuation_date: 日期
            - forward_return: 前瞻N日收益率
        """
        df_return = self.get_relative_returns().copy()
        df_return['valuation_date'] = pd.to_datetime(df_return['valuation_date'])
        df_return = df_return.sort_values('valuation_date')

        # 将收益率向后移动（即T日对应的是T+1日的收益率）
        df_return['forward_return'] = df_return['return'].shift(-n_days)
        df_return = df_return[['valuation_date', 'forward_return']].dropna()

        return df_return
