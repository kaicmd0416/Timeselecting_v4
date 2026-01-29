"""
因子加权器抽象基类 (base_weighter)

定义因子加权器的通用接口，所有具体的加权实现都应继承此基类。

作者: TimeSelecting Team
版本: v1.0
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseWeighter(ABC):
    """
    因子加权器抽象基类

    定义因子加权的通用接口，子类需实现具体的权重计算逻辑。

    Attributes:
    -----------
    lookback_window : int
        计算权重时使用的回溯窗口大小（交易日数）
    min_periods : int
        最小观测数量，低于此值时使用等权
    """

    def __init__(self, lookback_window: int = 252, min_periods: int = 252):
        """
        初始化加权器

        Parameters:
        -----------
        lookback_window : int
            计算权重时使用的回溯窗口大小，默认252（一年交易日）
        min_periods : int
            最小观测数量，低于此值时使用等权，默认252
        """
        self.lookback_window = lookback_window
        self.min_periods = min_periods

    @abstractmethod
    def calculate_weights(self, df_signals: pd.DataFrame, df_returns: pd.DataFrame,
                          target_date: str) -> pd.Series:
        """
        计算单日权重

        Parameters:
        -----------
        df_signals : pd.DataFrame
            信号数据，index为日期，columns为因子名称
        df_returns : pd.DataFrame
            收益率数据，需包含 'valuation_date' 和 'return' 列
        target_date : str
            目标日期，格式为 'YYYY-MM-DD'

        Returns:
        --------
        pd.Series
            权重序列，index为因子名称，values为权重值
            权重之和为1（已归一化）
        """
        pass

    @abstractmethod
    def calculate_weights_series(self, df_signals: pd.DataFrame,
                                  df_returns: pd.DataFrame) -> pd.DataFrame:
        """
        计算时间序列权重

        为每个日期计算对应的因子权重

        Parameters:
        -----------
        df_signals : pd.DataFrame
            信号数据，index为日期，columns为因子名称
        df_returns : pd.DataFrame
            收益率数据，需包含 'valuation_date' 和 'return' 列

        Returns:
        --------
        pd.DataFrame
            权重DataFrame，index为日期，columns为因子名称，values为权重
        """
        pass

    def normalize_weights(self, weights: pd.Series, remove_negative: bool = True) -> pd.Series:
        """
        权重归一化

        Parameters:
        -----------
        weights : pd.Series
            原始权重序列
        remove_negative : bool
            是否将负权重设为0，默认True

        Returns:
        --------
        pd.Series
            归一化后的权重，总和为1
            如果所有权重都为0或负，返回等权
        """
        weights = weights.copy()

        # 处理负权重
        if remove_negative:
            weights[weights < 0] = 0

        # 归一化
        total = weights.sum()
        if total == 0 or np.isnan(total):
            # 如果所有权重为0，返回等权
            n = len(weights)
            return pd.Series(1.0 / n, index=weights.index)

        return weights / total

    def apply_weights(self, df_signals: pd.DataFrame, df_weights: pd.DataFrame) -> pd.Series:
        """
        应用权重计算加权平均信号

        Parameters:
        -----------
        df_signals : pd.DataFrame
            信号数据，index为日期，columns为因子名称
        df_weights : pd.DataFrame
            权重数据，index为日期，columns为因子名称

        Returns:
        --------
        pd.Series
            加权平均后的信号值，index为日期
        """
        # 确保信号和权重的列对齐
        common_cols = df_signals.columns.intersection(df_weights.columns)
        df_signals_aligned = df_signals[common_cols]
        df_weights_aligned = df_weights[common_cols]

        # 确保索引对齐
        common_dates = df_signals_aligned.index.intersection(df_weights_aligned.index)
        df_signals_aligned = df_signals_aligned.loc[common_dates]
        df_weights_aligned = df_weights_aligned.loc[common_dates]

        # 计算加权平均
        weighted_signal = (df_signals_aligned * df_weights_aligned).sum(axis=1)

        return weighted_signal

    def get_equal_weights(self, factor_names: list) -> pd.Series:
        """
        获取等权权重

        Parameters:
        -----------
        factor_names : list
            因子名称列表

        Returns:
        --------
        pd.Series
            等权权重序列
        """
        n = len(factor_names)
        return pd.Series(1.0 / n, index=factor_names)
