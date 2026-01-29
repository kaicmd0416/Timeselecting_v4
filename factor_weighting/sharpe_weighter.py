"""
夏普比率加权器模块 (sharpe_weighter)

基于历史夏普比率的因子加权实现。
直接使用每个因子的历史风险调整收益来分配权重。

作者: TimeSelecting Team
版本: v1.0
"""

import pandas as pd
import numpy as np
from .base_weighter import BaseWeighter


class SharpeWeighter(BaseWeighter):
    """
    夏普比率加权器

    使用每个L1因子的历史夏普比率作为权重。
    夏普比率 = 超额收益均值 / 超额收益标准差

    这里的"超额收益"是指：当信号为1时持有小盘、信号为0时持有大盘，相对于等权基准的超额收益。
    """

    def __init__(self, lookback_window: int = 504, min_periods: int = 504):
        """
        初始化夏普比率加权器

        Parameters:
        -----------
        lookback_window : int
            回溯窗口大小，默认504（两年）
        min_periods : int
            最小观测数量，低于此值时使用等权，默认504
        """
        super().__init__(lookback_window, min_periods)

    def _calculate_factor_return(self, signal_series: pd.Series, return_series: pd.Series) -> pd.Series:
        """
        计算单个因子的收益序列

        Parameters:
        -----------
        signal_series : pd.Series
            信号序列 (0, 0.5, 1)
        return_series : pd.Series
            相对收益序列 (小盘 - 大盘)

        Returns:
        --------
        pd.Series
            因子收益序列
        """
        # signal=1 时，收益 = 相对收益（看多小盘）
        # signal=0 时，收益 = -相对收益（看多大盘）
        # signal=0.5 时，收益 = 0（中性）
        factor_return = signal_series.copy()
        factor_return = (signal_series - 0.5) * 2 * return_series
        return factor_return

    def _prepare_data(self, df_signals: pd.DataFrame, df_returns: pd.DataFrame) -> tuple:
        """准备数据，确保信号和收益率正确对齐（同一天配对）"""
        df_signals = df_signals.copy()
        if isinstance(df_signals.index, pd.DatetimeIndex):
            df_signals.index = df_signals.index.strftime('%Y-%m-%d')

        df_returns = df_returns.copy()
        df_returns['valuation_date'] = pd.to_datetime(df_returns['valuation_date']).dt.strftime('%Y-%m-%d')
        df_returns.set_index('valuation_date', inplace=True)

        common_dates = df_signals.index.intersection(df_returns.index)
        df_signals_aligned = df_signals.loc[common_dates]
        returns_aligned = df_returns.loc[common_dates, 'return']

        return df_signals_aligned, returns_aligned

    def calculate_weights(self, df_signals: pd.DataFrame, df_returns: pd.DataFrame,
                          target_date: str) -> pd.Series:
        """计算单日权重"""
        factor_names = df_signals.columns.tolist()
        target_date_dt = pd.to_datetime(target_date)

        # 筛选目标日期之前的数据
        df_signals_filtered = df_signals.copy()
        if not isinstance(df_signals_filtered.index, pd.DatetimeIndex):
            df_signals_filtered.index = pd.to_datetime(df_signals_filtered.index)
        df_signals_filtered = df_signals_filtered[df_signals_filtered.index < target_date_dt]
        df_signals_filtered.index = df_signals_filtered.index.strftime('%Y-%m-%d')

        df_returns_filtered = df_returns.copy()
        df_returns_filtered['valuation_date'] = pd.to_datetime(df_returns_filtered['valuation_date'])
        df_returns_filtered = df_returns_filtered[df_returns_filtered['valuation_date'] < target_date_dt]
        df_returns_filtered['valuation_date'] = df_returns_filtered['valuation_date'].dt.strftime('%Y-%m-%d')

        # 准备对齐后的数据
        df_signals_aligned, returns_aligned = self._prepare_data(df_signals_filtered, df_returns_filtered)

        n_obs = len(df_signals_aligned)
        if n_obs < self.min_periods:
            return self.get_equal_weights(factor_names)

        # 只使用最近lookback_window天的数据
        if n_obs > self.lookback_window:
            df_signals_aligned = df_signals_aligned.iloc[-self.lookback_window:]
            returns_aligned = returns_aligned.iloc[-self.lookback_window:]

        # 计算每个因子的夏普比率
        sharpe_values = {}
        for factor in factor_names:
            factor_return = self._calculate_factor_return(df_signals_aligned[factor], returns_aligned)

            if len(factor_return.dropna()) >= 20:
                mean_return = factor_return.mean()
                std_return = factor_return.std()

                if std_return > 0:
                    sharpe = mean_return / std_return * np.sqrt(252)  # 年化
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0

            sharpe_values[factor] = sharpe

        sharpe_series = pd.Series(sharpe_values)

        # 夏普为负的设为0
        sharpe_series[sharpe_series < 0] = 0

        # 归一化权重
        weights = self.normalize_weights(sharpe_series, remove_negative=True)

        return weights

    def calculate_weights_series(self, df_signals: pd.DataFrame,
                                  df_returns: pd.DataFrame) -> pd.DataFrame:
        """计算时间序列权重"""
        dates = df_signals.index.tolist()
        weights_list = []

        for date in dates:
            weights = self.calculate_weights(df_signals, df_returns, str(date))
            weights.name = date
            weights_list.append(weights)

        df_weights = pd.DataFrame(weights_list)
        df_weights.index = dates

        return df_weights
