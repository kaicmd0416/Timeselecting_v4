"""
IC_IR加权器模块 (icir_weighter)

基于IC信息比率（IC均值/IC标准差）的因子加权实现。
相比简单IC加权，IC_IR考虑了IC的稳定性，更加稳健。

作者: TimeSelecting Team
版本: v1.0
"""

import pandas as pd
import numpy as np
from .base_weighter import BaseWeighter


class ICIRWeighter(BaseWeighter):
    """
    IC_IR加权器

    使用IC的信息比率（IC均值/IC标准差）作为权重，
    这样不仅考虑IC的大小，还考虑IC的稳定性。

    IC_IR = mean(IC) / std(IC)
    """

    def __init__(self, lookback_window: int = 504, min_periods: int = 504,
                 ic_window: int = 20, use_abs: bool = False):
        """
        初始化IC_IR加权器

        Parameters:
        -----------
        lookback_window : int
            总回溯窗口大小，默认504（两年）
        min_periods : int
            最小观测数量，低于此值时使用等权，默认504
        ic_window : int
            计算单次IC的窗口大小，默认20（一个月）
        use_abs : bool
            是否使用IC_IR绝对值，默认False
        """
        super().__init__(lookback_window, min_periods)
        self.ic_window = ic_window
        self.use_abs = use_abs

    def _calculate_rolling_ic(self, signal_series: pd.Series, return_series: pd.Series) -> pd.Series:
        """
        计算滚动IC序列

        Parameters:
        -----------
        signal_series : pd.Series
            信号序列
        return_series : pd.Series
            收益率序列

        Returns:
        --------
        pd.Series
            滚动IC序列
        """
        ic_list = []
        dates = signal_series.index.tolist()

        for i in range(self.ic_window, len(dates) + 1):
            window_signal = signal_series.iloc[i - self.ic_window:i]
            window_return = return_series.iloc[i - self.ic_window:i]

            # 计算窗口内的相关系数
            if len(window_signal.dropna()) >= 10:
                ic = window_signal.corr(window_return)
            else:
                ic = np.nan

            ic_list.append(ic)

        # 前面填充NaN
        ic_series = pd.Series([np.nan] * (self.ic_window - 1) + ic_list, index=dates)
        return ic_series

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

        # 计算每个因子的IC_IR
        icir_values = {}
        for factor in factor_names:
            ic_series = self._calculate_rolling_ic(df_signals_aligned[factor], returns_aligned)
            ic_series = ic_series.dropna()

            if len(ic_series) >= 10:
                ic_mean = ic_series.mean()
                ic_std = ic_series.std()
                if ic_std > 0:
                    icir = ic_mean / ic_std
                else:
                    icir = 0.0
            else:
                icir = 0.0

            if self.use_abs:
                icir = abs(icir)

            icir_values[factor] = icir

        icir_series = pd.Series(icir_values)

        # IC_IR为负的设为0
        icir_series[icir_series < 0] = 0

        # 归一化权重
        weights = self.normalize_weights(icir_series, remove_negative=True)

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
