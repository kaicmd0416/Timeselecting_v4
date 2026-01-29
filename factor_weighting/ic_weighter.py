"""
时序IC加权器模块 (ic_weighter)

基于时序IC（Information Coefficient）的因子加权实现。

核心逻辑:
    1. IC = corr(signal_t, return_{t+1})
    2. 使用滚动窗口计算每个因子的IC
    3. IC为负的因子权重设为0
    4. 剩余因子按IC值归一化为权重
    5. 冷启动期（数据不足）使用等权

避免前瞻性偏差:
    计算T日权重时，只使用 valuation_date < T 的数据。
    最后一对可用数据: signal_{T-2} vs return_{T-1}

作者: TimeSelecting Team
版本: v1.0
"""

import pandas as pd
import numpy as np
from typing import Optional
from .base_weighter import BaseWeighter


class ICWeighter(BaseWeighter):
    """
    时序IC加权器

    基于历史IC值对因子进行加权，IC高的因子权重更大，IC为负的因子被剔除。

    Attributes:
    -----------
    lookback_window : int
        IC计算的滚动窗口大小（交易日数），默认252
    min_periods : int
        最小观测数量，低于此值时使用等权，默认252
    ic_threshold : float
        IC阈值，低于此值的因子权重设为0，默认0
    use_abs_ic : bool
        是否使用IC绝对值计算权重，默认False
    """

    def __init__(self, lookback_window: int = 252, min_periods: int = 252,
                 ic_threshold: float = 0.0, use_abs_ic: bool = False):
        """
        初始化IC加权器

        Parameters:
        -----------
        lookback_window : int
            IC计算的滚动窗口大小，默认252（一年交易日）
        min_periods : int
            最小观测数量，低于此值时使用等权，默认252
        ic_threshold : float
            IC阈值，低于此值的因子权重设为0，默认0（即剔除负IC因子）
        use_abs_ic : bool
            是否使用IC绝对值计算权重，默认False
        """
        super().__init__(lookback_window, min_periods)
        self.ic_threshold = ic_threshold
        self.use_abs_ic = use_abs_ic

    def _calculate_ic(self, signal_series: pd.Series, return_series: pd.Series) -> float:
        """
        计算单个因子的IC值

        Parameters:
        -----------
        signal_series : pd.Series
            信号序列
        return_series : pd.Series
            收益率序列（需与信号序列日期对齐）

        Returns:
        --------
        float
            IC值（皮尔逊相关系数）
        """
        # 确保两个序列索引对齐
        common_idx = signal_series.index.intersection(return_series.index)
        if len(common_idx) < 10:  # 至少需要10个观测值
            return np.nan

        signal_aligned = signal_series.loc[common_idx]
        return_aligned = return_series.loc[common_idx]

        # 移除缺失值
        valid_mask = ~(signal_aligned.isna() | return_aligned.isna())
        signal_clean = signal_aligned[valid_mask]
        return_clean = return_aligned[valid_mask]

        if len(signal_clean) < 10:
            return np.nan

        # 计算相关系数
        return signal_clean.corr(return_clean)

    def _prepare_data(self, df_signals: pd.DataFrame,
                      df_returns: pd.DataFrame) -> tuple:
        """
        准备数据，确保信号和收益率正确对齐

        关键点：
        - T日的信号对应T日的收益率（同一天）
        - 即 signal_t 与 return_t 配对
        - 因为 signal_t 是用于 T 日交易的信号，获得 T 日的收益

        Parameters:
        -----------
        df_signals : pd.DataFrame
            信号数据，index为日期
        df_returns : pd.DataFrame
            收益率数据，需包含 'valuation_date' 和 'return' 列

        Returns:
        --------
        tuple
            (df_signals_aligned, returns_aligned)
            信号数据和同日收益率数据，两者日期对齐
        """
        # 确保日期格式统一
        df_signals = df_signals.copy()
        if isinstance(df_signals.index, pd.DatetimeIndex):
            df_signals.index = df_signals.index.strftime('%Y-%m-%d')

        df_returns = df_returns.copy()
        df_returns['valuation_date'] = pd.to_datetime(df_returns['valuation_date']).dt.strftime('%Y-%m-%d')
        df_returns.set_index('valuation_date', inplace=True)

        # 对齐信号和收益率（同一天配对）
        common_dates = df_signals.index.intersection(df_returns.index)
        df_signals_aligned = df_signals.loc[common_dates]
        returns_aligned = df_returns.loc[common_dates, 'return']

        return df_signals_aligned, returns_aligned

    def calculate_weights(self, df_signals: pd.DataFrame, df_returns: pd.DataFrame,
                          target_date: str) -> pd.Series:
        """
        计算单日权重

        严格避免前瞻性偏差：
        - 计算T日权重时，只使用 valuation_date < T 的数据
        - 最后一对可用数据: signal_{T-2} vs return_{T-1}

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
        """
        factor_names = df_signals.columns.tolist()

        # 筛选目标日期之前的数据（严格小于，避免前瞻性偏差）
        target_date_dt = pd.to_datetime(target_date)

        df_signals_filtered = df_signals.copy()
        if isinstance(df_signals_filtered.index, pd.DatetimeIndex):
            df_signals_filtered = df_signals_filtered[df_signals_filtered.index < target_date_dt]
        else:
            df_signals_filtered.index = pd.to_datetime(df_signals_filtered.index)
            df_signals_filtered = df_signals_filtered[df_signals_filtered.index < target_date_dt]
            df_signals_filtered.index = df_signals_filtered.index.strftime('%Y-%m-%d')

        df_returns_filtered = df_returns.copy()
        df_returns_filtered['valuation_date'] = pd.to_datetime(df_returns_filtered['valuation_date'])
        df_returns_filtered = df_returns_filtered[df_returns_filtered['valuation_date'] < target_date_dt]
        df_returns_filtered['valuation_date'] = df_returns_filtered['valuation_date'].dt.strftime('%Y-%m-%d')

        # 准备对齐后的数据
        df_signals_aligned, forward_returns = self._prepare_data(df_signals_filtered, df_returns_filtered)

        # 检查数据是否足够
        n_obs = len(df_signals_aligned)
        if n_obs < self.min_periods:
            # 冷启动期，使用等权
            return self.get_equal_weights(factor_names)

        # 只使用最近lookback_window天的数据
        if n_obs > self.lookback_window:
            df_signals_aligned = df_signals_aligned.iloc[-self.lookback_window:]
            forward_returns = forward_returns.iloc[-self.lookback_window:]

        # 计算每个因子的IC
        ic_values = {}
        for factor in factor_names:
            ic = self._calculate_ic(df_signals_aligned[factor], forward_returns)
            ic_values[factor] = ic if not np.isnan(ic) else 0.0

        ic_series = pd.Series(ic_values)

        # 处理IC值
        if self.use_abs_ic:
            ic_series = ic_series.abs()

        # 低于阈值的IC设为0
        ic_series[ic_series < self.ic_threshold] = 0

        # 归一化权重
        weights = self.normalize_weights(ic_series, remove_negative=True)

        return weights

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
            权重DataFrame，index为日期，columns为因子名称
        """
        factor_names = df_signals.columns.tolist()
        dates = df_signals.index.tolist()

        weights_list = []
        for date in dates:
            weights = self.calculate_weights(df_signals, df_returns, str(date))
            weights.name = date
            weights_list.append(weights)

        df_weights = pd.DataFrame(weights_list)
        df_weights.index = dates

        return df_weights

    def calculate_ic_series(self, df_signals: pd.DataFrame,
                            df_returns: pd.DataFrame) -> pd.DataFrame:
        """
        计算IC时间序列（用于分析和诊断）

        Parameters:
        -----------
        df_signals : pd.DataFrame
            信号数据，index为日期，columns为因子名称
        df_returns : pd.DataFrame
            收益率数据，需包含 'valuation_date' 和 'return' 列

        Returns:
        --------
        pd.DataFrame
            IC DataFrame，index为日期，columns为因子名称
        """
        factor_names = df_signals.columns.tolist()
        dates = df_signals.index.tolist()

        # 准备对齐后的数据
        df_signals_aligned, forward_returns = self._prepare_data(df_signals, df_returns)

        ic_list = []
        for i, date in enumerate(df_signals_aligned.index):
            # 获取截至当前日期的数据
            current_idx = df_signals_aligned.index.get_loc(date)

            # 需要足够的历史数据
            if current_idx < self.min_periods - 1:
                ic_values = {factor: np.nan for factor in factor_names}
            else:
                # 使用滚动窗口
                start_idx = max(0, current_idx - self.lookback_window + 1)
                df_window_signals = df_signals_aligned.iloc[start_idx:current_idx + 1]
                window_returns = forward_returns.iloc[start_idx:current_idx + 1]

                ic_values = {}
                for factor in factor_names:
                    ic = self._calculate_ic(df_window_signals[factor], window_returns)
                    ic_values[factor] = ic

            ic_series = pd.Series(ic_values, name=date)
            ic_list.append(ic_series)

        df_ic = pd.DataFrame(ic_list)
        df_ic.index = df_signals_aligned.index

        return df_ic
