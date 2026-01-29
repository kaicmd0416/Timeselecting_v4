"""
因子加权模块 (factor_weighting)

本模块提供因子加权功能，支持多种加权方式用于信号聚合。

主要组件:
    - BaseWeighter: 加权器抽象基类
    - ICWeighter: 基于时序IC的加权器
    - ReturnCalculator: 收益率计算工具

使用示例:
    from factor_weighting import ICWeighter, ReturnCalculator

    # 计算收益率
    return_calc = ReturnCalculator(start_date, end_date, big_index, small_index)
    df_returns = return_calc.get_relative_returns()

    # 创建IC加权器并计算权重
    weighter = ICWeighter(lookback_window=252, min_periods=252)
    weights = weighter.calculate_weights_series(df_signals, df_returns)

    # 应用权重
    df_weighted = weighter.apply_weights(df_signals, weights)

作者: TimeSelecting Team
版本: v1.0
"""

from .base_weighter import BaseWeighter
from .ic_weighter import ICWeighter
from .icir_weighter import ICIRWeighter
from .sharpe_weighter import SharpeWeighter
from .momentum_weighter import MomentumWeighter
from .return_calculator import ReturnCalculator

__all__ = [
    'BaseWeighter',
    'ICWeighter',
    'ICIRWeighter',
    'SharpeWeighter',
    'MomentumWeighter',
    'ReturnCalculator'
]
