"""
信号构建模块 (signal_constructing)

本模块是L3信号生成的核心逻辑层，负责将因子数据转换为择时信号。
包含两个主要类：
1. signal_construct: 信号构建类，提供各种信号生成方法
2. factor_processing: 因子处理类，提供数据切片等辅助功能

信号构建方法分类:
    均线类方法:
        - MA_difference_signal_construct: 短期均线与长期均线差值信号
        - M1M2_signal_construct: M1/M2货币供应量专用信号

    技术指标类方法:
        - technical_signal_construct: 综合技术指标信号（MACD/KDJ/BBANDS/PSA等）
        - RSRS_construct: 阻力支撑相对强度信号

    季节性/事件类方法:
        - Monthly_effect_signal_construct: 月度效应信号
        - fixed_time_signal_construct: 固定时间触发信号
        - Monthly_data_construct: 月度数据信号

信号值含义:
    - 0: 看多大盘（如沪深300/上证50）
    - 1: 看多小盘（如中证2000）
    - 0.5: 中性（大小盘各配50%）

Forward-Looking防护:
    factor_processing.slice_processing使用严格小于运算符:
    df = df[df['valuation_date'] < target_date]
    确保信号生成只使用target_date之前的数据。

作者: TimeSelecting Team
版本: v3.0
"""

# ==================== 标准库导入 ====================
import os
import sys

# ==================== 第三方库导入 ====================
import pandas as pd
import numpy as np
from numpy import mean  # 用于计算均值
from pykalman import KalmanFilter  # 卡尔曼滤波器（用于RSRS因子）

# ==================== 自定义模块导入 ====================
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
import global_setting.global_dic as glv


class signal_construct:
    """
    信号构建类

    提供各种信号构建方法，将因子数据转换为择时信号。
    是L3信号生成的核心计算引擎。

    核心方法:
        - direction_decision: 正向方向决策（差值>0→买大盘）
        - direction_decision2: 反向方向决策（差值<0→买大盘）
        - MA_difference_signal_construct: 均线差异信号构建
        - M1M2_signal_construct: M1M2信号构建
        - Monthly_effect_signal_construct: 季节性效应信号构建
        - fixed_time_signal_construct: 固定时间触发信号构建
        - RSRS_construct: RSRS信号构建
        - technical_signal_construct: 技术指标信号构建
    """

    def direction_decision(self, x):
        """
        正向方向决策函数

        用于正向因子（因子值越大，越看好大盘）。
        当短期均线 > 长期均线时，买入大盘。

        逻辑:
            x > 0 → 返回0（买大盘）
            x <= 0 → 返回1（买小盘）

        Parameters:
        -----------
        x : float
            因子差值（通常是短期均线 - 长期均线）

        Returns:
        --------
        int
            0（买大盘）或 1（买小盘）

        适用模式:
            mode_1, mode_3, mode_8, mode_10
        """
        if x > 0:
            return 0  # 买大盘
        else:
            return 1  # 买小盘

    def direction_decision2(self, x):
        """
        反向方向决策函数

        用于反向因子（因子值越大，越看好小盘）。
        当短期均线 > 长期均线时，买入小盘。

        逻辑:
            x < 0 → 返回0（买大盘）
            x >= 0 → 返回1（买小盘）

        Parameters:
        -----------
        x : float
            因子差值（通常是短期均线 - 长期均线）

        Returns:
        --------
        int
            0（买大盘）或 1（买小盘）

        适用模式:
            mode_2, mode_4, mode_9, mode_11
        """
        if x < 0:
            return 0  # 买大盘
        else:
            return 1  # 买小盘

    def MA_difference_signal_construct(self, df, rolling_window_list, mode_type):
        """
        均线差异信号构建

        计算短期均线和长期均线的差值，根据mode_type决定信号方向。
        这是最常用的信号构建方法，适用于大多数因子。

        计算逻辑:
            1. 提取因子值序列（DataFrame第二列）
            2. 计算短期均线: MA_short = mean(最近N个值)
            3. 计算长期均线: MA_long = mean(最近M个值)
            4. 计算差值: difference = MA_short - MA_long
            5. 根据mode_type调用对应的方向决策函数

        Parameters:
        -----------
        df : pd.DataFrame
            包含因子数据的DataFrame
            - 第一列: valuation_date（日期）
            - 第二列: 因子值
        rolling_window_list : list
            滚动窗口列表，如[5, 20]
            - 较小的值作为短期窗口
            - 较大的值作为长期窗口
        mode_type : str
            模式类型:
            - 正向模式: 'mode_1', 'mode_3', 'mode_8', 'mode_10'
              差值>0 → 买大盘
            - 反向模式: 'mode_2', 'mode_4', 'mode_9', 'mode_11'
              差值>0 → 买小盘

        Returns:
        --------
        int
            最终信号: 0（买大盘）或 1（买小盘）

        Raises:
        -------
        ValueError
            当mode_type不在支持范围内时抛出异常

        示例:
            >>> df = pd.DataFrame({'valuation_date': dates, 'factor': values})
            >>> signal = sc.MA_difference_signal_construct(df, [5, 20], 'mode_1')
        """
        # 获取因子值列名（第二列）
        signal_name = df.columns.tolist()[1]

        # 确定短期和长期窗口
        rolling_window_short = min(rolling_window_list)
        rolling_window_long = max(rolling_window_list)

        # 计算短期和长期均线
        MA_short = mean(df[signal_name].tolist()[-rolling_window_short:])
        MA_long = mean(df[signal_name].tolist()[-rolling_window_long:])

        # 计算差值
        difference = MA_short - MA_long

        # 根据模式类型选择方向决策函数
        if mode_type in ['mode_1', 'mode_3', 'mode_8', 'mode_10']:
            # 正向模式：差值>0 → 买大盘
            final_signal = self.direction_decision(difference)
        elif mode_type in ['mode_2', 'mode_4', 'mode_9', 'mode_11']:
            # 反向模式：差值>0 → 买小盘
            final_signal = self.direction_decision2(difference)
        else:
            final_signal = None
            print(f'不支持的mode_type: {mode_type}')
            raise ValueError(f'不支持的mode_type: {mode_type}')

        return final_signal

    def M1M2_signal_construct(self, df, rolling_window):
        """
        M1M2信号构建

        基于M2货币供应量和M1-M2剪刀差的均线构建信号。
        这是M1M2因子（mode_5）的专用信号构建方法。

        信号逻辑:
            - M2 < M2均线 且 剪刀差 < 剪刀差均线 → 买大盘
            - M2 > M2均线 且 剪刀差 > 剪刀差均线 → 买大盘
            - 其他情况 → 买小盘

        经济含义:
            - M2收缩+剪刀差收窄: 流动性紧缩，大盘相对抗跌
            - M2扩张+剪刀差扩大: 流动性宽松传导中，大盘受益
            - 其他组合: 信号不明确，偏向小盘

        Parameters:
        -----------
        df : pd.DataFrame
            包含以下列的DataFrame:
            - valuation_date: 日期
            - M2: M2货币供应量同比增速
            - difference: M1-M2剪刀差
        rolling_window : int
            滚动窗口大小，用于计算均线

        Returns:
        --------
        int
            最终信号: 0（买大盘）或 1（买小盘）
        """
        # 计算M2和剪刀差的均线
        MA_1 = mean(df['M2'].tolist()[-rolling_window:])
        MA_2 = mean(df['difference'].tolist()[-rolling_window:])

        # 获取最新值
        M2 = df['M2'].tolist()[-1]
        M1_M2 = df['difference'].tolist()[-1]

        # 计算与均线的偏离
        difference_1 = M2 - MA_1        # M2相对于均线的位置
        difference_2 = M1_M2 - MA_2     # 剪刀差相对于均线的位置

        # 信号判断
        if difference_1 < 0 and difference_2 < 0:
            # M2收缩 + 剪刀差收窄 → 买大盘
            final_signal = 0
        elif difference_1 > 0 and difference_2 > 0:
            # M2扩张 + 剪刀差扩大 → 买大盘
            final_signal = 0
        else:
            # 其他情况 → 买小盘
            final_signal = 1

        return final_signal

    def Monthly_effect_signal_construct(self, df):
        """
        季节性效应信号构建

        适用于Monthly_Effect（月度效应）和Post_Holiday_Effect（节后效应）因子。
        基于历史同期平均收益率差值构建信号。

        信号逻辑:
            - 效应值 > 0: 历史上该时期大盘表现更好 → 买大盘
            - 效应值 < 0: 历史上该时期小盘表现更好 → 买小盘
            - 效应值 = 0.5: 中性信号

        Parameters:
        -----------
        df : pd.DataFrame
            包含效应值列的DataFrame，自动识别列名:
            - monthly_effect: 月度效应值
            - post_holiday_effect: 节后效应值

        Returns:
        --------
        float
            最终信号: 0（买大盘）、1（买小盘）或 0.5（中性）

        Raises:
        -------
        ValueError
            当DataFrame中找不到效应值列时抛出异常
        """
        # 自动识别效应值列名
        effect_cols = ['monthly_effect', 'post_holiday_effect']
        value_col = None
        for col in effect_cols:
            if col in df.columns:
                value_col = col
                break

        if value_col is None:
            raise ValueError(f"DataFrame中未找到效应值列，期望列名: {effect_cols}")

        # 获取最新的效应值
        a = df[value_col].tolist()[-1]

        # 根据效应值决定信号
        if a == 0.5:
            final_signal = 0.5  # 中性
        elif a > 0:
            final_signal = 0    # 买大盘
        else:
            final_signal = 1    # 买小盘

        return final_signal

    def fixed_time_signal_construct(self, df):
        """
        固定时间触发信号构建

        适用于Earnings_Season（财报季）等固定时间触发因子。
        这类因子在特定时间段固定买入大盘或小盘，不需要回测选择参数。

        信号逻辑:
            - 因子值 > 0 → 买大盘（如财报披露前2周）
            - 因子值 < 0 → 买小盘（如财报披露后2周）
            - 因子值 = 0.5 → 中性

        Parameters:
        -----------
        df : pd.DataFrame
            包含因子值列的DataFrame，自动识别列名:
            - earnings_season: 财报季因子值

        Returns:
        --------
        float
            最终信号: 0（买大盘）、1（买小盘）或 0.5（中性）

        Raises:
        -------
        ValueError
            当DataFrame中找不到因子值列时抛出异常
        """
        # 自动识别因子值列名
        factor_cols = ['earnings_season']
        value_col = None
        for col in factor_cols:
            if col in df.columns:
                value_col = col
                break

        if value_col is None:
            raise ValueError(f"DataFrame中未找到因子值列，期望列名: {factor_cols}")

        # 获取最新的因子值
        a = df[value_col].tolist()[-1]

        # 直接根据因子值决定信号
        if a == 0.5:
            final_signal = 0.5  # 中性
        elif a > 0:
            final_signal = 0    # 买大盘
        else:
            final_signal = 1    # 买小盘

        return final_signal

    def Monthly_data_construct(self, df):
        """
        月度数据信号构建

        基于月度因子值的变化趋势构建信号。
        比较最近两个不同的因子值，判断趋势方向。

        信号逻辑:
            - 最新值 > 前一个值 → 买大盘（趋势向上）
            - 最新值 <= 前一个值 → 买小盘（趋势向下）

        Parameters:
        -----------
        df : pd.DataFrame
            包含月度因子数据的DataFrame
            - 第一列: valuation_date
            - 第二列: 因子值

        Returns:
        --------
        int
            最终信号: 0（买大盘）或 1（买小盘）

        注意:
            使用unique()去重后比较，适用于月度数据
            （同一月份的多个交易日因子值相同）
        """
        # 获取因子值列名
        signal_name = df.columns.tolist()[1]

        # 获取去重后的最后两个值
        unique_values = df[signal_name].unique().tolist()
        a = unique_values[-1]   # 最新值
        b = unique_values[-2]   # 前一个值

        # 趋势判断
        if a > b:
            final_signal = 0  # 趋势向上 → 买大盘
        else:
            final_signal = 1  # 趋势向下 → 买小盘

        return final_signal

    def RSRS_construct(self, df, x_list):
        """
        RSRS信号构建

        RSRS（阻力支撑相对强度）是一个基于回归斜率的技术指标。
        根据不同的阈值x生成对应的信号列表。

        信号逻辑:
            - RSRS值 > x → 买大盘（上涨动能强）
            - RSRS值 < -x → 买小盘（下跌动能强）
            - -x <= RSRS值 <= x → 中性

        Parameters:
        -----------
        df : pd.DataFrame
            包含RSRS因子数据的DataFrame
            - 第一列: valuation_date
            - 第二列: RSRS值（通常经过标准化处理）
        x_list : list
            阈值列表，如[0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

        Returns:
        --------
        list
            对应每个x阈值的信号列表，每个元素为:
            0（买大盘）、1（买小盘）或 0.5（中性）

        特殊说明:
            RSRS使用mode_13，与其他模式不同，
            它直接为每个x值生成独立的信号。
        """
        final_signal_list = []

        # 获取因子值列名
        signal_name = df.columns.tolist()[1]

        # 获取最新的RSRS值
        a = df[signal_name].unique().tolist()[-1]

        # 对每个阈值x生成信号
        for x in x_list:
            if a > x:
                final_signal = 0    # RSRS高 → 买大盘
            elif a < -x:
                final_signal = 1    # RSRS低 → 买小盘
            else:
                final_signal = 0.5  # 中性
            final_signal_list.append(final_signal)

        return final_signal_list

    def technical_signal_construct(self, df, signal_name):
        """
        技术指标信号构建

        综合处理多种技术指标信号，是mode_6的核心方法。
        根据signal_name调用对应的技术指标信号逻辑。

        支持的技术指标:
            - TargetIndex_MACD: MACD金叉死叉
            - TargetIndex_MOMENTUM2: 动量指标
            - TargetIndex_BBANDS: 布林带突破
            - TargetIndex_KDJ: KDJ金叉死叉
            - TargetIndex_PSA: 抛物线SAR
            - TargetIndex_REVERSE: VIX结合的反转策略
            - TargetIndex_REVERSE2: 分位数反转策略

        Parameters:
        -----------
        df : pd.DataFrame
            包含技术指标数据的DataFrame，列名因指标而异
        signal_name : str
            信号名称，用于选择对应的处理逻辑

        Returns:
        --------
        float
            最终信号: 0（买大盘）、1（买小盘）或 0.5（中性）
        """
        # 初始化默认信号
        final_signal = 0.5

        # ==================== MACD指标 ====================
        if signal_name == 'TargetIndex_MACD':
            # MACD > 信号线 → 买大盘（金叉状态）
            macd = df['MACD'].tolist()[-1]
            macd_s = df['MACD_s'].tolist()[-1]
            if macd > macd_s:
                final_signal = 0
            else:
                final_signal = 1

        # ==================== 动量指标 ====================
        elif signal_name == 'TargetIndex_MOMENTUM2':
            # 动量 > 0 → 买大盘
            x = df['TargetIndex_MOMENTUM'].tolist()[-1]
            if x > 0:
                final_signal = 0
            else:
                final_signal = 1

        # ==================== 布林带 ====================
        elif signal_name == 'TargetIndex_BBANDS':
            # 价格突破上轨 → 买大盘
            # 价格突破下轨 → 买小盘
            # 价格在带内 → 中性
            upper = df['upper'].tolist()[-1]
            lower = df['lower'].tolist()[-1]
            target_index = df['target_index'].tolist()[-1]
            if target_index >= upper:
                final_signal = 0
            elif target_index <= lower:
                final_signal = 1
            else:
                final_signal = 0.5

        # ==================== KDJ指标 ====================
        elif signal_name == 'TargetIndex_KDJ':
            final_signal = 0.5
            # 获取KDJ数据
            K = df['K_9_3'].tolist()
            D = df['D_9_3'].tolist()
            J = df['J_9_3'].tolist()

            # 检查是否有足够的数据
            if len(K) < 10:
                return final_signal

            # 查找过去10天中的金叉信号（K在低位向上交叉D）
            case1_trigger_index = -1
            for i in range(1, min(11, len(K))):
                if (K[-i-1] <= D[-i-1] and K[-i] > D[-i] and
                        K[-i-1] <= 25 and D[-i-1] <= 25):
                    case1_trigger_index = -i
                    break

            # 查找过去10天中的死叉信号（K在高位向下交叉D）
            case2_trigger_index = -1
            for i in range(1, min(11, len(K))):
                if (K[-i-1] >= D[-i-1] and K[-i] < D[-i] and
                        K[-i-1] >= 75 and D[-i-1] >= 75):
                    case2_trigger_index = -i
                    break

            # 检查金叉后是否被死叉否定
            if case1_trigger_index != -1:
                case2_occurred = False
                for i in range(case1_trigger_index, 0):
                    if K[i] < D[i] or (K[i] > 80 and D[i] > 80):
                        case2_occurred = True
                        break
                if not case2_occurred:
                    final_signal = 1  # 金叉有效 → 买小盘（反转逻辑）
                    return final_signal

            # 检查死叉后是否被金叉否定
            if case2_trigger_index != -1:
                case1_occurred = False
                for i in range(case2_trigger_index, 0):
                    if K[i] > D[i] or (K[i] < 20 and D[i] < 20):
                        case1_occurred = True
                        break
                if not case1_occurred:
                    final_signal = 0  # 死叉有效 → 买大盘（反转逻辑）
                    return final_signal

        # ==================== 抛物线SAR ====================
        elif signal_name == 'TargetIndex_PSA':
            PSAL = df['PSARl_0.02_0.2'].tolist()[-1]  # 多头SAR
            PSAS = df['PSARs_0.02_0.2'].tolist()[-1]  # 空头SAR

            # 两个值都是NaN → 中性
            if np.isnan(PSAL) and np.isnan(PSAS):
                final_signal = 0.5
            # 只有多头SAR有值 → 买大盘（多头趋势）
            elif not np.isnan(PSAL) and np.isnan(PSAS):
                final_signal = 0
            # 只有空头SAR有值 → 买小盘（空头趋势）
            elif not np.isnan(PSAS) and np.isnan(PSAL):
                final_signal = 1
            else:
                final_signal = 0.5

        # ==================== VIX反转策略 ====================
        elif signal_name == 'TargetIndex_REVERSE':
            past_differences = df['difference'].tolist()[-10:]

            # 获取VIX数据判断市场恐慌程度
            df_vix_300 = df[['valuation_date', 'hs300']].copy()
            df_vix_1000 = df[['valuation_date', 'zz1000']].copy()
            df_vix_300.dropna(inplace=True)
            df_vix_1000.dropna(inplace=True)

            # 判断是否处于高波动状态
            if len(df_vix_300) == 0 and len(df_vix_1000) == 0:
                vix = True
            else:
                if len(df_vix_1000) < 252:
                    df_vix = df_vix_300
                else:
                    df_vix = df_vix_1000
                if len(df_vix) < 252:
                    vix = True
                else:
                    df_vix['quantile_09'] = df_vix[df_vix.columns.tolist()[1]].rolling(252).quantile(0.8)
                    vix_last = df_vix[df_vix.columns.tolist()[1]].tolist()[-1]
                    vix_quantile = df_vix['quantile_09'].tolist()[-1]
                    vix = vix_last >= vix_quantile

            final_signal = 0.5

            # 查找超涨信号
            case1_trigger_index = -1
            for i in range(1, min(11, len(past_differences))):
                if past_differences[-i] > 0.075 and vix:
                    case1_trigger_index = -i
                    break

            # 查找超跌信号
            case2_trigger_index = -1
            for i in range(1, min(11, len(past_differences))):
                if past_differences[-i] < -0.3:
                    case2_trigger_index = -i
                    break

            # 超涨信号有效性检查
            if case1_trigger_index != -1:
                case1_occurred = False
                for i in range(case1_trigger_index, 0):
                    if past_differences[i] < -0.05:
                        case1_occurred = True
                        break
                if not case1_occurred:
                    final_signal = 1  # 超涨后反转 → 买小盘

            # 超跌信号有效性检查
            if case2_trigger_index != -1:
                case2_occurred = False
                for i in range(case2_trigger_index, 0):
                    if past_differences[i] > 0.02:
                        case2_occurred = True
                        break
                if not case2_occurred:
                    final_signal = 0  # 超跌后反转 → 买大盘

        # ==================== 分位数反转策略 ====================
        elif signal_name == 'TargetIndex_REVERSE2':
            # 计算历史分位数
            df['quantile_0.8'] = df['difference'].rolling(500).quantile(0.9)
            df['quantile_0.2'] = df['difference'].rolling(500).quantile(0.1)

            difference = df['difference'].tolist()[-1]
            quantile_08 = df['quantile_0.8'].tolist()[-1]
            quantile_02 = df['quantile_0.2'].tolist()[-1]

            # 超过90%分位 → 买小盘（反转）
            if difference > quantile_08:
                final_signal = 1
            # 低于10%分位 → 买大盘（反转）
            elif difference < quantile_02:
                final_signal = 0
            else:
                final_signal = 0.5

        return final_signal


class factor_processing:
    """
    因子处理类

    提供因子数据处理的辅助功能，主要是数据切片操作。
    确保信号生成时只使用目标日期之前的数据，避免forward-looking。
    """

    def slice_processing(self, df, target_date):
        """
        数据切片处理（核心方法）

        获取target_date之前的所有数据，不包括target_date当天。
        这是避免forward-looking的关键步骤。

        使用场景:
            在L3_signal_main中，对于每个交易日target_date，
            调用此方法获取该日之前的数据用于信号计算。

        Parameters:
        -----------
        df : pd.DataFrame
            完整的因子数据DataFrame，必须包含'valuation_date'列
        target_date : str
            目标日期，格式为 'YYYY-MM-DD'

        Returns:
        --------
        pd.DataFrame
            只包含target_date之前数据的DataFrame（索引重置）

        示例:
            >>> df_slice = fp.slice_processing(df_full, '2024-01-15')
            >>> # df_slice只包含2024-01-14及之前的数据

        注意:
            使用严格小于运算符（<），确保不包含当天数据
        """
        df = df[df['valuation_date'] < target_date]
        df.reset_index(inplace=True, drop=True)
        return df

