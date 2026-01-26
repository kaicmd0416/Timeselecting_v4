"""
信号构建模块 (signal_constructing)

本模块提供信号构建的核心逻辑，包括：
- 方向决策函数
- 均线差异信号构建
- M1M2信号构建
- 技术指标信号构建
- 月度效应信号构建
- 因子处理函数

作者: TimeSelecting Team
版本: v3.0
"""

import pandas as pd
import os
import sys
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
import global_setting.global_dic as glv
from numpy import *
from pykalman import KalmanFilter
import numpy as np

class signal_construct:
    """
    信号构建类
    
    提供各种信号构建方法，将因子数据转换为择时信号。
    """
    
    def direction_decision(self, x):
        """
        方向决策函数1（正向）
        
        当x > 0时，返回0（看多沪深300）
        当x <= 0时，返回1（看多中证2000）
        
        Parameters:
        -----------
        x : float
            因子值或差值
        
        Returns:
        --------
        int
            0（沪深300）或1（中证2000）
        """
        if x > 0:
            return 0  # 沪深300
        else:
            return 1  # 中证2000
    def direction_decision2(self, x):
        """
        方向决策函数2（反向）
        
        当x < 0时，返回0（看多沪深300）
        当x >= 0时，返回1（看多中证2000）
        
        Parameters:
        -----------
        x : float
            因子值或差值
        
        Returns:
        --------
        int
            0（沪深300）或1（中证2000）
        """
        if x < 0:
            return 0  # 沪深300
        else:
            return 1  # 中证2000
    def MA_difference_signal_construct(self, df, rolling_window_list, mode_type):
        """
        均线差异信号构建
        
        计算短期均线和长期均线的差值，根据mode_type决定信号方向。
        
        Parameters:
        -----------
        df : pd.DataFrame
            包含因子数据的DataFrame，第二列为因子值
        rolling_window_list : list
            滚动窗口列表，[短期窗口, 长期窗口]
        mode_type : str
            模式类型：
            - 'mode_1', 'mode_3', 'mode_8', 'mode_10': 正向direction
            - 'mode_2', 'mode_4', 'mode_9', 'mode_11': 反向direction
        
        Returns:
        --------
        int
            最终信号：0（沪深300）或1（中证2000）
        
        Raises:
        -------
        ValueError
            当mode_type不在支持范围内时抛出异常
        """
        signal_name=df.columns.tolist()[1]
        rolling_window_short=min(rolling_window_list)
        rolling_window_long=max(rolling_window_list)       
        MA_short=mean(df[signal_name].tolist()[-rolling_window_short:])
        MA_long=mean(df[signal_name].tolist()[-rolling_window_long :])
        difference = MA_short-MA_long
        if mode_type=='mode_1' or mode_type=='mode_3' or mode_type=='mode_8' or mode_type=='mode_10':
             final_signal = self.direction_decision(difference)
        elif mode_type=='mode_2' or mode_type=='mode_4' or mode_type=='mode_9' or mode_type=='mode_11':
            final_signal = self.direction_decision2(difference)
        else:
             final_signal=None
             print('没有mode_type')
             raise ValueError
        return final_signal
    def M1M2_signal_construct(self, df, rolling_window):
        """
        M1M2信号构建
        
        基于M2和M1-M2差值的均线构建信号。
        
        Parameters:
        -----------
        df : pd.DataFrame
            包含M1、M2和difference列的DataFrame
        rolling_window : int
            滚动窗口大小
        
        Returns:
        --------
        int
            最终信号：0（沪深300）或1（中证2000）
        """
        MA_1 = mean(df['M2'].tolist()[-rolling_window:])
        MA_2 = mean(df['difference'].tolist()[-rolling_window:])
        M2 = df['M2'].tolist()[-1]
        M1_M2 = df['difference'].tolist()[-1]
        difference_1 = M2 - MA_1
        difference_2 = M1_M2 - MA_2
        if difference_1 < 0 and difference_2 < 0:
            final_signal = 0
        elif difference_1 > 0 and difference_2 > 0:
            final_signal = 0
        else:
            final_signal = 1
        return final_signal
    def Monthly_effect_signal_construct(self, df):
        """
        季节性效应信号构建（适用于Monthly_Effect, Post_Holiday_Effect）

        基于历史平均收益率差值构建信号。
        自动识别数据列名（monthly_effect, post_holiday_effect）

        Parameters:
        -----------
        df : pd.DataFrame
            包含效应值列的DataFrame

        Returns:
        --------
        float
            最终信号：0（大盘）、1（小盘）、0.5（中性，各配50%）
        """
        # 自动识别效应值列名（仅用于需要回测历史数据的因子）
        effect_cols = ['monthly_effect', 'post_holiday_effect']
        value_col = None
        for col in effect_cols:
            if col in df.columns:
                value_col = col
                break

        if value_col is None:
            raise ValueError(f"DataFrame中未找到效应值列，期望列名: {effect_cols}")

        a = df[value_col].tolist()[-1]

        # 0.5为中性信号，直接返回
        if a == 0.5:
            final_signal = 0.5
        elif a > 0:
            final_signal = 0
        else:
            final_signal = 1
        return final_signal

    def fixed_time_signal_construct(self, df):
        """
        固定时间触发信号构建（适用于 Earnings_Season 等固定时间触发因子）

        这类因子在特定时间段固定买入大盘或小盘，不需要回测历史数据选择参数。
        因子值直接决定信号：
        - 因子值 > 0 → final_signal = 0 → 买大盘
        - 因子值 < 0 → final_signal = 1 → 买小盘
        - 因子值 = 0.5 → final_signal = 0.5 → 中性

        Parameters:
        -----------
        df : pd.DataFrame
            包含因子值列的DataFrame

        Returns:
        --------
        float
            最终信号：0（大盘）、1（小盘）、0.5（中性）
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

        a = df[value_col].tolist()[-1]

        # 直接根据因子值决定信号
        if a == 0.5:
            final_signal = 0.5
        elif a > 0:
            final_signal = 0  # 买大盘
        else:
            final_signal = 1  # 买小盘
        return final_signal

    def Monthly_data_construct(self, df):
        """
        月度效应信号构建

        基于历史同月份的平均收益率差值构建信号。

        Parameters:
        -----------
        df : pd.DataFrame
            包含monthly_effect列的DataFrame

        Returns:
        --------
        int
            最终信号：0（沪深300）或1（中证2000）
        """
        signal_name = df.columns.tolist()[1]
        a = df[signal_name].unique().tolist()[-1]
        b = df[signal_name].unique().tolist()[-2]
        if a > b:
            final_signal = 0
        else:
            final_signal = 1
        return final_signal
    def RSRS_construct(self, df,x_list):
        """
        月度效应信号构建

        基于历史同月份的平均收益率差值构建信号。

        Parameters:
        -----------
        df : pd.DataFrame
            包含monthly_effect列的DataFrame

        Returns:
        --------
        int
            最终信号：0（沪深300）或1（中证2000）
        """
        final_signal_list=[]
        signal_name = df.columns.tolist()[1]
        a = df[signal_name].unique().tolist()[-1]
        for x in x_list:
            if a > x:
                final_signal = 0
            elif a<-x:
                final_signal=1
            else:
                final_signal = 0.5
            final_signal_list.append(final_signal)
        return final_signal_list
    def technical_signal_construct(self, df, signal_name):
        """
        技术指标信号构建
        
        根据不同的技术指标构建信号：
        - TargetIndex_MACD: 基于MACD和MACD信号线
        - TargetIndex_MOMENTUM2: 基于动量值
        - TargetIndex_BBANDS: 基于布林带
        - TargetIndex_KDJ: 基于KDJ交叉
        - TargetIndex_PSA: 基于抛物线SAR
        
        Parameters:
        -----------
        df : pd.DataFrame
            包含技术指标数据的DataFrame
        signal_name : str
            信号名称
        
        Returns:
        --------
        float
            最终信号：0（沪深300）、1（中证2000）或0.5（中性）
        """
        # 初始化final_signal为默认值
        final_signal = 0.5
        
        if signal_name=='TargetIndex_MACD':
            macd = df['MACD'].tolist()[-1]
            macd_s = df['MACD_s'].tolist()[-1]
            if macd > macd_s:
                final_signal = 0
            else:
                final_signal = 1
        elif signal_name=='TargetIndex_MOMENTUM2':
            x=df['TargetIndex_MOMENTUM'].tolist()[-1]
            if x>0:
                final_signal=0
            else:
                final_signal=1
        elif signal_name=='TargetIndex_BBANDS':
            upper = df['upper'].tolist()[-1]
            lower = df['lower'].tolist()[-1]
            target_index=df['target_index'].tolist()[-1]
            if target_index>=upper:
                final_signal=0
            elif target_index<=lower:
                final_signal=1
            else:
                final_signal=0.5
        elif signal_name=='TargetIndex_KDJ':
            final_signal=0.5
            # 获取KDJ数据
            K = df['K_9_3'].tolist()
            D = df['D_9_3'].tolist()
            J = df['J_9_3'].tolist()
            # 检查是否有足够的数据
            if len(K) < 10:
                return final_signal
            # 查找过去10天中case1（K在20左右向上交叉D）的触发时间
            case1_trigger_index = -1
            for i in range(1, min(11, len(K))):
                if (K[-i-1] <= D[-i-1] and K[-i] > D[-i] and 
                    K[-i-1]<= 25 and D[-i-1]<= 25):
                    case1_trigger_index = -i
                    break
            # 查找过去10天中case2（K在80左右向下交叉D）的触发时间
            case2_trigger_index = -1
            for i in range(1, min(11, len(K))):
                if (K[-i-1] >= D[-i-1] and K[-i] < D[-i] and 
                    K[-i-1] >= 75 and D[-i-1] >= 75):
                    case2_trigger_index = -i
                    break
            # 检查从case1触发到现在是否有case2发生
            if case1_trigger_index != -1:
                case2_occurred = False
                for i in range(case1_trigger_index, 0):
                    if K[i]<D[i] or (K[i]>80 and D[i]>80):
                        case2_occurred = True
                        break
                if not case2_occurred:
                    final_signal = 1
                    #print(df.iloc[case2_trigger_index:],final_signal)
                    return final_signal
            # 检查从case2触发到现在是否有case1发生
            if case2_trigger_index != -1:
                case1_occurred = False
                for i in range(case2_trigger_index, 0):
                    if  K[i]>D[i] or (K[i]<20 and D[i]<20):
                        case1_occurred = True
                        break
                if not case1_occurred:
                    final_signal = 0
                   #print(df.iloc[case2_trigger_index:],final_signal)
                    return final_signal
        elif signal_name=='TargetIndex_PSA':
            PSAL=df['PSARl_0.02_0.2'].tolist()[-1]
            PSAS=df['PSARs_0.02_0.2'].tolist()[-1]
            
            # 检查两个值是否都是NaN
            if np.isnan(PSAL) and np.isnan(PSAS):
                final_signal = 0.5
            # 如果PSAL不是NaN，返回1
            elif not np.isnan(PSAL) and np.isnan(PSAS):
                final_signal = 0
            # 如果PSAS不是NaN，返回0
            elif not np.isnan(PSAS) and np.isnan(PSAL):
                final_signal = 1
            # 其他情况返回0.5
            else:
                final_signal = 0.5
        elif signal_name=='TargetIndex_REVERSE':
            past_differences = df['difference'].tolist()[-10:]  # 过去10天的difference值
            df_vix_300=df[['valuation_date','hs300']]
            df_vix_1000=df[['valuation_date','zz1000']]
            df_vix_300.dropna(inplace=True)
            df_vix_1000.dropna(inplace=True)
            if len(df_vix_300)==0 and len(df_vix_1000)==0:
                vix=True
            else:
                if len(df_vix_1000)<252:
                    df_vix=df_vix_300
                else:
                    df_vix=df_vix_1000
                if len(df_vix)<252:
                    vix=True
                else:
                    df_vix['quantile_09'] = df_vix[df_vix.columns.tolist()[1]].rolling(252).quantile(0.8)
                    vix_last = df_vix[df_vix.columns.tolist()[1]].tolist()[-1]
                    vix_quantile = df_vix['quantile_09'].tolist()[-1]
                    if vix_last >= vix_quantile:
                        vix = True
                    else:
                        vix = False
            final_signal = 0.5  # 默认值
            case1_trigger_index = -1
            for i in range(1, min(11, len(past_differences))):
                if past_differences[-i] > 0.075 and vix==True:
                    case1_trigger_index = -i
                    break
            # 查找过去10天中case2（K在80左右向下交叉D）的触发时间
            case2_trigger_index = -1
            for i in range(1, min(11, len(past_differences))):
                if past_differences[-i]<-0.3:
                    case2_trigger_index = -i
                    break
            if case1_trigger_index!=-1:
                case1_occurred = False
                for i in range(case1_trigger_index, 0):
                    if past_differences[i]<-0.05:
                        case1_occurred = True
                        break
                if not case1_occurred:
                    final_signal = 1
            if case2_trigger_index!=-1:
                case2_occurred = False
                for i in range(case2_trigger_index, 0):
                    if past_differences[i]>0.02:
                        case2_occurred = True
                        break
                if not case2_occurred:
                    final_signal = 0
        elif signal_name=='TargetIndex_REVERSE2':
            df['quantile_0.8'] = df['difference'].rolling(500).quantile(0.9)
            df['quantile_0.2'] = df['difference'].rolling(500).quantile(0.1)
            difference=df['difference'].tolist()[-1]
            quantile_08=df['quantile_0.8'].tolist()[-1]
            quantile_02=df['quantile_0.2'].tolist()[-1]
            if difference>quantile_08:
                final_signal=1
            elif difference<quantile_02:
                final_signal=0
            else:
                final_signal=0.5
        return final_signal


class factor_processing:
    def slice_processing(self,df,target_date):
        df = df[df['valuation_date'] < target_date]
        df.reset_index(inplace=True, drop=True)
        return df
    def slice_processing2(self, df, target_date):
        """
        数据切片处理（模式2）
        
        获取target_date之前的所有数据（不包括target_date）
        与slice_processing功能相同，但可能用于不同的场景
        
        Parameters:
        -----------
        df : pd.DataFrame
            完整的因子数据DataFrame
        target_date : str
            目标日期，格式为 'YYYY-MM-DD'
        
        Returns:
        --------
        pd.DataFrame
            切片后的DataFrame
        """
        df = df[df['valuation_date'] <= target_date]
        df.reset_index(inplace=True, drop=True)
        return df
    def slice_processing_Monthly(self, df, target_date):
        """
        月度数据切片处理
        
        用于月度因子，获取target_date之前的所有数据
        
        Parameters:
        -----------
        df : pd.DataFrame
            完整的因子数据DataFrame
        target_date : str
            目标日期，格式为 'YYYY-MM-DD'
        
        Returns:
        --------
        pd.DataFrame
            切片后的DataFrame
        """
        df_final=df.copy()
        if target_date<='2025-03-01':
            df_final.set_index('valuation_date', inplace=True, drop=True)
            df_final = df_final.shift(20)
            df_final.dropna(inplace=True)
            df_final.reset_index(inplace=True)
        df_final=df_final[df_final['valuation_date'] < target_date]
        return df_final