"""
L3级别信号生成模块 (L3_signal_main)

本模块负责生成L3级别的择时信号。L3信号是最底层的信号，基于原始因子计算。

支持的信号模式：
- mode_1: 正向direction（因子值越大，信号越强）
- mode_2: 反向direction（因子值越小，信号越强）
- mode_3: 月度因子专用
- mode_4: 月度反向因子专用
- mode_5: M1M2专用
- mode_6: 技术指标专用
- mode_8: 短周期均线正向direction
- mode_9: 短周期均线反向direction
- mode_10: 长周期均线正向direction
- mode_11: 长周期均线反向direction

作者: TimeSelecting Team
版本: v3.0
"""

from datetime import datetime
import pandas as pd
from numpy import *
from data.data_prepare import data_prepare
from data.data_processing import data_processing
from factor_processing.signal_constructing import signal_construct, factor_processing
import os
import sys
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
import global_setting.global_dic as glv
import os

class L3_signalConstruction:
    """
    L3级别信号生成类
    
    负责生成L3级别的择时信号，支持多种信号模式和参数组合。
    
    Attributes:
    -----------
    signal_name : str
        信号名称，如 'NLBP_difference'、'credit_spread_3M' 等
    mode : str
        模式，'prod'（生产模式）或 'test'（测试模式）
    start_date : str
        处理后的开始日期（考虑数据需求）
    end_date : str
        结束日期，格式为 'YYYY-MM-DD'
    dp : data_prepare
        数据准备类实例
    dpro : data_processing
        数据处理类实例
    fp : factor_processing
        因子处理类实例
    sc : signal_construct
        信号构建类实例
    df_signal : pd.DataFrame
        信号数据DataFrame
    sc_mode : str
        信号构建模式（mode_1到mode_11）
    """
    
    def __init__(self, signal_name, mode, start_date, end_date):
        """
        初始化L3信号生成类
        
        Parameters:
        -----------
        signal_name : str
            信号名称
        mode : str
            模式，'prod' 或 'test'
        start_date : str
            开始日期，格式为 'YYYY-MM-DD'
        end_date : str
            结束日期，格式为 'YYYY-MM-DD'
        """
        self.signal_name=signal_name
        self.mode = mode
        self.start_date=start_date
        self.end_date=end_date
        self.available_date=gt.last_workday_calculate(self.end_date)
        self.dp=data_prepare(self.start_date,self.available_date)
        self.dpro=data_processing(self.start_date,self.available_date)
        self.fp=factor_processing()
        self.sc=signal_construct()
        self.df_signal,self.sc_mode=self.raw_data_preparing()
        self.start_date=self.start_date_processing(start_date)
    def final_signal_construction(self, final_signal, x):
        """
        构建最终信号
        
        根据final_signal和阈值x，生成最终的择时信号：
        - final_signal > x: 返回1（看多沪深300）
        - final_signal < (1-x): 返回0（看多中证2000）
        - 其他情况: 返回0.5（中性）
        
        Parameters:
        -----------
        final_signal : float
            最终信号值（通常在0-1之间）
        x : float
            阈值参数（通常在0.5-1之间）
        
        Returns:
        --------
        float
            最终信号：1（沪深300）、0（中证2000）或0.5（中性）
        """
        if final_signal > x:
            return 1  # 沪深300
        elif final_signal<(1-x):
            return 0
        else:
            return 0.5  # 中证2000
    def raw_data_preparing(self):
        """
        准备原始数据并确定信号构建模式
        
        根据signal_name调用相应的数据获取函数，并确定信号构建模式：
        - mode_1: 正向direction（因子值越大，信号越强）
        - mode_2: 反向direction（因子值越小，信号越强）
        - mode_3: 月度因子专用
        - mode_4: 月度反向因子专用
        - mode_5: M1M2专用
        - mode_6: 技术指标专用
        - mode_7: 月度效应专用
        - mode_8: 短周期均线正向direction
        - mode_9: 短周期均线反向direction
        - mode_10: 长周期均线正向direction
        - mode_11: 长周期均线反向direction
        - mode_12: 月度单调反向
        Returns:
        --------
        tuple
            (df_signal, sc_mode)
            - df_signal: 信号数据DataFrame
            - sc_mode: 信号构建模式字符串
        """
        if self.signal_name=='Shibor_2W': #正
            df=self.dp.raw_shibor(period='2W')
            sc_mode='mode_1'
        elif self.signal_name=='Shibor_9M': #正
            df = self.dp.raw_shibor(period='9M')
            sc_mode = 'mode_1'
        elif self.signal_name=='Bond_3Y':
            df=self.dp.raw_bond(period='3Y')
            sc_mode = 'mode_1'
        elif self.signal_name=='Bond_10Y':
            df=self.dp.raw_bond(period='10Y')
            sc_mode = 'mode_1'
        elif self.signal_name=='USDX':
            df=self.dp.raw_usdx()
            sc_mode = 'mode_2'
        elif self.signal_name=='USBond_3Y':
            df=self.dp.raw_usbond('3Y')
            sc_mode = 'mode_1'
        elif self.signal_name=='USBond_10Y':
            df=self.dp.raw_usbond('10Y')
            sc_mode = 'mode_1'
        elif self.signal_name=='CreditSpread_3M':
            df=self.dpro.credit_spread_3M()
            sc_mode = 'mode_1'
        elif self.signal_name == 'CreditSpread_9M':
            df = self.dpro.credit_spread_9M()
            sc_mode = 'mode_1'
        elif self.signal_name == 'CreditSpread_5Y':
            df = self.dpro.credit_spread_5Y()
            sc_mode = 'mode_1'
        elif self.signal_name=='TermSpread_9Y':
            df=self.dpro.term_spread_9Y()
            sc_mode='mode_2'
        elif self.signal_name=='M1M2':
            df=self.dpro.M1M2()
            sc_mode = 'mode_5'  #专属mode
        elif self.signal_name=='USStock':
            df=self.dpro.US_stock()
            sc_mode = 'mode_1'
        elif self.signal_name=='RelativeVolume_std':
            df=self.dpro.relativeVolume_std()
            sc_mode='mode_1'
        elif self.signal_name=='RelativeReturn_std':
            df=self.dpro.relativeReturn_std()
            sc_mode='mode_1'
        elif self.signal_name=='EarningsYield_Reverse':
            df=self.dp.raw_index_earningsyield()
            sc_mode='mode_10'
        elif self.signal_name=='Growth':
            df=self.dp.raw_index_growth()
            sc_mode='mode_11'
        elif self.signal_name=='LHBProportion':
            df=self.dpro.LHBProportion()
            sc_mode='mode_2'
        elif self.signal_name=='NLBP_difference':
            df=self.dpro.NetLeverageBuying()
            sc_mode='mode_1'
        elif self.signal_name=='LargeOrder_difference':
            df=self.dpro.LargeOrder_difference()
            sc_mode='mode_8'
        elif self.signal_name=='Monthly_effect':
            df=self.dpro.monthly_effect()
            sc_mode='mode_7'
        elif self.signal_name=='CPI':
            df=self.dp.raw_CPI_withdraw()
            sc_mode = 'mode_4'
        elif self.signal_name=='SocialFinance':
            df=self.dp.raw_socialfinance()
            sc_mode = 'mode_4'
        elif self.signal_name=='PPI':
            df=self.dp.raw_PPI_withdraw()
            sc_mode = 'mode_3'
        elif self.signal_name=='PMI':
            df=self.dp.raw_PMI_withdraw()
            sc_mode = 'mode_3'
        elif self.signal_name=='Stock_HL':
            df = self.dpro.stock_highLow()
            sc_mode = 'mode_2'
        elif self.signal_name=='Stock_RT':
            df = self.dpro.stock_raisingtrend()
            sc_mode = 'mode_9'
        elif self.signal_name=='Rsi_difference':
            df=self.dpro.stock_rsi()
            sc_mode='mode_2'
        elif self.signal_name=='RaisingTrend_proportion':
            df=self.dpro.stock_trend()
            sc_mode='mode_2'
        elif self.signal_name=='International_Index':
            df=self.dp.raw_internationalIndex()
            sc_mode='mode_8'
        elif self.signal_name == 'ETF_Shares':
            df = self.dp.raw_fund()
            sc_mode = 'mode_8'
        elif self.signal_name=='TargetIndex_MACD':
            df=self.dpro.targetIndex_MACD()
            sc_mode='mode_6'
        elif self.signal_name=='TargetIndex_BBANDS':
            df=self.dpro.targetIndex_BOLLBAND()
            sc_mode='mode_6'
        elif self.signal_name=='TargetIndex_MOMENTUM':
            df=self.dpro.TargetIndex_MOMENTUM()
            sc_mode='mode_8'
        elif self.signal_name=='TargetIndex_MOMENTUM2':
            df=self.dpro.TargetIndex_MOMENTUM2()
            sc_mode='mode_6'
        elif self.signal_name=='TargetIndex_KDJ':
            df=self.dpro.TargetIndex_KDJ()
            sc_mode='mode_6'
        elif self.signal_name == 'TargetIndex_RSRS':
            df = self.dpro.TargetIndex_RSRS()
            sc_mode = 'mode_13'
        elif self.signal_name=='TargetIndex_PSA':
            df=self.dpro.TargetIndex_PSA()
            sc_mode='mode_6'
        elif self.signal_name=='TargetIndex_REVERSE':
            df=self.dpro.TargetIndex_MOMENTUM3()
            sc_mode='mode_6'
        elif self.signal_name=='Future_difference':
            df=self.dpro.futureDifference()
            sc_mode='mode_9'
        elif self.signal_name=='Future_holding':
            df=self.dpro.futureHolding_analyse()
            sc_mode='mode_1'
        elif self.signal_name=='CopperGold':
            df=self.dp.raw_CopperGold()
            sc_mode='mode_1'
        elif self.signal_name=='BMCI':
            df=self.dp.raw_BMCI()
            sc_mode='mode_1'
        elif self.signal_name=='DBI':
            df=self.dp.raw_DBI()
            sc_mode='mode_1'
        elif self.signal_name=='PCT':
            df=self.dp.raw_PCT()
            sc_mode='mode_1'
        elif self.signal_name=='PTA':
            df=self.dp.raw_PTA()
            sc_mode='mode_1'
        elif self.signal_name=='Index_PE':
            df=self.dpro.index_PE()
            sc_mode='mode_8'
        elif self.signal_name=='Index_PS':
            df=self.dp.raw_PS_withdraw()
            sc_mode='mode_8'
        elif self.signal_name=='Index_PCF':
            df=self.dp.raw_PCF_withdraw()
            sc_mode='mode_8'
        elif self.signal_name == 'Index_Earning':
            df = self.dp.raw_Earning_withdraw()
            sc_mode = 'mode_8'
        elif self.signal_name == 'Index_NetProfit':
            df = self.dp.raw_NetProfit_withdraw()
            sc_mode = 'mode_8'
        elif self.signal_name == 'Index_ROE':
            df = self.dp.raw_ROE_withdraw()
            sc_mode = 'mode_8'
        elif self.signal_name=='Index_PB':
            df=self.dpro.index_PB()
            sc_mode='mode_8'
        elif self.signal_name=='RRScore_difference':
            df=self.dpro.rrscoreDifference()
            sc_mode='mode_1'
        elif self.signal_name=='Bank_Momentum':
            df=self.dp.BankMomentum_withdraw()
            sc_mode='mode_1'
        elif self.signal_name=='Relative_turnover':
            df=self.dpro.relativeTurnOver()
            sc_mode='mode_1'
        else:
            print('signal_name还没有纳入系统')
            raise ValueError
        if self.signal_name!='TargetIndex_PSA':
            df.dropna(inplace=True)
        return df,sc_mode
    def start_date_processing(self,start_date):
        self.df_signal.reset_index(inplace=True,drop=True)
        df_signal=self.df_signal.copy()
        
        # 检查df_signal是否为空
        if df_signal.empty:
            raise ValueError(f"信号 {self.signal_name} 的数据为空，无法处理开始日期。请检查数据源或日期范围。")
        
        slice_df=df_signal[df_signal['valuation_date']<start_date]
        if len(slice_df)==0:
            index=df_signal.index.tolist()[0]
        else:
            slice_df.reset_index(inplace=True,drop=True)
            index=slice_df.index.tolist()[-1]
        if self.signal_name=='TargetIndex_REVERSE':
            start_date= df_signal.iloc[index]['valuation_date']
        else:
            if self.sc_mode in ['mode_1','mode_2','mode_3','mode_4','mode_5','mode_10','mode_11','mode_7']:
                max_ma = 250
            elif self.sc_mode in ['mode_6','mode_8','mode_9']:
                max_ma=30
            else:
                max_ma=250
            if len(slice_df) < max_ma:
                index = index + max_ma - len(slice_df)
                start_date = df_signal.iloc[index]['valuation_date']
        return start_date

    def signal_main(self):
        df_sql=pd.DataFrame()
        inputpath_sql=glv.get('sql_path')
        if self.mode=='prod':
             sm = gt.sqlSaving_main(inputpath_sql, 'L3_signal_prod', delete=True)
        else:
            sm = gt.sqlSaving_main(inputpath_sql, 'L3_signal_test', delete=True)
        working_days_list = gt.working_days_list(self.start_date, self.end_date)
        for date in working_days_list:
            x_list = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
            df_final = pd.DataFrame()
            final_signal_list = []
            signal_list = []
            daily_df = self.fp.slice_processing(self.df_signal, date)
            if self.sc_mode in ['mode_1','mode_2','mode_3','mode_4','mode_8','mode_9','mode_10','mode_11']:
                # Define short and long windows
                if self.sc_mode in ['mode_1','mode_2']:
                    short_windows = [1, 5, 10, 15, 20]
                    long_windows = [5, 10, 15, 20, 30, 40, 60, 90, 120, 180, 250]
                elif self.sc_mode in ['mode_8','mode_9']:
                    short_windows = [1, 5, 10]
                    long_windows = [5, 10, 15, 20, 30]
                elif self.sc_mode in ['mode_10','mode_11']:
                    short_windows = [10, 15, 20]
                    long_windows = [30, 40, 60, 90, 120, 180, 250]
                else:
                    short_windows = [20,40]
                    long_windows = [20,40, 60, 80, 100,120, 180, 250]
                # Create combinations where short window is less than long window
                rolling_list = []
                for short_window in short_windows:
                    for long_window in long_windows:
                        if short_window < long_window:
                            rolling_list.append([short_window, long_window])
                for rolling_window in rolling_list:
                    signal = self.sc.MA_difference_signal_construct(daily_df, rolling_window, self.sc_mode)
                    signal_list.append(signal)
            elif self.sc_mode=='mode_5': #M1M2专属
                rolling_list = [20, 40, 60, 80, 100, 120, 180,250]
                for rolling_window in rolling_list:
                    signal = self.sc.M1M2_signal_construct(daily_df, rolling_window)
                    signal_list.append(signal)
            elif self.sc_mode=='mode_6':
                signal=self.sc.technical_signal_construct(daily_df,self.signal_name)
                signal_list.append(signal)
            elif self.sc_mode=='mode_7':
                signal=self.sc.Monthly_effect_signal_construct(daily_df)
                signal_list.append(signal)
            elif self.sc_mode=='mode_12':
                signal = self.sc.Monthly_data_construct(daily_df)
                signal_list.append(signal)
            if self.sc_mode!='mode_13':
                final_signal = mean(signal_list)
                for x in x_list:
                    final_signal2 = self.final_signal_construction(final_signal, x)
                    final_signal_list.append(final_signal2)
            else:
                 final_signal_list = self.sc.RSRS_construct(daily_df,x_list)
            df_final['x']=x_list
            df_final['final_signal']=final_signal_list
            df_final['valuation_date']=date
            df_final=df_final[['valuation_date','final_signal','x']]
            if len(df_final)>0:
                df_sql=pd.concat([df_sql,df_final])
        if self.signal_name == 'Future_holding':
            df_signal = self.dpro.hs300_weekly_close()
            df_sql = df_sql.merge(df_signal, on='valuation_date', how='left')
            df_sql.loc[df_sql['signal']=='normal',['final_signal']] = (df_sql[df_sql['signal']=='normal']['final_signal']-1)*-1
            df_sql.drop(columns='signal', inplace=True)
        df_sql['signal_name'] = self.signal_name
        df_sql['update_time']=datetime.now().replace(tzinfo=None)  # 当前时间
        sm.df_to_sql(df_sql,'signal_name',self.signal_name)

if __name__ == "__main__":

    # 其他mode (不等于1,2,3,4) 对应的signal_name列表
    other_mode_signal_names = ['Index_Earning','Index_NetProfit','Index_ROE'
    ]
    for signal_name in other_mode_signal_names:
        ssm=L3_signalConstruction(signal_name=signal_name,mode='test',start_date='2015-01-01',end_date='2026-01-07')
        ssm.signal_main()
    # for signal_name in other_mode_signal_names:
    #     ssm=single_signal_main(signal_name=signal_name,mode='test',start_date='2015-01-01',end_date='2025-09-27')
    #     ssm.signal_main()
    
    # ssm=single_signal_main(signal_name='TermSpread_9Y',mode='test',start_date='2015-01-01',end_date='2025-09-27')
    # ssm.signal_main()