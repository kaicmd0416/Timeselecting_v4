"""
L3级别信号生成模块 (L3_signal_main)

本模块负责生成L3级别的择时信号。L3信号是最底层的信号，基于原始因子数据计算。
每个L3因子对应一个具体的量化指标，通过特定的信号构建模式转换为择时信号。

信号体系层级结构:
    L0 (最终信号) ← 聚合所有L1信号
    L1 (一级因子) ← 聚合对应的L2信号
    L2 (二级因子) ← 聚合对应的L3信号
    L3 (三级因子) ← 原始因子数据计算  ← 【当前模块】

支持的信号构建模式（sc_mode）:
    - mode_1: 长周期均线正向 - 短MA > 长MA → 买大盘
              适用因子: Shibor, Bond, CreditSpread等利率类因子
    - mode_2: 长周期均线反向 - 短MA > 长MA → 买小盘
              适用因子: USDX, TermSpread, LHBProportion等
    - mode_3: 月度因子正向 - 当月值 > 历史均值 → 买大盘
              适用因子: PPI, PMI等宏观月度数据
    - mode_4: 月度因子反向 - 当月值 > 历史均值 → 买小盘
              适用因子: CPI等
    - mode_5: M1M2专用 - 综合M1、M2和剪刀差判断
              适用因子: M1M2
    - mode_6: 技术指标专用 - 直接根据技术指标状态判断
              适用因子: MACD, BBANDS, KDJ, PSA等
    - mode_7: 季节性效应 - 根据历史同期表现判断
              适用因子: Monthly_Effect, Post_Holiday_Effect
    - mode_8: 短周期均线正向 - 短MA > 长MA → 买大盘（窗口期较短）
              适用因子: LargeOrder, ETF_Shares, 期权因子等
    - mode_9: 短周期均线反向 - 短MA > 长MA → 买小盘
              适用因子: Stock_RT, Future_difference, Commodity_Downside等
    - mode_10: 长周期均线正向（慢速版）
              适用因子: EarningsYield_Reverse, IPO
    - mode_11: 长周期均线反向（慢速版）
              适用因子: Growth
    - mode_13: RSRS专用模式
              适用因子: TargetIndex_RSRS
    - mode_14: 固定时间触发 - 按照固定时间规则生成信号
              适用因子: Earnings_Season（财报季因子）

x参数说明:
    x是信号阈值参数，用于控制信号的敏感度：
    - x越大（如0.8），信号越谨慎，中性信号越多
    - x越小（如0.55），信号越激进，方向性信号越多
    - 可选值: [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

Forward-Looking防护:
    本模块使用slice_processing确保只使用target_date之前的数据：
    df = df[df['valuation_date'] < target_date]
    这样保证了信号生成不会使用未来数据。

作者: TimeSelecting Team
版本: v3.0
"""

# ==================== 标准库导入 ====================
from datetime import datetime
import os
import sys

# ==================== 第三方库导入 ====================
import pandas as pd
from numpy import *

# ==================== 自定义模块导入 ====================
from data.data_prepare import data_prepare
from data.data_processing import data_processing
from factor_processing.signal_constructing import signal_construct, factor_processing

# 添加全局工具函数路径
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
import global_setting.global_dic as glv

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
        elif self.signal_name=='Monthly_Effect':
            df=self.dpro.monthly_effect()
            sc_mode='mode_7'
        elif self.signal_name=='Post_Holiday_Effect':
            df=self.dpro.post_holiday_effect()
            sc_mode='mode_7'
        # ======================== 事件驱动因子（固定时间触发，不需要回测） ========================
        elif self.signal_name=='Earnings_Season':
            df=self.dpro.earnings_season()
            sc_mode='mode_14'  # 固定时间触发：财报前2周买大盘，财报后2周买小盘
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
        elif self.signal_name=='TargetIndex_REVERSE2':
            df=self.dpro.TargetIndex_REVERSE()
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
        elif self.signal_name == 'Index_NetProfit':
            df = self.dp.raw_NetProfit_withdraw()
            sc_mode = 'mode_8'
        elif self.signal_name=='Index_PB':
            df=self.dpro.index_PB()
            sc_mode='mode_8'
        elif self.signal_name=='IPO':
            df=self.dp.raw_IPO()
            sc_mode='mode_10'
        elif self.signal_name=='RRScore_difference':
            df=self.dpro.rrscoreDifference()
            sc_mode='mode_1'
        elif self.signal_name=='VP08Score_difference':
            df=self.dpro.vp08scoreDifference()
            sc_mode='mode_8'
        elif self.signal_name=='Bank_Momentum':
            df=self.dp.BankMomentum_withdraw()
            sc_mode='mode_1'
        elif self.signal_name=='Relative_turnover':
            df=self.dpro.relativeTurnOver()
            sc_mode='mode_1'
        # ======================== 商品期货因子 ========================
        elif self.signal_name=='Commodity_Upside':
            df=self.dpro.commodity_upside()
            sc_mode='mode_8'  # 正向：上游商品强→大盘占优
        elif self.signal_name=='Commodity_Downside':
            df=self.dpro.commodity_downside()
            sc_mode='mode_9'  # 反向：中下游商品强→小盘占优
        elif self.signal_name=='Commodity_Volume':
            df=self.dpro.commodity_volume()
            sc_mode='mode_9'  # 反向：活跃度高→小盘承压
        elif self.signal_name=='Commodity_PPI_Correl':
            df=self.dpro.commodity_ppi_correl()
            sc_mode='mode_1'  # 正向：联动度高→强化传导逻辑
        elif self.signal_name=='Commodity_Composite':
            df=self.dpro.commodity_composite()
            sc_mode='mode_1'  # 正向：同比上行→大盘占优
        elif self.signal_name=='Commodity_UpDown_Spread':
            df=self.dpro.commodity_updown_spread()
            sc_mode='mode_1'  # 长周期正向：上游/中下游比值上升→大盘占优
        elif self.signal_name=='Commodity_Volatility':
            df=self.dpro.commodity_volatility()
            sc_mode='mode_2'  # 长周期反向：波动率上升→小盘占优
        # ======================== 期权因子 ========================
        elif self.signal_name=='Option_PCR_OI':
            df=self.dpro.option_PCR_OI()
            sc_mode='mode_8'  # 短周期均线正向：基于持仓量，大盘PCR-小盘PCR，差值高→大盘更恐慌→买大盘
        elif self.signal_name=='Option_PCR_Amt':
            df=self.dpro.option_PCR_Amt()
            sc_mode='mode_8'  # 短周期均线正向：基于成交额，大盘PCR-小盘PCR，差值高→大盘更恐慌→买大盘
        elif self.signal_name=='Option_PCR_Volume':
            df=self.dpro.option_PCR_Volume()
            sc_mode='mode_8'  # 短周期均线正向：基于成交量，大盘PCR-小盘PCR，差值高→大盘更恐慌→买大盘
        elif self.signal_name=='Option_IV':
            df=self.dpro.option_IV()
            sc_mode='mode_8'  # 短周期均线正向：大盘IV-小盘IV，差值高→大盘更恐慌→买大盘
        elif self.signal_name=='Option_IVSkew':
            df=self.dpro.option_IVSkew()
            sc_mode='mode_8'  # 短周期均线正向：大盘IVSkew-小盘IVSkew，差值高→大盘更恐慌→买大盘
        elif self.signal_name=='Option_IV_Chg':
            df=self.dpro.option_IV_Chg()
            sc_mode='mode_8'  # 短周期均线正向：大盘IV变化率-小盘IV变化率，差值高→大盘恐慌升温→买大盘
        elif self.signal_name=='Option_OI_Chg':
            df=self.dpro.option_OI_Chg()
            sc_mode='mode_8'  # 短周期均线正向：大盘OI变化率-小盘OI变化率，差值高→大盘资金流入→买大盘
        elif self.signal_name=='Option_Turnover':
            df=self.dpro.option_Turnover()
            sc_mode='mode_8'  # 短周期均线正向：大盘换手率-小盘换手率，差值高→大盘更活跃→买大盘
        elif self.signal_name=='Option_CallPut_Spread':
            df=self.dpro.option_CallPut_Spread()
            sc_mode='mode_8'  # 短周期均线正向：大盘认购认沽涨跌差-小盘，差值高→大盘多头更强→买大盘
        # ======================== 港股联动因子 ========================
        elif self.signal_name=='HK_HSI_Momentum':
            df=self.dpro.hk_hsi_momentum()
            sc_mode='mode_8'  # 长周期均线正向：恒生指数强势→外资风险偏好高→买大盘
        elif self.signal_name=='HK_HSTECH_Momentum':
            df=self.dpro.hk_hstech_momentum()
            sc_mode='mode_9'  # 长周期均线反向：恒生科技强势→成长风格占优→买小盘
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
            elif self.sc_mode in ['mode_6','mode_8','mode_9','mode_14']:
                max_ma=30
            else:
                max_ma=250
            if len(slice_df) < max_ma:
                index = index + max_ma - len(slice_df)
                start_date = df_signal.iloc[index]['valuation_date']
        return start_date

    def signal_main(self):
        """
        L3信号生成主函数

        遍历每个交易日，根据sc_mode生成对应的择时信号。
        每天会生成多个x参数下的信号，用于后续best_x的选择。

        处理流程:
            1. 获取交易日列表
            2. 对每个交易日:
               a. 使用slice_processing获取该日之前的数据（避免forward-looking）
               b. 根据sc_mode选择信号构建方法
               c. 计算不同参数组合的信号
               d. 对信号取平均值
               e. 对每个x阈值生成最终信号
            3. 保存所有信号到数据库

        数据库表结构:
            - valuation_date: 日期
            - final_signal: 信号值（0/0.5/1）
            - x: 阈值参数
            - signal_name: 因子名称
            - update_time: 更新时间

        注意:
            - 每天会生成6条记录（对应6个不同的x值）
            - 后续L2层级会通过回测选择最优的x
        """
        # ==================== 初始化 ====================
        df_sql = pd.DataFrame()  # 用于收集所有日期的信号
        inputpath_sql = glv.get('sql_path')

        # 根据模式选择数据库表
        if self.mode == 'prod':
            sm = gt.sqlSaving_main(inputpath_sql, 'L3_signal_prod', delete=True)
        else:
            sm = gt.sqlSaving_main(inputpath_sql, 'L3_signal_test', delete=True)

        # 获取需要处理的交易日列表
        working_days_list = gt.working_days_list(self.start_date, self.end_date)

        # ==================== 遍历每个交易日 ====================
        for date in working_days_list:
            # x阈值列表，用于生成不同敏感度的信号
            x_list = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
            df_final = pd.DataFrame()
            final_signal_list = []
            signal_list = []

            # 【关键】使用slice_processing获取date之前的数据，避免forward-looking
            daily_df = self.fp.slice_processing(self.df_signal, date)
            # ==================== 根据sc_mode选择信号构建方法 ====================

            if self.sc_mode in ['mode_1', 'mode_2', 'mode_3', 'mode_4', 'mode_8', 'mode_9', 'mode_10', 'mode_11']:
                # ========== 均线差异类信号构建 ==========
                # 通过比较短期均线和长期均线的差值生成信号
                # 遍历多个窗口组合，最后取平均

                # 根据mode选择不同的窗口参数
                if self.sc_mode in ['mode_1', 'mode_2']:
                    # 长周期模式：窗口较大，适合趋势性因子
                    short_windows = [1, 5, 10, 15, 20]
                    long_windows = [5, 10, 15, 20, 30, 40, 60, 90, 120, 180, 250]
                elif self.sc_mode in ['mode_8', 'mode_9']:
                    # 短周期模式：窗口较小，适合高频因子
                    short_windows = [1, 5, 10]
                    long_windows = [5, 10, 15, 20, 30]
                elif self.sc_mode in ['mode_10', 'mode_11']:
                    # 超长周期模式：窗口更大，适合缓慢变化的因子
                    short_windows = [10, 15, 20]
                    long_windows = [30, 40, 60, 90, 120, 180, 250]
                else:
                    # 月度因子模式
                    short_windows = [20, 40]
                    long_windows = [20, 40, 60, 80, 100, 120, 180, 250]

                # 生成所有短窗口<长窗口的组合
                rolling_list = []
                for short_window in short_windows:
                    for long_window in long_windows:
                        if short_window < long_window:
                            rolling_list.append([short_window, long_window])

                # 对每个窗口组合计算信号
                for rolling_window in rolling_list:
                    signal = self.sc.MA_difference_signal_construct(daily_df, rolling_window, self.sc_mode)
                    signal_list.append(signal)

            elif self.sc_mode == 'mode_5':
                # ========== M1M2专用模式 ==========
                # 综合考虑M2和M1-M2剪刀差的信号
                rolling_list = [20, 40, 60, 80, 100, 120, 180, 250]
                for rolling_window in rolling_list:
                    signal = self.sc.M1M2_signal_construct(daily_df, rolling_window)
                    signal_list.append(signal)

            elif self.sc_mode == 'mode_6':
                # ========== 技术指标专用模式 ==========
                # 直接根据技术指标状态判断信号，如MACD金叉死叉
                signal = self.sc.technical_signal_construct(daily_df, self.signal_name)
                signal_list.append(signal)

            elif self.sc_mode == 'mode_7':
                # ========== 季节性效应模式 ==========
                # 根据历史同期表现判断
                signal = self.sc.Monthly_effect_signal_construct(daily_df)
                signal_list.append(signal)

            elif self.sc_mode == 'mode_12':
                # ========== 月度数据模式 ==========
                signal = self.sc.Monthly_data_construct(daily_df)
                signal_list.append(signal)

            elif self.sc_mode == 'mode_14':
                # ========== 固定时间触发模式 ==========
                # 如财报季因子，按固定规则生成信号
                signal = self.sc.fixed_time_signal_construct(daily_df)
                signal_list.append(signal)

            # ==================== 生成最终信号 ====================

            if self.sc_mode == 'mode_13':
                # RSRS模式：直接返回不同x下的信号列表
                final_signal_list = self.sc.RSRS_construct(daily_df, x_list)

            elif self.sc_mode == 'mode_14':
                # 固定时间触发模式：所有x使用相同信号（不需要回测选择）
                final_signal = signal_list[0]
                final_signal_list = [final_signal] * len(x_list)

            else:
                # 常规模式：取所有窗口信号的平均值，再根据x阈值转换
                final_signal = mean(signal_list)  # 信号平均值（0-1之间）
                for x in x_list:
                    # 根据阈值x将连续信号转换为离散信号
                    final_signal2 = self.final_signal_construction(final_signal, x)
                    final_signal_list.append(final_signal2)

            # ==================== 整理当天数据 ====================
            df_final['x'] = x_list
            df_final['final_signal'] = final_signal_list
            df_final['valuation_date'] = date
            df_final = df_final[['valuation_date', 'final_signal', 'x']]

            # 追加到总结果中
            if len(df_final) > 0:
                df_sql = pd.concat([df_sql, df_final])
        # ==================== 特殊因子后处理 ====================
        if self.signal_name == 'Future_holding':
            # Future_holding因子需要特殊处理：
            # 当沪深300处于"normal"状态时，信号需要反转
            df_signal = self.dpro.hs300_weekly_close()
            df_sql = df_sql.merge(df_signal, on='valuation_date', how='left')
            # 反转信号：0→1, 1→0, 0.5保持不变
            df_sql.loc[df_sql['signal'] == 'normal', ['final_signal']] = \
                (df_sql[df_sql['signal'] == 'normal']['final_signal'] - 1) * -1
            df_sql.drop(columns='signal', inplace=True)

        # ==================== 添加元数据并保存 ====================
        df_sql['signal_name'] = self.signal_name  # 添加因子名称
        df_sql['update_time'] = datetime.now().replace(tzinfo=None)  # 添加更新时间

        # 保存到数据库，第二和第三个参数用于删除旧数据
        sm.df_to_sql(df_sql, 'signal_name', self.signal_name)


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    """
    批量运行L3信号生成

    L3因子完整列表（按分类）:

    宏观流动性类 (MacroLiquidity):
        - Shibor_2W, Shibor_9M
        - Bond_3Y, Bond_10Y
        - CreditSpread_5Y, CreditSpread_9M
        - TermSpread_9Y
        - M1M2

    指数量价类 (IndexPriceVolume):
        - TargetIndex_MACD, TargetIndex_BBANDS, TargetIndex_PSA
        - TargetIndex_MOMENTUM, TargetIndex_MOMENTUM2, TargetIndex_REVERSE
        - TargetIndex_RSRS
        - RelativeVolume_std, RelativeReturn_std

    资金流向类 (StockCapital):
        - NLBP_difference, LargeOrder_difference
        - ETF_Shares
        - USDX, USBond_3Y, USBond_10Y

    市场情绪类 (StockEmotion):
        - Stock_HL, Stock_RT
        - Future_difference, Future_holding
        - Bank_Momentum
        - Relative_turnover

    宏观经济类 (MacroEconomy):
        - CPI, PPI, PMI
        - CopperGold, BMCI, DBI, PCT

    股票基本面类 (StockFundamentals):
        - Index_PE, Index_PB, Index_PS, Index_PCF
        - Index_NetProfit
        - EarningsYield_Reverse, Growth

    特殊因子类 (SpecialFactor):
        - Monthly_Effect, Post_Holiday_Effect
        - Earnings_Season
        - RRScore_difference, VP08Score_difference

    商品期货类 (Commodity):
        - Commodity_Upside, Commodity_Downside
        - Commodity_Volume, Commodity_PPI_Correl
        - Commodity_Composite
        - Commodity_UpDown_Spread, Commodity_Volatility

    期权因子类 (Option):
        - Option_PCR_OI, Option_PCR_Amt, Option_PCR_Volume
        - Option_IV, Option_IVSkew
        - Option_IV_Chg, Option_OI_Chg
        - Option_Turnover, Option_CallPut_Spread
    """
    # 要处理的L3因子列表
    other_mode_signal_names = ['Future_holding']

    # 批量处理因子
    for signal_name in other_mode_signal_names:
        ssm = L3_signalConstruction(
            signal_name=signal_name,
            mode='prod',           # 运行模式
            start_date='2015-01-01',  # 开始日期
            end_date='2026-01-26'     # 结束日期
        )
        ssm.signal_main()