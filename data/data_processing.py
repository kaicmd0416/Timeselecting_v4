"""
数据处理模块 (data_processing)

本模块负责对原始数据进行加工处理，计算各类技术指标和基本面因子，包括：
- 信用利差、期限利差
- 货币供应量相关因子
- 资金流向因子（融资融券、大单、龙虎榜等）
- 技术指标（MACD、RSI、布林带、KDJ等）
- 股票市场因子（涨跌停、RSI、趋势等）
- 其他复合因子

作者: TimeSelecting Team
版本: v3.0
"""

from matplotlib import pyplot as plt
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
import global_setting.global_dic as glv
from data.data_prepare import data_prepare
import pandas_ta as ta

class data_processing:
    """
    数据处理类
    
    基于data_prepare获取的原始数据，进行进一步加工处理，计算各类因子。
    
    Attributes:
    -----------
    dp : data_prepare
        data_prepare类的实例，用于获取原始数据
    """
    
    def __init__(self, start_date, end_date):
        """
        初始化数据处理类
        
        Parameters:
        -----------
        start_date : str
            开始日期，格式为 'YYYY-MM-DD'
        end_date : str
            结束日期，格式为 'YYYY-MM-DD'
        """
        self.dp=data_prepare(start_date,end_date)

    def hs300_weekly_close(self):
        """
        计算10季线的平均值
        一年四个季度：3月、6月、9月、12月的最后一个交易日
        当月取最新的收盘价作为当月季线的close
        然后取前面九个季度的close算平均值

        Returns:
            DataFrame: 包含日期和10季线平均值
        """
        df_close = self.dp.raw_index_close('沪深300')
        df_close['valuation_date'] = pd.to_datetime(df_close['valuation_date'])
        df_close = df_close.sort_values('valuation_date')

        # 获取指数名称（列名中除了valuation_date之外的列）
        index_col = [col for col in df_close.columns if col != 'valuation_date'][0]

        # 添加年份和月份列
        df_close['year'] = df_close['valuation_date'].dt.year
        df_close['month'] = df_close['valuation_date'].dt.month

        # 确定每个日期属于哪个季度
        def get_quarter(month):
            if month in [1, 2, 3]:
                return 1
            elif month in [4, 5, 6]:
                return 2
            elif month in [7, 8, 9]:
                return 3
            else:
                return 4

        df_close['quarter'] = df_close['month'].apply(get_quarter)

        # 找到每个季度最后一个交易日的收盘价（3月、6月、9月、12月的最后一个交易日）
        quarterly_close_dict = {}  # {(year, quarter): close}
        for year in df_close['year'].unique():
            for quarter in [1, 2, 3, 4]:
                quarter_month = quarter * 3  # 1->3, 2->6, 3->9, 4->12
                quarter_data = df_close[
                    (df_close['year'] == year) &
                    (df_close['month'] == quarter_month)
                    ]
                if not quarter_data.empty:
                    # 找到该月最后一个交易日
                    last_trading_day = quarter_data['valuation_date'].max()
                    last_close = quarter_data[quarter_data['valuation_date'] == last_trading_day][index_col].iloc[0]
                    quarterly_close_dict[(year, quarter)] = last_close

        # 对于每个日期，计算10季线平均值
        result_list = []
        for idx, row in df_close.iterrows():
            current_date = row['valuation_date']
            current_year = row['year']
            current_quarter = row['quarter']
            current_close = row[index_col]

            # 确定当前日期所在季度的收盘价
            # 如果还没到季度末（3月、6月、9月、12月的最后一个交易日），就用最新的收盘价（当天的收盘价）
            # 如果已经到了季度末，就用季度末的收盘价
            quarter_end_month = current_quarter * 3
            if current_date.month == quarter_end_month:
                # 是季度末月份，检查是否是最后一个交易日
                # 需要看整个月份的数据来判断最后一个交易日
                month_data_all = df_close[
                    (df_close['year'] == current_year) &
                    (df_close['month'] == quarter_end_month)
                    ]
                if not month_data_all.empty:
                    last_trading_day_of_month = month_data_all['valuation_date'].max()
                    # 只有当当前日期就是该月最后一个交易日时，才使用季度末收盘价
                    # 否则使用当前日期的收盘价
                    if current_date == last_trading_day_of_month:
                        # 是季度最后一个交易日，使用季度末收盘价
                        if (current_year, current_quarter) in quarterly_close_dict:
                            current_quarter_close = quarterly_close_dict[(current_year, current_quarter)]
                        else:
                            current_quarter_close = current_close
                    else:
                        # 还没到季度最后一个交易日，使用当前收盘价
                        current_quarter_close = current_close
                else:
                    # 该月没有数据，使用当前收盘价
                    current_quarter_close = current_close
            else:
                # 不是季度末月份，使用当前收盘价
                current_quarter_close = current_close

            # 找到前面9个季度的收盘价
            prev_quarters = []
            check_year = current_year
            check_quarter = current_quarter - 1  # 从上一个季度开始

            while len(prev_quarters) < 9:
                if check_quarter < 1:
                    check_quarter = 4
                    check_year -= 1

                # 查找该季度的收盘价
                if (check_year, check_quarter) in quarterly_close_dict:
                    prev_quarters.append(quarterly_close_dict[(check_year, check_quarter)])
                else:
                    # 如果找不到该季度的数据，停止查找
                    break

                check_quarter -= 1

            # 计算平均值（当前季度 + 前面9个季度）
            if len(prev_quarters) == 9:
                avg_10_quarter = (current_quarter_close + sum(prev_quarters)) / 10
            else:
                avg_10_quarter = None

            result_list.append({
                'valuation_date': current_date,
                'quarterly_avg_10': avg_10_quarter
            })

        result_df = pd.DataFrame(result_list)
        result_df['valuation_date'] = result_df['valuation_date'].dt.strftime('%Y-%m-%d')
        df_close['valuation_date'] = df_close['valuation_date'].dt.strftime('%Y-%m-%d')
        df_close = df_close[['valuation_date', '沪深300']]
        result_df.dropna(inplace=True)
        result_df = result_df.merge(df_close, on='valuation_date', how='left')
        result_df['difference'] = result_df['沪深300'] - result_df['quarterly_avg_10']
        result_df['signal'] = 'normal'
        result_df.loc[result_df['difference'] >= 0, ['signal']] = 'hedge'
        result_df = result_df[['valuation_date', 'signal']]
        return result_df
    def futureHolding_analyse(self):
        df=self.dp.raw_futureHolding()
        df.dropna(inplace=True)
        df['net_holding']=df['long_hld']-df['short_hld']
        df['net_chg']=df['long_chg']-df['short_chg']
        df=df[['valuation_date','net_holding','net_chg','new_symbol']]
        df=df.groupby(['valuation_date','new_symbol']).sum()
        df.reset_index(inplace=True)
        df_big=df[df['new_symbol'].isin(['IH','IF'])]
        df_small = df[df['new_symbol'].isin(['IC', 'IM'])]
        df_big.drop(columns='new_symbol',inplace=True)
        df_small.drop(columns='new_symbol', inplace=True)
        df_big=df_big.groupby(['valuation_date']).sum()
        df_small =df_small.groupby(['valuation_date']).sum()
        df_big=df_big[['net_holding']]
        df_small = df_small[['net_holding']]
        df_big.columns=['big_holding']
        df_small.columns=['small_holding']
        df_big.reset_index(inplace=True)
        df_small.reset_index(inplace=True)
        df_final=df_big.merge(df_small,on='valuation_date',how='left')
        df_final.rename(columns={'valuation_date':'valuation_date'},inplace=True)
        df_final['difference']=df_final['big_holding']/df_final['small_holding']
        df_final.dropna(inplace=True)
        df_final=df_final[['valuation_date','difference']]
        return df_final
    def futureHoldingchg_analyse(self):
        df = self.dp.raw_futureHolding()
        df.dropna(inplace=True)
        df['net_holding'] = df['long_hld'] - df['short_hld']
        df['total_holding']=abs(df['long_hld']) + abs(df['short_hld'])
        #df = df[~(df['net_holding'] == 0)]
        df['net_chg'] = df['long_chg'] - df['short_chg']
        df['net_chg'] = df['net_chg'] / df['total_holding']
        df = df[['valuation_date', 'net_holding', 'net_chg', 'new_symbol','total_holding']]
        # df2=df.groupby(['valuation_date', 'new_symbol']).sum()
        # df2.reset_index(inplace=True)
        # df2=df2[['valuation_date','new_symbol','total_holding']]
        # df2.rename(columns={'total_holding':'symbol_holding'},inplace=True)
        # df=df.merge(df2,on=['valuation_date','new_symbol'],how='left')
        #df['net_chg']=df['net_chg']*df['total_holding']/df['symbol_holding']
        df = df.groupby(['valuation_date', 'new_symbol']).sum()
        df.reset_index(inplace=True)
        df_big = df[df['new_symbol'].isin(['IH', 'IF'])]
        df_small = df[df['new_symbol'].isin(['IC', 'IM'])]
        df_big.drop(columns='new_symbol', inplace=True)
        df_small.drop(columns='new_symbol', inplace=True)
        df_big = df_big.groupby(['valuation_date']).sum()
        df_small = df_small.groupby(['valuation_date']).sum()
        df_big = df_big[['net_chg']]
        df_small = df_small[['net_chg']]
        df_big.columns = ['big_chg']
        df_small.columns = ['small_chg']
        df_big.reset_index(inplace=True)
        df_small.reset_index(inplace=True)
        df_final = df_big.merge(df_small, on='valuation_date', how='left')
        df_final['difference'] = df_final['big_chg'] - df_final['small_chg']
        df_final = df_final[['valuation_date', 'difference']]
        df_final.columns = ['valuation_date', 'netHolding_chg']
        df_final['netHolding_chg'] = df_final['netHolding_chg'].rolling(10).sum()
        #df_final = df_final.merge(df_signal, on='valuation_date', how='left')
        #df_final['netHolding_chg'] = df_final['netHolding_chg'] * df_final['signal']
        df_final.dropna(inplace=True)
        df_final = df_final[['valuation_date', 'netHolding_chg']]
        return df_final
    def credit_spread_3M(self):
        """
        计算3个月信用利差
        
        信用利差 = |中债中短3M - 国开债3M|
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - credit_spread_3M: 3个月信用利差值
        """
        df_zzgk = self.dp.raw_ZZGK(period='3M')
        df_zzzd = self.dp.raw_ZZZD(period='3M')
        df_zzzd = df_zzzd.merge(df_zzgk, on='valuation_date', how='outer')
        df_zzzd.dropna(inplace=True)
        df_zzzd['credit_spread_3M'] = abs(df_zzzd['CMTN_3M'] - df_zzzd['CDBB_3M'])
        df_zzzd = df_zzzd[['valuation_date', 'credit_spread_3M']]
        return df_zzzd
    def credit_spread_9M(self):
        """
        计算9个月信用利差
        
        信用利差 = |中债中短9M - 国开债9M|
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - credit_spread_9M: 9个月信用利差值
        """
        df_zzgk = self.dp.raw_ZZGK(period='9M')
        df_zzzd = self.dp.raw_ZZZD(period='9M')
        df_zzzd = df_zzzd.merge(df_zzgk, on='valuation_date', how='outer')
        df_zzzd.dropna(inplace=True)
        df_zzzd['credit_spread_9M'] = abs(df_zzzd['CMTN_9M'] - df_zzzd['CDBB_9M'])
        df_zzzd = df_zzzd[['valuation_date', 'credit_spread_9M']]
        return df_zzzd
    def credit_spread_5Y(self):
        """
        计算5年信用利差
        
        信用利差 = |中债中短5Y - 国开债5Y|
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - credit_spread_5Y: 5年信用利差值
        """
        df_zzgk = self.dp.raw_ZZGK(period='5Y')
        df_zzzd = self.dp.raw_ZZZD(period='5Y')
        df_zzzd = df_zzzd.merge(df_zzgk, on='valuation_date', how='outer')
        df_zzzd.dropna(inplace=True)
        df_zzzd['credit_spread_5Y'] = abs(df_zzzd['CMTN_5Y'] - df_zzzd['CDBB_5Y'])
        df_zzzd = df_zzzd[['valuation_date', 'credit_spread_5Y']]
        return df_zzzd
    def term_spread_9Y(self):
        """
        计算9年期限利差
        
        期限利差 = |国开债10Y - 国开债1Y|
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - term_spread_9Y: 9年期限利差值
        """
        df_10Y = self.dp.raw_ZZGK(period='10Y')
        df_1Y = self.dp.raw_ZZGK(period='1Y')
        df_zzzd = df_1Y.merge(df_10Y, on='valuation_date', how='outer')
        df_zzzd.dropna(inplace=True)
        df_zzzd = df_zzzd[df_zzzd['valuation_date'] > '2017-08-01']
        df_zzzd['term_spread_9Y'] = abs(df_zzzd['CDBB_10Y'] - df_zzzd['CDBB_1Y'])
        df_zzzd = df_zzzd[['valuation_date', 'term_spread_9Y']]
        return df_zzzd
    def M1M2(self):
        """
        计算M1-M2差值
        
        获取M1和M2货币供应量数据，计算它们的差值，并对非工作日进行前向填充
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - M1: M1货币供应量
            - M2: M2货币供应量
            - difference: M1-M2差值
        """
        df_M1 = self.dp.raw_M1M2(signal_name='M1')
        df_M2 = self.dp.raw_M1M2(signal_name='M2')
        df_M1M2 = df_M1.merge(df_M2, on='valuation_date', how='outer')
        df_M1M2.dropna(inplace=True)
        working_days_list = gt.working_days_list(df_M1M2['valuation_date'].tolist()[0],
                                                 df_M1M2['valuation_date'].tolist()[-1])
        df_final = pd.DataFrame()
        df_final['valuation_date'] = working_days_list
        df_final = df_final.merge(df_M1M2, on='valuation_date', how='outer')
        df_final.fillna(method='ffill', inplace=True)
        df_final = df_final[df_final['valuation_date'].isin(working_days_list)]
        df_final['difference'] = df_final['M1'] - df_final['M2']
        return df_final
    def US_stock(self):
         df_D=self.dp.raw_DJUS()
         df_N=self.dp.raw_NDAQ()
         df_us=df_D.merge(df_N,on='valuation_date',how='outer')
         df_us.dropna(inplace=True)
         df_us.set_index('valuation_date', inplace=True, drop=True)
         df_us = (1 + df_us).cumprod()
         df_us['D/N'] = df_us['DJUS'] / df_us['NDAQ']
         df_us.reset_index(inplace=True)
         df_us['valuation_date']=pd.to_datetime(df_us['valuation_date'])
         df_us['valuation_date']=df_us['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
         return df_us
    def relativeVolume_std(self):
        """
        计算相对成交量标准差
        
        计算沪深300和中证2000成交额的40日滚动标准差，然后计算它们的比值
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - RelativeVolume_std: 相对成交量标准差（沪深300标准差 / 中证2000标准差）
        """
        df_300 = self.dp.raw_index_amt(index_name='沪深300')
        df_2000 = self.dp.raw_index_amt(index_name='中证2000')
        df_300['std_300'] = df_300['沪深300'].rolling(40).std()
        df_2000['std_2000'] = df_2000['中证2000'].rolling(40).std()
        df_final = df_300.merge(df_2000, on='valuation_date', how='left')
        df_final.dropna(inplace=True)
        df_final['RelativeVolume_std'] = df_final['std_300'] / df_final['std_2000']
        df_final = df_final[['valuation_date', 'RelativeVolume_std']]
        return df_final
    def relativeReturn_std(self):
        """
        计算相对收益率标准差
        
        计算沪深300和中证2000收益率的40日滚动标准差，然后计算它们的比值
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - RelativeReturn_std: 相对收益率标准差（沪深300标准差 / 中证2000标准差）
        """
        df = self.dp.index_return_withdraw()
        df['std_300'] = df['上证50'].rolling(40).std()
        df['std_2000'] = df['中证2000'].rolling(40).std()
        df.dropna(inplace=True)
        df['RelativeReturn_std'] = df['std_300'] / df['std_2000']
        df = df[['valuation_date', 'RelativeReturn_std']]
        return df
    def relativeTurnOver(self):
        """
        计算相对换手率差值
        
        计算沪深300与国证2000换手率的差值
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - turnover_difference: 换手率差值（沪深300 - 国证2000）
        """
        df_sz50 = self.dp.raw_index_turnover(index_name='上证50')
        df_hs300 = self.dp.raw_index_turnover(index_name='沪深300')
        df_zz2000 = self.dp.raw_index_turnover(index_name='国证2000')
        df_zz2000.dropna(inplace=True)
        df = df_hs300.merge(df_zz2000, on='valuation_date', how='left')
        df = df.merge(df_sz50,on='valuation_date',how='left')
        df['turnover_difference'] = (df['上证50']+df['沪深300'])/2 - df['国证2000']
        df = df[['valuation_date', 'turnover_difference']]
        return df
    def LargeOrder_difference(self):
        """
        计算大单资金流入差异
        
        计算沪深300、中证1000、国证2000的大单资金流入相对于成交额的比例，
        然后计算20日滚动求和后的差值（沪深300 - 国证2000）
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - LargeOrder_difference: 大单差异值（20日滚动求和）
        """
        df=self.dp.raw_LargeOrder_withdraw()
        df_hs300amt=self.dp.raw_index_amt(index_name='沪深300')
        df_hs300amt.columns=['valuation_date','hs300amt']
        df_zz1000amt=self.dp.raw_index_amt(index_name='中证1000')
        df_zz1000amt.columns=['valuation_date','zz1000amt']
        df_zz2000amt=self.dp.raw_index_amt(index_name='国证2000')
        df_zz2000amt.columns=['valuation_date','zz2000amt']
        df=df[['valuation_date','000300.SH','000852.SH','399303.SZ']]
        df=df.merge(df_hs300amt,on='valuation_date',how='left')
        df=df.merge(df_zz1000amt,on='valuation_date',how='left')
        df=df.merge(df_zz2000amt,on='valuation_date',how='left')
        df['000300.SH']=df['000300.SH'].astype(float)/df['hs300amt']
        df['000852.SH']=df['000852.SH'].astype(float)/df['zz1000amt']
        df['399303.SZ']=df['399303.SZ'].astype(float)/df['zz2000amt']
        df.set_index('valuation_date', inplace=True)
        df = df.rolling(20).sum()
        df['LargeOrder_difference']=df['000300.SH']-df['399303.SZ']
        df=df[['LargeOrder_difference']]
        df.reset_index(inplace=True)
        df.dropna(inplace=True)
        return df
    def NetLeverageBuying(self):
        """
        计算净杠杆买入比例差异
        
        计算沪深300和国证2000成分股的净杠杆买入（融资买入-融资偿还）相对于成交额的比例，
        然后计算差值（沪深300 - 国证2000）
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - NetLeverageAMTProportion_difference: 净杠杆买入比例差异值
            
        Note:
        -----
        如果数据为空，返回空的DataFrame
        """
        df_leverage = self.dp.raw_NetLeverageBuying_withdraw()
        df_leverage['NetLeverage_buying'] = df_leverage['mrg_long_amt']-df_leverage['mrg_long_repay']
        # 获取沪深300和国证2000的成分股（时间序列数据）
        hs300_stocks = self.dp.raw_index_weight('沪深300')
        gz2000_stocks = self.dp.raw_index_weight('国证2000')
        # 确保日期格式为字符串格式
        hs300_stocks['valuation_date'] = pd.to_datetime(hs300_stocks['valuation_date']).dt.strftime('%Y-%m-%d')
        gz2000_stocks['valuation_date'] = pd.to_datetime(gz2000_stocks['valuation_date']).dt.strftime('%Y-%m-%d')
        df_leverage['valuation_date'] = pd.to_datetime(df_leverage['valuation_date']).dt.strftime('%Y-%m-%d')
        
        # 创建df_leverage2，按日期和指数分组求和
        df_leverage2_list = []
        
        # 对每个指数按日期处理
        for df_stocks, org_name in [(hs300_stocks, 'hs300'), (gz2000_stocks, 'gz2000')]:
            # 按日期分组，获取每个日期的成分股列表
            daily_stocks = df_stocks.groupby('valuation_date')['code'].apply(list).to_dict()
            
            # 对每个日期，获取该日期的成分股并求和
            daily_sum_list = []
            for date, stocks_list in daily_stocks.items():
                # 获取该日期该指数的成分股数据
                df_date_index = df_leverage[
                    (df_leverage['valuation_date'] == date) & 
                    (df_leverage['code'].isin(stocks_list))
                ].copy()
                
                # 对该日期的成分股求和
                if not df_date_index.empty:
                    daily_sum = df_date_index['NetLeverage_buying'].sum()
                    daily_sum_list.append({
                        'valuation_date': date,
                        'value': daily_sum,
                        'organization': org_name
                    })
            
            if daily_sum_list:
                df_index_sum = pd.DataFrame(daily_sum_list)
                df_leverage2_list.append(df_index_sum)
        
        # 合并所有指数的数据
        if df_leverage2_list:
            df_leverage2 = pd.concat(df_leverage2_list, ignore_index=True)
        else:
            # 如果没有数据，创建一个空的DataFrame
            df_leverage2 = pd.DataFrame(columns=['valuation_date', 'value', 'organization'])
        
        # 获取指数成交额并转换为统一格式
        amt_hs300 = self.dp.raw_index_amt('沪深300')
        amt_gz2000 = self.dp.raw_index_amt('国证2000')

        # 将成交额数据转换为统一格式
        amt_hs300.rename(columns={'沪深300': 'amt'}, inplace=True)
        amt_hs300['organization'] = 'hs300'
        amt_gz2000.rename(columns={'国证2000': 'amt'}, inplace=True)
        amt_gz2000['organization'] = 'gz2000'
        
        # 确保成交额数据的日期格式为字符串格式
        amt_hs300['valuation_date'] = pd.to_datetime(amt_hs300['valuation_date']).dt.strftime('%Y-%m-%d')
        amt_gz2000['valuation_date'] = pd.to_datetime(amt_gz2000['valuation_date']).dt.strftime('%Y-%m-%d')

        # 合并成交额数据
        df_amt = pd.concat([amt_hs300[['valuation_date', 'organization', 'amt']],
                           amt_gz2000[['valuation_date', 'organization', 'amt']]], ignore_index=True)
        # 如果df_leverage2为空，直接返回空DataFrame
        if df_leverage2.empty:
            return pd.DataFrame(columns=['valuation_date', 'NetLeverageAMTProportion_difference'])
        
        # Merge df_leverage2 和 df_amt
        df_merged = df_leverage2.merge(df_amt, on=['valuation_date', 'organization'], how='left')
        # 计算相对于成交额的比例
        df_merged['nlb_ratio'] = df_merged.apply(
            lambda row: row['value'] / row['amt'] if pd.notna(row['amt']) and row['amt'] != 0 else 0, axis=1
        )
        # 计算NetLeverageAMTProportion_difference
        df_pivot = df_merged.pivot_table(index='valuation_date', columns='organization', values='nlb_ratio', aggfunc='first')
        
        # 确保所有需要的列存在，如果不存在则填充0
        required_cols = ['hs300', 'gz2000']
        for col in required_cols:
            if col not in df_pivot.columns:
                df_pivot[col] = 0
        
        # 计算差值
        df_pivot['NetLeverageAMTProportion_difference'] = df_pivot['hs300'] - df_pivot['gz2000']
        # 转换为长格式
        df_final = df_pivot.reset_index().melt(id_vars='valuation_date', 
                                                value_vars=['hs300', 'gz2000', 'NetLeverageAMTProportion_difference'],
                                                var_name='organization', value_name='value')
        df_final['type'] = 'NetLeverageBuying'
        df_final = df_final.dropna(subset=['value'])
        df_final=df_final[df_final['organization']=='NetLeverageAMTProportion_difference']
        df_final.rename(columns={'value':'NetLeverageAMTProportion_difference'},inplace=True)
        df_final=df_final[['valuation_date','NetLeverageAMTProportion_difference']]
        df_final['NetLeverageAMTProportion_difference'] = df_final['NetLeverageAMTProportion_difference'].rolling(20).sum()
        df_final.dropna(inplace=True)
        # df_final.to_csv('NetLeverageBuying.csv')
        return df_final
    def LHBProportion(self):
        """
        计算龙虎榜占比
        
        计算龙虎榜成交金额占全市场成交金额的比例
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - LHBProportion: 龙虎榜占比（龙虎榜金额 / 全市场成交金额）
        """
        df_lhb = self.dp.raw_LHBProportion_withdraw()
        # 只保留.SH和.SZ后缀的股票
        df_lhb = df_lhb[df_lhb['ts_code'].str.endswith(('.SH', '.SZ'))]
        df_index = self.dp.raw_index_amt(None)
        
        # 确保日期格式为字符串格式
        df_lhb['valuation_date'] = pd.to_datetime(df_lhb['valuation_date']).dt.strftime('%Y-%m-%d')
        df_index['valuation_date'] = pd.to_datetime(df_index['valuation_date']).dt.strftime('%Y-%m-%d')
        
        # 排除特定指数
        exclude_codes = ['932000.CSI', '999004.SSI', '000510.CSI']
        df_index_filtered = df_index[~df_index['code'].isin(exclude_codes)]
        
        # 按日期分组处理
        result_list = []
        
        # 获取所有日期
        all_dates = sorted(set(df_lhb['valuation_date'].unique()) & set(df_index_filtered['valuation_date'].unique()))
        
        for date in all_dates:
            # 获取该日期的指数成交额数据
            df_index_date = df_index_filtered[df_index_filtered['valuation_date'] == date]
            amt_sum = df_index_date['amt'].sum()
            
            # 获取该日期的龙虎榜数据
            df_lhb_date = df_lhb[df_lhb['valuation_date'] == date]
            
            # 对该日期的每个股票取最小amount（idxmin）
            if not df_lhb_date.empty:
                df_lhb_unique = df_lhb_date.loc[df_lhb_date.groupby('ts_code')['amount'].idxmin()]
                lhb_sum = df_lhb_unique['amount'].sum()
                
                # 计算占比
                if amt_sum != 0:
                    proportion = lhb_sum / amt_sum
                    result_list.append({
                        'valuation_date': date,
                        'LHBProportion': proportion
                    })
        
        # 转换为DataFrame
        if result_list:
            df_final = pd.DataFrame(result_list)
        else:
            df_final = pd.DataFrame(columns=['valuation_date', 'LHBProportion'])
        
        return df_final
    def index_future_withdraw(self, x):
        """
        根据期货代码判断对应的指数类型
        
        Parameters:
        -----------
        x : str
            期货代码
        
        Returns:
        --------
        str
            指数类型：'hs300'、'zz500'、'zz1000' 或 'other'
        """
        if 'IF' in x:
            return 'hs300'
        elif 'IC' in x:
            return 'zz500'
        elif 'IM' in x:
            return 'zz1000'
        else:
            return 'other'
    def futureDifference(self):
        """
        计算期货与现货价差差异
        
        计算沪深300、中证500/中证1000的期货与现货价差，然后计算总差异
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - indexFuture_difference: 期货价差差异值
        """
        df_future = self.dp.future_difference_withdraw()
        df_index = self.dp.raw_index_close(None)
        # 确保日期格式为字符串格式
        if 'valuation_date' in df_future.columns:
            df_future['valuation_date'] = pd.to_datetime(df_future['valuation_date']).dt.strftime('%Y-%m-%d')
        df_index['valuation_date'] = pd.to_datetime(df_index['valuation_date']).dt.strftime('%Y-%m-%d')
        
        # 添加指数类型标识
        df_future['is_index'] = df_future['code'].apply(lambda x: self.index_future_withdraw(x))
        df_future = df_future[~(df_future['is_index'] == 'other')]
        df_future['len'] = df_future['code'].apply(lambda x: len(x))
        df_future = df_future[df_future['len'] == 6]
        # 为每个日期和指数类型添加排序索引，用于去掉第一个
        df_future = df_future.sort_values(['valuation_date', 'is_index', 'code']).reset_index(drop=True)
        df_future['rank'] = df_future.groupby(['valuation_date', 'is_index']).cumcount()
        df_future = df_future[df_future['rank'] > 0]  # 去掉第一个
        
        # 按日期和指数类型分组计算均值
        df_future_mean = df_future.groupby(['valuation_date', 'is_index'])['close'].mean().reset_index()
        df_future_mean = df_future_mean.pivot(index='valuation_date', columns='is_index', values='close').reset_index()

        # 确保所有需要的列都存在，如果不存在则填充 NaN
        required_cols = ['hs300', 'zz1000', 'zz500']
        for col in required_cols:
            if col not in df_future_mean.columns:
                df_future_mean[col] = pd.NA
        
        # 判断每个日期使用哪个指数（zz1000 还是 zz500）
        # 直接检查 pivot 后的列是否存在 zz1000
        df_future_mean['has_zz1000'] = df_future_mean['zz1000'].notna()
        # 准备指数数据：创建宽格式
        df_index_hs300 = df_index[df_index['code'] == '000300.SH'][['valuation_date', 'close']].copy()
        df_index_hs300.columns = ['valuation_date', 'index_close_hs300']
        
        df_index_zz1000 = df_index[df_index['code'] == '000852.SH'][['valuation_date', 'close']].copy()
        df_index_zz1000.columns = ['valuation_date', 'index_close_zz1000']
        
        df_index_zz500 = df_index[df_index['code'] == '000905.SH'][['valuation_date', 'close']].copy()
        df_index_zz500.columns = ['valuation_date', 'index_close_zz500']
        
        # 合并所有指数数据
        df_index_merged = df_index_hs300.merge(df_index_zz1000, on='valuation_date', how='outer')
        df_index_merged = df_index_merged.merge(df_index_zz500, on='valuation_date', how='outer')
        
        # 合并期货和指数数据
        df_merged = df_future_mean.merge(df_index_merged, on='valuation_date', how='inner')
        
        # 计算差值
        # hs300 差值
        df_merged['difference_hs300'] = df_merged['index_close_hs300'] - df_merged['hs300']
        
        # zz 差值（根据是否有 zz1000 选择，使用向量化操作）
        # 先计算 zz1000 的差值
        df_merged['difference_zz1000'] = df_merged['index_close_zz1000'] - df_merged['zz1000']
        # 再计算 zz500 的差值
        df_merged['difference_zz500'] = df_merged['index_close_zz500'] - df_merged['zz500']

        # 根据 has_zz1000 选择使用哪个差值
        df_merged['difference_zz'] = df_merged['difference_zz1000'].where(
            df_merged['has_zz1000'] & df_merged['zz1000'].notna(),
            df_merged['difference_zz500']
        )
        df_merged2=df_merged.copy()
        df_merged2=df_merged2[['valuation_date','hs300','zz500']]
        df_merged2.dropna(inplace=True)
        # 总差值
        df_merged['difference_future'] = df_merged['difference_hs300']-df_merged['difference_zz']
        
        # 选择需要的列并转换为长格式
        df_result = df_merged[['valuation_date', 'difference_hs300', 'difference_zz', 'difference_future']].copy()
        df_final = df_result.melt(
            id_vars='valuation_date',
            value_vars=['difference_hs300', 'difference_zz', 'difference_future'],
            var_name='organization',
            value_name='value'
        )
        
        # 重命名 organization 列的值
        org_mapping = {
            'difference_hs300': 'hs300',
            'difference_zz': 'zz1000',
            'difference_future': 'indexFuture_difference'
        }
        df_final['organization'] = df_final['organization'].map(org_mapping)
        
        # 添加 type 列
        df_final['type'] = 'FutureDifference'
        # 删除 NaN 值
        df_final = df_final.dropna(subset=['value'])
        
        # 重新排列列顺序
        df_final = df_final[['valuation_date', 'organization', 'type', 'value']]
        df_final=df_final[df_final['organization']=='indexFuture_difference']
        df_final=df_final[['valuation_date','value']]
        df_final.columns=['valuation_date','indexFuture_difference']
        return df_final
    def rrscoreDifference(self):
        """
        计算RR评分差异
        
        计算沪深300和国证2000成分股的平均RR评分，然后计算它们的差值
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - rrscoreDifference: RR评分差异值（沪深300 - 国证2000）
        """
        df_score = self.dp.raw_rrscore_withdraw()
        df_index_hs300 = self.dp.raw_index_weight('沪深300')
        df_index_gz2000 = self.dp.raw_index_weight('国证2000')
        
        # 确保日期格式为字符串格式
        df_score['valuation_date'] = pd.to_datetime(df_score['valuation_date']).dt.strftime('%Y-%m-%d')
        df_index_hs300['valuation_date'] = pd.to_datetime(df_index_hs300['valuation_date']).dt.strftime('%Y-%m-%d')
        df_index_gz2000['valuation_date'] = pd.to_datetime(df_index_gz2000['valuation_date']).dt.strftime('%Y-%m-%d')
        
        # 为指数成分股添加标识
        df_index_hs300['organization'] = 'hs300'
        df_index_gz2000['organization'] = 'gz2000'
        
        # 合并两个指数的成分股数据
        df_index_combined = pd.concat([
            df_index_hs300[['valuation_date', 'code', 'organization']],
            df_index_gz2000[['valuation_date', 'code', 'organization']]
        ], ignore_index=True)
        
        # 将 score 数据与成分股数据进行 merge，匹配日期和 code
        df_merged = df_score.merge(
            df_index_combined,
            on=['valuation_date', 'code'],
            how='inner'
        )
        
        # 按日期和指数类型分组计算均值
        df_result = df_merged.groupby(['valuation_date', 'organization'])['final_score'].mean().reset_index()
        df_result.rename(columns={'final_score': 'value'}, inplace=True)
        df_result['type'] = 'rrIndexScore'
        
        # 转换为宽格式以计算 difference
        df_pivot = df_result.pivot(index='valuation_date', columns='organization', values='value').reset_index()
        
        # 填充缺失值
        df_pivot.fillna(method='ffill', inplace=True)
        
        # 计算 difference
        df_pivot['difference'] = df_pivot['hs300'] - df_pivot['gz2000']
        
        # 将 difference 转换为长格式，匹配 df_result 的格式
        df_final = df_pivot[['valuation_date', 'difference']].copy()
        df_final['organization'] = 'difference'
        df_final['type'] = 'rrIndexScore'
        df_final.rename(columns={'difference': 'value'}, inplace=True)
        
        # 重新排列列顺序，匹配 df_result 的格式
        df_final = df_final[['valuation_date', 'value']]
        df_final.columns=['valuation_date','rrscoreDifference']
        return df_final
    def index_PB(self):
        """
        计算指数PB相对值
        
        计算沪深300与中证500的PB比值（相对值）
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - relative_value: PB相对值（000300.SH / 000905.SH）
        """
        df=self.dp.raw_indexBasic()
        df=df[['valuation_date','ts_code','pb']]
        df=df[df['ts_code'].isin(['000300.SH','000905.SH'])]
        df=gt.sql_to_timeseries(df)
        df['relative_value']=df['000300.SH']/df['000905.SH']
        df=df[['valuation_date','relative_value']]
        # 确保日期列为datetime类型，并按日期排序
        df['valuation_date'] = pd.to_datetime(df['valuation_date'])
        df = df.sort_values('valuation_date').reset_index(drop=True)
        
        # 定义函数：计算当前值在过去252*3天中的分位数位置
        def calculate_quantile_position(series, window=252*3):
            """
            计算每一天的值在过去window天中所对应的分位数位置
            返回0到1之间的值，0表示最小值，1表示最大值
            如果窗口数据长度小于window，则返回None
            """
            result = []
            for i in range(len(series)):
                # 获取过去window天的数据（包括当前天）
                start_idx = max(0, i - window + 1)
                end_idx = i + 1
                window_data = series.iloc[start_idx:end_idx].dropna()
                
                # 如果窗口数据长度小于window，返回None
                if len(window_data) < window:
                    result.append(None)
                elif len(window_data) == 0:
                    result.append(None)
                else:
                    current_value = series.iloc[i]
                    if pd.isna(current_value):
                        result.append(None)
                    else:
                        # 计算当前值在窗口数据中的分位数位置
                        # 使用rank方法，然后归一化到0-1之间
                        window_series = pd.Series(window_data)
                        # 计算当前值的排名（从1开始）
                        rank = (window_series <= current_value).sum()
                        # 转换为分位数位置 (rank - 1) / (len - 1)
                        if len(window_data) > 1:
                            quantile_pos = (rank - 1) / (len(window_data) - 1)
                        else:
                            quantile_pos = 0.5
                        result.append(quantile_pos)
            return pd.Series(result, index=series.index)
        # df['quantile_0.2']=df['relative_value'].rolling(252*3).quantile(0.2)
        # df['quantile_0.8'] = df['relative_value'].rolling(252 * 3).quantile(0.8)
        #df['relative_quantile'] = calculate_quantile_position(df['relative_value'], window=252 * 3)
        # 将日期转换回字符串格式
        df['valuation_date'] = df['valuation_date'].dt.strftime('%Y-%m-%d')
        df.dropna(inplace=True)
        df=df[['valuation_date','relative_value']]
        return df

    def index_PE(self):
        """
        计算指数PE相对值
        
        计算沪深300与中证500的PE比值（相对值）
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - relative_value: PE相对值（000300.SH / 000905.SH）
        """
        df = self.dp.raw_indexBasic()
        df = df[['valuation_date', 'ts_code', 'pe_ttm']]
        df = df[df['ts_code'].isin(['000300.SH', '000905.SH'])]
        df = gt.sql_to_timeseries(df)
        df['relative_value'] = df['000300.SH'] / df['000905.SH']
        df = df[['valuation_date', 'relative_value']]
        # 确保日期列为datetime类型，并按日期排序
        df['valuation_date'] = pd.to_datetime(df['valuation_date'])
        df = df.sort_values('valuation_date').reset_index(drop=True)

        # 定义函数：计算当前值在过去252*3天中的分位数位置
        def calculate_quantile_position(series, window=252 * 3):
            """
            计算每一天的值在过去window天中所对应的分位数位置
            返回0到1之间的值，0表示最小值，1表示最大值
            如果窗口数据长度小于window，则返回None
            """
            result = []
            for i in range(len(series)):
                # 获取过去window天的数据（包括当前天）
                start_idx = max(0, i - window + 1)
                end_idx = i + 1
                window_data = series.iloc[start_idx:end_idx].dropna()

                # 如果窗口数据长度小于window，返回None
                if len(window_data) < window:
                    result.append(None)
                elif len(window_data) == 0:
                    result.append(None)
                else:
                    current_value = series.iloc[i]
                    if pd.isna(current_value):
                        result.append(None)
                    else:
                        # 计算当前值在窗口数据中的分位数位置
                        # 使用rank方法，然后归一化到0-1之间
                        window_series = pd.Series(window_data)
                        # 计算当前值的排名（从1开始）
                        rank = (window_series <= current_value).sum()
                        # 转换为分位数位置 (rank - 1) / (len - 1)
                        if len(window_data) > 1:
                            quantile_pos = (rank - 1) / (len(window_data) - 1)
                        else:
                            quantile_pos = 0.5
                        result.append(quantile_pos)
            return pd.Series(result, index=series.index)

        # df['quantile_0.2']=df['relative_value'].rolling(252*3).quantile(0.2)
        # df['quantile_0.8'] = df['relative_value'].rolling(252 * 3).quantile(0.8)
        # df['relative_quantile'] = calculate_quantile_position(df['relative_value'], window=252 * 3)
        # 将日期转换回字符串格式
        df['valuation_date'] = df['valuation_date'].dt.strftime('%Y-%m-%d')
        df.dropna(inplace=True)
        df = df[['valuation_date', 'relative_value']]
        return df
    def monthly_effect(self):
        """
        计算月度效应因子
        
        计算历史同月份的沪深300与国证2000收益率差值的平均数（不包括当年的数据）
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - monthly_effect: 历史同月份平均收益率差值
        """
        """
        计算历史同月份的沪深300与国证2000收益率差值的平均数
        不包括当年的数据
        数据日期为available_date，但输出目标日期（下一个工作日）
        计算时使用目标日期的月份来查找历史同月份数据
        最后结果多输出一天，使用最后一条数据的monthly_effect值

        Returns:
            DataFrame: 包含日期和历史同月份平均收益率差值
        """
        # 获取原始数据
        df_return = self.dp.index_return_withdraw2()

        # 转换日期格式
        df_return['valuation_date'] = pd.to_datetime(df_return['valuation_date'])

        # 添加月份和年份列
        df_return['month'] = df_return['valuation_date'].dt.month
        df_return['year'] = df_return['valuation_date'].dt.year

        # 计算沪深300和国证2000的收益率差值
        df_return['return_diff'] = df_return['上证50'] - df_return['中证2000']

        # 创建结果DataFrame，基于available_date
        result_df = pd.DataFrame()
        result_df['available_date'] = df_return['valuation_date'].unique()
        result_df['available_date'] = pd.to_datetime(result_df['available_date'])

        # 计算对应的target_date（下一个工作日）
        result_df['target_date'] = result_df['available_date'].apply(
            lambda x: pd.to_datetime(gt.next_workday_calculate(x.strftime('%Y-%m-%d')))
        )

        # 对每个日期计算历史同月份的平均收益率差值
        # 使用target_date的月份来查找历史数据
        def calc_historical_avg(row):
            available_date = row['available_date']
            target_date = row['target_date']
            target_month = target_date.month
            target_year = target_date.year

            # 使用target_date的月份来查找历史同月份数据
            # 历史数据需要小于available_date（因为available_date是数据可用日期）
            historical_data = df_return[
                (df_return['valuation_date'] < target_date) &
                (df_return['month'] == target_month) &
                (df_return['year'] < target_year)  # 排除当年的数据
                ]
            return historical_data['return_diff'].mean() if not historical_data.empty else None

        # 应用计算函数
        result_df['monthly_effect'] = result_df.apply(calc_historical_avg, axis=1)

        # 使用target_date作为输出日期
        result_df['valuation_date'] = result_df['available_date'].dt.strftime('%Y-%m-%d')
        result_df.dropna(inplace=True)
        result_df = result_df[['valuation_date', 'monthly_effect']]

        return result_df[['valuation_date', 'monthly_effect']]
    def stock_highLow(self):
        """
        计算股票高低点信号差异
        
        对每只股票：
        - 如果当天股价大于过去三个月最高价，信号为1
        - 如果当天股价小于过去三个月最低价，信号为-1
        - 其他情况信号为0
        
        然后计算每天信号为1的股票数量减去信号为-1的股票数量的差值
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - stock_highLow: 高低点信号差值
        """
        """
        计算每只股票过去三个月的最高价和最低价信号
        如果当天股价大于过去三个月最高价返回1
        如果当天股价小于过去三个月最低价返回-1
        其他情况返回0
        最后计算每天信号为1的股票数量减去信号为-1的股票数量的差值
        
        Returns:
            DataFrame: 包含日期和信号差值
        """
        # 获取股票收盘价数据
        df_stock = self.dp.raw_stockClose_withdraw()
        
        # 转换日期列为datetime类型，并删除无效日期
        df_stock['valuation_date'] = pd.to_datetime(df_stock['valuation_date'])
        df_stock = df_stock.dropna(subset=['valuation_date'])
        
        # 设置日期为索引以便进行时间序列操作
        df_stock.set_index('valuation_date', inplace=True)
        
        # 计算过去三个月的最高价和最低价（不包含当天）
        high_3m = df_stock.rolling(window='90D', min_periods=1).max().shift(1)
        low_3m = df_stock.rolling(window='90D', min_periods=1).min().shift(1)
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=df_stock.index)
        
        # 对每只股票计算信号
        for col in df_stock.columns:
            # 获取当前价格
            current_price = df_stock[col]
            # 获取历史最高价和最低价（不包含当天）
            hist_high = high_3m[col]
            hist_low = low_3m[col]
            
            # 计算信号
            signal = pd.Series(0, index=current_price.index)
            # 只在有足够历史数据时计算信号
            valid_mask = current_price.notna() & hist_high.notna() & hist_low.notna()
            
            # 计算信号
            signal[valid_mask & (current_price > hist_high)] = 1
            signal[valid_mask & (current_price < hist_low)] = -1
            
            # 将信号添加到结果DataFrame
            result[col] = signal

        
        # 计算每天的有效股票数量（收盘价不为空的股票）
        valid_stocks = df_stock.notna().sum(axis=1)
        
        # 计算每天信号为1和-1的股票数量
        signal_1_count = (result == 1).sum(axis=1)
        signal_minus_1_count = (result == -1).sum(axis=1)
        
        # 计算信号差值
        signal_diff = signal_1_count - signal_minus_1_count
        
        # 创建最终结果DataFrame
        final_result = pd.DataFrame({
            'valuation_date': signal_diff.index,
            'stock_highLow': signal_diff,
            'valid_stocks': valid_stocks
        })
        
        # 去掉前90天的数据
        final_result = final_result.iloc[90:]
        
        # 将日期转换回字符串格式，确保没有NaT值
        final_result = final_result.dropna(subset=['valuation_date'])
        final_result['valuation_date'] = final_result['valuation_date'].dt.strftime('%Y-%m-%d')
        final_result.reset_index(inplace=True,drop=True)
        
        return final_result[['valuation_date', 'stock_highLow']]
    
    def stock_raisingtrend(self):
        """
        计算股票上升趋势数量
        
        对每只股票：
        - 计算30日移动平均线（30日线）
        - 判断当天股价是否在30日线之上
        
        然后计算每天在30日线之上的股票数量
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - raising_number: 在30日线之上的股票数量
        """
        # 获取股票收盘价数据
        df_stock = self.dp.raw_stockClose_withdraw()
        
        # 转换日期列为datetime类型，并删除无效日期
        df_stock['valuation_date'] = pd.to_datetime(df_stock['valuation_date'])
        df_stock = df_stock.dropna(subset=['valuation_date'])
        
        # 设置日期为索引以便进行时间序列操作
        df_stock.set_index('valuation_date', inplace=True)
        
        # 计算30日移动平均线
        ma_30 = df_stock.rolling(window=30, min_periods=1).mean()
        
        # 创建结果DataFrame，用于存储每只股票是否在30日线之上（1表示在之上，0表示不在）
        result = pd.DataFrame(index=df_stock.index)
        
        # 对每只股票判断是否在30日线之上
        for col in df_stock.columns:
            # 获取当前价格
            current_price = df_stock[col]
            # 获取30日移动平均线
            ma_30_col = ma_30[col]
            
            # 计算信号：在30日线之上为1，否则为0
            signal = pd.Series(0, index=current_price.index)
            # 只在有足够数据时计算信号
            valid_mask = current_price.notna() & ma_30_col.notna()
            
            # 判断是否在30日线之上
            signal[valid_mask & (current_price > ma_30_col)] = 1
            
            # 将信号添加到结果DataFrame
            result[col] = signal
        
        # 计算每天在30日线之上的股票数量
        raising_count = (result == 1).sum(axis=1)
        
        # 创建最终结果DataFrame
        final_result = pd.DataFrame({
            'valuation_date': raising_count.index,
            'raising_number': raising_count
        })
        
        # 去掉前30天的数据（因为前30天可能没有完整的30日线数据）
        final_result = final_result.iloc[30:]
        
        # 将日期转换回字符串格式，确保没有NaT值
        final_result = final_result.dropna(subset=['valuation_date'])
        final_result['valuation_date'] = final_result['valuation_date'].dt.strftime('%Y-%m-%d')
        final_result.reset_index(inplace=True, drop=True)
        
        return final_result[['valuation_date', 'raising_number']]
    
    def stock_number(self):
        """
        计算涨跌停家数差值
        
        计算每天股票涨跌停家数之差：
        - 00、60开头：涨跌停阈值9.5%
        - 68、30开头：涨跌停阈值19.5%
        - 正数为涨停多于跌停，负数为跌停多于涨停
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - stock_number: 涨跌停家数差值
        """
        """
        计算每天股票涨跌停家数之差
        正数为涨停，负数为跌停
        00、60开头：涨跌停阈值9.5%
        68、30开头：涨跌停阈值19.5%

        Returns:
            DataFrame: 包含日期和涨跌停家数差值的DataFrame
        """
        # 获取股票收益率数据
        df_stock = self.dp.raw_stockPct_withdraw()

        # 转换日期列为datetime类型，并删除无效日期
        df_stock['valuation_date'] = pd.to_datetime(df_stock['valuation_date'])
        df_stock = df_stock.dropna(subset=['valuation_date'])

        # 过滤股票代码，只保留00、60、68、30开头的股票
        valid_stocks = []
        for col in df_stock.columns:
            if col != 'valuation_date':
                # 检查股票代码开头
                if (col.startswith('00') or col.startswith('60') or 
                    col.startswith('68') or col.startswith('30')):
                    valid_stocks.append(col)
        
        # 只保留符合条件的股票列
        df_stock = df_stock[['valuation_date'] + valid_stocks]

        # 设置日期为索引以便进行时间序列操作
        df_stock.set_index('valuation_date', inplace=True)

        # 创建结果DataFrame
        result = pd.DataFrame(index=df_stock.index)

        # 对每只股票计算涨跌停信号
        for col in df_stock.columns:
            # 获取收益率序列（已经是百分比形式）
            returns = df_stock[col]
            
            # 根据股票代码开头确定涨跌停阈值
            if col.startswith('00') or col.startswith('60'):
                limit_threshold = 0.095  # 9.5%
            elif col.startswith('68') or col.startswith('30'):
                limit_threshold = 0.195 # 19.5%
            else:
                continue  # 跳过不符合条件的股票

            # 计算涨跌停信号
            signal = pd.Series(0, index=returns.index)
            signal[returns >= limit_threshold] = 1    # 涨停
            signal[returns <= -limit_threshold] = -1  # 跌停
            # 其他情况保持为0（正常波动）

            # 将信号添加到结果DataFrame
            result[col] = signal

        # 计算每天涨停和跌停的股票数量
        limit_up_count = (result == 1).sum(axis=1)    # 涨停股票数量
        limit_down_count = (result == -1).sum(axis=1)  # 跌停股票数量

        # 计算涨跌停家数差值（正数为涨停多于跌停，负数为跌停多于涨停）
        stock_diff = limit_up_count - limit_down_count

        # 创建最终结果DataFrame
        final_result = pd.DataFrame({
            'valuation_date': stock_diff.index,
            'stock_number': stock_diff
        })

        # 将日期转换回字符串格式，确保没有NaT值
        final_result = final_result.dropna(subset=['valuation_date'])
        final_result['valuation_date'] = final_result['valuation_date'].dt.strftime('%Y-%m-%d')
        final_result.reset_index(inplace=True, drop=True)
        return final_result

    def stock_rsi(self):
        """
        计算股票RSI差异
        
        计算每只股票的RSI指标（14天窗口），然后计算每日RSI>70的股票数量
        与RSI<30的股票数量之差
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - rsi_difference: RSI差异值（超买股票数量 - 超卖股票数量）
        """
        """
        计算每只股票的RSI指标
        1. 计算每只股票的日收益率
        2. 使用14天窗口计算RSI
        3. RSI = A/(A+abs(B))*100，其中A为14天内正收益之和，B为14天内负收益之和
        4. 计算每日RSI>70的股票数量与RSI<30的股票数量之差
        
        Returns:
            DataFrame: 包含日期和RSI差值（RSI>70的股票数量 - RSI<30的股票数量）
        """
        # 获取股票收盘价数据
        df_stock = self.dp.raw_stockClose_withdraw()
        
        # 转换日期列为datetime类型，并删除无效日期
        df_stock['valuation_date'] = pd.to_datetime(df_stock['valuation_date'])
        df_stock = df_stock.dropna(subset=['valuation_date'])
        
        # 设置日期为索引以便进行时间序列操作
        df_stock.set_index('valuation_date', inplace=True)
        
        # 计算每只股票的日收益率
        df_returns = df_stock.pct_change()
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=df_stock.index)
        
        # 对每只股票计算RSI
        for col in df_returns.columns:
            # 获取收益率序列
            returns = df_returns[col]
            
            # 分离正收益和负收益
            positive_returns = returns.copy()
            negative_returns = returns.copy()
            positive_returns[positive_returns < 0] = 0
            negative_returns[negative_returns > 0] = 0
            
            # 计算14天窗口内的正收益和负收益之和
            positive_sum = positive_returns.rolling(window=14, min_periods=1).sum()
            negative_sum = negative_returns.rolling(window=14, min_periods=1).sum()
            
            # 计算RSI
            rsi = positive_sum / (positive_sum + abs(negative_sum)) * 100
            
            # 将RSI添加到结果DataFrame
            result[col] = rsi
        
        # 计算每天RSI大于70和小于30的股票数量
        rsi_high_count = (result > 70).sum(axis=1)
        rsi_low_count = (result < 30).sum(axis=1)
        
        # 计算信号差值（超买股票数量减去超卖股票数量）
        signal_diff = rsi_high_count - rsi_low_count
        
        # 创建最终结果DataFrame
        final_result = pd.DataFrame({
            'valuation_date': signal_diff.index,
            'rsi_difference': signal_diff
        })
        
        # 去掉前14天的数据（因为RSI需要14天数据）
        final_result = final_result.iloc[14:]
        
        # 将日期转换回字符串格式，确保没有NaT值
        final_result = final_result.dropna(subset=['valuation_date'])
        final_result['valuation_date'] = final_result['valuation_date'].dt.strftime('%Y-%m-%d')
        final_result.reset_index(inplace=True, drop=True)
        return final_result
    def stock_trend(self):
        """
        计算上涨趋势股票比例
        
        计算每日收盘价格在20日均线上方的股票比例
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - RaisingTrend_proportion: 上涨趋势股票比例（百分比）
        """
        """
        计算每日收盘价格在月均线上方的股票比例
        1. 计算每只股票的20日移动平均线
        2. 计算收盘价在均线上方的股票数量
        3. 计算该数量与有效股票总数的比例
        
        Returns:
            DataFrame: 包含日期和上涨趋势股票比例
        """
        # 获取股票收盘价数据
        df_stock = self.dp.raw_stockClose_withdraw()
        
        # 转换日期列为datetime类型，并删除无效日期
        df_stock['valuation_date'] = pd.to_datetime(df_stock['valuation_date'])
        df_stock = df_stock.dropna(subset=['valuation_date'])
        
        # 设置日期为索引以便进行时间序列操作
        df_stock.set_index('valuation_date', inplace=True)
        
        # 计算20日移动平均线
        ma20 = df_stock.rolling(window=20, min_periods=1).mean()
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=df_stock.index)
        
        # 对每只股票判断是否在均线上方
        for col in df_stock.columns:
            # 获取收盘价和均线
            close = df_stock[col]
            ma = ma20[col]
            
            # 判断是否在均线上方
            above_ma = (close > ma).astype(int)
            
            # 将结果添加到结果DataFrame
            result[col] = above_ma
        
        # 计算每日有效股票数量（收盘价不为0的股票）
        valid_stocks = (df_stock != 0).sum(axis=1)
        
        # 计算每日在均线上方的股票数量
        above_ma_count = result.sum(axis=1)
        
        # 计算比例
        raising_trend_proportion = above_ma_count / valid_stocks * 100
        
        # 创建最终结果DataFrame
        final_result = pd.DataFrame({
            'valuation_date': raising_trend_proportion.index,
            'RaisingTrend_proportion': raising_trend_proportion
        })
        
        # 去掉前20天的数据（因为MA需要20天数据）
        final_result = final_result.iloc[20:]
        
        # 将日期转换回字符串格式，确保没有NaT值
        final_result = final_result.dropna(subset=['valuation_date'])
        final_result['valuation_date'] = final_result['valuation_date'].dt.strftime('%Y-%m-%d')
        final_result.reset_index(inplace=True, drop=True)
        
        return final_result
    def targetIndex_MACD(self):
        """
        计算目标指数的MACD指标
        
        基于target_index计算MACD、MACD信号线和MACD柱状图
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - MACD: MACD值
            - MACD_h: MACD柱状图
            - MACD_s: MACD信号线
        """
        df=self.dp.target_index()
        df.dropna(inplace=True)
        df=df[['valuation_date','target_index']]
        df['MACD'] = ta.macd(df['target_index'])['MACD_12_26_9']
        df['MACD_h'] = ta.macd(df['target_index'])['MACDh_12_26_9']
        df['MACD_s'] = ta.macd(df['target_index'])['MACDs_12_26_9']
        df.dropna(inplace=True)
        df.reset_index(inplace=True,drop=True)
        return df
    def targetIndex_RSI(self):
        """
        计算目标指数的RSI指标
        
        基于target_index计算14日RSI
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - RSI: RSI值（0-100之间）
        """
        df=self.dp.target_index()
        df.dropna(inplace=True)
        df=df[['valuation_date','target_index']]
        df['RSI'] = ta.rsi(df['target_index'],14)
        df.dropna(inplace=True)
        df.reset_index(inplace=True,drop=True)
        df = df[['valuation_date', 'RSI']]
        return df
    def targetIndex_BOLLBAND(self):
        """
        计算目标指数的布林带指标
        
        基于target_index计算布林带（20日均线，1.5倍标准差）
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - target_index: 目标指数值
            - upper: 布林带上轨
            - middle: 布林带中轨（20日均线）
            - lower: 布林带下轨
        """
        df = self.dp.target_index()
        df.dropna(inplace=True)
        df = df[['valuation_date', 'target_index']]
        # 计算布林带指标
        # 使用正确的方式获取布林带指标
        bbands = ta.bbands(df['target_index'], length=20,std=1.5)
        if bbands is not None:
            df['upper'] = bbands['BBU_20_1.5']
            df['middle'] = bbands['BBM_20_1.5']
            df['lower'] = bbands['BBL_20_1.5']
        else:
            # 如果bbands返回None，设置默认值
            df['upper'] = df['target_index']
            df['middle'] = df['target_index']
            df['lower'] = df['target_index']
            print("Warning: Bollinger Bands calculation returned None. Using default values.")
        df=df[['valuation_date','target_index','upper','middle','lower']]
        df.dropna(inplace=True)
        df.reset_index(inplace=True,drop=True)
        return df
    def TargetIndex_MOMENTUM(self):
        """
        获取目标指数动量值
        
        直接返回target_index的值作为动量指标
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - TargetIndex_MOMENTUM: 目标指数值
        """
        df = self.dp.target_index()
        df.dropna(inplace=True)
        df = df[['valuation_date', 'target_index']]
        df.columns=['valuation_date','TargetIndex_MOMENTUM']
        return df
    def TargetIndex_MOMENTUM2(self):
        df = self.dp.index_return_withdraw()
        df = df[['valuation_date', '上证50', '中证2000']]
        df.dropna(inplace=True)
        df.set_index('valuation_date',inplace=True,drop=True)
        df = df.rolling(15).sum()
        df.dropna(inplace=True)
        df['TargetIndex_MOMENTUM'] = df['上证50'] - df['中证2000']
        df.reset_index(inplace=True)
        df=df[['valuation_date','TargetIndex_MOMENTUM']]
        return df
    def TargetIndex_RSRS(self):
        """
        计算目标指数的RSRS（Right-side Regression Slope）标准分指标
        
        RSRS计算步骤：
        1. 对于每一天，取过去N天（默认20天）的最高价和最低价
        2. 对这N天的数据进行线性回归（high对low回归），计算斜率
        3. 取过去M天（默认300天）的斜率时间序列
        4. 计算当日斜率在这M天斜率序列中的标准分（z-score）
        
        Parameters:
        -----------
        N : int, default=20
            用于计算斜率的滚动窗口大小（天数）
        M : int, default=300
            用于计算标准分的历史斜率序列长度（天数）
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - RSRS: RSRS标准分值
        """
        # 获取K线数据
        df = self.dp.target_index_candle()
        df.dropna(inplace=True)
        df.reset_index(inplace=True,drop=True)
        # 确保日期列为datetime类型，并按日期排序
        df['valuation_date'] = pd.to_datetime(df['valuation_date'])
        df = df.sort_values('valuation_date').reset_index(drop=True)
        
        # 参数设置
        N = 18  # 计算斜率的滚动窗口
        M = 300  # 计算标准分的历史斜率序列长度
        
        # 存储每日的斜率值
        slope_list = []
        
        # 对每一天计算斜率
        for i in range(len(df)):
            # 获取过去N天的数据（包括当前天）
            start_idx = max(0, i - N + 1)
            end_idx = i + 1
            window_data = df.iloc[start_idx:end_idx]
            
            # 如果窗口数据不足N天，跳过
            if len(window_data) < N:
                slope_list.append(None)
                continue
            
            # 提取high和low序列
            high_series = window_data['high'].values
            low_series = window_data['low'].values
            
            # 进行线性回归：high = slope * low + intercept
            # 使用scipy.stats.linregress计算斜率
            slope, intercept, r_value, p_value, std_err = stats.linregress(low_series, high_series)
            
            slope_list.append(slope)
        
        # 将斜率添加到DataFrame
        df['slope'] = slope_list
        
        # 计算RSRS标准分
        rsrs_list = []
        
        for i in range(len(df)):
            # 如果当前斜率为None，跳过
            if pd.isna(df.iloc[i]['slope']):
                rsrs_list.append(None)
                continue
            
            # 获取过去M天的斜率序列（不包括当前天）
            start_idx = max(0, i - M)
            end_idx = i
            slope_window = df.iloc[start_idx:end_idx]['slope'].dropna()
            
            # 如果历史斜率序列不足M天，跳过
            if len(slope_window) < M:
                rsrs_list.append(None)
                continue
            
            # 计算当前斜率在历史斜率序列中的标准分（z-score）
            current_slope = df.iloc[i]['slope']
            mean_slope = slope_window.mean()
            std_slope = slope_window.std()
            
            # 避免除零错误
            if std_slope == 0:
                rsrs_list.append(0)
            else:
                z_score = (current_slope - mean_slope) / std_slope
                rsrs_list.append(z_score)
        
        # 将RSRS添加到DataFrame
        df['RSRS'] = rsrs_list
        
        # 选择需要的列并删除NaN值
        df_result = df[['valuation_date', 'RSRS']].copy()
        df_result.dropna(inplace=True)
        
        # 将日期转换回字符串格式
        df_result['valuation_date'] = df_result['valuation_date'].dt.strftime('%Y-%m-%d')
        df_result.reset_index(inplace=True, drop=True)
        return df_result
    def TargetIndex_KDJ(self):
        """
        计算目标指数的KDJ指标
        
        基于target_index_candle计算KDJ指标（9,3参数）
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - K_9_3: K值
            - D_9_3: D值
            - J_9_3: J值
        """
        df = self.dp.target_index_candle()
        df.dropna(inplace=True)
        df_kdj = ta.kdj(df['high'], df['low'], df['close'])
        # 将 KDJ 指标合并到原 DataFrame 中
        df = pd.concat([df, df_kdj], axis=1)
        df.dropna(inplace=True)
        df=df[['valuation_date','K_9_3', 'D_9_3', 'J_9_3']]
        return df
    def TargetIndex_PSA(self):
        """
        计算目标指数的抛物线指标（PSAR）
        
        基于target_index_candle计算抛物线SAR指标
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - close: 收盘价差值
            - PSARl_0.02_0.2: 长期PSAR值
            - PSARs_0.02_0.2: 短期PSAR值
        """
        df = self.dp.target_index_candle()
        df.dropna(inplace=True)
        # 计算抛物线指标
        psar = ta.psar(df['high'], df['low'])
        # 将抛物线指标合并到原 DataFrame 中
        df = pd.concat([df, psar], axis=1)
        df=df[['valuation_date','close','PSARl_0.02_0.2','PSARs_0.02_0.2']]
        return df

    def TargetIndex_MOMENTUM3(self):
        """
        计算目标指数动量3（过去10天涨跌幅累乘）
        
        计算沪深300和国证2000过去10天的涨跌幅累乘，然后计算它们的差值，
        最后与VIX数据合并
        
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - difference: 动量差值（沪深300 - 国证2000）
            - hs300: 沪深300的VIX值
            - zz1000: 中证1000的VIX值
        """
        """
        计算过去20天的涨跌幅累乘

        Returns:
            DataFrame: 包含日期和过去20天涨跌幅累乘的结果
        """
        df = self.dp.index_return_withdraw2()
        df.set_index('valuation_date', inplace=True)

        # 对每一列分别计算20天涨跌幅累乘
        for col in df.columns:
            df[col] = df[col].rolling(10).apply(lambda x: (1 + x).prod() - 1)

        # 重置索引
        df.reset_index(inplace=True)
        df['difference'] = df['上证50'] - df['中证2000']
        df = df[['valuation_date', 'difference']]
        df.dropna(inplace=True)
        df_vix = self.dp.raw_vix_withdraw()
        df = df.merge(df_vix, on='valuation_date', how='left')
        return df

    def TargetIndex_REVERSE(self):
        """
        计算目标指数动量3（过去10天涨跌幅累乘）

        计算沪深300和国证2000过去10天的涨跌幅累乘，然后计算它们的差值，
        最后与VIX数据合并

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - difference: 动量差值（沪深300 - 国证2000）
            - hs300: 沪深300的VIX值
            - zz1000: 中证1000的VIX值
        """
        """
        计算过去20天的涨跌幅累乘

        Returns:
            DataFrame: 包含日期和过去20天涨跌幅累乘的结果
        """
        df = self.dp.index_return_withdraw2()
        df.set_index('valuation_date', inplace=True)

        # 对每一列分别计算20天涨跌幅累乘
        for col in df.columns:
            df[col] = df[col].rolling(5).apply(lambda x: (1 + x).prod() - 1)

        # 重置索引
        df.reset_index(inplace=True)
        df['difference'] = df['上证50'] - df['中证2000']
        df = df[['valuation_date', 'difference']]
        df.dropna(inplace=True)
        return df

    # ======================== 商品期货因子 ========================
    # 上游品种列表（原油/煤炭/有色/贵金属/黑色）
    COMMODITY_UPSIDE = [
        'SC',   # 原油
        'ZC', 'JM', 'J',  # 煤炭：动力煤、焦煤、焦炭
        'CU', 'AL', 'ZN', 'PB', 'NI', 'SN',  # 有色：铜、铝、锌、铅、镍、锡
        'AU', 'AG',  # 贵金属：黄金、白银
        'RB', 'HC', 'I'  # 黑色：螺纹钢、热轧卷板、铁矿石
    ]

    # 中下游品种列表（农产品/轻工）
    COMMODITY_DOWNSIDE = [
        'A', 'M', 'Y', 'P', 'OI', 'RM', 'CF', 'SR', 'C',  # 农产品
        'RU', 'FG', 'MA', 'PP', 'L', 'TA'  # 轻工
    ]

    def _get_commodity_main_contracts(self, df_commodity):
        """
        筛选主力连续合约（纯字母代码，如A、CU、RB等）

        Parameters:
        -----------
        df_commodity : pd.DataFrame
            商品期货数据

        Returns:
        --------
        pd.DataFrame
            筛选后的主力连续合约数据，包含symbol列标识品种
        """
        if df_commodity.empty:
            return pd.DataFrame(columns=['valuation_date', 'code', 'close', 'volume', 'open_interest', 'symbol'])

        # 主力连续合约的代码是纯字母（如A、CU、RB），不包含数字
        # 具体合约代码是字母+数字（如A1201、CU2001）
        import re

        def is_main_contract(code):
            """判断是否为主力连续合约（纯字母代码）"""
            code_str = str(code).strip()
            # 去掉可能的交易所后缀
            code_clean = re.sub(r'\.[A-Z]+$', '', code_str)
            # 纯字母代码为主力连续合约
            return bool(re.match(r'^[A-Za-z]+$', code_clean))

        df_main = df_commodity[df_commodity['code'].apply(is_main_contract)].copy()

        if df_main.empty:
            return pd.DataFrame(columns=['valuation_date', 'code', 'close', 'volume', 'open_interest', 'symbol'])

        # symbol就是code本身（转大写）
        df_main['symbol'] = df_main['code'].str.upper()
        # 去掉可能的交易所后缀
        df_main['symbol'] = df_main['symbol'].str.replace(r'\.[A-Z]+$', '', regex=True)

        return df_main

    def _calculate_nanhua_weights(self, df_main, date, symbol_list):
        """
        计算南华权重（基于过去一年交易金额，符合南华编制规则）

        南华规则：
        1. 权重基于流动性（交易金额）计算
        2. 单品种权重上限25%，下限2%
        3. 每年6月第一个交易日调整权重

        Parameters:
        -----------
        df_main : pd.DataFrame
            主力连续合约数据
        date : str
            计算权重的日期
        symbol_list : list
            需要计算权重的品种列表

        Returns:
        --------
        pd.DataFrame
            包含品种和权重的DataFrame
        """
        date_dt = pd.to_datetime(date)
        # 获取过去一年的数据
        year_start = (date_dt - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        df_year = df_main[(df_main['valuation_date'] >= year_start) &
                          (df_main['valuation_date'] <= date) &
                          (df_main['symbol'].isin(symbol_list))].copy()

        if df_year.empty:
            return pd.DataFrame(columns=['symbol', 'weight'])

        # 计算交易金额（流动性指标）
        df_year['trade_value'] = df_year['volume'] * df_year['close']

        # 按品种汇总年度交易金额
        df_agg = df_year.groupby('symbol').agg({
            'trade_value': 'sum'
        }).reset_index()

        # 计算流动性权重
        total_trade = df_agg['trade_value'].sum()
        if total_trade == 0:
            return pd.DataFrame(columns=['symbol', 'weight'])

        df_agg['weight'] = df_agg['trade_value'] / total_trade

        # 应用南华权重限制规则（品种数>=5时）
        if len(df_agg) >= 5:
            # 迭代调整权重，确保满足上下限约束
            for _ in range(10):  # 最多迭代10次
                # 上限约束：单品种不超过25%
                df_agg.loc[df_agg['weight'] > 0.25, 'weight'] = 0.25
                # 下限约束：单品种不低于2%
                df_agg.loc[df_agg['weight'] < 0.02, 'weight'] = 0.02
                # 重新归一化
                weight_sum = df_agg['weight'].sum()
                if weight_sum > 0:
                    df_agg['weight'] = df_agg['weight'] / weight_sum
                # 检查是否满足约束
                if df_agg['weight'].max() <= 0.25 and df_agg['weight'].min() >= 0.02:
                    break
        else:
            # 品种数<5时，只归一化，不限制上下限
            weight_sum = df_agg['weight'].sum()
            if weight_sum > 0:
                df_agg['weight'] = df_agg['weight'] / weight_sum

        return df_agg[['symbol', 'weight']]

    def _build_commodity_index(self, df_main, symbol_list):
        """
        构建商品指数（符合南华编制规则）

        南华规则：
        1. 每年6月第一个交易日调整权重
        2. 权重基于过去一年交易金额计算
        3. 单品种权重上限25%，下限2%
        4. 收益率截断±15%防止换月跳空

        Parameters:
        -----------
        df_main : pd.DataFrame
            主力连续合约数据
        symbol_list : list
            品种代码列表

        Returns:
        --------
        pd.DataFrame
            包含日期和指数值的DataFrame
        """
        if df_main.empty:
            return pd.DataFrame(columns=['valuation_date', 'value'])

        # 筛选指定品种
        df_filtered = df_main[df_main['symbol'].isin(symbol_list)].copy()

        if df_filtered.empty:
            return pd.DataFrame(columns=['valuation_date', 'value'])

        # 按日期和品种排序
        df_filtered = df_filtered.sort_values(['symbol', 'valuation_date'])

        # 计算每个品种的日收益率
        df_filtered['return'] = df_filtered.groupby('symbol')['close'].pct_change()

        # 处理异常收益率（主力合约切换可能导致跳空）
        # 限制单日收益率在 -15% 到 +15% 之间
        df_filtered['return'] = df_filtered['return'].clip(-0.15, 0.15)

        # 获取所有日期
        dates = sorted(df_filtered['valuation_date'].unique())

        index_values = []
        index_value = 1000.0  # 基期指数值1000（符合南华规则）
        current_weights = None  # 当前使用的权重（t-1时刻的权重）
        current_weight_year = None  # 当前权重对应的年份
        pending_weights = None  # 待生效的新权重

        for date in dates:
            date_dt = pd.to_datetime(date)
            df_date = df_filtered[df_filtered['valuation_date'] == date].copy()

            # 如果有待生效的权重，在新的一天开始时生效
            if pending_weights is not None:
                current_weights = pending_weights
                pending_weights = None

            # 首次计算时初始化权重
            if current_weights is None:
                df_weights = self._calculate_nanhua_weights(df_main, date, symbol_list)
                if not df_weights.empty:
                    current_weights = df_weights
                    current_weight_year = date_dt.year

            # 使用t-1时刻的权重计算当天指数（符合公式 CI_t = CI_{t-1} × Σ[ω_{k,t-1} × I_{k,t}/I_{k,t-1}]）
            if current_weights is None or current_weights.empty:
                # 如果无法计算权重，使用等权重
                df_date_valid = df_date[df_date['return'].notna()]
                if not df_date_valid.empty:
                    weighted_return = df_date_valid['return'].mean()
                else:
                    weighted_return = 0
            else:
                # 合并权重
                df_date = df_date.merge(current_weights, on='symbol', how='left')
                df_date['weight'] = df_date['weight'].fillna(0)

                # 计算加权收益率
                df_date_valid = df_date[(df_date['return'].notna()) & (df_date['weight'] > 0)]
                if not df_date_valid.empty:
                    # 重新归一化权重（处理缺失品种）
                    weight_sum = df_date_valid['weight'].sum()
                    if weight_sum > 0:
                        weighted_return = (df_date_valid['return'] * df_date_valid['weight']).sum() / weight_sum
                    else:
                        weighted_return = 0
                else:
                    weighted_return = 0

            # 更新指数值
            index_value = index_value * (1 + weighted_return)
            index_values.append({
                'valuation_date': date,
                'value': index_value
            })

            # 计算完当天指数后，检查是否需要更新权重（每年6月第一个交易日）
            # 新权重从下一个交易日开始生效
            if date_dt.month == 6 and current_weight_year != date_dt.year:
                june_dates = [d for d in dates if d.startswith(f"{date_dt.year}-06")]
                if june_dates and date == june_dates[0]:
                    df_weights = self._calculate_nanhua_weights(df_main, date, symbol_list)
                    if not df_weights.empty:
                        pending_weights = df_weights
                        current_weight_year = date_dt.year

        return pd.DataFrame(index_values)

    def commodity_upside(self):
        """
        计算上游商品指数（原油/煤炭/有色/贵金属/黑色）

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - value: 上游商品指数值
        """
        df_commodity = self.dp.raw_futureData_commodity()
        if df_commodity.empty:
            print("警告: commodity_upside - 商品期货数据为空")
            return pd.DataFrame(columns=['valuation_date', 'value'])

        df_main = self._get_commodity_main_contracts(df_commodity)
        if df_main.empty:
            print("警告: commodity_upside - 未找到主力连续合约数据")
            return pd.DataFrame(columns=['valuation_date', 'value'])

        df_index = self._build_commodity_index(df_main, self.COMMODITY_UPSIDE)
        df_index.dropna(inplace=True)
        return df_index

    def commodity_downside(self):
        """
        计算中下游商品指数（农产品/轻工）

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - value: 中下游商品指数值
        """
        df_commodity = self.dp.raw_futureData_commodity()
        if df_commodity.empty:
            print("警告: commodity_downside - 商品期货数据为空")
            return pd.DataFrame(columns=['valuation_date', 'value'])

        df_main = self._get_commodity_main_contracts(df_commodity)
        if df_main.empty:
            print("警告: commodity_downside - 未找到主力连续合约数据")
            return pd.DataFrame(columns=['valuation_date', 'value'])

        df_index = self._build_commodity_index(df_main, self.COMMODITY_DOWNSIDE)
        df_index.dropna(inplace=True)
        return df_index

    def commodity_volume(self):
        """
        计算期货交易活跃度综合指数变化

        计算逻辑：
        1. 单品种活跃度 = 0.5×(当日成交量/20日均值) + 0.5×(当日持仓量/20日均值)
        2. 按南华权重加权得到综合活跃度指数
        3. 输出周度变化率（5日变化率）

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - value: 活跃度指数5日变化率
        """
        df_commodity = self.dp.raw_futureData_commodity()
        if df_commodity.empty:
            print("警告: commodity_volume - 商品期货数据为空")
            return pd.DataFrame(columns=['valuation_date', 'value'])

        df_main = self._get_commodity_main_contracts(df_commodity)
        if df_main.empty:
            print("警告: commodity_volume - 未找到主力连续合约数据")
            return pd.DataFrame(columns=['valuation_date', 'value'])

        # 获取所有品种
        all_symbols = self.COMMODITY_UPSIDE + self.COMMODITY_DOWNSIDE
        df_filtered = df_main[df_main['symbol'].isin(all_symbols)].copy()

        if df_filtered.empty:
            return pd.DataFrame(columns=['valuation_date', 'value'])

        # 按日期和品种排序
        df_filtered = df_filtered.sort_values(['symbol', 'valuation_date'])

        # 计算20日均值
        df_filtered['volume_ma20'] = df_filtered.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        df_filtered['oi_ma20'] = df_filtered.groupby('symbol')['open_interest'].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )

        # 计算单品种活跃度
        df_filtered['volume_ratio'] = df_filtered['volume'] / df_filtered['volume_ma20']
        df_filtered['oi_ratio'] = df_filtered['open_interest'] / df_filtered['oi_ma20']
        df_filtered['activity'] = 0.5 * df_filtered['volume_ratio'] + 0.5 * df_filtered['oi_ratio']

        # 按日期计算加权活跃度（使用年度权重调整，符合南华规则）
        dates = sorted(df_filtered['valuation_date'].unique())
        activity_values = []
        current_weights = None  # 当前使用的权重（t-1时刻的权重）
        current_weight_year = None  # 当前权重对应的年份
        pending_weights = None  # 待生效的新权重

        for date in dates:
            date_dt = pd.to_datetime(date)
            df_date = df_filtered[df_filtered['valuation_date'] == date].copy()

            # 如果有待生效的权重，在新的一天开始时生效
            if pending_weights is not None:
                current_weights = pending_weights
                pending_weights = None

            # 首次计算时初始化权重
            if current_weights is None:
                df_weights = self._calculate_nanhua_weights(df_main, date, all_symbols)
                if not df_weights.empty:
                    current_weights = df_weights
                    current_weight_year = date_dt.year

            # 使用t-1时刻的权重计算当天活跃度
            if current_weights is None or current_weights.empty:
                # 等权重
                df_date_valid = df_date[df_date['activity'].notna()]
                if not df_date_valid.empty:
                    weighted_activity = df_date_valid['activity'].mean()
                else:
                    weighted_activity = np.nan
            else:
                df_date = df_date.merge(current_weights, on='symbol', how='left')
                df_date['weight'] = df_date['weight'].fillna(0)

                df_date_valid = df_date[(df_date['activity'].notna()) & (df_date['weight'] > 0)]
                if not df_date_valid.empty:
                    weight_sum = df_date_valid['weight'].sum()
                    if weight_sum > 0:
                        weighted_activity = (df_date_valid['activity'] * df_date_valid['weight']).sum() / weight_sum
                    else:
                        weighted_activity = np.nan
                else:
                    weighted_activity = np.nan

            activity_values.append({
                'valuation_date': date,
                'activity': weighted_activity
            })

            # 计算完当天后，检查是否需要更新权重（每年6月第一个交易日）
            if date_dt.month == 6 and current_weight_year != date_dt.year:
                june_dates = [d for d in dates if d.startswith(f"{date_dt.year}-06")]
                if june_dates and date == june_dates[0]:
                    df_weights = self._calculate_nanhua_weights(df_main, date, all_symbols)
                    if not df_weights.empty:
                        pending_weights = df_weights
                        current_weight_year = date_dt.year

        df_result = pd.DataFrame(activity_values)

        # 计算5日变化率
        df_result['value'] = df_result['activity']
        df_result = df_result[['valuation_date', 'value']]
        df_result.dropna(inplace=True)

        return df_result

    def commodity_ppi_correl(self):
        """
        计算商品综合指数与PPI的20日滚动相关性

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - value: 商品指数与PPI的20日滚动相关性
        """
        df_commodity = self.dp.raw_futureData_commodity()
        if df_commodity.empty:
            print("警告: commodity_ppi_correl - 商品期货数据为空")
            return pd.DataFrame(columns=['valuation_date', 'value'])

        df_main = self._get_commodity_main_contracts(df_commodity)
        if df_main.empty:
            print("警告: commodity_ppi_correl - 未找到主力连续合约数据")
            return pd.DataFrame(columns=['valuation_date', 'value'])

        # 构建全品种商品综合指数
        all_symbols = self.COMMODITY_UPSIDE + self.COMMODITY_DOWNSIDE
        df_index = self._build_commodity_index(df_main, all_symbols)

        if df_index.empty:
            return pd.DataFrame(columns=['valuation_date', 'value'])

        # 获取PPI数据
        df_ppi = self.dp.raw_PPI_withdraw()

        # 合并数据
        df_merged = df_index.merge(df_ppi, on='valuation_date', how='left')

        # PPI为月度数据，需要前向填充
        df_merged['PPI'] = df_merged['PPI'].fillna(method='ffill')

        # 计算商品指数的日收益率
        df_merged = df_merged.sort_values('valuation_date')
        df_merged['index_return'] = df_merged['value'].pct_change()

        # 计算PPI的变化率（用差分近似）
        df_merged['ppi_change'] = df_merged['PPI'].diff()

        # 计算20日滚动相关性
        df_merged['correlation'] = df_merged['index_return'].rolling(20).corr(df_merged['ppi_change'])

        df_result = df_merged[['valuation_date', 'correlation']].copy()
        df_result.columns = ['valuation_date', 'value']
        df_result.dropna(inplace=True)

        return df_result

    def commodity_composite(self):
        """
        计算南华商品综合指数

        使用全部30个商品期货品种（上游+中下游）构建综合指数

        信号逻辑：指数上行 = 大盘占优，下行 = 小盘占优 (mode_1)

        Returns:
        --------
        pd.DataFrame
            包含 valuation_date 和 value（指数值）的 DataFrame
        """
        df_commodity = self.dp.raw_futureData_commodity()

        if df_commodity.empty:
            print("警告: commodity_composite - 未获取到期货数据")
            return pd.DataFrame(columns=['valuation_date', 'value'])

        df_main = self._get_commodity_main_contracts(df_commodity)
        if df_main.empty:
            print("警告: commodity_composite - 未找到主力连续合约数据")
            return pd.DataFrame(columns=['valuation_date', 'value'])

        # 构建全品种商品综合指数
        all_symbols = self.COMMODITY_UPSIDE + self.COMMODITY_DOWNSIDE
        df_index = self._build_commodity_index(df_main, all_symbols)

        df_index.dropna(inplace=True)
        return df_index


if __name__ == "__main__":
    dp = data_prepare('2015-01-03', '2025-12-29')
    df2 = dp.target_index()
    df2 = df2[['valuation_date', 'target_index']]
    dpro=data_processing('2015-01-03', '2025-01-15')
    df=dpro.commodity_composite()
    print(df)
    df = df.merge(df2, on='valuation_date', how='left')
    #df = df[['valuation_date', 'difference', 'target_index']]
    df.set_index('valuation_date', inplace=True, drop=True)
    df = (df - df.min()) / (df.max() - df.min())
    df.plot()
    plt.show()