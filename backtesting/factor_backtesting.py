import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
import global_setting.global_dic as glv
from backtesting.backtesting_tools import Back_testing_processing
config_path=glv.get('config_path')
class factor_backtesting:
    def __init__(self,signal_name,start_date,end_date,cost,mode,signal_type,big_indexName,small_indexName,base_indexName,x=None):

        self.signal_name=signal_name
        self.start_date=start_date
        self.end_date=end_date
        self.cost=cost
        self.mode=mode
        self.x=x
        self.big_indexName = big_indexName
        self.small_indexName = small_indexName
        self.base_indexName=base_indexName
        self.big_indexCode = gt.index_mapping(self.big_indexName, 'code')
        self.small_indexCode = gt.index_mapping(self.small_indexName, 'code')
        if self.base_indexName!=None:
            self.base_indexCode = gt.index_mapping(self.base_indexName, 'code')
        else:
            self.base_indexCode=None
        self.signal_type=signal_type
        self.inputpath_base = glv.get(str(signal_type)+'_signalData_'+str(mode))
        self.start_date=self.start_date_processing()
        self.df_index_return=self.index_return_withdraw()
    def sql_path_withdraw(self):
        workspace_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(workspace_path, 'config_project', 'timeselecting_sql.yaml')
        return config_path
    def start_date_processing(self):
        inputpath = self.inputpath_base
        if self.signal_type != 'L0':
            inputpath = str(
                inputpath) + f" Where signal_name='{self.signal_name}'"
        df = gt.data_getting(inputpath, config_path)
        df.sort_values(by='valuation_date', inplace=True)
        running_date = df['valuation_date'].unique().tolist()[0]
        running_date = gt.strdate_transfer(running_date)
        if self.start_date < running_date:
            start_date = running_date
            print(self.start_date + '目前没有数据，已经自动调整到:' + str(running_date))
        else:
            start_date = self.start_date
        return start_date

    def index_return_withdraw(self):
        df_return = gt.indexData_withdraw(None, self.start_date, self.end_date, ['pct_chg'])
        df_return = gt.sql_to_timeseries(df_return)
        if self.base_indexCode!=None:
            df_return = df_return[['valuation_date', self.big_indexCode, self.small_indexCode,self.base_indexCode]]
            df_return.columns = ['valuation_date', self.big_indexName, self.small_indexName,self.base_indexName]
            df_return[self.base_indexName] = df_return[self.base_indexName].astype(float)
        else:
            df_return = df_return[['valuation_date', self.big_indexCode, self.small_indexCode]]
            df_return.columns = ['valuation_date', self.big_indexName, self.small_indexName]
        df_return[self.big_indexName] = df_return[self.big_indexName].astype(float)
        df_return[self.small_indexName] = df_return[self.small_indexName].astype(float)
        return df_return
    def raw_signal_withdraw(self):
        if self.signal_type!='L0':
            sql=self.inputpath_base+f" Where signal_name='{self.signal_name}' And valuation_date BETWEEN '{self.start_date}' AND '{self.end_date}'"
        else:
            sql = self.inputpath_base + f" Where valuation_date BETWEEN '{self.start_date}' AND '{self.end_date}'"
        inputpath_config=glv.get('config_path')
        df=gt.data_getting(sql,inputpath_config)
        if self.signal_type=='L3':
            df=df[df['x']==self.x]
        df=df[['valuation_date','final_signal']]
        df.sort_values('valuation_date',inplace=True)
        return df
    def probability_processing(self,df_signal):
        df_index = self.index_return_withdraw()
        df_signal = df_signal.merge(df_index, on='valuation_date', how='left')
        df_final=pd.DataFrame()
        df_signal['target'] = df_signal[self.big_indexName] - df_signal[self.small_indexName]
        df_signal.loc[df_signal['target'] > 0, ['target']] = 0
        df_signal.loc[df_signal['target'] < 0, ['target']] = 1
        df_signal['target'] = df_signal['target'].shift(-1)
        df_signal.dropna(inplace=True)
        number_0 = len(df_signal[df_signal['final_signal'] == 0])
        number_1 = len(df_signal[df_signal['final_signal'] == 1])
        number_0_correct = len(df_signal[(df_signal['final_signal'] == 0) & (df_signal['target'] == 0)])
        number_1_correct = len(df_signal[(df_signal['final_signal'] == 1) & (df_signal['target'] == 1)])
        if number_0==0:
            number_0=1
        if number_1==0:
            number_1=1
        pb_0_correct = number_0_correct / number_0
        pb_0_wrong = 1 - pb_0_correct
        pb_1_correct=number_1_correct/number_1
        pb_1_wrong=1-pb_1_correct
        df_final[self.big_indexName]=[pb_0_correct,pb_0_wrong]
        df_final[self.small_indexName]=[pb_1_correct,pb_1_wrong]
        return df_final

    def signal_return_processing(self,df_signal,index_name):
        df_index = self.index_return_withdraw()
        if self.base_indexName==None:
              df_index['大小盘等权']=0.5*df_index[self.big_indexName]+0.5*df_index[self.small_indexName]
        else:
              df_index['大小盘等权'] = df_index[self.base_indexName]
        df_signal = df_index.merge(df_signal, on='valuation_date', how='left')
        df_signal.dropna(inplace=True)
        df_signal['signal_return'] = 0
        df_signal.loc[df_signal['final_signal'] == 0, ['signal_return']] = \
        df_signal.loc[df_signal['final_signal'] == 0][self.big_indexName].tolist()
        df_signal.loc[df_signal['final_signal'] == 1, ['signal_return']] = \
        df_signal.loc[df_signal['final_signal'] == 1][self.small_indexName].tolist()
        if index_name==self.big_indexName:
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5][self.big_indexName].tolist()
        elif index_name==self.small_indexName:
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5][self.small_indexName].tolist()
        else:
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5]['大小盘等权'].tolist()
        df_signal['turn_over'] = df_signal['final_signal'] - df_signal['final_signal'].shift(1)
        df_signal['turn_over'] = abs(df_signal['turn_over']) * 2
        df_signal.fillna(method='ffill',inplace=True)
        df_signal.fillna(method='bfill',inplace=True)
        df_signal['turn_over'] = df_signal['turn_over'] * self.cost
        df_signal['portfolio'] = df_signal['signal_return'].astype(float) - df_signal['turn_over']
        df_signal = df_signal[['valuation_date', 'portfolio',index_name]]
        df_signal.rename(columns={index_name:'index'},inplace=True)
        return df_signal
    def backtesting_main_sql(self):
        inputpath_sql=self.sql_path_withdraw()
        sm = gt.sqlSaving_main(inputpath_sql,str(self.signal_type)+'_signal_'+str(self.mode)+'_backtest')
        df_signal=self.raw_signal_withdraw()
        df_portfolio = self.signal_return_processing(df_signal, '大小盘等权')
        df_portfolio.columns = ['valuation_date', 'portfolio', 'benchmark']
        df_portfolio['excess_return'] = df_portfolio['portfolio'] - df_portfolio['benchmark']
        df_portfolio['signal_name']=self.signal_name
        df_portfolio['update_time']=datetime.now()
        if len(df_portfolio)>0:
            sm.df_to_sql(df_portfolio)

    def backtesting_main(self):
        bp = Back_testing_processing(self.df_index_return,self.big_indexName,self.small_indexName,self.base_indexName)
        outputpath = glv.get('backtest_output')
        outputpath=os.path.join(outputpath,self.mode)
        outputpath=os.path.join(outputpath,self.signal_type)
        outputpath=os.path.join(outputpath,self.signal_name)
        df_signal=self.raw_signal_withdraw()
        df_prob=self.probability_processing(df_signal)
        outputpath_prob=os.path.join(outputpath,'positive_negative_probabilities.xlsx')
        gt.folder_creator2(outputpath)
        df_prob.to_excel(outputpath_prob, index=False)
        for index_name in [self.big_indexName,self.small_indexName,'大小盘等权']:
            if index_name=='大小盘等权':
                index_type='combine'
            else:
                index_type='single'
            df_portfolio = self.signal_return_processing(df_signal, index_name)
            bp.back_testing_history(df_portfolio, outputpath, index_type, index_name, self.signal_name)
        return outputpath
class L3factor_backtesting:
    def __init__(self,signal_name,start_date,end_date,cost,mode,big_indexName,small_indexName,big_proportion,small_proportion):
        self.signal_name=signal_name
        self.start_date=start_date
        self.end_date=end_date
        self.cost=cost
        self.mode=mode
        self.big_indexName=big_indexName
        self.small_indexName=small_indexName
        self.big_proportion=big_proportion
        self.small_proportion=small_proportion
        self.big_indexCode=gt.index_mapping(self.big_indexName,'code')
        self.small_indexCode = gt.index_mapping(self.small_indexName, 'code')
        if self.mode == 'prod':
            self.inputpath_base = glv.get('L3_signalData_prod')
        else:
            self.inputpath_base = glv.get('L3_signalData_test')
        self.start_date=self.start_date_processing()
        self.df_index_return = self.index_return_withdraw()
    def start_date_processing(self):
        inputpath = self.inputpath_base
        inputpath = str(
            inputpath) + f" Where signal_name='{self.signal_name}'"
        df = gt.data_getting(inputpath, config_path)
        df.sort_values(by='valuation_date',inplace=True)
        running_date=df['valuation_date'].unique().tolist()[0]
        running_date=gt.strdate_transfer(running_date)
        if self.start_date<running_date:
            start_date=running_date
            print(self.start_date+'目前没有数据，已经自动调整到:'+str(running_date))
        else:
            start_date=self.start_date
        return start_date
    def index_return_withdraw(self):
        df_return = gt.indexData_withdraw(None,self.start_date,self.end_date,['pct_chg'])
        df_return = gt.sql_to_timeseries(df_return)
        df_return = df_return[['valuation_date', self.big_indexCode, self.small_indexCode]]
        df_return.columns=['valuation_date',self.big_indexName,self.small_indexName]
        df_return[self.big_indexName] = df_return[self.big_indexName].astype(float)
        df_return[self.small_indexName] = df_return[self.small_indexName].astype(float)
        return df_return

    def raw_signal_withdraw(self):
        inputpath = self.inputpath_base
        inputpath = str(
            inputpath) + f" Where signal_name='{self.signal_name}' And valuation_date between '{self.start_date}' and '{self.end_date}'"
        df = gt.data_getting(inputpath, config_path)
        df=df[['valuation_date','final_signal','x']]
        return df
    def signal_return_processing(self,df_signal,index_name):
        x=str(df_signal['x'].unique().tolist()[0])
        df_index = self.df_index_return.copy()
        df_index['大小盘等权']=0.5*df_index[self.big_indexName]+0.5*df_index[self.small_indexName]
        df_signal = df_index.merge(df_signal, on='valuation_date', how='left')
        df_signal.dropna(inplace=True)
        df_signal['signal_return'] = 0
        df_signal.loc[df_signal['final_signal'] == 0, ['signal_return']] = \
        df_signal.loc[df_signal['final_signal'] == 0][self.big_indexName].tolist()
        df_signal.loc[df_signal['final_signal'] == 1, ['signal_return']] = \
        df_signal.loc[df_signal['final_signal'] == 1][self.small_indexName].tolist()
        if index_name==self.big_indexName:
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5][self.big_indexName].tolist()
        elif index_name==self.small_indexName:
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5][self.small_indexName].tolist()
        else:
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5]['大小盘等权'].tolist()
        df_signal['turn_over'] = df_signal['final_signal'] - df_signal['final_signal'].shift(1)
        df_signal['turn_over'] = abs(df_signal['turn_over']) * 2
        df_signal.fillna(method='ffill',inplace=True)
        df_signal.fillna(method='bfill',inplace=True)
        df_signal['turn_over'] = df_signal['turn_over'] * self.cost
        df_signal['portfolio'] = df_signal['signal_return'].astype(float) - df_signal['turn_over']
        df_signal = df_signal[['valuation_date', 'portfolio',index_name]]
        df_signal.rename(columns={index_name:'index'},inplace=True)
        df_signal['excess_return']=df_signal['portfolio']-df_signal['index']
        df_signal=df_signal[['valuation_date','excess_return']]
        df_signal[self.signal_name+'_'+x]=(1+df_signal['excess_return']).cumprod()
        df_signal=df_signal[['valuation_date',self.signal_name+'_'+x]]
        return df_signal
    def technical_signal_calculator(self,df):
        # Initialize result DataFrame
        result_df = pd.DataFrame(columns=['portfolio_name', 'annual_return', 'regression_annual_return', 'max_drawdown'])

        # Get portfolio columns (excluding valuation_date)
        portfolio_cols = [col for col in df.columns if col != 'valuation_date']

        for portfolio in portfolio_cols:
            # Calculate annualized return
            nav0 = df[portfolio].iloc[0]
            navt = df[portfolio].iloc[-1]
            total_return = navt / nav0
            t = len(df)  # Use total number of trading days
            annual_return = (total_return ** (365 / t) - 1) * 100  # Convert to annual rate and percentage

            # Calculate regression annualized return using ln(navt) - ln(navo) = kt
            k = (np.log(navt) - np.log(nav0)) / t
            regression_annual_return = k * 252 * 100  # Convert to annual rate and percentage

            # Calculate maximum drawdown
            rolling_max = df[portfolio].expanding().max()
            drawdowns = (df[portfolio] - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min()) * 100

            # Calculate longest new high days
            rolling_max = df[portfolio].expanding().max()
            new_highs = df[portfolio] >= rolling_max
            longest_streak = 0
            current_streak = 0
            for is_new_high in new_highs:
                if is_new_high:
                    current_streak += 1
                    longest_streak = max(longest_streak, current_streak)
                else:
                    current_streak = 0

            # Add results to DataFrame
            result_df = pd.concat([result_df, pd.DataFrame({
                'portfolio_name': [portfolio],
                'annual_return': [annual_return],
                'regression_annual_return': [regression_annual_return],
                'max_drawdown': [-max_drawdown]
            })], ignore_index=True)

        # 对除了portfolio_name以外的列进行排序，最小值为0，最大值为len(df)
        numeric_columns = ['annual_return', 'regression_annual_return', 'max_drawdown']
        rank_columns = []
        for col in numeric_columns:
            if col in result_df.columns:
                # 使用rank方法进行排序，method='min'确保相同值获得相同排名
                result_df[col + '_rank'] = result_df[col].rank(method='min', ascending=True) - 1
                # 确保排名从0开始到len(df)-1
                result_df[col + '_rank'] = result_df[col + '_rank'].astype(int)
                rank_columns.append(col + '_rank')
        
        # 计算每行rank的平均值
        if rank_columns:
            result_df['rank_average'] = result_df[rank_columns].mean(axis=1)
        # 输出portfolio_name和rank_average
        output_df = result_df[['portfolio_name', 'rank_average']].copy()
        return output_df
    def backtesting_main(self):
        df_signal=self.raw_signal_withdraw()
        x_list=df_signal['x'].unique().tolist()
        df_final = None
        
        for base_index in [self.big_indexName ,self.small_indexName,'大小盘等权']:
            print(base_index)
            proportion=self.big_proportion if base_index==self.big_indexName else self.small_proportion if base_index==self.small_indexName else (1-self.big_proportion-self.small_proportion)
            
            # 为当前base_index计算df_nav
            df_nav=pd.DataFrame()
            n=1
            for x in x_list:
                slice_df_signal = df_signal[df_signal['x'] == x]
                df_x = self.signal_return_processing(slice_df_signal,base_index)
                if n == 1:
                    df_nav = df_x
                    n += 1
                else:
                    df_nav = df_nav.merge(df_x, on='valuation_date', how='left')
            
            # 计算当前base_index的df_output
            df_output = self.technical_signal_calculator(df_nav)
            # 将df_output按照proportion相乘
            df_output_weighted = df_output.copy()
            numeric_columns = [col for col in df_output.columns if col != 'portfolio_name']
            for col in numeric_columns:
                df_output_weighted[col] = df_output[col] * proportion
            # 累加到最终结果
            if df_final is None:
                df_final = df_output_weighted
            else:
                # 确保portfolio_name列对齐，然后相加数值列
                df_final = df_final.set_index('portfolio_name')
                df_output_weighted = df_output_weighted.set_index('portfolio_name')
                df_final = df_final.add(df_output_weighted, fill_value=0)
                df_final = df_final.reset_index()

        # 处理portfolio_name，按照最后一个_分割
        df_final['x'] = df_final['portfolio_name'].str.extract(r'_([^_]+)$')
        df_final['portfolio_name'] = df_final['portfolio_name'].str.replace(r'_[^_]+$', '', regex=True)
        
        # 重新排列列的顺序，将x列放在portfolio_name之后
        cols = df_final.columns.tolist()
        cols.remove('x')
        cols.insert(1, 'x')  # 将x列插入到第2个位置（portfolio_name之后）
        df_final = df_final[cols]
        return df_final

if __name__ == "__main__":
    factor_list=['StockEmotion']
    for factor_name in factor_list:
        fbm=factor_backtesting(factor_name,'2015-01-01',"2025-12-31",0.00006,'prod','L1','上证50','中证2000',None,None)
        fbm.backtesting_main()
        # fbm = L3factor_backtesting(factor_name, '2015-01-01', '2025-11-31', 0.00006, 'test', '上证50',
        #                            '中证2000', 0.15, 0.15)
        # fbm.backtesting_main()


