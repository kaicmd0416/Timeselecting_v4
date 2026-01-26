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
        self.running_date=self.start_date_processing()

        if self.start_date < self.running_date:
            self.start_date = self.running_date
        self.df_index_return = self.index_return_withdraw()
        self.valuation_date_list=gt.working_days_list(self.start_date,self.end_date)
    def start_date_processing(self):
        inputpath = self.inputpath_base
        inputpath = str(
            inputpath) + f" Where signal_name='{self.signal_name}'"
        df = gt.data_getting(inputpath, config_path)
        df.sort_values(by='valuation_date',inplace=True)
        running_date=df['valuation_date'].unique().tolist()[0]
        running_date=gt.strdate_transfer(running_date)
        return running_date
    def index_return_withdraw(self):
        df_return = gt.indexData_withdraw(None,self.running_date,self.end_date,['pct_chg'])
        df_return = gt.sql_to_timeseries(df_return)
        df_return = df_return[['valuation_date', self.big_indexCode, self.small_indexCode]]
        df_return.columns=['valuation_date',self.big_indexName,self.small_indexName]
        df_return[self.big_indexName] = df_return[self.big_indexName].astype(float)
        df_return[self.small_indexName] = df_return[self.small_indexName].astype(float)
        return df_return

    def raw_signal_withdraw(self):
        inputpath = self.inputpath_base
        inputpath = str(
            inputpath) + f" Where signal_name='{self.signal_name}' And valuation_date between '{self.running_date}' and '{self.end_date}'"
        df = gt.data_getting(inputpath, config_path)
        df=df[['valuation_date','final_signal','x']]
        return df
    def target_raw_signal_withdraw(self):
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
        # 确保df按日期排序
        df = df.sort_values(by='valuation_date').reset_index(drop=True)
        # 获取portfolio列（排除valuation_date）
        portfolio_cols = [col for col in df.columns if col != 'valuation_date']
        # 如果数据长度小于500，返回所有rank都为0的DataFrame
        if len(df) < 500:
            # 创建所有日期和portfolio的组合，rank_average都为0
            # 使用self.valuation_date_list作为时间列
            result_list = []
            for date in self.valuation_date_list:
                available_date=gt.last_workday_calculate(date)
                for portfolio in portfolio_cols:
                    result_list.append({
                        'valuation_date': available_date,
                        'portfolio_name': portfolio,
                        'rank_average': 0
                    })
            output_df = pd.DataFrame(result_list)
            # 将长格式转换为宽格式：每个portfolio作为列
            if len(output_df) > 0:
                output_df = output_df.pivot(index='valuation_date', columns='portfolio_name', values='rank_average')
                output_df = output_df.reset_index()
                output_df.columns.name = None  # 移除列名索引名称
            return output_df
        
        # 初始化结果列表（使用列表收集结果，最后一次性concat，避免循环中concat的性能问题）
        result_list = []
        # 遍历self.valuation_date_list中的每个日期
        for target_date in self.valuation_date_list:
            # 检查该日期是否在df中
            date=gt.last_workday_calculate(target_date)
            date_mask = df['valuation_date'] == date
            if not date_mask.any():
                # 如果日期不在df中，跳过
                continue
            
            # 获取该日期在df中的索引位置
            date_indices = df[date_mask].index.tolist()
            if len(date_indices) == 0:
                continue
            index = date_indices[0]  # 取第一个匹配的索引
            # 只有当索引大于等于500时才进行计算（确保有足够的历史数据）
            if index >= 500:
                current_date = date
                # 使用从开始到当前日期前一个工作日的数据
                df_window = df.iloc[:index+1]
                # 初始化当前日期的结果列表
                portfolio_results = []
                
                for portfolio in portfolio_cols:
                    # 计算年化收益率
                    nav0 = df_window[portfolio].iloc[0]
                    navt = df_window[portfolio].iloc[-1]
                    total_return = navt / nav0
                    t = len(df_window)  # 使用窗口内的交易日数量
                    if t > 0 and nav0 > 0:
                        annual_return = (total_return ** (365 / t) - 1) * 100  # 转换为年化率和百分比
                    else:
                        annual_return = 0
                    
                    # 使用 ln(navt) - ln(navo) = kt 计算回归年化收益率
                    if t > 0 and nav0 > 0 and navt > 0:
                        k = (np.log(navt) - np.log(nav0)) / t
                        regression_annual_return = k * 252 * 100  # 转换为年化率和百分比
                    else:
                        regression_annual_return = 0
                    
                    # 计算最大回撤
                    if len(df_window) > 0:
                        rolling_max = df_window[portfolio].expanding().max()
                        drawdowns = (df_window[portfolio] - rolling_max) / rolling_max
                        max_drawdown = abs(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0
                    else:
                        max_drawdown = 0
                    
                    # 添加结果到列表
                    portfolio_results.append({
                        'portfolio_name': portfolio,
                        'annual_return': annual_return,
                        'regression_annual_return': regression_annual_return,
                        'max_drawdown': -max_drawdown
                    })
                
                # 转换为DataFrame进行排名计算
                result_df = pd.DataFrame(portfolio_results)
                
                # 对除了portfolio_name以外的列进行排序，最小值为0，最大值为len(df_window)
                numeric_columns = ['annual_return', 'regression_annual_return', 'max_drawdown']
                rank_columns = []
                for col in numeric_columns:
                    if col in result_df.columns:
                        # 使用rank方法进行排序，method='min'确保相同值获得相同排名
                        result_df[col + '_rank'] = result_df[col].rank(method='min', ascending=True) - 1
                        # 确保排名从0开始到len(df_window)-1
                        result_df[col + '_rank'] = result_df[col + '_rank'].astype(int)
                        rank_columns.append(col + '_rank')
                
                # 计算每行rank的平均值
                if rank_columns:
                    result_df['rank_average'] = result_df[rank_columns].mean(axis=1)
                
                # 为当前日期添加结果到列表
                for _, row in result_df.iterrows():
                    result_list.append({
                        'valuation_date': current_date,
                        'portfolio_name': row['portfolio_name'],
                        'rank_average': row['rank_average']
                    })
        
        # 一次性创建DataFrame（避免循环中concat的性能问题）
        output_df = pd.DataFrame(result_list)
        # 将长格式转换为宽格式：每个portfolio作为列
        if len(output_df) > 0:
            output_df = output_df.pivot(index='valuation_date', columns='portfolio_name', values='rank_average')
            output_df = output_df.reset_index()
            output_df.columns.name = None  # 移除列名索引名称
        
        return output_df
    def backtesting_main(self):
        df_signal=self.raw_signal_withdraw()
        df_signal_target=self.target_raw_signal_withdraw()
        valuation_date_list=df_signal_target['valuation_date'].unique().tolist()
        x_list=df_signal['x'].unique().tolist()
        df_final = None
        for base_index in [self.big_indexName ,self.small_indexName,'大小盘等权']:
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
            df_output.set_index('valuation_date',inplace=True,drop=True)
            # 将df_output按照proportion相乘
            df_output_weighted = df_output.copy()
            df_output_weighted = df_output_weighted * proportion
            # 累加到最终结果
            if df_final is None:
                df_final = df_output_weighted
            else:
                # 确保索引和列都对齐后再相加
                df_final = df_final.add(df_output_weighted, fill_value=0)
        # 循环结束后，将索引重置为列
        if df_final is not None:
            df_final = df_final.reset_index()
        # 处理df_final：对于每一天，找到rank值最大的列名
        # 如果有相同的最大rank值，看前面N天哪个列名出现的最早，就用哪个
        if df_final is not None and len(df_final) > 0:
            # 获取所有portfolio列（排除valuation_date）
            portfolio_cols = [col for col in df_final.columns if col != 'valuation_date']
            
            # 从所有列名中提取x值
            x_values = []
            col_to_x = {}
            for col in portfolio_cols:
                x_match = pd.Series([col]).str.extract(r'_([\d.]+)$')
                # 检查是否成功提取到x值
                if not pd.isna(x_match.iloc[0, 0]):
                    x_val = float(x_match.iloc[0, 0])
                    x_values.append(x_val)
                    col_to_x[col] = x_val
            
            # 判断df_final中所有列的值是否都相同（不是x值，而是列的实际值）
            all_values_same = False
            if len(portfolio_cols) > 0 and len(df_final) > 0:
                # 获取第一行的所有值
                first_row_values = df_final[portfolio_cols].iloc[0].values
                # 检查所有行的所有列值是否都相同
                all_values_same = (df_final[portfolio_cols] == first_row_values).all().all()
            
            # 如果所有列的值都相同，找到最小的x对应的列名
            if all_values_same and len(x_values) > 0:
                min_x = min(x_values)
                min_x_column = [col for col, x in col_to_x.items() if x == min_x][0]
                # 直接使用最小的x对应的列名
                result_list = []
                for idx in range(len(df_final)):
                    current_date = df_final['valuation_date'].iloc[idx]
                    result_list.append({
                        'valuation_date': current_date,
                        'rank': min_x_column
                    })
            else:
                # 初始化结果DataFrame
                result_list = []
                
                for idx in range(len(df_final)):
                    current_date = df_final['valuation_date'].iloc[idx]
                    current_row = df_final.iloc[idx]
                    
                    # 获取当前行的所有portfolio值
                    portfolio_values = {col: current_row[col] for col in portfolio_cols if pd.notna(current_row[col])}
                    
                    if len(portfolio_values) > 0:
                        # 找到最大值
                        max_value = max(portfolio_values.values())
                        
                        # 找到所有等于最大值的列名
                        max_columns = [col for col, val in portfolio_values.items() if val == max_value]
                        
                        if len(max_columns) == 1:
                            # 只有一个最大值，直接使用
                            selected_column = max_columns[0]
                        else:
                            # 有多个相同的最大值，看前面N天哪个列名出现的最早
                            # 回溯历史，找到这些列名中最早达到最大值的
                            earliest_idx = len(df_final)  # 初始化为一个很大的值
                            selected_column = max_columns[0]  # 默认选择第一个
                            
                            for col in max_columns:
                                # 从当前行往前回溯，找到这个列名最早达到当前最大值的索引
                                col_earliest_idx = len(df_final)  # 记录这个列名最早达到最大值的索引
                                for prev_idx in range(idx, -1, -1):
                                    prev_row = df_final.iloc[prev_idx]
                                    if col in portfolio_cols and pd.notna(prev_row[col]):
                                        if prev_row[col] == max_value:
                                            # 更新这个列名最早达到最大值的索引
                                            col_earliest_idx = prev_idx
                                
                                # 比较这个列名的最早索引，选择最早的那个
                                if col_earliest_idx < earliest_idx:
                                    earliest_idx = col_earliest_idx
                                    selected_column = col
                        
                        result_list.append({
                            'valuation_date': current_date,
                            'rank': selected_column
                        })
            
            df_result = pd.DataFrame(result_list)
            
            # 提取rank列中下划线后面的数字作为best_x
            df_result['x'] = df_result['rank'].str.extract(r'_([\d.]+)$')
            # 只保留valuation_date和best_x两列
            df_result = df_result[['valuation_date', 'x']]
            df_result['valuation_date']=df_result['valuation_date'].apply(lambda x: gt.next_workday_calculate(x))
            df_output=pd.DataFrame()
            df_output['valuation_date']=valuation_date_list
            df_output=df_output.merge(df_result,on='valuation_date',how='left')
            df_output.fillna(method='bfill',inplace=True)
            df_output.fillna(0.5, inplace=True)
            return df_output

if __name__ == "__main__":
    fbm = L3factor_backtesting('TargetIndex_REVERSE', '2026-01-26','2026-01-26', 0.00006, 'prod', '上证50',
                               '中证2000', 0.15, 0.15)
    print(fbm.backtesting_main())
    # factor_list=['Monthly_Effect','Holiday_Effect']
    # for factor_name in factor_list:
    #     fbm=factor_backtesting(factor_name,'2015-01-01',"2026-01-21",0.00006,'test','L2','上证50','中证2000',None,None)
    #     fbm.backtesting_main()
        # fbm = L3factor_backtesting(factor_name, '2015-01-01', '2025-11-31', 0.00006, 'test', '上证50',
        #                            '中证2000', 0.15, 0.15)
        # fbm.backtesting_main()


