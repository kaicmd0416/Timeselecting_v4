import os
import pandas as pd
import os
import sys
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt

for i in range(0, 10):
    j = i / 10
    k = 0.1 + 0.1 * i
    quantile_lower_digit = 1 - k
    quantile_upper_digit = 1 - j
    quantile_lower_digit=round(quantile_lower_digit,1)
    quantile_upper_digit = round(quantile_upper_digit, 1)
    print(quantile_lower_digit,quantile_upper_digit)
# date_list=gt.working_days_list('2024-01-01','2026-01-05')
# for date in date_list:
#     date_yes=gt.last_workday_calculate(date)
#     for i in ['沪深300', '中证500', '中证1000', '中证A500']:
#         df1 = gt.index_weight_withdraw(i,  date)
#         df1 = df1[['code', 'weight']]
#         df2 = gt.index_weight_withdraw(i, date_yes)
#         df2 = df2[['code', 'weight']]
#         df2.columns = ['code', 'weight_yes']
#         df1 = df1.merge(df2, on='code', how='outer')
#         df1.fillna(0, inplace=True)
#         df1['difference'] = df1['weight'] - df1['weight_yes']
#         difference=abs(df1['difference']).sum()
#         if difference>0.005:
#            print(f"{i}在{date}的权重变化为{difference}")
