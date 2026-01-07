import os
import sys
import pandas as pd
import yaml
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
from datetime import datetime
import global_setting.global_dic as glv
from backtesting.factor_backtesting import factor_backtesting
config_path=glv.get('config_path')
pd.set_option('display.max_rows', None)
def best_x_compare(start_date,end_date):
    inputpath1=glv.get('L3_best_x_prod')
    inputpath2=glv.get('V3_L3_best_x_prod')
    inputpath1=str(inputpath1)+f" Where valuation_date between '{start_date}' and '{end_date}'"
    inputpath2 = str(inputpath2) + f" Where valuation_date between '{start_date}' and '{end_date}'"
    df_v4=gt.data_getting(inputpath1,config_path)
    df_v3=gt.data_getting(inputpath2,config_path)
    df_v3.rename(columns={'x':'x_v3'},inplace=True)
    df_v4=df_v4.merge(df_v3,on=['valuation_date','signal_name'],how='outer')
    df_v4.fillna(0,inplace=True)
    df_v4['difference']=df_v4['x']-df_v4['x_v3']
    df_difference=df_v4[df_v4['difference']!=0]
    
    # 读取signal_dictionary.yaml文件
    config_dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config_project', 'signal_dictionary.yaml')
    with open(config_dict_path, 'r', encoding='utf-8') as f:
        signal_dict = yaml.safe_load(f)
    
    # 创建L3_factor到L2_factor和L1_factor的映射
    # 通过L3_factor字段来匹配，而不是通过key（因为key和L3_factor可能有大小写差异）
    l3_to_l2 = {}
    l3_to_l1 = {}
    for key, value in signal_dict.items():
        if isinstance(value, dict):
            l3_factor = value.get('L3_factor')
            l2_factor = value.get('L2_factor')
            l1_factor = value.get('L1_factor')
            # 使用L3_factor作为key，如果没有L3_factor则使用YAML的key
            l3_key = l3_factor if l3_factor else key
            if l2_factor:
                l3_to_l2[l3_key] = l2_factor
            if l1_factor:
                l3_to_l1[l3_key] = l1_factor
    
    # 根据signal_name（L3_factor）添加L2和L1两列
    df_difference['L2'] = df_difference['signal_name'].map(l3_to_l2)
    df_difference['L1'] = df_difference['signal_name'].map(l3_to_l1)
    
    print(df_difference)
if __name__ == "__main__":
    best_x_compare('2026-01-08', '2026-01-08')