import os
import pandas as pd
import numpy as np
from scipy import stats

def get_best_x(signal_name):
    """获取最佳x值，使用三个指标的加权平均：
    1/3 annual_return + 1/3 regression_annual_return - 1/3 max_drawdown
    """
    backtest_path = r'D:\Signal_v2\signal_backtest\test'
    report_path = os.path.join(backtest_path, signal_name, '综合回测报告.xlsx')
    
    if not os.path.exists(report_path):
        print(f"Warning: {report_path} does not exist")
        return None
        
    df_report = pd.read_excel(report_path)
    
    # 从portfolio_name中提取x值
    df_report['x'] = df_report['portfolio_name'].str.split('_').str[-1].astype(float)
    
    # 计算综合得分
    # 对每个指标进行标准化处理
    annual_return_norm = (df_report['annual_return'] - df_report['annual_return'].min()) / (df_report['annual_return'].max() - df_report['annual_return'].min())
    regression_return_norm = (df_report['regression_annual_return'] - df_report['regression_annual_return'].min()) / (df_report['regression_annual_return'].max() - df_report['regression_annual_return'].min())
    # max_drawdown是反向指标，所以用1减去标准化值
    max_drawdown_norm = 1 - (df_report['max_drawdown'] - df_report['max_drawdown'].min()) / (df_report['max_drawdown'].max() - df_report['max_drawdown'].min())
    
    # 计算加权得分
    df_report['综合得分'] = (annual_return_norm + regression_return_norm + max_drawdown_norm) / 3
    
    # 选择得分最高的x值
    best_x = df_report.loc[df_report['综合得分'].idxmax(), 'x']
    return best_x

def get_signal_data(signal_name, x):
    """获取指定因子和x值的信号数据"""
    signal_path = r'D:\Signal_v2\signal_data\test'
    signal_folder = os.path.join(signal_path, signal_name)
    
    if not os.path.exists(signal_folder):
        print(f"Warning: {signal_folder} does not exist")
        return None
        
    all_signals = []
    for file in os.listdir(signal_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(signal_folder, file)
            df = pd.read_csv(file_path)
            # 筛选指定x值的信号
            df = df[df['x'] == x]
            if not df.empty:
                all_signals.append(df[['valuation_date', 'final_signal']])
    
    if not all_signals:
        return None
        
    # 合并所有信号数据
    df_combined = pd.concat(all_signals, ignore_index=True)
    df_combined['valuation_date'] = pd.to_datetime(df_combined['valuation_date'])
    df_combined = df_combined.sort_values('valuation_date')
    return df_combined

def correlation_analysis(signal_names):
    """进行相关性分析"""
    # 存储所有因子的信号数据
    signal_data = {}
    
    # 获取每个因子的最佳x值和信号数据
    for signal_name in signal_names:
        best_x = get_best_x(signal_name)
        if best_x is not None:
            df_signal = get_signal_data(signal_name, best_x)
            if df_signal is not None:
                signal_data[signal_name] = df_signal
    
    if not signal_data:
        print("No valid signal data found")
        return
    
    # 创建相关性矩阵
    dates = pd.date_range(
        start=min(df['valuation_date'].min() for df in signal_data.values()),
        end=max(df['valuation_date'].max() for df in signal_data.values()),
        freq='D'
    )
    
    # 创建完整的时间序列数据框
    df_corr = pd.DataFrame(index=dates)
    for signal_name, df in signal_data.items():
        df_corr[signal_name] = df.set_index('valuation_date')['final_signal']
    
    # 删除任何包含NaN的行
    df_corr = df_corr.dropna()
    
    if df_corr.empty:
        print("No overlapping dates found between signals")
        return
    
    # 计算相关性矩阵
    corr_matrix = df_corr.corr()
    
    # 输出结果
    output_path = os.path.join(os.path.dirname(__file__), 'correlation_results.xlsx')
    with pd.ExcelWriter(output_path) as writer:
        corr_matrix.to_excel(writer, sheet_name='Correlation Matrix')
        
        # 添加p值矩阵
        p_values = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
        for i in corr_matrix.index:
            for j in corr_matrix.columns:
                if i != j:
                    # 使用对齐后的数据计算相关性
                    x = df_corr[i].values
                    y = df_corr[j].values
                    corr, p_value = stats.pearsonr(x, y)
                    p_values.loc[i, j] = p_value
                else:
                    p_values.loc[i, j] = 1.0
        p_values.to_excel(writer, sheet_name='P Values')
        
        # 添加数据覆盖信息
        coverage_info = pd.DataFrame({
            'signal_name': list(signal_data.keys()),
            'start_date': [df['valuation_date'].min() for df in signal_data.values()],
            'end_date': [df['valuation_date'].max() for df in signal_data.values()],
            'data_points': [len(df) for df in signal_data.values()],
            'common_dates': len(df_corr)
        })
        coverage_info.to_excel(writer, sheet_name='Data Coverage')
    
    print(f"Correlation analysis results saved to {output_path}")
    print(f"Number of common dates used: {len(df_corr)}")
    return corr_matrix, p_values

if __name__ == "__main__":
    # 示例使用
    backtest_path = r'D:\Signal_v2\signal_backtest\test'
    signal_names = os.listdir(backtest_path)
    correlation_analysis(signal_names) 