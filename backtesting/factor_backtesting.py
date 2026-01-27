"""
因子回测模块 (factor_backtesting)

本模块负责对各级别信号进行回测分析，评估信号的历史表现。
包含两个核心类：
1. factor_backtesting: 通用回测类，适用于L0/L1/L2级别信号
2. L3factor_backtesting: L3专用回测类，包含x参数动态优化功能

回测指标说明:
    - 年化收益率: (期末净值/期初净值)^(365/交易日数) - 1
    - 回归年化收益率: 使用对数回归 ln(navt) - ln(nav0) = k*t, k*252即为回归年化
    - 最大回撤: 从历史峰值到低谷的最大跌幅
    - 胜率: 信号正确预测的比例

信号含义:
    - final_signal = 0: 买大盘（如沪深300/上证50）
    - final_signal = 1: 买小盘（如中证2000）
    - final_signal = 0.5: 中性（大小盘各配50%或跟踪基准）

作者: TimeSelecting Team
版本: v3.0
"""

# ==================== 标准库导入 ====================
import os
import sys
from datetime import datetime

# ==================== 第三方库导入 ====================
import pandas as pd
import numpy as np

# ==================== 自定义模块导入 ====================
# 添加全局工具函数路径
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
import global_setting.global_dic as glv
from backtesting.backtesting_tools import Back_testing_processing

# ==================== 全局配置 ====================
config_path = glv.get('config_path')  # 获取全局配置文件路径


class factor_backtesting:
    """
    通用因子回测类

    适用于L0/L1/L2级别信号的回测分析。
    计算信号的历史收益、胜率、最大回撤等指标，并输出回测报告。

    工作流程:
        1. 从数据库获取信号数据
        2. 获取指数收益率数据
        3. 计算组合收益（考虑交易成本）
        4. 分析信号胜率
        5. 输出回测报告（Excel + 图表）

    Attributes:
    -----------
    signal_name : str
        信号名称，L0为'final_signal'，L1/L2为具体因子名称
    start_date : str
        回测开始日期，格式为 'YYYY-MM-DD'
    end_date : str
        回测结束日期，格式为 'YYYY-MM-DD'
    cost : float
        单边交易成本，如 0.00006 表示万分之0.6
    mode : str
        运行模式: 'prod'（生产环境）或 'test'（测试环境）
    signal_type : str
        信号层级: 'L0'、'L1'、'L2' 或 'L3'
    big_indexName : str
        大盘代表指数名称，如 '沪深300'、'上证50'
    small_indexName : str
        小盘代表指数名称，如 '中证2000'
    base_indexName : str
        基准指数名称，如 None 则使用大小盘等权
    x : float, optional
        L3信号的x参数值（仅L3需要）
    """

    def __init__(self, signal_name, start_date, end_date, cost, mode, signal_type,
                 big_indexName, small_indexName, base_indexName, x=None):
        """
        初始化因子回测类

        Parameters:
        -----------
        signal_name : str
            信号名称
        start_date : str
            开始日期，格式为 'YYYY-MM-DD'
        end_date : str
            结束日期，格式为 'YYYY-MM-DD'
        cost : float
            单边交易成本（如 0.00006 = 万分之0.6）
        mode : str
            模式，'prod'（生产环境）或 'test'（测试环境）
        signal_type : str
            信号层级，'L0'、'L1'、'L2' 或 'L3'
        big_indexName : str
            大盘指数名称，如 '沪深300'、'上证50'
        small_indexName : str
            小盘指数名称，如 '中证2000'
        base_indexName : str
            基准指数名称，None 表示使用大小盘等权作为基准
        x : float, optional
            L3信号的x参数（默认为 None，仅L3需要）
        """
        # 保存基本参数
        self.signal_name = signal_name
        self.start_date = start_date
        self.end_date = end_date
        self.cost = cost
        self.mode = mode
        self.x = x
        self.signal_type = signal_type

        # 保存指数配置
        self.big_indexName = big_indexName
        self.small_indexName = small_indexName
        self.base_indexName = base_indexName

        # 将指数名称映射为代码（用于数据库查询）
        self.big_indexCode = gt.index_mapping(self.big_indexName, 'code')
        self.small_indexCode = gt.index_mapping(self.small_indexName, 'code')
        if self.base_indexName is not None:
            self.base_indexCode = gt.index_mapping(self.base_indexName, 'code')
        else:
            self.base_indexCode = None

        # 根据信号层级和模式获取对应的数据表路径
        # 例如: 'L1_signalData_prod' 或 'L2_signalData_test'
        self.inputpath_base = glv.get(str(signal_type) + '_signalData_' + str(mode))

        # 处理开始日期（确保数据存在）
        self.start_date = self.start_date_processing()

        # 预加载指数收益率数据
        self.df_index_return = self.index_return_withdraw()

    def sql_path_withdraw(self):
        """
        获取SQL配置文件路径

        返回项目中SQL配置文件的绝对路径，用于数据库连接。

        Returns:
        --------
        str
            SQL配置文件的绝对路径
        """
        workspace_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(workspace_path, 'config_project', 'timeselecting_sql.yaml')
        return config_path

    def start_date_processing(self):
        """
        处理回测开始日期

        检查信号数据的最早可用日期，如果用户指定的开始日期早于数据可用日期，
        则自动调整为数据可用的最早日期。

        Returns:
        --------
        str
            调整后的开始日期，格式为 'YYYY-MM-DD'

        说明:
            - L0信号不需要按signal_name筛选（只有一个最终信号）
            - L1/L2/L3信号需要按signal_name筛选对应的因子
        """
        inputpath = self.inputpath_base

        # L0不需要按signal_name筛选
        if self.signal_type != 'L0':
            inputpath = str(inputpath) + f" Where signal_name='{self.signal_name}'"

        # 查询信号数据
        df = gt.data_getting(inputpath, config_path)
        df.sort_values(by='valuation_date', inplace=True)

        # 获取数据中最早的日期
        running_date = df['valuation_date'].unique().tolist()[0]
        running_date = gt.strdate_transfer(running_date)

        # 如果用户指定的开始日期早于数据可用日期，自动调整
        if self.start_date < running_date:
            start_date = running_date
            print(self.start_date + '目前没有数据，已经自动调整到:' + str(running_date))
        else:
            start_date = self.start_date

        return start_date

    def index_return_withdraw(self):
        """
        获取指数收益率数据

        从数据库获取大盘指数、小盘指数（以及可选的基准指数）的日收益率数据。

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期
            - {big_indexName}: 大盘指数日收益率（%）
            - {small_indexName}: 小盘指数日收益率（%）
            - {base_indexName}: 基准指数日收益率（%，可选）

        说明:
            收益率单位为百分比，如 1.5 表示涨1.5%
        """
        # 获取所有指数的收益率数据
        df_return = gt.indexData_withdraw(None, self.start_date, self.end_date, ['pct_chg'])
        df_return = gt.sql_to_timeseries(df_return)

        # 根据是否有基准指数，选择不同的列
        if self.base_indexCode is not None:
            df_return = df_return[['valuation_date', self.big_indexCode, self.small_indexCode, self.base_indexCode]]
            df_return.columns = ['valuation_date', self.big_indexName, self.small_indexName, self.base_indexName]
            df_return[self.base_indexName] = df_return[self.base_indexName].astype(float)
        else:
            df_return = df_return[['valuation_date', self.big_indexCode, self.small_indexCode]]
            df_return.columns = ['valuation_date', self.big_indexName, self.small_indexName]

        # 确保收益率列为float类型
        df_return[self.big_indexName] = df_return[self.big_indexName].astype(float)
        df_return[self.small_indexName] = df_return[self.small_indexName].astype(float)

        return df_return

    def raw_signal_withdraw(self):
        """
        从数据库获取原始信号数据

        根据信号层级、名称和日期范围，查询对应的信号数据。

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期
            - final_signal: 信号值（0/0.5/1）

        SQL查询逻辑:
            - L0: 直接按日期查询（不需要signal_name筛选）
            - L1/L2: 按signal_name和日期查询
            - L3: 还需要额外按x参数筛选
        """
        # 根据信号层级构建不同的SQL查询
        if self.signal_type != 'L0':
            sql = self.inputpath_base + f" Where signal_name='{self.signal_name}' And valuation_date BETWEEN '{self.start_date}' AND '{self.end_date}'"
        else:
            sql = self.inputpath_base + f" Where valuation_date BETWEEN '{self.start_date}' AND '{self.end_date}'"

        inputpath_config = glv.get('config_path')
        df = gt.data_getting(sql, inputpath_config)

        # L3信号需要按x参数筛选
        if self.signal_type == 'L3':
            df = df[df['x'] == self.x]

        # 只保留需要的列
        df = df[['valuation_date', 'final_signal']]
        df.sort_values('valuation_date', inplace=True)

        return df

    def probability_processing(self, df_signal):
        """
        计算信号胜率

        统计信号为0（买大盘）和信号为1（买小盘）的预测准确率。
        胜率 = 正确预测的次数 / 该信号出现的总次数

        Parameters:
        -----------
        df_signal : pd.DataFrame
            包含 'valuation_date' 和 'final_signal' 列的信号数据

        Returns:
        --------
        pd.DataFrame
            包含两行的胜率统计：
            - 第一行: 预测正确的概率
            - 第二行: 预测错误的概率
            列名为大盘指数名和小盘指数名

        判断逻辑:
            - target = 0: 当日大盘跑赢小盘（大盘收益 > 小盘收益）
            - target = 1: 当日小盘跑赢大盘
            - 如果 final_signal == target，则预测正确
        """
        # 合并指数收益率数据
        df_index = self.index_return_withdraw()
        df_signal = df_signal.merge(df_index, on='valuation_date', how='left')
        df_final = pd.DataFrame()

        # 计算目标标签：大盘-小盘 > 0 则 target=0（应买大盘）
        df_signal['target'] = df_signal[self.big_indexName] - df_signal[self.small_indexName]
        df_signal.loc[df_signal['target'] > 0, ['target']] = 0  # 大盘跑赢
        df_signal.loc[df_signal['target'] < 0, ['target']] = 1  # 小盘跑赢

        # 使用下一日的结果评判当日信号（信号T日发出，T+1日验证）
        df_signal['target'] = df_signal['target'].shift(-1)
        df_signal.dropna(inplace=True)

        # 统计各类信号的数量
        number_0 = len(df_signal[df_signal['final_signal'] == 0])  # 信号为0的次数
        number_1 = len(df_signal[df_signal['final_signal'] == 1])  # 信号为1的次数

        # 统计正确预测的次数
        number_0_correct = len(df_signal[(df_signal['final_signal'] == 0) & (df_signal['target'] == 0)])
        number_1_correct = len(df_signal[(df_signal['final_signal'] == 1) & (df_signal['target'] == 1)])

        # 避免除零错误
        if number_0 == 0:
            number_0 = 1
        if number_1 == 0:
            number_1 = 1

        # 计算胜率
        pb_0_correct = number_0_correct / number_0  # 买大盘信号的胜率
        pb_0_wrong = 1 - pb_0_correct               # 买大盘信号的错误率
        pb_1_correct = number_1_correct / number_1  # 买小盘信号的胜率
        pb_1_wrong = 1 - pb_1_correct               # 买小盘信号的错误率

        # 组装结果DataFrame
        df_final[self.big_indexName] = [pb_0_correct, pb_0_wrong]
        df_final[self.small_indexName] = [pb_1_correct, pb_1_wrong]

        return df_final

    def signal_return_processing(self, df_signal, index_name):
        """
        计算组合收益（考虑交易成本）

        根据信号将资金配置到大盘或小盘，计算每日收益并扣除换仓成本。

        Parameters:
        -----------
        df_signal : pd.DataFrame
            包含 'valuation_date' 和 'final_signal' 列的信号数据
        index_name : str
            用于处理中性信号（0.5）的参考指数名称：
            - big_indexName: 中性时配置大盘
            - small_indexName: 中性时配置小盘
            - '大小盘等权': 中性时配置等权或基准

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期
            - portfolio: 组合日收益（扣除交易成本后）
            - index: 对比指数的日收益

        收益计算逻辑:
            - signal=0 → 收益 = 大盘收益
            - signal=1 → 收益 = 小盘收益
            - signal=0.5 → 收益取决于 index_name 参数

        交易成本计算:
            - 换仓比例 = |当日信号 - 前日信号| * 2
            - 交易成本 = 换仓比例 * cost
            - 例如: 从 signal=0 换到 signal=1，换仓比例=2（全部卖出再全部买入）
        """
        # 获取指数收益率数据
        df_index = self.index_return_withdraw()

        # 计算基准收益（等权或指定基准）
        if self.base_indexName is None:
            df_index['大小盘等权'] = 0.5 * df_index[self.big_indexName] + 0.5 * df_index[self.small_indexName]
        else:
            df_index['大小盘等权'] = df_index[self.base_indexName]

        # 合并信号和收益数据
        df_signal = df_index.merge(df_signal, on='valuation_date', how='left')
        df_signal.dropna(inplace=True)

        # 根据信号分配收益
        df_signal['signal_return'] = 0

        # signal=0: 买大盘
        df_signal.loc[df_signal['final_signal'] == 0, ['signal_return']] = \
            df_signal.loc[df_signal['final_signal'] == 0][self.big_indexName].tolist()

        # signal=1: 买小盘
        df_signal.loc[df_signal['final_signal'] == 1, ['signal_return']] = \
            df_signal.loc[df_signal['final_signal'] == 1][self.small_indexName].tolist()

        # signal=0.5: 中性信号，根据 index_name 决定配置
        if index_name == self.big_indexName:
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5][self.big_indexName].tolist()
        elif index_name == self.small_indexName:
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5][self.small_indexName].tolist()
        else:
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5]['大小盘等权'].tolist()

        # 计算换仓比例（信号变化量的绝对值 * 2）
        df_signal['turn_over'] = df_signal['final_signal'] - df_signal['final_signal'].shift(1)
        df_signal['turn_over'] = abs(df_signal['turn_over']) * 2

        # 填充缺失值（首日没有前日数据）
        df_signal.fillna(method='ffill', inplace=True)
        df_signal.fillna(method='bfill', inplace=True)

        # 计算交易成本
        df_signal['turn_over'] = df_signal['turn_over'] * self.cost

        # 计算组合净收益 = 毛收益 - 交易成本
        df_signal['portfolio'] = df_signal['signal_return'].astype(float) - df_signal['turn_over']

        # 整理输出格式
        df_signal = df_signal[['valuation_date', 'portfolio', index_name]]
        df_signal.rename(columns={index_name: 'index'}, inplace=True)

        return df_signal

    def backtesting_main_sql(self):
        """
        回测主函数（SQL存储版本）

        执行回测并将结果存储到数据库。

        数据库表结构:
            - valuation_date: 日期
            - portfolio: 组合日收益
            - benchmark: 基准日收益
            - excess_return: 超额收益（组合 - 基准）
            - signal_name: 信号名称
            - update_time: 更新时间
        """
        # 获取SQL配置路径
        inputpath_sql = self.sql_path_withdraw()

        # 初始化数据库保存器
        sm = gt.sqlSaving_main(inputpath_sql, str(self.signal_type) + '_signal_' + str(self.mode) + '_backtest')

        # 获取信号数据
        df_signal = self.raw_signal_withdraw()

        # 计算组合收益（以等权为基准）
        df_portfolio = self.signal_return_processing(df_signal, '大小盘等权')
        df_portfolio.columns = ['valuation_date', 'portfolio', 'benchmark']

        # 计算超额收益
        df_portfolio['excess_return'] = df_portfolio['portfolio'] - df_portfolio['benchmark']
        df_portfolio['signal_name'] = self.signal_name
        df_portfolio['update_time'] = datetime.now()

        # 保存到数据库
        if len(df_portfolio) > 0:
            sm.df_to_sql(df_portfolio)

    def backtesting_main(self):
        """
        回测主函数（文件输出版本）

        执行完整的回测分析流程：
        1. 获取信号数据
        2. 计算胜率并输出
        3. 分别计算以大盘、小盘、等权为基准的组合表现
        4. 生成回测报告和图表

        Returns:
        --------
        str
            回测结果输出目录路径

        输出文件:
            - positive_negative_probabilities.xlsx: 胜率统计
            - 各指数对比的回测报告和图表
        """
        # 初始化回测处理器
        bp = Back_testing_processing(self.df_index_return, self.big_indexName,
                                     self.small_indexName, self.base_indexName)

        # 构建输出路径：backtest_output/{mode}/{signal_type}/{signal_name}/
        outputpath = glv.get('backtest_output')
        outputpath = os.path.join(outputpath, self.mode)
        outputpath = os.path.join(outputpath, self.signal_type)
        outputpath = os.path.join(outputpath, self.signal_name)

        # 获取信号数据
        df_signal = self.raw_signal_withdraw()

        # 计算并保存胜率
        df_prob = self.probability_processing(df_signal)
        outputpath_prob = os.path.join(outputpath, 'positive_negative_probabilities.xlsx')
        gt.folder_creator2(outputpath)  # 创建输出目录
        df_prob.to_excel(outputpath_prob, index=False)

        # 分别以大盘、小盘、等权为基准进行回测
        for index_name in [self.big_indexName, self.small_indexName, '大小盘等权']:
            if index_name == '大小盘等权':
                index_type = 'combine'  # 等权基准
            else:
                index_type = 'single'   # 单一指数基准

            # 计算组合收益
            df_portfolio = self.signal_return_processing(df_signal, index_name)

            # 生成回测报告
            bp.back_testing_history(df_portfolio, outputpath, index_type, index_name, self.signal_name)

        return outputpath


class L3factor_backtesting:
    """
    L3因子回测类（含x参数优化）

    专门用于L3级别信号的回测分析，核心功能是动态选择最优的x参数。
    x参数通常代表因子计算中的回看窗口、阈值等可调参数。

    优化逻辑:
        1. 对每个x值分别计算历史表现
        2. 使用年化收益率、回归年化收益率、最大回撤三个指标进行综合排名
        3. 按大盘、小盘、等权三个维度加权计算最终得分
        4. 选择综合得分最高的x作为当日最优参数
        5. 将最优x延迟一天使用（防止前视偏差）

    防止前视偏差机制:
        - technical_signal_calculator 使用历史数据计算各x的表现
        - 最终选出的best_x通过 gt.next_workday_calculate() 延迟一天
        - 确保T日选出的x用于T+1日的信号生成

    Attributes:
    -----------
    signal_name : str
        L3因子名称
    start_date : str
        回测开始日期
    end_date : str
        回测结束日期
    cost : float
        单边交易成本
    mode : str
        运行模式: 'prod' 或 'test'
    big_indexName : str
        大盘代表指数名称
    small_indexName : str
        小盘代表指数名称
    big_proportion : float
        大盘维度的权重（用于综合评分）
    small_proportion : float
        小盘维度的权重
    """

    def __init__(self, signal_name, start_date, end_date, cost, mode,
                 big_indexName, small_indexName, big_proportion, small_proportion):
        """
        初始化L3因子回测类

        Parameters:
        -----------
        signal_name : str
            L3因子名称，如 'RSRS'、'MACD_Hist' 等
        start_date : str
            开始日期，格式为 'YYYY-MM-DD'
        end_date : str
            结束日期，格式为 'YYYY-MM-DD'
        cost : float
            单边交易成本
        mode : str
            模式，'prod'（生产环境）或 'test'（测试环境）
        big_indexName : str
            大盘指数名称，如 '上证50'
        small_indexName : str
            小盘指数名称，如 '中证2000'
        big_proportion : float
            大盘维度权重，如 0.15 表示15%
        small_proportion : float
            小盘维度权重，如 0.15 表示15%
            等权维度权重 = 1 - big_proportion - small_proportion
        """
        # 保存基本参数
        self.signal_name = signal_name
        self.start_date = start_date
        self.end_date = end_date
        self.cost = cost
        self.mode = mode

        # 保存指数配置
        self.big_indexName = big_indexName
        self.small_indexName = small_indexName
        self.big_proportion = big_proportion
        self.small_proportion = small_proportion

        # 将指数名称映射为代码
        self.big_indexCode = gt.index_mapping(self.big_indexName, 'code')
        self.small_indexCode = gt.index_mapping(self.small_indexName, 'code')

        # 根据模式获取L3信号数据表路径
        if self.mode == 'prod':
            self.inputpath_base = glv.get('L3_signalData_prod')
        else:
            self.inputpath_base = glv.get('L3_signalData_test')

        # 获取数据可用的最早日期
        self.running_date = self.start_date_processing()

        # 如果开始日期早于数据可用日期，自动调整
        if self.start_date < self.running_date:
            self.start_date = self.running_date

        # 预加载指数收益率数据
        self.df_index_return = self.index_return_withdraw()

        # 生成回测日期列表
        self.valuation_date_list = gt.working_days_list(self.start_date, self.end_date)

    def start_date_processing(self):
        """
        处理回测开始日期

        获取该L3因子在数据库中最早的可用日期。

        Returns:
        --------
        str
            数据可用的最早日期，格式为 'YYYY-MM-DD'
        """
        inputpath = self.inputpath_base
        inputpath = str(inputpath) + f" Where signal_name='{self.signal_name}'"
        df = gt.data_getting(inputpath, config_path)
        df.sort_values(by='valuation_date', inplace=True)
        running_date = df['valuation_date'].unique().tolist()[0]
        running_date = gt.strdate_transfer(running_date)
        return running_date

    def index_return_withdraw(self):
        """
        获取指数收益率数据

        从数据库获取大盘和小盘指数的日收益率数据。

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期
            - {big_indexName}: 大盘指数日收益率（%）
            - {small_indexName}: 小盘指数日收益率（%）
        """
        df_return = gt.indexData_withdraw(None, self.running_date, self.end_date, ['pct_chg'])
        df_return = gt.sql_to_timeseries(df_return)
        df_return = df_return[['valuation_date', self.big_indexCode, self.small_indexCode]]
        df_return.columns = ['valuation_date', self.big_indexName, self.small_indexName]
        df_return[self.big_indexName] = df_return[self.big_indexName].astype(float)
        df_return[self.small_indexName] = df_return[self.small_indexName].astype(float)
        return df_return

    def raw_signal_withdraw(self):
        """
        获取原始L3信号数据（全部x参数）

        从数据可用的最早日期开始获取所有x参数的信号数据，
        用于计算历史表现和选择最优x。

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期
            - final_signal: 信号值（0/0.5/1）
            - x: 参数值
        """
        inputpath = self.inputpath_base
        inputpath = str(inputpath) + f" Where signal_name='{self.signal_name}' And valuation_date between '{self.running_date}' and '{self.end_date}'"
        df = gt.data_getting(inputpath, config_path)
        df = df[['valuation_date', 'final_signal', 'x']]
        return df

    def target_raw_signal_withdraw(self):
        """
        获取目标日期范围的L3信号数据

        从用户指定的开始日期获取信号数据，用于确定输出的日期范围。

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期
            - final_signal: 信号值
            - x: 参数值
        """
        inputpath = self.inputpath_base
        inputpath = str(inputpath) + f" Where signal_name='{self.signal_name}' And valuation_date between '{self.start_date}' and '{self.end_date}'"
        df = gt.data_getting(inputpath, config_path)
        df = df[['valuation_date', 'final_signal', 'x']]
        return df

    def signal_return_processing(self, df_signal, index_name):
        """
        计算单个x参数的组合超额收益

        计算使用特定x参数时的累计超额收益曲线。

        Parameters:
        -----------
        df_signal : pd.DataFrame
            单个x参数的信号数据
        index_name : str
            基准指数名称（大盘/小盘/等权）

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期
            - {signal_name}_{x}: 累计超额收益净值

        说明:
            超额收益 = 组合收益 - 基准收益
            净值曲线 = (1 + 超额收益).cumprod()
        """
        # 获取当前x值
        x = str(df_signal['x'].unique().tolist()[0])

        # 复制指数收益数据
        df_index = self.df_index_return.copy()
        df_index['大小盘等权'] = 0.5 * df_index[self.big_indexName] + 0.5 * df_index[self.small_indexName]

        # 合并信号和指数数据
        df_signal = df_index.merge(df_signal, on='valuation_date', how='left')
        df_signal.dropna(inplace=True)

        # 根据信号分配收益
        df_signal['signal_return'] = 0
        df_signal.loc[df_signal['final_signal'] == 0, ['signal_return']] = \
            df_signal.loc[df_signal['final_signal'] == 0][self.big_indexName].tolist()
        df_signal.loc[df_signal['final_signal'] == 1, ['signal_return']] = \
            df_signal.loc[df_signal['final_signal'] == 1][self.small_indexName].tolist()

        # 处理中性信号
        if index_name == self.big_indexName:
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5][self.big_indexName].tolist()
        elif index_name == self.small_indexName:
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5][self.small_indexName].tolist()
        else:
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5]['大小盘等权'].tolist()

        # 计算换仓成本
        df_signal['turn_over'] = df_signal['final_signal'] - df_signal['final_signal'].shift(1)
        df_signal['turn_over'] = abs(df_signal['turn_over']) * 2
        df_signal.fillna(method='ffill', inplace=True)
        df_signal.fillna(method='bfill', inplace=True)
        df_signal['turn_over'] = df_signal['turn_over'] * self.cost

        # 计算组合净收益
        df_signal['portfolio'] = df_signal['signal_return'].astype(float) - df_signal['turn_over']
        df_signal = df_signal[['valuation_date', 'portfolio', index_name]]
        df_signal.rename(columns={index_name: 'index'}, inplace=True)

        # 计算超额收益和累计净值
        df_signal['excess_return'] = df_signal['portfolio'] - df_signal['index']
        df_signal = df_signal[['valuation_date', 'excess_return']]
        df_signal[self.signal_name + '_' + x] = (1 + df_signal['excess_return']).cumprod()
        df_signal = df_signal[['valuation_date', self.signal_name + '_' + x]]

        return df_signal

    def technical_signal_calculator(self, df):
        """
        计算各x参数的综合评分

        对每个x参数计算三个指标（年化收益率、回归年化收益率、最大回撤），
        然后进行排名，计算平均排名作为综合评分。

        Parameters:
        -----------
        df : pd.DataFrame
            包含各x参数累计净值曲线的DataFrame
            列格式: valuation_date, {signal_name}_{x1}, {signal_name}_{x2}, ...

        Returns:
        --------
        pd.DataFrame
            包含每日各x参数综合评分的DataFrame
            列格式: valuation_date, {signal_name}_{x1}, {signal_name}_{x2}, ...
            值为综合排名分数（越高越好）

        计算逻辑:
            1. 年化收益率 = (NAV_t / NAV_0)^(365/t) - 1
            2. 回归年化收益率 = k * 252，其中 k = (ln(NAV_t) - ln(NAV_0)) / t
            3. 最大回撤 = max((峰值 - 当前值) / 峰值)
            4. 综合评分 = mean(年化收益排名, 回归年化排名, 回撤排名)

        防止前视偏差:
            - 每个日期只使用该日期之前的历史数据
            - 使用 gt.last_workday_calculate() 确保使用前一个工作日的数据
            - 需要至少500个交易日的数据才开始计算（约2年）
        """
        # 确保df按日期排序
        df = df.sort_values(by='valuation_date').reset_index(drop=True)

        # 获取portfolio列（排除valuation_date）
        portfolio_cols = [col for col in df.columns if col != 'valuation_date']

        # 如果数据长度小于500，返回所有rank都为0的DataFrame
        # 这确保了有足够的历史数据才进行优化
        if len(df) < 500:
            result_list = []
            for date in self.valuation_date_list:
                available_date = gt.last_workday_calculate(date)
                for portfolio in portfolio_cols:
                    result_list.append({
                        'valuation_date': available_date,
                        'portfolio_name': portfolio,
                        'rank_average': 0
                    })
            output_df = pd.DataFrame(result_list)

            # 将长格式转换为宽格式
            if len(output_df) > 0:
                output_df = output_df.pivot(index='valuation_date', columns='portfolio_name', values='rank_average')
                output_df = output_df.reset_index()
                output_df.columns.name = None
            return output_df

        # 初始化结果列表
        result_list = []

        # 遍历每个目标日期
        for target_date in self.valuation_date_list:
            # 使用前一个工作日的数据（防止前视偏差）
            date = gt.last_workday_calculate(target_date)
            date_mask = df['valuation_date'] == date

            if not date_mask.any():
                continue

            # 获取该日期在df中的索引位置
            date_indices = df[date_mask].index.tolist()
            if len(date_indices) == 0:
                continue
            index = date_indices[0]

            # 只有当索引大于等于500时才进行计算（确保有足够的历史数据）
            if index >= 500:
                current_date = date

                # 使用从开始到当前日期的数据（不包含未来数据）
                df_window = df.iloc[:index + 1]

                # 初始化当前日期的结果列表
                portfolio_results = []

                for portfolio in portfolio_cols:
                    # ========== 计算年化收益率 ==========
                    nav0 = df_window[portfolio].iloc[0]   # 期初净值
                    navt = df_window[portfolio].iloc[-1]  # 期末净值
                    total_return = navt / nav0            # 总收益率
                    t = len(df_window)                    # 交易日数量

                    if t > 0 and nav0 > 0:
                        # 年化收益率 = (总收益)^(365/交易日数) - 1
                        annual_return = (total_return ** (365 / t) - 1) * 100
                    else:
                        annual_return = 0

                    # ========== 计算回归年化收益率 ==========
                    # 使用对数回归: ln(navt) - ln(nav0) = k * t
                    if t > 0 and nav0 > 0 and navt > 0:
                        k = (np.log(navt) - np.log(nav0)) / t
                        regression_annual_return = k * 252 * 100  # 转换为年化（252交易日）
                    else:
                        regression_annual_return = 0

                    # ========== 计算最大回撤 ==========
                    if len(df_window) > 0:
                        rolling_max = df_window[portfolio].expanding().max()  # 历史峰值
                        drawdowns = (df_window[portfolio] - rolling_max) / rolling_max  # 回撤率
                        max_drawdown = abs(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0
                    else:
                        max_drawdown = 0

                    # 添加结果到列表（回撤取负值使其排名方向一致）
                    portfolio_results.append({
                        'portfolio_name': portfolio,
                        'annual_return': annual_return,
                        'regression_annual_return': regression_annual_return,
                        'max_drawdown': -max_drawdown  # 负值使得回撤小的排名靠后（更好）
                    })

                # 转换为DataFrame进行排名计算
                result_df = pd.DataFrame(portfolio_results)

                # 对三个指标进行排名
                numeric_columns = ['annual_return', 'regression_annual_return', 'max_drawdown']
                rank_columns = []
                for col in numeric_columns:
                    if col in result_df.columns:
                        # 排名从0开始，越大越好
                        result_df[col + '_rank'] = result_df[col].rank(method='min', ascending=True) - 1
                        result_df[col + '_rank'] = result_df[col + '_rank'].astype(int)
                        rank_columns.append(col + '_rank')

                # 计算综合排名（三个指标的平均排名）
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
            output_df.columns.name = None

        return output_df

    def backtesting_main(self):
        """
        L3回测主函数

        执行完整的L3信号优化和回测流程：
        1. 获取所有x参数的信号数据
        2. 计算各x在大盘、小盘、等权三个维度的历史表现
        3. 按权重加权计算综合评分
        4. 选择每日最优的x参数
        5. 将最优x延迟一天输出（防止前视偏差）

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期
            - x: 该日应使用的最优x参数

        处理逻辑:
            1. 对每个基准指数分别计算各x的评分
            2. df_final = big_score * big_proportion + small_score * small_proportion + equal_score * (1-big-small)
            3. 每日选择综合评分最高的x
            4. 如果多个x评分相同，选择历史上最早达到该评分的x（稳定性考虑）
            5. 通过 gt.next_workday_calculate() 延迟一天

        重要：防止前视偏差
            第497行的 next_workday_calculate 确保了：
            - T日计算出的最优x在T+1日才能使用
            - 避免使用当日信息做当日决策
        """
        # 获取所有x参数的信号数据
        df_signal = self.raw_signal_withdraw()
        df_signal_target = self.target_raw_signal_withdraw()
        valuation_date_list = df_signal_target['valuation_date'].unique().tolist()
        x_list = df_signal['x'].unique().tolist()

        df_final = None

        # 分别计算大盘、小盘、等权三个维度的评分
        for base_index in [self.big_indexName, self.small_indexName, '大小盘等权']:
            # 确定该维度的权重
            proportion = self.big_proportion if base_index == self.big_indexName \
                else self.small_proportion if base_index == self.small_indexName \
                else (1 - self.big_proportion - self.small_proportion)

            # 计算各x参数在该维度的累计净值曲线
            df_nav = pd.DataFrame()
            n = 1
            for x in x_list:
                slice_df_signal = df_signal[df_signal['x'] == x]
                df_x = self.signal_return_processing(slice_df_signal, base_index)
                if n == 1:
                    df_nav = df_x
                    n += 1
                else:
                    df_nav = df_nav.merge(df_x, on='valuation_date', how='left')

            # 计算该维度各x的综合评分
            df_output = self.technical_signal_calculator(df_nav)
            df_output.set_index('valuation_date', inplace=True, drop=True)

            # 按权重加权
            df_output_weighted = df_output.copy()
            df_output_weighted = df_output_weighted * proportion

            # 累加到最终结果
            if df_final is None:
                df_final = df_output_weighted
            else:
                df_final = df_final.add(df_output_weighted, fill_value=0)

        # 将索引重置为列
        if df_final is not None:
            df_final = df_final.reset_index()

        # 处理df_final：对于每一天，找到rank值最大的列名
        if df_final is not None and len(df_final) > 0:
            portfolio_cols = [col for col in df_final.columns if col != 'valuation_date']

            # 从列名中提取x值
            x_values = []
            col_to_x = {}
            for col in portfolio_cols:
                x_match = pd.Series([col]).str.extract(r'_([\d.]+)$')
                if not pd.isna(x_match.iloc[0, 0]):
                    x_val = float(x_match.iloc[0, 0])
                    x_values.append(x_val)
                    col_to_x[col] = x_val

            # 判断所有列的值是否都相同
            all_values_same = False
            if len(portfolio_cols) > 0 and len(df_final) > 0:
                first_row_values = df_final[portfolio_cols].iloc[0].values
                all_values_same = (df_final[portfolio_cols] == first_row_values).all().all()

            # 如果所有列的值都相同（无法区分），选择最小的x（保守策略）
            if all_values_same and len(x_values) > 0:
                min_x = min(x_values)
                min_x_column = [col for col, x in col_to_x.items() if x == min_x][0]
                result_list = []
                for idx in range(len(df_final)):
                    current_date = df_final['valuation_date'].iloc[idx]
                    result_list.append({
                        'valuation_date': current_date,
                        'rank': min_x_column
                    })
            else:
                # 正常逻辑：选择评分最高的x
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
                            # 有多个相同的最大值，选择历史上最早达到该评分的
                            # 这样可以保持参数的稳定性
                            earliest_idx = len(df_final)
                            selected_column = max_columns[0]

                            for col in max_columns:
                                col_earliest_idx = len(df_final)
                                for prev_idx in range(idx, -1, -1):
                                    prev_row = df_final.iloc[prev_idx]
                                    if col in portfolio_cols and pd.notna(prev_row[col]):
                                        if prev_row[col] == max_value:
                                            col_earliest_idx = prev_idx

                                if col_earliest_idx < earliest_idx:
                                    earliest_idx = col_earliest_idx
                                    selected_column = col

                        result_list.append({
                            'valuation_date': current_date,
                            'rank': selected_column
                        })

            df_result = pd.DataFrame(result_list)

            # 从列名中提取x值
            df_result['x'] = df_result['rank'].str.extract(r'_([\d.]+)$')
            df_result = df_result[['valuation_date', 'x']]

            # 【关键】将日期延迟一天，防止前视偏差
            # T日计算出的最优x，在T+1日才能使用
            df_result['valuation_date'] = df_result['valuation_date'].apply(lambda x: gt.next_workday_calculate(x))

            # 创建输出DataFrame
            df_output = pd.DataFrame()
            df_output['valuation_date'] = valuation_date_list
            df_output = df_output.merge(df_result, on='valuation_date', how='left')

            # 填充缺失值
            df_output.fillna(method='bfill', inplace=True)  # 向后填充
            df_output.fillna(0.5, inplace=True)             # 默认值

            return df_output


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    """
    测试入口

    可以分别测试 factor_backtesting 和 L3factor_backtesting 类。
    """
    # L3因子回测示例（注释状态）
    # fbm = L3factor_backtesting('SpecialFactor', '2015-01-01','2026-01-26', 0.00006, 'prod', '上证50',
    #                            '中证2000', 0.15, 0.15)
    # print(fbm.backtesting_main())

    # L1因子回测示例
    factor_list = ['SpecialFactor']
    for factor_name in factor_list:
        fbm = factor_backtesting(factor_name, '2015-01-01', "2026-01-26", 0.00006, 'prod', 'L1',
                                 '上证50', '中证2000', None, None)
        fbm.backtesting_main()

        # L3因子回测示例（注释状态）
        # fbm = L3factor_backtesting(factor_name, '2015-01-01', '2025-11-31', 0.00006, 'test', '上证50',
        #                            '中证2000', 0.15, 0.15)
        # fbm.backtesting_main()
