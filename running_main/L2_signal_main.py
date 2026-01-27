"""
L2级别信号生成模块 (L2_signal_main)

本模块负责生成L2级别的择时信号。L2信号是二级分类信号，
通过聚合同一L2分类下的所有L3信号得到，并动态选择最优的x参数。

信号体系层级结构:
    L0 (最终信号) ← 聚合所有L1信号
    L1 (一级因子) ← 聚合对应的L2信号
    L2 (二级因子) ← 聚合对应的L3信号  ← 【当前模块】
    L3 (三级因子) ← 原始因子数据计算

L2因子分类示例（以MacroLiquidity为例）:
    - Bond: 包含Bond_3Y、Bond_10Y等L3因子
    - Shibor: 包含Shibor_2W、Shibor_9M等L3因子
    - CreditSpread: 包含CreditSpread_5Y、CreditSpread_9M等L3因子

关键特性:
    - L2层级会进行best_x的动态选择
    - 通过L3factor_backtesting对不同x参数进行回测
    - 选择历史表现最优的x参数用于生成信号

信号含义:
    - final_signal = 0: 看多大盘（如上证50）
    - final_signal = 1: 看多小盘（如中证2000）
    - final_signal = 0.5: 中性（大小盘各配50%）

作者: TimeSelecting Team
版本: v3.0
"""

# ==================== 标准库导入 ====================
import os
import sys
from datetime import datetime

# ==================== 第三方库导入 ====================
import pandas as pd
import yaml

# ==================== 自定义模块导入 ====================
# 添加全局工具函数路径
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
import global_setting.global_dic as glv
from backtesting.factor_backtesting import L3factor_backtesting, factor_backtesting

# ==================== 全局配置 ====================
config_path = glv.get('config_path')  # 获取全局配置文件路径
pd.set_option('display.max_rows', None)  # 设置pandas显示所有行


class L2_signalConstruction:
    """
    L2级别信号生成类

    负责生成L2级别的择时信号，通过以下步骤实现：
    1. 获取该L2分类下所有L3因子的信号
    2. 对每个L3因子进行回测，选择最优的x参数
    3. 使用最优x参数获取L3信号
    4. 聚合所有L3信号生成L2信号

    关键概念 - x参数:
        x是信号生成的阈值参数，用于将连续的因子值转换为离散信号：
        - 当 final_signal > x 时，返回1（买小盘）
        - 当 final_signal < (1-x) 时，返回0（买大盘）
        - 其他情况返回0.5（中性）
        x的可选值: [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    Attributes:
    -----------
    signal_name : str
        L2因子名称，如 'Bond'、'Shibor'、'CreditSpread' 等
    start_date : str
        回测/生成信号的开始日期，格式为 'YYYY-MM-DD'
    end_date : str
        回测/生成信号的结束日期，格式为 'YYYY-MM-DD'
    cost : float
        交易成本，用于回测计算，如 0.00006（万分之0.6）
    mode : str
        运行模式: 'prod'（生产模式）或 'test'（测试模式）
    big_indexName : str
        大盘指数名称，如 '上证50'
    small_indexName : str
        小盘指数名称，如 '中证2000'
    big_proportion : float
        选择best_x时大盘指数的权重，如 0.15
    small_proportion : float
        选择best_x时小盘指数的权重，如 0.15
    inputpath_base : str
        L3信号数据的SQL查询基础路径
    """

    def __init__(self, signal_name, start_date, end_date, cost, mode,
                 big_indexName, small_indexName, big_proportion, small_proportion):
        """
        初始化L2信号构建类

        Parameters:
        -----------
        signal_name : str
            L2因子名称，如 'Bond'、'Shibor' 等
        start_date : str
            开始日期，格式为 'YYYY-MM-DD'
        end_date : str
            结束日期，格式为 'YYYY-MM-DD'
        cost : float
            交易成本（单边），如 0.00006
        mode : str
            模式，'prod'（生产模式）或 'test'（测试模式）
        big_indexName : str
            大盘指数名称，如 '上证50'
        small_indexName : str
            小盘指数名称，如 '中证2000'
        big_proportion : float
            选择best_x时大盘指数的权重（0-1之间）
        small_proportion : float
            选择best_x时小盘指数的权重（0-1之间）
        """
        # 保存日期范围和交易参数
        self.start_date = start_date
        self.end_date = end_date
        self.cost = cost

        # 保存指数配置
        self.big_indexName = big_indexName
        self.small_indexName = small_indexName
        self.big_proportion = big_proportion
        self.small_proportion = small_proportion

        # 保存基本参数
        self.signal_name = signal_name
        self.mode = mode

        # 根据模式选择对应的数据表路径
        if self.mode == 'prod':
            self.inputpath_base = glv.get('L3_signalData_prod')  # 生产环境L3信号表
        else:
            self.inputpath_base = glv.get('L3_signalData_test')  # 测试环境L3信号表

    def get_factor_info(self, factor_name, name=True):
        """
        根据因子名称查找相关信息

        根据参数name的值，可以实现两种查询：
        1. name=True: 根据L2因子名称，查找所有对应的L3因子
        2. name=False: 根据L3因子名称，查找对应的Best_x值

        Parameters:
        -----------
        factor_name : str
            当name=True时，输入L2因子名称（如'Shibor'、'Bond'等）
            当name=False时，输入L3因子名称（如'Shibor_2W'、'Bond_10Y'等）
        name : bool
            True: 输入L2因子名称，返回对应的L3因子名称列表
            False: 输入L3因子名称，返回对应的Best_x值

        Returns:
        --------
        list or float
            当name=True时，返回包含所有匹配的L3因子名称的列表
            当name=False时，返回对应的x值

        示例:
            get_factor_info('Shibor', name=True)
            → ['Shibor_2W', 'Shibor_9M']

            get_factor_info('Shibor_2W', name=False)
            → 0.65  # 假设Best_x配置为0.65
        """
        try:
            # 构建配置文件的绝对路径
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),  # 项目根目录
                'config_project',
                'signal_dictionary.yaml'
            )

            # 读取并解析YAML配置文件
            with open(config_path, 'r', encoding='utf-8') as file:
                signal_dict = yaml.safe_load(file)

            if name:
                # ==================== 查找L2因子对应的L3因子列表 ====================
                l3_factors = []
                for factor_key, factor_info in signal_dict.items():
                    if factor_info.get('L2_factor') == factor_name:
                        l3_factors.append(factor_info.get('L3_factor'))

                print(f"找到L2因子 '{factor_name}' 对应的L3因子: {l3_factors}")
                return l3_factors
            else:
                # ==================== 查找L3因子对应的Best_x值 ====================
                for factor_key, factor_info in signal_dict.items():
                    if factor_info.get('L3_factor') == factor_name:
                        x_value = factor_info.get('Best_x')
                        print(f"找到L3因子 '{factor_name}' 对应的x值: {x_value}")
                        return x_value

                print(f"未找到L3因子 '{factor_name}' 对应的x值")
                return None

        except FileNotFoundError:
            print(f"配置文件未找到: {config_path}")
            return [] if name else None
        except yaml.YAMLError as e:
            print(f"YAML文件解析错误: {e}")
            return [] if name else None
        except Exception as e:
            print(f"查找因子信息时出错: {e}")
            return [] if name else None

    def raw_signal_withdraw(self, signal_name, df_x):
        """
        从数据库获取指定L3因子的信号数据，并根据最优x筛选

        根据信号名称、日期范围和最优x参数，从数据库查询对应的L3信号数据。
        这个方法会将df_x（包含每天最优x值的DataFrame）与原始信号数据合并，
        确保每天使用的是当天对应的最优x参数下的信号。

        Parameters:
        -----------
        signal_name : str
            L3因子名称，如 'Shibor_2W'、'Bond_10Y' 等
        df_x : pd.DataFrame
            包含每日最优x值的DataFrame，列包括：
            - valuation_date: 日期
            - x: 当日最优x值

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - {signal_name}: 该因子在最优x下的信号值（0/0.5/1）

        实现逻辑:
            1. 从数据库查询该L3因子的所有信号数据（包含所有x值）
            2. 将df_x与信号数据按(valuation_date, x)进行合并
            3. 这样每天只保留该天最优x对应的信号
        """
        # 构建SQL查询语句
        inputpath = self.inputpath_base
        inputpath = str(inputpath) + \
            f" Where signal_name='{signal_name}' And valuation_date between '{self.start_date}' and '{self.end_date}'"

        # 执行查询
        df = gt.data_getting(inputpath, config_path)

        # ==================== 数据类型统一 ====================
        # 确保x列的类型一致（转为float），以便正确合并
        if 'x' in df_x.columns:
            df_x['x'] = df_x['x'].astype(float)
        if 'x' in df.columns:
            df['x'] = df['x'].astype(float)

        # ==================== 合并最优x与原始信号 ====================
        # 通过(valuation_date, x)合并，只保留每天最优x对应的信号
        df = df_x.merge(df, on=['valuation_date', 'x'], how='left')

        # 提取需要的列并重命名
        df = df[['valuation_date', 'final_signal']]
        df.columns = ['valuation_date', signal_name]

        return df

    def L2_construction_main(self):
        """
        L2信号构建主函数

        执行完整的L2信号生成流程：
        1. 获取该L2因子下所有L3因子列表
        2. 对每个L3因子进行回测，选择每天最优的x参数
        3. 使用最优x参数获取L3信号
        4. 聚合所有L3信号，计算平均值
        5. 根据平均值生成最终的0/0.5/1信号
        6. 保存到数据库

        数据库表:
            - L2信号: L2_signal_prod/test
            - 最优x记录: L3_bext_x_prod/test（用于追溯和分析）

        注意:
            best_x的选择是通过L3factor_backtesting类实现的，
            该类会对不同x参数下的信号进行历史回测，
            综合考虑年化收益率、最大回撤等指标选择最优x。
        """
        # ==================== 初始化数据库连接 ====================
        inputpath_sql = glv.get('sql_path')
        if self.mode == 'prod':
            # L2信号表，delete=True表示先删除旧数据
            sm = gt.sqlSaving_main(inputpath_sql, 'L2_signal_prod', delete=True)
            # L3最优x记录表，用于追溯每天选择的x值
            sm2 = gt.sqlSaving_main(inputpath_sql, 'L3_bext_x_prod')
        else:
            sm = gt.sqlSaving_main(inputpath_sql, 'L2_signal_test', delete=True)
            sm2 = gt.sqlSaving_main(inputpath_sql, 'L3_bext_x_test')

        # ==================== 遍历L3因子，获取最优x和信号 ====================
        n = 1
        df_final = pd.DataFrame()
        factor_name_list = self.get_factor_info(self.signal_name, True)  # 获取该L2下的所有L3因子

        for factor_name in factor_name_list:
            # ========== 步骤1: 对L3因子进行回测，选择最优x ==========
            L3fb = L3factor_backtesting(
                factor_name,           # L3因子名称
                self.start_date,       # 开始日期
                self.end_date,         # 结束日期
                self.cost,             # 交易成本
                self.mode,             # 运行模式
                self.big_indexName,    # 大盘指数
                self.small_indexName,  # 小盘指数
                self.big_proportion,   # 大盘权重
                self.small_proportion  # 小盘权重
            )
            # 返回DataFrame，包含每天的最优x
            df_x = L3fb.backtesting_main()

            # ========== 步骤2: 保存最优x记录到数据库 ==========
            df_x_sql = df_x.copy()
            df_x_sql['signal_name'] = factor_name  # 添加因子名称
            df_x_sql['update_time'] = datetime.now().replace(tzinfo=None)  # 添加更新时间
            sm2.df_to_sql(df_x_sql)

            # ========== 步骤3: 获取该L3因子在最优x下的信号 ==========
            df = self.raw_signal_withdraw(factor_name, df_x)

            # ========== 步骤4: 合并L3信号 ==========
            if n == 1:
                df_final = df  # 第一个因子直接赋值
                n += 1
            else:
                # 后续因子通过外连接合并
                df_final = df_final.merge(df, on='valuation_date', how='outer')

        # ==================== 信号处理 ====================
        df_final.fillna(0.5, inplace=True)  # 缺失值填充为中性信号
        df_final.set_index('valuation_date', inplace=True, drop=True)

        def x_processing(x):
            """
            将平均值转换为离散信号

            Parameters:
            -----------
            x : float
                L3信号的平均值

            Returns:
            --------
            float
                离散化后的信号：0（买大盘）、0.5（中性）、1（买小盘）
            """
            if x < 0.5:
                return 0      # 平均值偏向大盘 → 买大盘
            elif x == 0.5:
                return 0.5    # 平均值中性 → 保持中性
            else:
                return 1      # 平均值偏向小盘 → 买小盘

        # 计算所有L3信号的平均值
        signal_columns = [col for col in df_final.columns if col != 'valuation_date']
        df_final[self.signal_name] = df_final[signal_columns].mean(axis=1)

        # 将平均值转换为离散信号
        df_final[self.signal_name] = df_final[self.signal_name].apply(lambda x: x_processing(x))

        # ==================== 整理输出格式 ====================
        df_final.reset_index(inplace=True)
        df_final = df_final[['valuation_date', self.signal_name]]
        df_final['signal_name'] = self.signal_name  # 添加信号名称列
        df_final['update_time'] = datetime.now().replace(tzinfo=None)  # 添加更新时间戳
        df_final.rename(columns={self.signal_name: 'final_signal'}, inplace=True)

        # ==================== 保存到数据库 ====================
        sm.df_to_sql(df_final, 'signal_name', self.signal_name)

    def L2_backtest_main(self):
        """
        L2信号回测主函数

        先构建L2信号，然后对生成的信号进行回测分析。
        回测会计算信号的历史收益、最大回撤等指标。

        回测参数:
            - 大盘指数: 使用初始化时传入的big_indexName
            - 小盘指数: 使用初始化时传入的small_indexName
            - 交易成本: 0.00006（万分之0.6，单边）
        """
        # 先构建L2信号
        self.L2_construction_main()

        # 执行回测分析
        fb = factor_backtesting(
            self.signal_name,      # L2因子名称
            self.start_date,       # 开始日期
            self.end_date,         # 结束日期
            0.00006,               # 交易成本
            self.mode,             # 运行模式
            'L2',                  # 信号层级
            self.big_indexName,    # 大盘指数名称
            self.small_indexName,  # 小盘指数名称
            None,                  # 基准指数
            None                   # x参数（L2层级不需要）
        )
        fb.backtesting_main()


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    """
    批量运行L2信号生成和回测

    可选的L2因子列表示例:
        宏观流动性类: 'Bond', 'Shibor', 'CreditSpread', 'TermSpread', 'M1M2'
        指数量价类: 'TargetIndex_Technical', 'TargetIndex_Momentum', 'RelativeIndex_Std'
        资金流向类: 'NLBP_difference', 'LargeOrder_difference', 'ETF_Shares', 'USDX', 'USBond'
        市场情绪类: 'IndividualStock_Emotion', 'Future_difference', 'Bank_Momentum'
        特殊因子类: 'Seasonality_Effect', 'EventDriven', 'RRScore_difference', 'VP08Score_difference'
        期权因子类: 'OptionPCR', 'OptionIV', 'OptionIVMomentum', 'OptionOIMomentum', 等
    """
    # 要处理的L2因子列表
    signal_name_list = ['VP08Score_difference']

    # 参数配置
    mode = "prod"               # 运行模式：prod-生产环境，test-测试环境
    start_date = "2020-01-02"   # 回测开始日期
    end_date = "2026-01-15"     # 回测结束日期
    cost = 0.00006              # 交易成本（万分之0.6）

    # 指数配置
    big_indexName = "上证50"     # 大盘代表指数
    small_indexName = "中证2000"  # 小盘代表指数

    # 选择best_x时的权重配置
    # big_proportion + small_proportion + 等权部分 = 1
    big_proportion = 0.15       # 大盘指数权重
    small_proportion = 0.15     # 小盘指数权重
    # 剩余0.7为等权部分的权重

    # 批量处理L2因子
    for signal_name in signal_name_list:
        signal_constructor = L2_signalConstruction(
            signal_name, start_date, end_date, cost, mode,
            big_indexName, small_indexName, big_proportion, small_proportion
        )
        signal_constructor.L2_backtest_main()

    # ==================== 其他使用示例（已注释） ====================

    # # 示例1：查找L2因子对应的L3因子
    # l2_factor_name = "Shibor"  # 示例L2因子名称
    # l3_factors = signal_constructor.get_factor_info(l2_factor_name, name=True)
    # print(f"L2因子 '{l2_factor_name}' 对应的L3因子: {l3_factors}")
    #
    # # 示例2：查找L3因子对应的x值
    # l3_factor_name = "Shibor_2W"  # 示例L3因子名称
    # x_value = signal_constructor.get_factor_info(l3_factor_name, name=False)
    # print(f"L3因子 '{l3_factor_name}' 对应的x值: {x_value}")
