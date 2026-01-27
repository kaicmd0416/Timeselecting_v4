"""
L1级别信号生成模块 (L1_signal_main)

本模块负责生成L1级别的择时信号。L1信号是一级分类信号，
通过聚合同一L1分类下的所有L2信号得到。

信号体系层级结构:
    L0 (最终信号) ← 聚合所有L1信号
    L1 (一级因子) ← 聚合对应的L2信号  ← 【当前模块】
    L2 (二级因子) ← 聚合对应的L3信号
    L3 (三级因子) ← 原始因子数据计算

L1因子分类示例:
    - MacroLiquidity (宏观流动性): 包含Shibor、Bond、CreditSpread等L2因子
    - IndexPriceVolume (指数量价): 包含技术指标、动量等L2因子
    - StockCapital (资金流向): 包含NLBP、LargeOrder、ETF等L2因子
    - MacroEconomy (宏观经济): 包含CPI、PPI、PMI等L2因子
    - StockFundamentals (股票基本面): 包含PE、PB、盈利等L2因子
    - StockEmotion (市场情绪): 包含个股情绪、期货持仓等L2因子
    - SpecialFactor (特殊因子): 包含季节性、事件驱动等L2因子

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
from backtesting.factor_backtesting import factor_backtesting

# ==================== 全局配置 ====================
config_path = glv.get('config_path')  # 获取全局配置文件路径
pd.set_option('display.max_rows', None)  # 设置pandas显示所有行


class L1_signalConstruction:
    """
    L1级别信号生成类

    负责生成L1级别的择时信号，通过聚合同一L1分类下的所有L2信号实现。
    每个L1因子代表一个大类因子，如宏观流动性、指数量价等。

    工作流程:
        1. 根据L1因子名称，从signal_dictionary.yaml查找对应的L2因子列表
        2. 从数据库获取每个L2因子的信号数据
        3. 对所有L2信号取平均值
        4. 根据平均值生成最终的0/0.5/1信号
        5. 将结果保存到数据库

    Attributes:
    -----------
    signal_name : str
        L1因子名称，如 'MacroLiquidity'、'IndexPriceVolume' 等
    start_date : str
        回测/生成信号的开始日期，格式为 'YYYY-MM-DD'
    end_date : str
        回测/生成信号的结束日期，格式为 'YYYY-MM-DD'
    mode : str
        运行模式:
        - 'prod': 生产模式，使用生产环境的数据表
        - 'test': 测试模式，使用测试环境的数据表
    inputpath_base : str
        L2信号数据的SQL查询基础路径
    """

    def __init__(self, signal_name, start_date, end_date, mode):
        """
        初始化L1信号构建类

        Parameters:
        -----------
        signal_name : str
            L1因子名称，如 'MacroLiquidity'、'IndexPriceVolume' 等
        start_date : str
            开始日期，格式为 'YYYY-MM-DD'
        end_date : str
            结束日期，格式为 'YYYY-MM-DD'
        mode : str
            模式，'prod'（生产模式）或 'test'（测试模式）
        """
        # 保存基本参数
        self.start_date = start_date
        self.end_date = end_date
        self.signal_name = signal_name
        self.mode = mode

        # 根据模式选择对应的数据表路径
        if self.mode == 'prod':
            self.inputpath_base = glv.get('L2_signalData_prod')  # 生产环境L2信号表
        else:
            self.inputpath_base = glv.get('L2_signalData_test')  # 测试环境L2信号表

    def get_factor_info(self, factor_name):
        """
        根据L1因子名称查找对应的所有L2因子名称

        从signal_dictionary.yaml配置文件中，找出所有属于指定L1分类的L2因子。

        Parameters:
        -----------
        factor_name : str
            L1因子名称，如 'MacroLiquidity'、'IndexPriceVolume' 等

        Returns:
        --------
        list
            包含所有匹配的L2因子名称的列表，如：
            - MacroLiquidity → ['Bond', 'Shibor', 'CreditSpread', 'TermSpread', 'M1M2']
            - IndexPriceVolume → ['TargetIndex_Technical', 'TargetIndex_Momentum', 'RelativeIndex_Std']

        示例:
            get_factor_info('MacroLiquidity')
            → ['Bond', 'Shibor', 'CreditSpread', 'TermSpread', 'M1M2']
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

            # 遍历所有因子，找出L1_factor匹配的L2因子
            l2_factors = []
            for factor_key, factor_info in signal_dict.items():
                if factor_info.get('L1_factor') == factor_name:
                    l2_factor = factor_info.get('L2_factor')
                    if l2_factor not in l2_factors:  # 去重（多个L3可能属于同一个L2）
                        l2_factors.append(l2_factor)

            print(f"找到L1因子 '{factor_name}' 对应的L2因子: {l2_factors}")
            return l2_factors

        except FileNotFoundError:
            print(f"配置文件未找到: {config_path}")
            return []
        except yaml.YAMLError as e:
            print(f"YAML文件解析错误: {e}")
            return []
        except Exception as e:
            print(f"查找因子信息时出错: {e}")
            return []

    def raw_signal_withdraw(self, signal_name):
        """
        从数据库获取指定L2因子的信号数据

        根据信号名称和日期范围，从数据库查询对应的L2信号数据。

        Parameters:
        -----------
        signal_name : str
            L2因子名称，如 'Bond'、'Shibor'、'CreditSpread' 等

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - {signal_name}: 该因子的信号值（0/0.5/1）

        SQL查询逻辑:
            SELECT * FROM L2_signal_prod/test
            WHERE signal_name='{signal_name}'
            AND valuation_date BETWEEN '{start_date}' AND '{end_date}'
        """
        # 构建SQL查询语句
        inputpath = self.inputpath_base
        inputpath = str(inputpath) + \
            f" Where signal_name='{signal_name}' And valuation_date between '{self.start_date}' and '{self.end_date}'"

        # 执行查询并提取需要的列
        df = gt.data_getting(inputpath, config_path)
        df = df[['valuation_date', 'final_signal']]
        df.columns = ['valuation_date', signal_name]  # 重命名列为因子名称

        return df

    def L1_construction_main(self):
        """
        L1信号构建主函数

        执行完整的L1信号生成流程：
        1. 获取该L1因子下所有L2因子的信号数据
        2. 合并所有L2信号（按日期外连接）
        3. 对缺失值填充0.5（中性信号）
        4. 计算所有L2信号的平均值
        5. 根据平均值生成最终信号（0/0.5/1）
        6. 保存到数据库

        信号生成逻辑:
            final_value = mean(所有L2信号)
            if final_value < 0.5:
                final_signal = 0  # 买大盘
            elif final_value == 0.5:
                final_signal = 0.5  # 中性
            else:
                final_signal = 1  # 买小盘

        数据库表:
            - 生产环境: L1_signal_prod
            - 测试环境: L1_signal_test
        """
        # ==================== 初始化数据库连接 ====================
        inputpath_sql = glv.get('sql_path')
        if self.mode == 'prod':
            # delete=True 表示先删除该signal_name的旧数据再插入
            sm = gt.sqlSaving_main(inputpath_sql, 'L1_signal_prod', delete=True)
        else:
            sm = gt.sqlSaving_main(inputpath_sql, 'L1_signal_test', delete=True)

        # ==================== 获取并合并所有L2信号 ====================
        n = 1
        df_final = pd.DataFrame()
        factor_name_list = self.get_factor_info(self.signal_name)  # 获取该L1下的所有L2因子

        for factor_name in factor_name_list:
            df = self.raw_signal_withdraw(factor_name)  # 获取单个L2因子的信号
            if n == 1:
                df_final = df  # 第一个因子直接赋值
                n += 1
            else:
                # 后续因子通过外连接合并，保留所有日期
                df_final = df_final.merge(df, on='valuation_date', how='outer')

        # ==================== 信号处理 ====================
        df_final.set_index('valuation_date', inplace=True, drop=True)
        df_final.fillna(0.5, inplace=True)  # 缺失值填充为中性信号

        def x_processing(x):
            """
            将平均值转换为离散信号

            Parameters:
            -----------
            x : float
                L2信号的平均值

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

        # 计算所有L2信号的平均值
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
        # 第二个参数是用于删除旧数据的字段名，第三个参数是对应的值
        sm.df_to_sql(df_final, 'signal_name', self.signal_name)

    def L1_backtest_main(self):
        """
        L1信号回测主函数

        先构建L1信号，然后对生成的信号进行回测分析。
        回测会计算信号的历史收益、最大回撤等指标。

        回测参数:
            - 大盘指数: 上证50
            - 小盘指数: 中证2000
            - 交易成本: 0.00006（万分之0.6，单边）

        注意: L1层级使用上证50作为大盘代表，与L0层级的沪深300不同
        """
        # 定义回测使用的指数（L1层级使用上证50）
        big_indexName = '上证50'        # 大盘代表指数
        small_indexName = '中证2000'    # 小盘代表指数

        # 先构建L1信号
        self.L1_construction_main()

        # 执行回测分析
        fb = factor_backtesting(
            self.signal_name,     # L1因子名称
            self.start_date,      # 开始日期
            self.end_date,        # 结束日期
            0.00006,              # 交易成本（万分之0.6）
            self.mode,            # 运行模式
            'L1',                 # 信号层级
            big_indexName,        # 大盘指数名称
            small_indexName,      # 小盘指数名称
            None,                 # 基准指数（None表示使用等权）
            None                  # x参数（L1层级不需要）
        )
        fb.backtesting_main()


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    """
    批量运行L1信号生成和回测

    可选的L1因子列表:
        - MacroLiquidity: 宏观流动性因子
        - IndexPriceVolume: 指数量价因子
        - SpecialFactor: 特殊因子（季节性、事件驱动等）
        - StockCapital: 资金流向因子
        - MacroEconomy: 宏观经济因子
        - StockFundamentals: 股票基本面因子
        - StockEmotion: 市场情绪因子
        - Commodity: 商品期货因子
        - Option: 期权因子
    """
    # 要处理的L1因子列表
    for signal_name in ['SpecialFactor']:
        # 参数配置
        mode = "test"              # 运行模式：prod-生产环境，test-测试环境
        start_date = "2015-01-01"  # 回测开始日期
        end_date = '2026-01-23'    # 回测结束日期

        # 创建L1信号构建器并执行回测
        signal_constructor = L1_signalConstruction(signal_name, start_date, end_date, mode)
        signal_constructor.L1_backtest_main()
