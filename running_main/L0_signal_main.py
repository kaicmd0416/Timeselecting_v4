"""
L0级别信号生成模块 (L0_signal_main)

本模块负责生成最终的L0级别择时信号，是整个因子体系的最顶层。
L0信号通过聚合所有L1级别信号得到，代表最终的大小盘择时决策。

信号体系层级结构:
    L0 (最终信号) ← 聚合所有L1信号
    L1 (一级因子) ← 聚合对应的L2信号
    L2 (二级因子) ← 聚合对应的L3信号
    L3 (三级因子) ← 原始因子数据计算

信号含义:
    - final_signal = 0: 看多大盘（如沪深300）
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
from dotenv import load_dotenv

# ==================== 自定义模块导入 ====================
# 添加全局工具函数路径
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
import global_setting.global_dic as glv
from backtesting.factor_backtesting import factor_backtesting
from factor_weighting import ICWeighter, ICIRWeighter, SharpeWeighter, MomentumWeighter, ReturnCalculator

# ==================== 全局配置 ====================
config_path = glv.get('config_path')  # 获取全局配置文件路径
pd.set_option('display.max_rows', None)  # 设置pandas显示所有行


class L0_signalConstruction:
    """
    L0级别信号生成类

    负责生成最终的L0级别择时信号，通过聚合所有L1信号实现。
    L0信号是整个因子体系的最终输出，用于指导实际的大小盘配置决策。

    工作流程:
        1. 从signal_dictionary.yaml读取所有L1因子列表
        2. 从数据库获取每个L1因子的信号数据
        3. 对所有L1信号取平均值
        4. 根据平均值生成最终的0/0.5/1信号
        5. 通过飞书机器人发送结果通知
        6. 将结果保存到数据库

    Attributes:
    -----------
    start_date : str
        回测/生成信号的开始日期，格式为 'YYYY-MM-DD'
    end_date : str
        回测/生成信号的结束日期，格式为 'YYYY-MM-DD'
    mode : str
        运行模式:
        - 'prod': 生产模式，使用生产环境的数据表
        - 'test': 测试模式，使用测试环境的数据表
    webhook_url : str
        飞书机器人的Webhook URL，用于发送通知
    inputpath_base : str
        L1信号数据的SQL查询基础路径
    """

    def __init__(self, start_date, end_date, mode, weighting_method='ic'):
        """
        初始化L0信号构建类

        Parameters:
        -----------
        start_date : str
            开始日期，格式为 'YYYY-MM-DD'
        end_date : str
            结束日期，格式为 'YYYY-MM-DD'
        mode : str
            模式，'prod'（生产模式）或 'test'（测试模式）
        weighting_method : str
            加权方式，'equal'（等权，默认）或 'ic'（IC加权）
        """
        # 加载环境变量（包含飞书Webhook URL等敏感信息）
        load_dotenv()
        self.webhook_url = os.getenv('FEISHU_WEBHOOK_URL')

        # 保存日期范围和运行模式
        self.start_date = start_date
        self.end_date = end_date
        self.mode = mode
        self.weighting_method = weighting_method

        # 根据模式选择对应的数据表路径
        if self.mode == 'prod':
            self.inputpath_base = glv.get('L1_signalData_prod')  # 生产环境L1信号表
        else:
            self.inputpath_base = glv.get('L1_signalData_test')  # 测试环境L1信号表

    def get_factor_info(self):
        """
        从配置文件获取所有L1因子名称

        读取signal_dictionary.yaml配置文件，提取所有不重复的L1因子名称。
        L1因子是一级分类，如：MacroLiquidity（宏观流动性）、IndexPriceVolume（指数量价）等。

        Returns:
        --------
        list
            包含所有L1因子名称的列表，如：
            ['MacroLiquidity', 'IndexPriceVolume', 'SpecialFactor',
             'StockCapital', 'MacroEconomy', 'StockFundamentals', 'StockEmotion']

        Raises:
        -------
        打印错误信息并返回空列表（不抛出异常）
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

            # 遍历所有因子，提取不重复的L1因子名称
            l1_factors = []
            for factor_key, factor_info in signal_dict.items():
                l1_factor = factor_info.get('L1_factor')
                if l1_factor not in l1_factors:  # 去重
                    l1_factors.append(l1_factor)

            print(f"找到所有L1因子: {l1_factors}")
            return l1_factors

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
        从数据库获取指定L1因子的信号数据

        根据信号名称和日期范围，从数据库查询对应的L1信号数据。

        Parameters:
        -----------
        signal_name : str
            L1因子名称，如 'MacroLiquidity'、'IndexPriceVolume' 等

        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - valuation_date: 日期，格式为 'YYYY-MM-DD'
            - {signal_name}: 该因子的信号值（0/0.5/1）

        SQL查询逻辑:
            SELECT * FROM L1_signal_prod/test
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

    def L0_construction_main(self):
        """
        L0信号构建主函数

        执行完整的L0信号生成流程：
        1. 获取所有L1因子的信号数据
        2. 合并所有L1信号（按日期外连接）
        3. 对缺失值填充0.5（中性信号）
        4. 计算所有L1信号的平均值
        5. 根据平均值生成最终信号（0/0.5/1）
        6. 通过飞书发送结果通知
        7. 保存到数据库

        信号生成逻辑:
            final_value = mean(所有L1信号)
            if final_value < 0.5:
                final_signal = 0  # 买大盘
            elif final_value == 0.5:
                final_signal = 0.5  # 中性
            else:
                final_signal = 1  # 买小盘
        """
        # ==================== 初始化数据库连接 ====================
        inputpath_sql = glv.get('sql_path')
        if self.mode == 'prod':
            sm = gt.sqlSaving_main(inputpath_sql, 'L0_signal_prod')
        else:
            sm = gt.sqlSaving_main(inputpath_sql, 'L0_signal_test')

        # ==================== 获取并合并所有L1信号 ====================
        n = 1
        df_final = pd.DataFrame()
        factor_name_list = self.get_factor_info()  # 获取所有L1因子名称

        for factor_name in factor_name_list:
            df = self.raw_signal_withdraw(factor_name)  # 获取单个L1因子的信号
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
                L1信号的平均值

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

        # 计算所有L1信号的平均值或加权平均值
        signal_columns = [col for col in df_final.columns if col != 'valuation_date']

        if self.weighting_method != 'equal':
            # 使用非等权加权
            try:
                # 获取收益率数据
                return_calc = ReturnCalculator(
                    self.start_date, self.end_date,
                    big_index='沪深300', small_index='中证2000',
                    level='L0'
                )
                df_returns = return_calc.get_relative_returns()

                # 根据加权方式选择加权器
                if self.weighting_method == 'ic':
                    weighter = ICWeighter(lookback_window=504, min_periods=504)
                    method_name = "IC加权"
                elif self.weighting_method == 'icir':
                    weighter = ICIRWeighter(lookback_window=504, min_periods=504, ic_window=20)
                    method_name = "IC_IR加权"
                elif self.weighting_method == 'sharpe':
                    weighter = SharpeWeighter(lookback_window=504, min_periods=504)
                    method_name = "夏普比率加权"
                elif self.weighting_method == 'momentum':
                    weighter = MomentumWeighter(lookback_window=252, min_periods=252, decay=0.01)
                    method_name = "动量加权"
                else:
                    raise ValueError(f"未知的加权方式: {self.weighting_method}")

                # 计算权重序列
                df_weights = weighter.calculate_weights_series(df_final[signal_columns], df_returns)

                # 打印调试信息
                print("=" * 60)
                print(f"[调试] {method_name}权重信息:")
                print(f"信号数据形状: {df_final[signal_columns].shape}")
                print(f"权重数据形状: {df_weights.shape}")
                print(f"\n权重样本（最后一行）:\n{df_weights.tail(1).T}")
                print("=" * 60)

                # 应用权重计算加权平均
                df_final['final_value'] = weighter.apply_weights(df_final[signal_columns], df_weights)
                print(f"[{method_name}] L0信号使用{method_name}方式聚合L1信号")
            except Exception as e:
                print(f"[加权] 计算失败，回退到等权方式: {e}")
                import traceback
                traceback.print_exc()
                df_final['final_value'] = df_final[signal_columns].mean(axis=1)
        else:
            # 使用等权
            df_final['final_value'] = df_final[signal_columns].mean(axis=1)

        # 将平均值转换为离散信号
        df_final['final_signal'] = df_final['final_value'].apply(lambda x: x_processing(x))

        # ==================== 整理输出格式 ====================
        df_final.reset_index(inplace=True)
        df_final = df_final[['valuation_date', 'final_value', 'final_signal']]
        df_final['update_time'] = datetime.now().replace(tzinfo=None)  # 添加更新时间戳

        # ==================== 发送飞书通知 ====================
        sender = gt.FeishuBot(self.webhook_url)
        result_mean_str = df_final.to_string(index=False)
        sender.send_message(result_mean_str)

        # ==================== 保存到数据库 ====================
        sm.df_to_sql(df_final)

    def L0_backtest_main(self):
        """
        L0信号回测主函数

        先构建L0信号，然后对生成的信号进行回测分析。
        回测会计算信号的历史收益、最大回撤等指标。

        回测参数:
            - 大盘指数: 沪深300
            - 小盘指数: 中证2000
            - 交易成本: 0.00006（万分之0.6，单边）
        """
        # 定义回测使用的指数
        big_indexName = '上证50'      # 大盘代表指数
        small_indexName = '中证2000'   # 小盘代表指数

        # 先构建L0信号
        self.L0_construction_main()

        # 执行回测分析
        fb = factor_backtesting(
            'final_signal',       # 信号列名
            self.start_date,      # 开始日期
            self.end_date,        # 结束日期
            0.00006,              # 交易成本（万分之0.6）
            self.mode,            # 运行模式
            'L0',                 # 信号层级
            big_indexName,        # 大盘指数名称
            small_indexName,      # 小盘指数名称
            None,                 # 基准指数（None表示使用等权）
            None                  # x参数（L0层级不需要）
        )
        fb.backtesting_main()


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 参数配置
    mode = "test"              # 运行模式：prod-生产环境，test-测试环境
    start_date = "2015-01-01"  # 回测开始日期
    end_date = "2026-01-29"    # 回测结束日期

    # 创建L0信号构建器并执行回测
    # weighting_method 可选: 'equal', 'ic', 'icir', 'sharpe', 'momentum'
    signal_constructor = L0_signalConstruction(start_date, end_date, mode, weighting_method='equal')
    signal_constructor.L0_backtest_main()
