"""
数据检查模块 (data_check)

本模块提供两类数据检查器：
1. SignalChecker: 检查信号生成完整性
2. PortfolioChecker: 检查投资组合完整性

注意：原始数据完整性检查已合并到 data_prepare.py 中，每个函数返回前会自动检查

作者: TimeSelecting Team
版本: v3.0
"""

import pandas as pd
import os
import sys
import yaml
import logging
from datetime import datetime
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
import global_setting.global_dic as glv
config_path=glv.get('config_path')

class SignalChecker:
    """
    信号检查器类
    
    负责检查各层级信号（L0、L1、L2、L3）的生成完整性。
    
    Attributes:
    -----------
    start_date : str
        实际检查开始日期（输入日期向前推1年）
    end_date : str
        结束日期
    target_date : str
        目标日期
    mode : str
        模式，'prod' 或 'test'
    working_days : list
        工作日列表
    config : dict
        检查配置
    signal_config : dict
        信号字典配置
    logger : logging.Logger
        日志记录器
    """
    
    def __init__(self, start_date, end_date, mode='prod'):
        """
        初始化信号检查器
        
        Parameters:
        -----------
        start_date : str
            开始日期，格式为'YYYY-MM-DD'（实际检查会向前推1年）
        end_date : str
            结束日期，格式为'YYYY-MM-DD'
        mode : str
            模式，'test' 或 'prod'
        """
        self.target_date = end_date
        self.mode = mode
        
        # 计算实际检查开始日期（往回退1年）
        start_datetime = pd.to_datetime(start_date)
        actual_start_date = (start_datetime - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        self.start_date = actual_start_date
        self.end_date = end_date

        
        # 获取工作日列表
        self.working_days = gt.working_days_list(actual_start_date, self.target_date)
        
        self.config = self._load_config()
        self.signal_config = self._load_signal_config()
        self._ensure_directories()
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志"""
        log_filename = f"signalChecking_{datetime.now().strftime('%Y%m%d')}.log"
        self.log_path = os.path.join(self.report_dir, log_filename)
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_path, encoding='utf-8'),
                logging.StreamHandler()  # 同时输出到控制台
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"开始信号检查 - 检查期间: {self.start_date} 到 {self.end_date}")
        self.logger.info(f"模式: {self.mode}")
        self.logger.info(f"目标日期: {self.target_date}")
        self.logger.info(f"可用日期: {self.target_date}")
        self.logger.info(f"工作日总数: {len(self.working_days)}")
        
    def _load_config(self):
        """加载配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), 'config_checking.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_signal_config(self):
        """加载信号字典配置"""
        signal_config_path = os.path.join(project_root, 'config_project', 'signal_dictionary.yaml')
        with open(signal_config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        # 创建报告保存目录
        self.report_dir = os.path.join(project_root, 'data_check', 'reports')
        os.makedirs(self.report_dir, exist_ok=True)

    def get_signal_sql_query(self, level):
        """
        获取信号SQL查询语句
        
        Args:
            level: 信号级别 ('L0', 'L1', 'L2', 'L3')
            
        Returns:
            str: SQL查询语句
        """
        if level not in ['L0', 'L1', 'L2', 'L3']:
            raise ValueError(f"无效的信号级别: {level}")
        
        table_key = f"{level}_signalData_{self.mode}"
        base_sql = glv.get(table_key)
        
        if not base_sql:
            raise ValueError(f"未找到SQL配置: {table_key}")
        
        return base_sql

    def check_l0_date_completeness(self):
        """
        检查L0级别的日期完整性（不检查具体信号）
        
        Returns:
            dict: 检查结果
        """
        if self.mode=='prod':
            inputpath=glv.get('L0_signalData_prod')
        else:
            inputpath=glv.get('L0_signalData_test')
        try:
            
            # L0级别没有signal_name字段，直接查询日期
            sql_query = inputpath+f"""
            WHERE valuation_date Between '{self.start_date}' 
            AND '{self.target_date}'
            """
            
            # 执行查询
            df = gt.data_getting(sql_query, config_path)
            df.sort_values(by='valuation_date',inplace=True)
            if df is None or df.empty:
                return {
                    'status': 'error',
                    'message': f'L0级别无数据',
                    'missing_dates': self.working_days,
                    'total_missing': len(self.working_days)
                }
            
            # 确保日期格式正确
            df['valuation_date'] = pd.to_datetime(df['valuation_date'])
            signal_dates = set(df['valuation_date'].dt.strftime('%Y-%m-%d').tolist())
            
            # 检查是否包含所有工作日
            missing_dates = set(self.working_days) - signal_dates
            
            if missing_dates:
                latest_date = df['valuation_date'].max().strftime('%Y-%m-%d')
                return {
                    'status': 'error',
                    'message': f'L0级别数据不完整，缺少{len(missing_dates)}个工作日，最新更新日期为{latest_date}',
                    'missing_dates': sorted(list(missing_dates)),
                    'total_missing': len(missing_dates),
                    'latest_date': latest_date
                }
            else:
                return {
                    'status': 'normal',
                    'message': f'L0级别数据完整，包含{len(signal_dates)}个日期，最新更新日期为{self.target_date}',
                    'total_dates': len(signal_dates),
                    'latest_date': self.target_date
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'检查L0级别时出错: {str(e)}'
            }

    def check_signal_generation(self, signal_name, level, use_pagination=True):
        """
        检查单个信号的生成情况
        
        Args:
            signal_name: 信号名称
            level: 信号级别 ('L1', 'L2', 'L3')
            use_pagination: 是否使用分页查询优化性能
            
        Returns:
            dict: 检查结果
        """
        try:
            # 获取基础SQL查询
            base_sql = self.get_signal_sql_query(level)
            # 优化SQL查询，只选择需要的列，根据级别选择正确的表

            # 使用分页查询优化性能，添加更多限制条件
            sql_query = base_sql + f"""
                            WHERE signal_name = '{signal_name}' 
                            AND valuation_date Between '{self.start_date}' 
                            AND  '{self.target_date}'
                            """
            
            self.logger.info(f"    执行SQL查询: {sql_query[:100]}...")

            
            df=gt.data_getting(sql_query,config_path)
            # 确保日期格式正确
            df['valuation_date'] = pd.to_datetime(df['valuation_date'])
            signal_dates = set(df['valuation_date'].dt.strftime('%Y-%m-%d').tolist())
            
            # 检查是否包含所有工作日
            missing_dates = set(self.working_days) - signal_dates
            
            if missing_dates:
                latest_date = df['valuation_date'].max().strftime('%Y-%m-%d')
                return {
                    'status': 'error',
                    'message': f'信号 {signal_name} 在 {level} 级别数据不完整，缺少{len(missing_dates)}个工作日，最新更新日期为{latest_date}',
                    'missing_dates': sorted(list(missing_dates)),
                    'total_missing': len(missing_dates),
                    'latest_date': latest_date
                }
            else:
                return {
                    'status': 'normal',
                    'message': f'信号 {signal_name} 在 {level} 级别数据完整，包含{len(signal_dates)}个日期，最新更新日期为{self.target_date}',
                    'total_dates': len(signal_dates),
                    'latest_date': self.target_date
                }
                
        except TimeoutError:
            return {
                'status': 'error',
                'message': f'检查信号 {signal_name} 在 {level} 级别时超时（30秒）'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'检查信号 {signal_name} 在 {level} 级别时出错: {str(e)}'
            }

    def get_signal_names_by_level(self):
        """
        根据signal_dictionary.yaml获取每个级别的信号名称列表
        
        Returns:
            dict: 每个级别对应的信号名称列表
        """
        level_signals = {
            'L0': [],  # L0级别不需要检查具体信号，只检查日期完整性
            'L1': set(), 
            'L2': set(),
            'L3': set()
        }
        
        for signal_name, config in self.signal_config.items():
            # L1级别：使用L1_factor
            if 'L1_factor' in config:
                level_signals['L1'].add(config['L1_factor'])
            
            # L2级别：使用L2_factor
            if 'L2_factor' in config:
                level_signals['L2'].add(config['L2_factor'])
            
            # L3级别：使用L3_factor
            if 'L3_factor' in config:
                level_signals['L3'].add(config['L3_factor'])
        
        # 转换为列表并排序
        for level in ['L1', 'L2', 'L3']:
            level_signals[level] = sorted(list(level_signals[level]))
        
        return level_signals

    def check_l0_signals(self):
        """
        检查L0级别信号的生成情况（只检查日期完整性）
        
        Returns:
            tuple: (result, status)
                - result: L0级别检查结果
                - status: 'normal' 或 'error'
        """
        self.logger.info(f"\n=== 检查 L0 级别日期完整性 ===")
        l0_result = self.check_l0_date_completeness()
        
        # 记录L0结果
        if l0_result['status'] == 'normal':
            self.logger.info(f"  ✓ L0级别: {l0_result['message']}")
        else:
            self.logger.error(f"  ✗ L0级别: {l0_result['message']}")
            if 'missing_dates' in l0_result and l0_result['missing_dates']:
                missing_dates = l0_result['missing_dates']
                if len(missing_dates) <= 10:
                    self.logger.error(f"    缺失日期: {missing_dates}")
                else:
                    self.logger.error(f"    缺失日期（前10个）: {missing_dates[:10]}")
                    self.logger.error(f"    总共缺失 {len(missing_dates)} 个日期")
        
        status = l0_result['status']
        self.logger.info(f"L0级别检查整体状态: {status}")
        return l0_result, status

    def check_all_signals(self):
        """
        检查所有信号的生成情况（L1、L2、L3级别）
        
        Returns:
            tuple: (results, status)
                - results: 所有信号的检查结果
                - status: 'normal' 或 'error'
        """
        self.logger.info(f"\n开始检查L1、L2、L3级别信号生成情况...")
        
        # 获取每个级别的信号名称列表
        level_signals = self.get_signal_names_by_level()
        
        self.logger.info(f"L1级别信号数量: {len(level_signals['L1'])}")
        self.logger.info(f"L2级别信号数量: {len(level_signals['L2'])}")
        self.logger.info(f"L3级别信号数量: {len(level_signals['L3'])}")
        
        results = {}
        
        # 检查L1、L2、L3级别（检查具体信号）
        levels = ['L1', 'L2', 'L3']
        
        for level in levels:
            self.logger.info(f"\n=== 检查 {level} 级别信号 ===")
            level_results = {}
            
            for i, signal_name in enumerate(level_signals[level], 1):
                self.logger.info(f"  检查信号 ({i}/{len(level_signals[level])}): {signal_name}")
                self.logger.info(f"    开始时间: {datetime.now().strftime('%H:%M:%S')}")
                
                # 使用分页查询优化性能
                try:
                    result = self.check_signal_generation(signal_name, level, use_pagination=True)
                    level_results[signal_name] = result
                except Exception as e:
                    self.logger.error(f"    信号 {signal_name} 检查失败，跳过: {str(e)}")
                    level_results[signal_name] = {
                        'status': 'error',
                        'message': f'检查失败: {str(e)}'
                    }
                
                self.logger.info(f"    完成时间: {datetime.now().strftime('%H:%M:%S')}")
                
                # 记录结果
                if result['status'] == 'normal':
                    self.logger.info(f"    ✓ {signal_name}: {result['message']}")
                else:
                    self.logger.error(f"    ✗ {signal_name}: {result['message']}")
                    if 'missing_dates' in result and result['missing_dates']:
                        missing_dates = result['missing_dates']
                        if len(missing_dates) <= 10:
                            self.logger.error(f"      缺失日期: {missing_dates}")
                        else:
                            self.logger.error(f"      缺失日期（前10个）: {missing_dates[:10]}")
                            self.logger.error(f"      总共缺失 {len(missing_dates)} 个日期")
            
            results[level] = level_results
        
        # 输出总结
        self._log_signal_check_summary(results)
        
        # 计算整体状态
        status = 'normal'
        
        # 检查L1、L2、L3级别
        for level in ['L1', 'L2', 'L3']:
            if level in results:
                for signal_name, result in results[level].items():
                    if result['status'] != 'normal':
                        status = 'error'
                        break
                if status == 'error':
                    break
        
        self.logger.info(f"L1、L2、L3级别信号检查整体状态: {status}")
        return results, status

    def _log_signal_check_summary(self, results):
        """输出信号检查结果总结"""
        self.logger.info(f"\n=== L1、L2、L3级别信号检查结果总结 ===")
        
        total_checks = 0
        normal_count = 0
        error_count = 0
        
        for level, level_results in results.items():
            # L1、L2、L3级别
            level_normal = 0
            level_error = 0
            
            for signal_name, result in level_results.items():
                total_checks += 1
                if result['status'] == 'normal':
                    normal_count += 1
                    level_normal += 1
                else:
                    error_count += 1
                    level_error += 1
            
            self.logger.info(f"{level}级别: 正常 {level_normal} 个, 错误 {level_error} 个")
        
        self.logger.info(f"总检查项: {total_checks}")
        self.logger.info(f"总计: 正常 {normal_count} 个, 错误 {error_count} 个")
        self.logger.info(f"L1、L2、L3级别检查完成\n")


class PortfolioChecker:
    """
    投资组合检查器类
    
    负责检查投资组合数据的完整性。
    
    Attributes:
    -----------
    target_date : str
        目标日期，格式为'YYYY-MM-DD'
    mode : str
        模式，'prod' 或 'test'
    config : dict
        检查配置
    logger : logging.Logger
        日志记录器
    """
    
    def __init__(self, target_date, mode='prod'):
        """
        初始化投资组合检查器
        
        Parameters:
        -----------
        target_date : str
            目标日期，格式为'YYYY-MM-DD'
        mode : str
            模式，'test' 或 'prod'
        """
        self.target_date = target_date
        self.mode = mode

        
        self.config = self._load_config()
        self._ensure_directories()
        self._setup_logging()

        
    def _setup_logging(self):
        """设置日志"""
        log_filename = f"portfolioChecking_{datetime.now().strftime('%Y%m%d')}.log"
        self.log_path = os.path.join(self.report_dir, log_filename)
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_path, encoding='utf-8'),
                logging.StreamHandler()  # 同时输出到控制台
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"开始投资组合检查 - 目标日期: {self.target_date}")
        self.logger.info(f"模式: {self.mode}")
        self.logger.info(f"可用日期: { self.target_date}")
        
    def _load_config(self):
        """加载配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), 'config_checking.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        # 创建报告保存目录
        self.report_dir = os.path.join(project_root, 'data_check', 'reports')
        os.makedirs(self.report_dir, exist_ok=True)

    def check_portfolio_data(self, df_portfolio,portfolio_name):
        """
        检查单个投资组合的数据完整性
        
        Args:
            portfolio_name: 投资组合名称
            
        Returns:
            dict: 检查结果
        """
        df=df_portfolio[df_portfolio['portfolio_name']==portfolio_name]

        if df is None or df.empty:
            return {
                'status': 'error',
                'message': f'投资组合 {portfolio_name} 在 {self.target_date} 无数据',
                'missing_portfolio': portfolio_name
            }
        else:
            # 检查数据完整性
            row_count = len(df)
            self.logger.info(f"    投资组合 {portfolio_name} 数据行数: {row_count}")
            
            return {
                'status': 'normal',
                'message': f'投资组合 {portfolio_name} 数据完整，包含 {row_count} 行数据',
                'row_count': row_count,
                'check_date': self.target_date
            }
    def check_all_portfolios(self):
        """
        检查所有投资组合的数据完整性
        
        Returns:
            tuple: (results, status)
                - results: 所有投资组合的检查结果
                - status: 'normal' 或 'error'
        """
        self.logger.info(f"\n开始检查所有投资组合数据完整性...")
        
        results = {}
        inputpath = glv.get('portfolio')
        inputpath = inputpath + f" Where valuation_date='{self.target_date}'"
        df_portfolio = gt.data_getting(inputpath, config_path)
        portfolio_names=df_portfolio['portfolio_name'].unique().tolist()
        portfolio_names=[i for i in portfolio_names if 'Timeselecting_' in str(i) ]
        
        # 检查是否有投资组合数据
        if not portfolio_names:
            self.logger.error(f"  在 {self.target_date} 没有找到任何Timeselecting_开头的投资组合数据")
            self.logger.error(f"  所有投资组合均没有生成")
            
            # 创建错误结果
            error_result = {
                'status': 'error',
                'message': f'在 {self.target_date} 没有找到任何Timeselecting_开头的投资组合数据',
                'missing_date': self.target_date
            }
            results['all_portfolios'] = error_result
            
            # 输出总结
            self._log_portfolio_check_summary(results)
            
            self.logger.info(f"投资组合检查整体状态: error")
            return results, 'error'
        
        # 有投资组合数据，进行正常检查
        self.logger.info(f"找到 {len(portfolio_names)} 个Timeselecting_开头的投资组合")
        
        for i, portfolio_name in enumerate(portfolio_names, 1):
            self.logger.info(f"  检查投资组合 ({i}/{len(portfolio_names)}): {portfolio_name}")
            self.logger.info(f"    开始时间: {datetime.now().strftime('%H:%M:%S')}")

            # 检查投资组合数据
            result = self.check_portfolio_data(df_portfolio,portfolio_name)
            results[portfolio_name] = result
            
            self.logger.info(f"    完成时间: {datetime.now().strftime('%H:%M:%S')}")
            
            # 记录结果
            if result['status'] == 'normal':
                self.logger.info(f"    ✓ {portfolio_name}: {result['message']}")
            else:
                self.logger.error(f"    ✗ {portfolio_name}: {result['message']}")
        
        # 输出总结
        self._log_portfolio_check_summary(results)
        
        # 计算整体状态
        status = 'normal'
        for portfolio_name, result in results.items():
            if result['status'] != 'normal':
                status = 'error'
                break
        
        self.logger.info(f"投资组合检查整体状态: {status}")
        return results, status

    def _log_portfolio_check_summary(self, results):
        """输出投资组合检查结果总结"""
        self.logger.info(f"\n=== 投资组合检查结果总结 ===")
        
        total_checks = len(results)
        normal_count = 0
        error_count = 0
        
        for portfolio_name, result in results.items():
            if result['status'] == 'normal':
                normal_count += 1
                self.logger.info(f"✓ {portfolio_name}: {result['message']}")
            else:
                error_count += 1
                self.logger.error(f"✗ {portfolio_name}: {result['message']}")
        
        self.logger.info(f"总检查项: {total_checks}")
        self.logger.info(f"总计: 正常 {normal_count} 个, 错误 {error_count} 个")
        self.logger.info(f"投资组合检查完成\n")

if __name__ == "__main__":
    # 使用示例
    input_start_date = '2025-10-30'  # 输入的开始日期
    end_date = '2025-10-30'
    mode = 'prod'  # 或 'test'
    check_type = 'portfolio'  # 或 'signal' 或 'portfolio'
    
    print(f"输入开始日期: {input_start_date}")
    print(f"结束日期: {end_date}")
    print(f"检查类型: {check_type}")
    
    if check_type == 'signal':
        # 信号生成检查
        print("开始信号生成检查...")
        checker = SignalChecker(start_date=input_start_date, end_date=end_date, mode=mode)
        results, status = checker.check_all_signals()
        print(f"信号生成检查完成，整体状态: {status}")
        
    elif check_type == 'portfolio':
        # 投资组合检查
        print("开始投资组合检查...")
        checker = PortfolioChecker(target_date=end_date, mode=mode)
        results, status = checker.check_all_portfolios()
        print(f"投资组合检查完成，整体状态: {status}")
        
    else:
        print("无效的检查类型，请选择 'signal' 或 'portfolio'")
