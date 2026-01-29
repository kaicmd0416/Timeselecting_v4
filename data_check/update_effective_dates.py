"""
自动检测数据函数的实际起始日期并更新config_checking.yaml

此脚本会：
1. 读取config_checking.yaml中配置的所有函数
2. 调用每个函数获取数据（不做完整性检查）
3. 从返回的DataFrame中找到最早的valuation_date
4. 将正确的日期更新到config_checking.yaml中

使用方法：
    python update_effective_dates.py
"""

import os
import sys
import yaml
import pandas as pd

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 初始化全局变量
import global_setting.global_dic as glv
glv.init()

# 导入data_prepare，但需要临时禁用完整性检查
from data.data_prepare import data_prepare


class DataPrepareNoCheck(data_prepare):
    """继承data_prepare，但禁用完整性检查"""

    def _check_working_days_completeness(self, df, func_name, parameter=None):
        """覆盖父类方法，不做任何检查"""
        pass


def get_earliest_date(df):
    """从DataFrame中获取最早的valuation_date"""
    if df is None or df.empty:
        return None
    if 'valuation_date' not in df.columns:
        return None
    df_copy = df.copy()
    df_copy['valuation_date'] = pd.to_datetime(df_copy['valuation_date'])
    earliest = df_copy['valuation_date'].min()
    return earliest.strftime('%Y-%m-%d')


def test_parameterized_function(dp, func_name, parameter):
    """测试带参数的函数，返回最早日期"""
    try:
        func = getattr(dp, func_name)
        df = func(parameter)
        earliest = get_earliest_date(df)
        print(f"    {parameter}: {earliest}")
        return earliest
    except Exception as e:
        print(f"    {parameter}: 错误 - {str(e)[:50]}")
        return None


def test_parameterless_function(dp, func_name):
    """测试无参数的函数，返回最早日期"""
    try:
        func = getattr(dp, func_name)
        df = func()
        earliest = get_earliest_date(df)
        print(f"  {func_name}: {earliest}")
        return earliest
    except Exception as e:
        print(f"  {func_name}: 错误 - {str(e)[:50]}")
        return None


def main():
    print("=" * 60)
    print("有效开始日期检测和更新工具")
    print("=" * 60)

    # 配置文件路径
    config_path = os.path.join(current_dir, 'config_checking.yaml')

    # 读取当前配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 初始化data_prepare（禁用完整性检查），使用很早的日期以获取全部数据
    # 注意：data_prepare会自动将start_date向前推3年，所以用1990会变成1987
    print("\n初始化 DataPrepareNoCheck (1990-01-01 到 2026-12-31)...")
    print("注意：这个过程可能需要较长时间，因为需要运行所有函数...\n")

    try:
        dp = DataPrepareNoCheck('1990-01-01', '2026-12-31')
    except Exception as e:
        print(f"初始化失败: {e}")
        return 1

    print("=" * 60)
    print("检测带参数的函数")
    print("=" * 60)

    parameterized = config['data_prepare_functions'].get('parameterized_functions', {})

    for func_name, func_config in parameterized.items():
        print(f"\n{func_name}:")
        parameters = func_config.get('parameters', [])

        # 判断parameters是列表还是字典
        if isinstance(parameters, list):
            # 转换为字典格式
            new_params = {}
            earliest_overall = None
            for param in parameters:
                earliest = test_parameterized_function(dp, func_name, param)
                if earliest:
                    new_params[param] = earliest
                    if earliest_overall is None or earliest < earliest_overall:
                        earliest_overall = earliest

            # 更新配置
            func_config['parameters'] = new_params
            if earliest_overall:
                func_config['effective_start_date'] = earliest_overall

        elif isinstance(parameters, dict):
            # 已经是字典格式，更新每个参数的日期
            earliest_overall = None
            for param in list(parameters.keys()):
                earliest = test_parameterized_function(dp, func_name, param)
                if earliest:
                    parameters[param] = earliest
                    if earliest_overall is None or earliest < earliest_overall:
                        earliest_overall = earliest

            if earliest_overall:
                func_config['effective_start_date'] = earliest_overall

    print("\n" + "=" * 60)
    print("检测无参数的函数")
    print("=" * 60 + "\n")

    parameterless = config['data_prepare_functions'].get('parameterless_functions', [])

    for i, func_config in enumerate(parameterless):
        if isinstance(func_config, dict):
            func_name = func_config.get('name')
            if func_name:
                earliest = test_parameterless_function(dp, func_name)
                if earliest:
                    func_config['effective_start_date'] = earliest

    # 保存更新后的配置
    print("\n" + "=" * 60)
    print("保存更新后的配置")
    print("=" * 60)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\n配置已保存到: {config_path}")
    print("\n" + "=" * 60)
    print("SUCCESS: 有效开始日期检测和更新完成！")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
