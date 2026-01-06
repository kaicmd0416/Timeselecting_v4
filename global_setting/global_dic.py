"""
全局路径配置模块

这个模块负责管理和配置项目中使用的所有文件路径，主要功能包括：
1. 读取路径配置文件（JSON格式）
2. 构建完整的文件路径
3. 提供全局路径访问接口

配置文件结构 (tracking_path_config.json):
1. main_folder:
   - folder_type: 文件夹类型标识
   - path: 基础路径
   - disk: 磁盘路径（可选）
2. sub_folder:
   - data_type: 数据类型标识
   - folder_name: 文件夹名称
   - folder_type: 文件夹类型标识
3. components:
   - data_source: 数据源配置
     - mode: 数据源模式 (local/sql)

主要依赖：
- json：配置文件处理
- os：路径操作
- pathlib：路径处理
- pandas：数据处理
- pymysql：数据库连接

作者：[作者名]
创建时间：[创建时间]
"""

import json
import os
from pathlib import Path
import pandas as pd
import pymysql
from datetime import datetime

# 全局字典，存储所有配置信息
global_dic = {}

def init():
    """
    初始化全局字典，从配置文件加载设置
    
    读取tracking_path_config.json配置文件，解析其中的路径配置信息，
    并更新全局字典供其他函数使用
    
    Returns:
        bool: 初始化是否成功
    """
    global global_dic
    
    try:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 构建配置文件的完整路径
        config_path = os.path.join(current_dir, 'timeselecting_path_config.json')
        
        # 读取配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            
        # 更新全局字典
        global_dic.update(config_data)
        return True
        
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        global_dic = {}
        return False

def get(key):
    """
    获取全局变量值，支持本地文件系统和SQL数据库两种模式
    
    根据配置的数据源模式，返回对应的文件路径或SQL查询语句。
    支持动态切换本地文件和数据库模式。
    
    Args:
        key (str): 要获取的键名，对应配置文件中的data_type
    
    Returns:
        any: 对应的值，可能是文件路径或SQL查询语句，失败时返回None
    """
    global global_dic
    
    # 如果字典为空，尝试重新初始化
    if not global_dic:
        if not init():
            return None
    
    # 获取数据源模式
    data_source = global_dic.get('components', {}).get('data_source', {})
    mode = data_source.get('mode', 'local')
    
    # 特殊键处理
    if key == 'mode':
        return mode
    if key == 'config_path':
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'timeselecting_path_config.json')
        return config_path
    
    # 获取配置信息
    config = None
    for item in global_dic.get('sub_folder', []):
        if item.get('data_type') == key:
            config = item
            break
    
    if not config:
        return None
    
    if mode == 'sql':
        # SQL模式：返回 select * from db_name.table_name
        if 'sql_sheet' not in config or 'database' not in config:
            # 如果没有sql_sheet或database，降级到local模式
            mode = 'local'
        else:
            table_name = config['sql_sheet']
            db_name = config['database']
            return (f"SELECT * FROM {db_name}.{table_name}")
    
    if mode == 'local':
        # 本地模式：返回文件路径
        if 'folder_name' not in config:
            return None
            
        # 获取主文件夹路径
        main_folder = global_dic.get('main_folder', [])
        folder_type = config.get('folder_type', 'input_folder')
        
        # 查找对应的主文件夹
        base_path = None
        for folder in main_folder:
            if folder.get('folder_type') == folder_type:
                base_path = folder.get('path')
                # 尝试获取disk配置，如果没有则使用path中的磁盘路径
                disk = folder.get('disk', '')
                if disk:
                    # 确保disk以冒号结尾
                    if not disk.endswith(':'):
                        disk += ':'
                    # 确保base_path不以斜杠开头
                    base_path = base_path.lstrip('\\').lstrip('/')
                    # 构建完整路径
                    full_path = os.path.join(disk + os.sep, base_path, config['folder_name'])
                    # 标准化路径分隔符
                    return os.path.normpath(full_path)
                break
                
        if not base_path:
            return None
        
        # 如果没有disk配置，直接使用base_path
        full_path = os.path.join(base_path, config['folder_name'])
        full_path = os.path.normpath(full_path)
        return full_path

def set(key, value):
    """
    设置全局变量值
    
    手动设置全局字典中的键值对，用于动态配置
    
    Args:
        key (str): 要设置的键名
        value (any): 要设置的值
    """
    global global_dic
    global_dic[key] = value

# 初始化全局字典
init()

# 确保get函数可以被导出
__all__ = ['get', 'set', 'init']