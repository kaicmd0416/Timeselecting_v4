# DataChecker 重构说明

## 重构概述

本次重构对 `data_check` 文件夹中的 `DataChecker` 类进行了重大改进，主要变化如下：

## 主要变化

### 1. 构造函数参数变更

**之前：**
```python
def __init__(self, target_date):
    # 内部计算过去一年的日期范围
```

**现在：**
```python
def __init__(self, start_date, end_date):
    # 接受开始和结束日期参数，data_prepare类内部会自动使用三年前作为数据开始日期
```

### 2. 日期计算逻辑

- **input_start_date**: 保存原始输入的开始日期
- **start_date**: 直接使用输入的开始日期
- **actual_data_start_date**: data_prepare类内部使用的实际开始日期（输入日期的三年前）
- **available_date**: 使用 `gt.last_workday_calculate(end_date)` 计算
- **working_days**: 使用 `gt.working_days_list(actual_data_start_date, end_date)` 获取工作日列表

### 3. 新增功能

#### 检查 data_prepare.py 函数输出
新增 `check_data_prepare_functions()` 方法，可以检查 `data_prepare.py` 中每个函数的输出是否包含指定时间段内的所有工作日。

#### 有效开始日期检测
- **自动检测**: `detect_effective_start_dates()` 方法自动检测每个函数的有效开始日期
- **动态调整**: 如果函数的有效开始日期晚于三年前，则使用有效开始日期进行检查
- **配置更新**: `detect_and_update_effective_dates()` 方法将检测结果更新到配置文件

#### 支持的函数类型
- **有参数函数**: 如 `raw_shibor('2W')`, `raw_bond('3Y')` 等
- **无参数函数**: 如 `raw_usdx()`, `raw_CPI_withdraw()` 等

### 4. 配置文件更新

在 `config_checking.yaml` 中新增了 `data_prepare_functions` 配置节，用于定义要检查的函数列表。

## 使用方法

### 基本使用

#### 第一次运行（检测有效开始日期）

```python
from data_check.data_check import DataChecker

# 创建检查器实例
checker = DataChecker(start_date='2015-01-01', end_date='2025-12-31')

# 检测并更新有效开始日期（只需要运行一次）
effective_dates = checker.run_effective_date_detection_only()
```

或者使用独立脚本：
```bash
python data_check/update_effective_dates.py
```

#### 日常运行（从配置文件读取）

```python
from data_check.data_check import DataChecker

# 创建检查器实例
checker = DataChecker(start_date='2024-01-01', end_date='2024-12-31')

# 检查原始数据
status = checker.check_raw_data('Shibor_2W')

# 检查信号数据
status, latest_date = checker.check_signal_data('Shibor_2W_45D_9_combine')

# 检查data_prepare函数输出（从配置文件读取有效开始日期）
results = checker.check_data_prepare_functions()
```

### 检查结果

`check_data_prepare_functions()` 方法返回一个字典，包含每个函数的检查结果：

```python
{
    'raw_usdx': {
        'status': 'normal',  # 'normal', 'warning', 'error'
        'message': '数据完整，包含250个日期',
        'total_dates': 250
    },
    'raw_shibor(2W)': {
        'status': 'error',
        'message': '缺少5个工作日',
        'missing_dates': ['2024-01-02', '2024-01-03', ...],
        'total_missing': 5
    }
}
```

## 配置说明

### data_prepare_functions 配置

```yaml
data_prepare_functions:
  # 有参数的函数
  parameterized_functions:
    raw_shibor:
      parameters: ['2W', '9M']
      description: "Shibor利率数据"
      effective_start_date: "2015-01-01"  # 有效开始日期
    raw_bond:
      parameters: ['3Y', '10Y']
      description: "国债收益率数据"
      effective_start_date: "2015-01-01"
    # ... 更多函数
  
  # 无参数函数
  parameterless_functions:
    - name: raw_usdx
      effective_start_date: "2015-01-01"
    - name: raw_CPI_withdraw
      effective_start_date: "2015-01-01"
    # ... 更多函数
```

## 日志输出

重构后的类会输出更详细的日志信息：

```
开始数据检查 - 检查期间: 2021-01-01 到 2024-12-31
原始输入开始日期: 2024-01-01
data_prepare实际使用开始日期: 2021-01-01 (三年前)
目标日期: 2024-12-31
可用日期: 2024-12-30
工作日总数: 1043

开始检查data_prepare.py中的函数输出...
检查函数: raw_usdx
  有效开始日期: 2015-01-05
  三年前日期: 2021-01-01
  实际检查开始日期: 2021-01-01
✓ raw_usdx: 数据完整，包含1250个日期（其中1043个工作日）

=== data_prepare函数检查结果总结 ===
✓ raw_usdx: 数据完整，包含1250个日期（其中1043个工作日）
✓ raw_shibor(2W): 数据完整，包含1250个日期（其中1043个工作日）
✗ raw_index_earningsyield: 缺少372个工作日

总结: 正常 37 个, 警告 0 个, 错误 2 个
检查完成
```

## 注意事项

1. 确保 `global_tools` 模块可用（通过环境变量 `GLOBAL_TOOLSFUNC` 路径）
2. 确保 `data.data_prepare` 模块可用
3. 实际检查的日期范围会根据函数的有效开始日期动态调整：
   - 如果有效开始日期晚于三年前，使用有效开始日期
   - 如果有效开始日期早于三年前，使用三年前日期
4. 某些函数可能需要特定的数据源，确保数据源可用
5. **重要**：有效开始日期检测只需要运行一次，之后日常运行会直接从配置文件读取
6. 如果需要更新有效开始日期，可以重新运行 `update_effective_dates.py` 脚本
7. 配置文件中的 `effective_start_date` 字段会在检测后自动更新

## 兼容性

- 保持了原有的 `check_raw_data()` 和 `check_signal_data()` 方法
- 新增的方法不会影响现有功能
- 配置文件向后兼容
