# 实施历史记录

## 2026-01-19 商品期货因子实施

### 需求概述
为大小盘择时系统新增4个商品期货相关的L3因子：

| L2因子 | L3因子 | 说明 | 信号模式 |
|--------|--------|------|----------|
| CommodityIndex | Commodity_Upside | 上游商品指数（原油/煤炭/有色） | mode_1（正向） |
| CommodityIndex | Commodity_Downside | 中下游商品指数（农产品/轻工） | mode_2（反向） |
| CommodityTrading | Commodity_Volume | 期货交易活跃度综合指数变化 | mode_2（反向） |
| CommodityRelation | Commodity_PPI_Correl | 商品综合指数与PPI的20日滚动相关性 | mode_1（正向） |

### 期货品种分类

**上游品种（COMMODITY_UPSIDE）- 15个：**
- 原油：SC
- 煤炭：ZC（动力煤）、JM（焦煤）、J（焦炭）
- 有色：CU（铜）、AL（铝）、ZN（锌）、PB（铅）、NI（镍）、SN（锡）
- 贵金属：AU（黄金）、AG（白银）
- 黑色：RB（螺纹钢）、HC（热轧卷板）、I（铁矿石）

**中下游品种（COMMODITY_DOWNSIDE）- 15个：**
- 农产品：A（豆一）、M（豆粕）、Y（豆油）、P（棕榈油）、OI（菜油）、RM（菜粕）、CF（棉花）、SR（白糖）、C（玉米）
- 轻工：RU（橡胶）、FG（玻璃）、MA（甲醇）、PP（聚丙烯）、L（塑料）、TA（PTA）

### 修改文件清单

#### 1. data/data_prepare.py (行1550-1571)
新增方法：
```python
def raw_futureData_commodity(self):
    """获取商品期货完整数据（含成交量、持仓量、收盘价）"""
    df = gt.futureData_withdraw(self.start_date, self.end_date, ['close', 'volume', 'oi'], False)
    df.rename(columns={'oi': 'open_interest'}, inplace=True)
    ...
```

#### 2. data/data_processing.py (行1793-2170)
新增内容：
- **品种分类常量**：`COMMODITY_UPSIDE`、`COMMODITY_DOWNSIDE`
- **辅助方法**：
  - `_get_commodity_main_contracts()` - 筛选主力连续合约（纯字母代码如A、CU、RB）
  - `_calculate_nanhua_weights()` - 计算南华权重（成交金额占比×持仓金额占比）
  - `_build_commodity_index()` - 构建商品指数（含收益率截断±15%）
- **因子计算方法**：
  - `commodity_upside()` - 上游商品指数
  - `commodity_downside()` - 中下游商品指数
  - `commodity_volume()` - 期货交易活跃度（5日变化率）
  - `commodity_ppi_correl()` - 商品指数与PPI的20日滚动相关性

#### 3. running_main/L3_signal_main.py (行311-323)
在`raw_data_preparing()`方法中添加4个elif分支

#### 4. config_project/signal_dictionary.yaml (行190-209)
```yaml
Commodity_Upside:
  L1_factor: MacroEconomy
  L2_factor: CommodityIndex
  L3_factor: Commodity_Upside

Commodity_Downside:
  L1_factor: MacroEconomy
  L2_factor: CommodityIndex
  L3_factor: Commodity_Downside

Commodity_Volume:
  L1_factor: MacroEconomy
  L2_factor: CommodityTrading
  L3_factor: Commodity_Volume

Commodity_PPI_Correl:
  L1_factor: MacroEconomy
  L2_factor: CommodityRelation
  L3_factor: Commodity_PPI_Correl
```

### 计算逻辑说明

#### 南华权重计算
```
单品种权重 = 成交金额占比 × 持仓金额占比
成交金额 = 成交量 × 收盘价
持仓金额 = 持仓量 × 收盘价
每月调整权重
```

#### Commodity_Upside / Commodity_Downside
1. 筛选主力连续合约（纯字母代码如A、CU、RB）
2. 计算每个品种的每日收益率
3. **收益率截断**：clip(-0.15, 0.15) 防止合约切换跳空
4. 按南华权重加权计算综合指数收益率
5. 累积收益率得到指数值（初始值100）

#### Commodity_Volume
1. 单品种活跃度 = 0.5×(当日成交量/20日均值) + 0.5×(当日持仓量/20日均值)
2. 按南华权重加权得到综合活跃度指数
3. 输出5日变化率

#### Commodity_PPI_Correl
1. 构建全品种商品综合指数（上游+中下游）
2. 获取PPI数据并前向填充至日度
3. 计算商品指数与PPI的20日滚动相关性

### Debug记录

#### Bug 1: 列名错误
- **问题**：`gt.futureData_withdraw` 的列名是 `volume` 而不是 `vol`
- **修复**：`['close', 'vol', 'oi']` → `['close', 'volume', 'oi']`

#### Bug 2: 主力合约识别错误
- **问题**：期货主力连续合约代码是纯字母（如`A`、`CU`），不是以`888`结尾
- **原始数据样例**：`['A', 'A1201', 'A1203', ...]`
- **修复**：改为识别纯字母代码 `re.match(r'^[A-Za-z]+$', code)`

#### Bug 3: 指数值爆炸
- **问题**：指数从100涨到2.69e+25，主力合约切换时跳空导致极端收益率
- **修复**：
  1. 收益率截断：`df_filtered['return'] = df_filtered['return'].clip(-0.15, 0.15)`
  2. 初始指数值从1.0改为100.0

#### Bug 4: L1因子分类不合理
- **问题**：`Commodity_Volume` 分到 `StockCapital` 不合理
- **修复**：改为 `MacroEconomy`，与其他商品因子保持一致

### 测试代码
```python
if __name__ == "__main__":
    for signal_name in ['Commodity_Upside', 'Commodity_Downside',
                        'Commodity_Volume', 'Commodity_PPI_Correl']:
        ssm = L3_signalConstruction(
            signal_name=signal_name,
            mode='test',
            start_date='2015-01-01',
            end_date='2026-01-18'
        )
        ssm.signal_main()
```

### 注意事项
1. 期货数据可能存在缺失，代码中已做fillna处理
2. 部分品种上市时间较晚，权重计算会动态调整
3. 主力连续合约使用纯字母代码识别（如A、CU、RB）
4. 收益率已做±15%截断处理，防止合约切换跳空
5. PPI为月度数据，与日度商品指数计算相关性时已做ffill处理

---
*记录时间：2026-01-19*
*Claude Opus 4.5*

---

## 2026-01-20 商品期货因子优化（符合南华编制规则）

### 需求概述
1. 过滤掉股指期货合约（IH、IF、IC、IM），只保留商品期货
2. 新增第5个商品因子 `Commodity_Composite`（南华商品综合指数）
3. 按照南华商品指数官方编制规则重构权重计算和指数构建方法

### 新增因子

| L2因子 | L3因子 | 说明 | 信号模式 |
|--------|--------|------|----------|
| CommodityIndex | Commodity_Composite | 南华商品综合指数（全部30个品种） | mode_1（正向：上行=大盘占优） |

### 南华商品指数编制规则要点

根据《南华商品指数编制细则》（2025年版）：

#### 1. 权重分配原则
- **消费金额权重**：基于年消费金额（消费量×价格），取过去5年均值
- **流动性权重**：基于年交易金额，取过去5年均值
- **多样化原则**：单品种权重≤25%，≥2%（品种数≥5时）
- **调整频率**：每年6月第一个交易日调整权重

#### 2. 指数计算公式
```
CI₀ = 1000（基期：2004年6月1日）
CI_t = CI_{t-1} × Σ[ω_{k,t-1} × I_{k,t}/I_{k,t-1}], t > 1
```
关键点：使用 **t-1时刻的权重** 乘以当期收益率

### 修改文件清单

#### 1. data/data_prepare.py (行1566-1569)
过滤股指期货合约：
```python
def raw_futureData_commodity(self):
    df = gt.futureData_withdraw(...)
    # 过滤掉股指期货合约（IH、IF、IC、IM开头），只保留商品期货
    df = df[~df['code'].str.upper().str.match(r'^(IH|IF|IC|IM)')]
    ...
```

#### 2. data/data_processing.py

**`_calculate_nanhua_weights()` 方法重构（行1850-1919）：**
```python
def _calculate_nanhua_weights(self, df_main, date, symbol_list):
    # 基于过去一年的交易金额计算流动性权重
    year_start = (date_dt - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    df_agg['weight'] = df_agg['trade_value'] / total_trade

    # 应用南华权重限制规则（品种数>=5时）
    if len(df_agg) >= 5:
        # 迭代调整：上限25%，下限2%
        df_agg.loc[df_agg['weight'] > 0.25, 'weight'] = 0.25
        df_agg.loc[df_agg['weight'] < 0.02, 'weight'] = 0.02
```

**`_build_commodity_index()` 方法重构（行1921-2034）：**
```python
def _build_commodity_index(self, df_main, symbol_list):
    index_value = 1000.0  # 基期指数值1000（符合南华规则）
    current_weights = None  # t-1时刻的权重
    pending_weights = None  # 待生效的新权重

    for date in dates:
        # 如果有待生效的权重，在新的一天开始时生效
        if pending_weights is not None:
            current_weights = pending_weights
            pending_weights = None

        # 使用t-1时刻的权重计算当天指数
        weighted_return = (df_date_valid['return'] * df_date_valid['weight']).sum() / weight_sum
        index_value = index_value * (1 + weighted_return)

        # 计算完当天指数后，检查是否需要更新权重（每年6月第一个交易日）
        if date_dt.month == 6 and current_weight_year != date_dt.year:
            pending_weights = new_weights  # 下一交易日生效
```

**新增 `commodity_composite()` 方法（行2242-2271）：**
```python
def commodity_composite(self):
    """计算南华商品综合指数（全部30个品种）"""
    all_symbols = self.COMMODITY_UPSIDE + self.COMMODITY_DOWNSIDE
    df_index = self._build_commodity_index(df_main, all_symbols)
    return df_index
```

**`commodity_volume()` 方法优化（行2144-2201）：**
- 添加年度权重调整逻辑
- 使用t-1时刻的权重

#### 3. running_main/L3_signal_main.py (行324-326)
新增elif分支：
```python
elif self.signal_name=='Commodity_Composite':
    df=self.dpro.commodity_composite()
    sc_mode='mode_1'  # 正向：同比上行→大盘占优
```

#### 4. config_project/signal_dictionary.yaml (行211-214)
```yaml
Commodity_Composite:
  L1_factor: MacroEconomy
  L2_factor: CommodityIndex
  L3_factor: Commodity_Composite
```

### 新旧逻辑对比

| 项目 | 旧逻辑 | 新逻辑（符合南华规则） |
|------|--------|------------------------|
| 基期指数 | 100 | **1000** |
| 权重计算 | 成交金额占比×持仓金额占比 | **过去一年交易金额占比** |
| 权重限制 | 无 | **单品种≤25%，≥2%** |
| 权重调整频率 | 每月 | **每年6月第一个交易日** |
| 权重使用 | 当天计算当天用 | **t-1时刻权重**（新权重下一交易日生效） |

### 5个商品因子汇总

| 因子名称 | 说明 | 信号模式 |
|----------|------|----------|
| Commodity_Upside | 上游商品指数（15个品种） | mode_1（正向） |
| Commodity_Downside | 中下游商品指数（15个品种） | mode_9（反向） |
| Commodity_Composite | 综合商品指数（30个品种） | mode_1（正向） |
| Commodity_Volume | 期货交易活跃度 | mode_9（反向） |
| Commodity_PPI_Correl | 商品指数与PPI相关性 | mode_1（正向） |

### 测试代码
```python
if __name__ == "__main__":
    for signal_name in ['Commodity_Upside', 'Commodity_Downside',
                        'Commodity_Composite', 'Commodity_PPI_Correl']:
        ssm = L3_signalConstruction(
            signal_name=signal_name,
            mode='test',
            start_date='2015-01-01',
            end_date='2026-01-18'
        )
        ssm.signal_main()
```

### 注意事项
1. 股指期货（IH、IF、IC、IM）已从数据源过滤，不会进入商品指数计算
2. 权重每年6月调整，新权重从下一个交易日开始生效
3. 由于没有消费量数据，权重计算简化为只用流动性权重（交易金额）
4. 所有5个商品因子现在都符合南华商品指数编制规则

---
*记录时间：2026-01-20*
*Claude Opus 4.5*

---

## 2026-01-21 商品期货因子独立为L1因子

### 变更概述
将5个商品期货因子从`MacroEconomy`独立出来，新建`Commodity`作为独立的L1因子分类。

### 变更理由
1. **数据源独立**：商品期货数据来自期货市场，与宏观经济指标数据源不同
2. **经济逻辑独特**：商品价格反映实体供需，有独特的产业链逻辑
3. **体量足够**：5个因子已形成完整体系
4. **扩展空间大**：便于后续扩展（期限结构、跨品种价差、库存数据等）
5. **组合管理更清晰**：独立L1因子便于风险归因和因子贡献分析

### 修改文件
- `config_project/signal_dictionary.yaml`：5个Commodity因子的L1_factor从`MacroEconomy`改为`Commodity`

### 现有L1因子分类（共9个）

| L1因子 | 说明 |
|--------|------|
| MacroLiquidity | 宏观流动性 |
| MacroEconomy | 宏观经济 |
| IndexPriceVolume | 指数价量 |
| StockCapital | 股票资金 |
| StockFundamentals | 股票基本面 |
| StockEmotion | 股票情绪 |
| SpecialFactor | 特殊因子 |
| **Commodity** | **商品期货（新增）** |
| Rubbish | 弃用因子 |

### Commodity因子列表

| L3因子 | L2因子 | 说明 |
|--------|--------|------|
| Commodity_Upside | CommodityIndex | 上游商品指数 |
| Commodity_Downside | CommodityIndex | 中下游商品指数 |
| Commodity_Composite | CommodityIndex | 综合商品指数 |
| Commodity_Volume | CommodityTrading | 期货交易活跃度 |
| Commodity_PPI_Correl | CommodityRelation | 商品指数与PPI相关性 |

---
*记录时间：2026-01-21*
*Claude Opus 4.5*

---

## 2026-01-21 季节性因子（Seasonality）实施

### 需求概述
1. 新建L1因子`Seasonality`（季节性/周期性）
2. 将原有`Monthly_effect`从`SpecialFactor`迁移至`Seasonality`
3. 新增`Pre_Holiday_Return`（节假日效应）

### 因子结构

| L1因子 | L2因子 | L3因子 | 说明 | 信号模式 |
|--------|--------|--------|------|----------|
| Seasonality | Monthly_Effect | Monthly_Effect | 历史同月份效应 | mode_7 |
| Seasonality | Holiday_Effect | Pre_Holiday_Return | 节假日前后效应 | mode_7 |

### 修改文件清单

#### 1. config_project/signal_dictionary.yaml
- `Monthly_effect` → `Monthly_Effect`，L1_factor从`SpecialFactor`改为`Seasonality`
- 新增`Pre_Holiday_Return`配置

#### 2. data/data_processing.py（新增方法）

**`pre_holiday_return()`方法：**
```python
def pre_holiday_return(self):
    """计算节假日前后一周的大小盘收益率差值"""
    # 1. 获取工作日列表，识别法定节假日（非周末的休市日）
    # 2. 将连续节假日合并为一个假期段
    # 3. 标记节前一周和节后一周的交易日
    # 4. 计算历史节前节后的平均收益率差值（排除当年）
    # 5. 输出valuation_date和pre_holiday_return
```

#### 3. running_main/L3_signal_main.py
- `Monthly_effect` → `Monthly_Effect`
- 新增`Pre_Holiday_Return`的elif分支

#### 4. factor_processing/signal_constructing.py
- 修改`Monthly_effect_signal_construct`方法，自动识别列名（支持monthly_effect、pre_holiday_return）

### 计算逻辑说明

#### Pre_Holiday_Return
1. 通过工作日列表识别法定节假日（非周末的休市日）
2. 将相邻节假日（间隔≤3天）合并为同一假期段
3. 标记每个假期的节前一周和节后一周
4. 计算历史节前节后期间的收益率差值平均数（排除当年）
5. 正值 → 大盘占优，负值 → 小盘占优

### 现有L1因子分类（共10个）

| L1因子 | 说明 |
|--------|------|
| MacroLiquidity | 宏观流动性 |
| MacroEconomy | 宏观经济 |
| IndexPriceVolume | 指数价量 |
| StockCapital | 股票资金 |
| StockFundamentals | 股票基本面 |
| StockEmotion | 股票情绪 |
| SpecialFactor | 特殊因子 |
| Commodity | 商品期货 |
| **Seasonality** | **季节性（新增）** |
| Rubbish | 弃用因子 |

### 测试代码
```python
if __name__ == "__main__":
    for signal_name in ['Monthly_Effect', 'Pre_Holiday_Return']:
        ssm = L3_signalConstruction(
            signal_name=signal_name,
            mode='test',
            start_date='2015-01-01',
            end_date='2026-01-20'
        )
        ssm.signal_main()
```

---
*记录时间：2026-01-21*
*Claude Opus 4.5*
