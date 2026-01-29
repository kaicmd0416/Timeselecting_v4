# 港股联动因子实现记录

## 日期：2026-01-28

---

## 1. 需求讨论

### 1.1 因子设计背景

蔡老板提出添加港股联动因子，用于捕捉港股与A股之间的联动关系。

**初始建议的因子：**
- VIX / 恐慌指数
- 北向资金流（已停止披露，排除）
- 两融余额
- AH溢价
- DR007

**最终确定方向：** 港股联动，作为北向资金的替代指标

### 1.2 因子归类讨论

**问题：** 港股联动应该放在哪个一级因子？

**讨论结果：**
- 港股联动本质上反映外资/南向资金的风险偏好和配置行为
- StockCapital下已有USDX（美元指数）、USBond（美债）等海外因子
- **结论：** 放入 `StockCapital`（股票资金）一级因子

### 1.3 二级因子拆分

**相关性分析：**
| 因子对 | 预期相关性 | 原因 |
|--------|-----------|------|
| HSI vs HSTECH | 较高 (0.6-0.8) | 都是港股动量，方向通常一致 |
| HSI vs AH溢价 | 较低 | 不同维度：趋势 vs 估值 |
| HSTECH vs AH溢价 | 较低 | 同上 |

**最终方案：**
- `HKStock_Momentum` (L2) → HK_HSI_Momentum, HK_HSTECH_Momentum
- `AH_Premium` (L2) → AH_Premium

---

## 2. 数据源确认

### 2.1 数据库信息
- **数据库：** tusharedb
- **数据表：** index_global

### 2.2 可用指数代码
| ts_code | 名称 | 数据行数 |
|---------|------|---------|
| HSI | 恒生指数 | 2723 |
| HKTECH | 恒生科技指数 | 1354 |
| HKAH | 恒生AH股溢价指数 | 2723 |

**注意：** HKTECH从2020年7月才有数据

---

## 3. 代码实现

### 3.1 data_prepare.py

新增函数：

```python
def raw_hk_index(self, index_type='HSI'):
    """
    获取港股指数数据

    Parameters:
    -----------
    index_type : str
        - 'HSI': 恒生指数
        - 'HKTECH': 恒生科技指数
        - 'HKAH': 恒生AH股溢价指数

    Returns:
    --------
    pd.DataFrame: [valuation_date, close, pct_chg]
    """
    # 港股和A股交易日历不同，港股休市时用前值填充
    # close用前值填充，pct_chg缺失日设为0
```

```python
def raw_ah_premium(self):
    """
    直接获取恒生AH股溢价指数(HKAH)

    Returns:
    --------
    pd.DataFrame: [valuation_date, ah_premium]
    """
```

### 3.2 data_processing.py

新增函数：

```python
def hk_hsi_momentum(self):
    """
    计算恒生指数动量因子
    逻辑：港股强势时，外资风险偏好高，利好A股大盘

    计算方式：恒生指数累计收益 / 沪深300累计收益
    """

def hk_hstech_momentum(self):
    """
    计算恒生科技指数动量因子
    逻辑：港股科技强势时，成长风格占优

    计算方式：恒生科技累计收益 / 中证1000累计收益
    """

def ah_premium_factor(self):
    """
    计算AH溢价因子
    逻辑：
    - AH溢价上升 → A股相对贵 → 外资可能流出 → 利空大盘
    - AH溢价下降 → A股相对便宜 → 外资可能流入 → 利好大盘
    """
```

### 3.3 L3_signal_main.py

新增信号生成逻辑：

```python
# ======================== 港股联动因子 ========================
elif self.signal_name=='HK_HSI_Momentum':
    df=self.dpro.hk_hsi_momentum()
    sc_mode='mode_1'  # 长周期均线正向：恒生指数强势→外资风险偏好高→买大盘

elif self.signal_name=='HK_HSTECH_Momentum':
    df=self.dpro.hk_hstech_momentum()
    sc_mode='mode_2'  # 长周期均线反向：恒生科技强势→成长风格占优→买小盘

elif self.signal_name=='AH_Premium':
    df=self.dpro.ah_premium_factor()
    sc_mode='mode_2'  # 长周期均线反向：AH溢价上升→A股相对贵→买小盘
```

### 3.4 signal_dictionary.yaml

新增配置：

```yaml
# ======================== 港股联动因子 ========================
# 港股动量因子（HSI和HSTECH相关性较高，共用L2）
HK_HSI_Momentum:
  L1_factor: StockCapital
  L2_factor: HKStock_Momentum
  L3_factor: HK_HSI_Momentum

HK_HSTECH_Momentum:
  L1_factor: StockCapital
  L2_factor: HKStock_Momentum
  L3_factor: HK_HSTECH_Momentum

# AH溢价因子（独立L2）
AH_Premium:
  L1_factor: StockCapital
  L2_factor: AH_Premium
  L3_factor: AH_Premium
```

---

## 4. 测试结果

### 4.1 数据获取测试

| 因子 | 数据行数 | 数据范围 | 状态 |
|------|---------|---------|------|
| HK_HSI_Momentum | 2198 | 2017-01-03 ~ 2026-01-20 | ✅ |
| HK_HSTECH_Momentum | 1332 | 2020-07-27 ~ 2026-01-20 | ✅ |
| AH_Premium | 2198 | 2017-01-03 ~ 2026-01-20 | ✅ |

### 4.2 L3信号生成测试（test模式）

| 因子 | 信号模式 | 生成信号条数 | 状态 |
|------|---------|-------------|------|
| HK_HSI_Momentum | mode_1 | 8802 | ✅ |
| HK_HSTECH_Momentum | mode_2 | 6492 | ✅ |
| AH_Premium | mode_2 | 8802 | ✅ |

---

## 5. 待办事项

- [ ] 生成prod模式正式信号
- [ ] 运行回测查看效果
- [ ] 根据回测结果调整信号模式（如有必要）
- [ ] 生成L2层级信号

---

## 6. 回测命令参考

```python
from running_main.L3_signal_main import L3_signalConstruction

# 生成prod模式信号
for signal_name in ['HK_HSI_Momentum', 'HK_HSTECH_Momentum', 'AH_Premium']:
    ssm = L3_signalConstruction(
        signal_name=signal_name,
        mode='prod',
        start_date='2015-01-01',
        end_date='2026-01-28'
    )
    ssm.signal_main()
```

---

## 7. 因子统计更新

实现后的因子统计：

| 层级 | 原数量 | 新增 | 现数量 |
|------|-------|------|-------|
| L1因子 | 9 | 0 | 9 |
| L2因子 | 48 | 2 | 50 |
| L3因子 | 65 | 3 | 68 |

StockCapital下的L2因子：
- NLBP_difference
- LargeOrder_difference
- USDX
- USBond
- ETF_Shares
- **HKStock_Momentum** (新增)
- **AH_Premium** (新增)
