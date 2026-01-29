"""
信号盈亏分析模块 (signal_pnl_analysis)

分析L0信号在不同状态下的多头和空头收益情况。

功能:
- 从portfolio_new数据库提取指定portfolio的持仓数据
- 获取L0因子信号数据
- 分析signal=1时多头/空头各赚多少
- 分析signal=0时多头/空头各赚多少
- 绘制不同信号下的多头/空头盈亏曲线

作者: TimeSelecting Team
版本: v1.0
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

# 添加全局工具函数路径
path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt
import global_setting.global_dic as glv

config_path = glv.get('config_path')


class SignalPnLAnalysis:
    """
    信号盈亏分析类

    分析指定portfolio在不同L0信号状态下的多头/空头收益。
    """

    def __init__(self, portfolio_name: str, start_date: str, end_date: str,
                 account_money: float = 10000000):
        """
        初始化

        Parameters:
        -----------
        portfolio_name : str
            组合名称，如 'Timeselecting_future_hs300'
        start_date : str
            开始日期
        end_date : str
            结束日期
        account_money : float
            账户资金，默认1000万
        """
        self.portfolio_name = portfolio_name
        self.start_date = start_date
        self.end_date = end_date
        self.account_money = account_money

    def get_portfolio_data(self) -> pd.DataFrame:
        """
        从portfolio_new数据库获取portfolio数据

        Returns:
        --------
        pd.DataFrame
            包含 valuation_date, code, weight 的DataFrame
        """
        inputpath = f"SELECT * FROM portfolio_new.portfolio WHERE portfolio_name='{self.portfolio_name}' AND valuation_date BETWEEN '{self.start_date}' AND '{self.end_date}'"

        df = gt.data_getting(inputpath, config_path)
        df = df[['valuation_date', 'code', 'weight']]
        df['valuation_date'] = pd.to_datetime(df['valuation_date']).dt.strftime('%Y-%m-%d')
        df = df.sort_values('valuation_date')

        print(f"[Portfolio] 获取到 {len(df)} 条数据，日期范围: {df['valuation_date'].min()} ~ {df['valuation_date'].max()}")
        return df

    def get_l0_signal(self) -> pd.DataFrame:
        """
        获取L0信号数据

        Returns:
        --------
        pd.DataFrame
            包含 valuation_date, final_signal, final_value 的DataFrame
        """
        inputpath = glv.get('L0_signalData_prod')
        inputpath = f"{inputpath} WHERE valuation_date BETWEEN '{self.start_date}' AND '{self.end_date}'"

        df = gt.data_getting(inputpath, config_path)
        df = df[['valuation_date', 'final_signal', 'final_value']]
        df['valuation_date'] = pd.to_datetime(df['valuation_date']).dt.strftime('%Y-%m-%d')
        df = df.sort_values('valuation_date')

        print(f"[L0信号] 获取到 {len(df)} 条数据")
        return df

    def calculate_daily_pnl(self, df_portfolio: pd.DataFrame, df_signal: pd.DataFrame) -> pd.DataFrame:
        """
        计算每日盈亏

        Parameters:
        -----------
        df_portfolio : pd.DataFrame
            持仓数据
        df_signal : pd.DataFrame
            信号数据

        Returns:
        --------
        pd.DataFrame
            每日盈亏数据
        """
        # 合并信号
        df_signal_map = df_signal.set_index('valuation_date')[['final_signal']].to_dict()['final_signal']

        results = []
        dates = sorted(df_portfolio['valuation_date'].unique())

        for date in dates:
            signal = df_signal_map.get(date, 0.5)

            # 获取当日持仓
            df_day = df_portfolio[df_portfolio['valuation_date'] == date]

            # 分离多头和空头
            df_long = df_day[df_day['weight'] > 0]
            df_short = df_day[df_day['weight'] < 0]

            result = {
                'valuation_date': date,
                'signal': signal,
                'long_codes': ','.join(df_long['code'].tolist()) if not df_long.empty else '',
                'short_codes': ','.join(df_short['code'].tolist()) if not df_short.empty else '',
                'long_weight_sum': df_long['weight'].sum() if not df_long.empty else 0,
                'short_weight_sum': df_short['weight'].sum() if not df_short.empty else 0,
            }
            results.append(result)

        df_result = pd.DataFrame(results)
        return df_result

    def analyze_pnl_by_signal(self) -> dict:
        """
        按信号分析盈亏

        Returns:
        --------
        dict
            分析结果
        """
        # 获取数据
        df_portfolio = self.get_portfolio_data()
        df_signal = self.get_l0_signal()

        if df_portfolio.empty:
            print("警告: portfolio数据为空")
            return {}

        # 计算每日盈亏
        df_daily = self.calculate_daily_pnl(df_portfolio, df_signal)

        # 计算整体portfolio的盈亏
        print("\n[计算] 使用 gt.portfolio_analyse 计算portfolio盈亏...")
        df_pnl_long = pd.DataFrame()
        df_pnl_short = pd.DataFrame()
        try:
            result = gt.portfolio_analyse(
                df_holding=df_portfolio,
                account_money=self.account_money,
                cost_stock=0.00085,
                cost_etf=0.0003,
                cost_future=0.00006,
                cost_option=0,
                cost_convertiblebond=0.0007,
                realtime=False,
                weight_standardize=False
            )
            print(f"[计算] 完成，返回类型: {type(result)}")

            # gt.portfolio_analyse 返回两个DataFrame: (df_long, df_short)
            if isinstance(result, tuple) and len(result) == 2:
                df_pnl_long, df_pnl_short = result
                print(f"[计算] 多头盈亏数据: {len(df_pnl_long)} 行, 列名: {df_pnl_long.columns.tolist() if not df_pnl_long.empty else 'N/A'}")
                print(f"[计算] 空头盈亏数据: {len(df_pnl_short)} 行, 列名: {df_pnl_short.columns.tolist() if not df_pnl_short.empty else 'N/A'}")
            elif isinstance(result, pd.DataFrame):
                df_pnl_long = result
                print(f"[计算] 返回单个DataFrame，列名: {df_pnl_long.columns.tolist()}")
        except Exception as e:
            print(f"[错误] 计算portfolio盈亏时出错: {e}")
            import traceback
            traceback.print_exc()

        # 按信号分组统计
        summary = self._summarize_by_signal(df_daily, df_pnl_long, df_signal)

        return {
            'daily': df_daily,
            'df_pnl_long': df_pnl_long,
            'df_pnl_short': df_pnl_short,
            'summary': summary,
            'df_portfolio': df_portfolio,
            'df_signal': df_signal
        }

    def _summarize_by_signal(self, df_daily: pd.DataFrame, df_pnl: pd.DataFrame,
                             df_signal: pd.DataFrame) -> dict:
        """
        按信号汇总统计

        Parameters:
        -----------
        df_daily : pd.DataFrame
            每日持仓数据
        df_pnl : pd.DataFrame
            盈亏数据
        df_signal : pd.DataFrame
            信号数据

        Returns:
        --------
        dict
            汇总统计
        """
        summary = {}

        for signal_val in [0, 0.5, 1]:
            df_sig_days = df_daily[df_daily['signal'] == signal_val]

            summary[f'signal_{signal_val}'] = {
                'days': len(df_sig_days),
                'dates': df_sig_days['valuation_date'].tolist() if not df_sig_days.empty else [],
            }

        return summary

    def plot_pnl_by_signal(self, results: dict, save_path: str = None):
        """
        绘制不同信号下的多头/空头盈亏曲线

        gt.portfolio_analyse 返回:
        - df1 (df_pnl_long): portfolio级别的汇总盈亏
        - df2 (df_pnl_short): 持仓明细，包含每个code的profit

        从持仓明细中按weight分离多头(weight>0)和空头(weight<0)

        Parameters:
        -----------
        results : dict
            分析结果
        save_path : str
            保存路径，如果为None则直接显示
        """
        df_portfolio_summary = results.get('df_pnl_long', pd.DataFrame())  # portfolio汇总
        df_holding_detail = results.get('df_pnl_short', pd.DataFrame())    # 持仓明细
        df_signal = results.get('df_signal', pd.DataFrame())

        if df_holding_detail.empty:
            print("警告: 持仓明细数据为空，无法绘图")
            return

        # 准备信号映射
        df_signal_map = df_signal.set_index('valuation_date')['final_signal'].to_dict()

        # 处理持仓明细数据
        df_detail = df_holding_detail.copy()
        df_detail['valuation_date'] = pd.to_datetime(df_detail['valuation_date']).dt.strftime('%Y-%m-%d')

        # 添加信号列
        df_detail['signal'] = df_detail['valuation_date'].map(df_signal_map)

        # 从持仓明细中分离多头和空头
        df_long_positions = df_detail[df_detail['weight'] > 0].copy()
        df_short_positions = df_detail[df_detail['weight'] < 0].copy()

        print(f"[绘图] 多头持仓记录数: {len(df_long_positions)}")
        print(f"[绘图] 空头持仓记录数: {len(df_short_positions)}")

        # 按日期汇总profit
        def aggregate_by_date(df):
            if df.empty:
                return pd.DataFrame()
            return df.groupby('valuation_date').agg({
                'profit': 'sum',
                'signal': 'first'
            }).reset_index()

        df_long_daily = aggregate_by_date(df_long_positions)
        df_short_daily = aggregate_by_date(df_short_positions)

        # 分离signal=1和signal=0的数据
        df_long_sig1 = df_long_daily[df_long_daily['signal'] == 1].copy() if not df_long_daily.empty else pd.DataFrame()
        df_long_sig0 = df_long_daily[df_long_daily['signal'] == 0].copy() if not df_long_daily.empty else pd.DataFrame()
        df_short_sig1 = df_short_daily[df_short_daily['signal'] == 1].copy() if not df_short_daily.empty else pd.DataFrame()
        df_short_sig0 = df_short_daily[df_short_daily['signal'] == 0].copy() if not df_short_daily.empty else pd.DataFrame()

        # 计算累计盈亏
        def calc_cumulative(df):
            if df.empty:
                return df
            df = df.sort_values('valuation_date')
            df['cum_pnl'] = df['profit'].cumsum()
            return df

        df_long_sig1 = calc_cumulative(df_long_sig1)
        df_long_sig0 = calc_cumulative(df_long_sig0)
        df_short_sig1 = calc_cumulative(df_short_sig1)
        df_short_sig0 = calc_cumulative(df_short_sig0)

        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # 图1: Signal=1 时的多头/空头盈亏
        ax1 = axes[0]
        if not df_long_sig1.empty:
            ax1.plot(pd.to_datetime(df_long_sig1['valuation_date']), df_long_sig1['cum_pnl'],
                     'b-', label='多头盈亏', linewidth=1.5)
        if not df_short_sig1.empty:
            ax1.plot(pd.to_datetime(df_short_sig1['valuation_date']), df_short_sig1['cum_pnl'],
                     'r-', label='空头盈亏', linewidth=1.5)
        ax1.set_title(f'Signal=1 (看多小盘) 时的多头/空头盈亏曲线\n{self.portfolio_name}', fontsize=12)
        ax1.set_xlabel('日期')
        ax1.set_ylabel('累计盈亏')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # 图2: Signal=0 时的多头/空头盈亏
        ax2 = axes[1]
        if not df_long_sig0.empty:
            ax2.plot(pd.to_datetime(df_long_sig0['valuation_date']), df_long_sig0['cum_pnl'],
                     'b-', label='多头盈亏', linewidth=1.5)
        if not df_short_sig0.empty:
            ax2.plot(pd.to_datetime(df_short_sig0['valuation_date']), df_short_sig0['cum_pnl'],
                     'r-', label='空头盈亏', linewidth=1.5)
        ax2.set_title(f'Signal=0 (看多大盘) 时的多头/空头盈亏曲线\n{self.portfolio_name}', fontsize=12)
        ax2.set_xlabel('日期')
        ax2.set_ylabel('累计盈亏')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[绘图] 图表已保存到: {save_path}")
        else:
            plt.show()

        # 打印统计信息
        print("\n【盈亏统计】")
        print(f"  Signal=1 (看多小盘) 天数: {len(df_long_sig1)}")
        if not df_long_sig1.empty:
            print(f"    多头累计盈亏: {df_long_sig1['cum_pnl'].iloc[-1]:,.2f}")
        if not df_short_sig1.empty:
            print(f"    空头累计盈亏: {df_short_sig1['cum_pnl'].iloc[-1]:,.2f}")

        print(f"  Signal=0 (看多大盘) 天数: {len(df_long_sig0)}")
        if not df_long_sig0.empty:
            print(f"    多头累计盈亏: {df_long_sig0['cum_pnl'].iloc[-1]:,.2f}")
        if not df_short_sig0.empty:
            print(f"    空头累计盈亏: {df_short_sig0['cum_pnl'].iloc[-1]:,.2f}")

    def print_analysis_report(self, results: dict):
        """
        打印分析报告

        Parameters:
        -----------
        results : dict
            分析结果
        """
        if not results:
            print("没有分析结果")
            return

        df_daily = results.get('daily', pd.DataFrame())
        df_pnl_long = results.get('df_pnl_long', pd.DataFrame())
        df_pnl_short = results.get('df_pnl_short', pd.DataFrame())
        summary = results.get('summary', {})

        print("\n" + "=" * 70)
        print(f"  信号盈亏分析报告")
        print(f"  Portfolio: {self.portfolio_name}")
        print(f"  分析区间: {self.start_date} ~ {self.end_date}")
        print("=" * 70)

        # 信号分布
        print("\n【1. 信号分布】")
        for signal_val in [0, 0.5, 1]:
            key = f'signal_{signal_val}'
            days = summary.get(key, {}).get('days', 0)
            signal_name = {0: '看多大盘', 0.5: '中性', 1: '看多小盘'}.get(signal_val, '')
            print(f"  Signal={signal_val} ({signal_name}): {days} 天")

        # 持仓分析
        print("\n【2. 持仓分析】")
        if not df_daily.empty:
            # Signal=1 时的持仓
            df_sig1 = df_daily[df_daily['signal'] == 1]
            if not df_sig1.empty:
                print(f"\n  Signal=1 (看多小盘) 时的持仓:")
                print(f"    多头 (weight>0): {df_sig1['long_codes'].iloc[0] if not df_sig1.empty else 'N/A'}")
                print(f"    空头 (weight<0): {df_sig1['short_codes'].iloc[0] if not df_sig1.empty else 'N/A'}")

            # Signal=0 时的持仓
            df_sig0 = df_daily[df_daily['signal'] == 0]
            if not df_sig0.empty:
                print(f"\n  Signal=0 (看多大盘) 时的持仓:")
                print(f"    多头 (weight>0): {df_sig0['long_codes'].iloc[0] if not df_sig0.empty else 'N/A'}")
                print(f"    空头 (weight<0): {df_sig0['short_codes'].iloc[0] if not df_sig0.empty else 'N/A'}")

        # Portfolio盈亏
        print("\n【3. Portfolio盈亏】")
        print("  多头盈亏数据:")
        if isinstance(df_pnl_long, pd.DataFrame) and not df_pnl_long.empty:
            print(f"    列名: {df_pnl_long.columns.tolist()}")
            print(f"    数据量: {len(df_pnl_long)}")
            print(f"    数据预览:")
            print(df_pnl_long.head(5).to_string(index=False))
        else:
            print("    无数据")

        print("\n  空头盈亏数据:")
        if isinstance(df_pnl_short, pd.DataFrame) and not df_pnl_short.empty:
            print(f"    列名: {df_pnl_short.columns.tolist()}")
            print(f"    数据量: {len(df_pnl_short)}")
            print(f"    数据预览:")
            print(df_pnl_short.head(5).to_string(index=False))
        else:
            print("    无数据")

        print("\n" + "=" * 70)


def main():
    """主函数"""
    # 参数配置 - 可修改
    portfolio_name = "Timeselecting_future_sz50_pro"
    start_date = "2022-07-27"
    end_date = "2026-01-28"
    account_money = 10000000

    print(f"开始分析...")
    print(f"  Portfolio: {portfolio_name}")
    print(f"  日期范围: {start_date} ~ {end_date}")
    print(f"  账户资金: {account_money:,.0f}")

    # 创建分析器
    analyzer = SignalPnLAnalysis(
        portfolio_name=portfolio_name,
        start_date=start_date,
        end_date=end_date,
        account_money=account_money
    )

    # 执行分析
    results = analyzer.analyze_pnl_by_signal()

    # 打印报告
    analyzer.print_analysis_report(results)

    # 绘制盈亏曲线
    if results:
        save_path = os.path.join(
            os.path.dirname(__file__),
            f'signal_pnl_{portfolio_name}_{start_date}_{end_date}.png'
        )
        analyzer.plot_pnl_by_signal(results, save_path=save_path)

    return results


if __name__ == "__main__":
    results = main()
