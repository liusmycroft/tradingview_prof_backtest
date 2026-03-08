import numpy as np
import pandas as pd

from factors.base import BaseFactor


class VWPINFactor(BaseFactor):
    """基于物理时间交易量加权的知情交易概率 VWPIN 因子"""

    name = "VWPIN"
    category = "高频资金流"
    description = "基于物理时间交易量加权的知情交易概率，衡量知情交易者的活跃程度"

    def compute(
        self,
        daily_buy_volume: pd.DataFrame,
        daily_sell_volume: pd.DataFrame,
        daily_total_volume: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 VWPIN 因子。

        公式:
            order_imbalance = |buy_volume - sell_volume| / total_volume
            VWPIN = rolling_mean(order_imbalance, T)

        VWPIN 基于物理时间窗口内的交易量加权订单不平衡来估计知情交易概率。

        Args:
            daily_buy_volume: 每日主动买入成交量，index=日期，columns=股票代码。
            daily_sell_volume: 每日主动卖出成交量，index=日期，columns=股票代码。
            daily_total_volume: 每日总成交量，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: VWPIN 因子值 (T日滚动均值)。
        """
        order_imbalance = (daily_buy_volume - daily_sell_volume).abs() / daily_total_volume
        result = order_imbalance.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# VWPIN (Volume-Weighted Probability of Informed Trading) 是对经典 PIN 模型
# 的改进，使用物理时间窗口内的交易量加权订单不平衡来估计知情交易概率。
# 相比传统 PIN，VWPIN 计算更简便且更适合高频数据。
# 因子值越大，说明知情交易者越活跃，信息不对称程度越高。
#
# 【使用示例】
#
#   from factors.vwpin import VWPINFactor
#   factor = VWPINFactor()
#   result = factor.compute(
#       daily_buy_volume=buy_df, daily_sell_volume=sell_df,
#       daily_total_volume=total_df, T=20
#   )
