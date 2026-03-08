import pandas as pd

from factors.base import BaseFactor


class ImprovedLargeRatioActiveFactor(BaseFactor):
    """引入主动买卖特征的改进大单交易占比因子"""

    name = "IMPROVED_LARGE_RATIO_ACTIVE"
    category = "高频资金流"
    description = "引入主动买卖特征的改进大单交易占比，区分主动买入和主动卖出的大单"

    def compute(
        self,
        active_big_buy: pd.DataFrame,
        active_big_sell: pd.DataFrame,
        total_amount: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算引入主动买卖特征的改进大单交易占比因子。

        公式: rolling_mean((active_big_buy - active_big_sell) / total_amount, T)

        Args:
            active_big_buy: 主动买入大单成交额，index=日期，columns=股票代码。
            active_big_sell: 主动卖出大单成交额，index=日期，columns=股票代码。
            total_amount: 总成交额，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值 (T日滚动均值)。
        """
        daily_ratio = (active_big_buy - active_big_sell) / total_amount
        result = daily_ratio.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 在传统大单交易占比因子的基础上，引入主动买卖特征进行改进。
# 通过区分主动买入大单和主动卖出大单，更精确地刻画大资金的方向性意图。
# 主动买入大单占优时因子值为正，预示看多力量较强。
#
# 【使用示例】
#
#   from factors.improved_large_ratio_active import ImprovedLargeRatioActiveFactor
#   factor = ImprovedLargeRatioActiveFactor()
#   result = factor.compute(
#       active_big_buy=buy_df, active_big_sell=sell_df,
#       total_amount=total_df, T=20
#   )
