import numpy as np
import pandas as pd

from factors.base import BaseFactor


class CorrectedAmbiguitySpreadFactor(BaseFactor):
    """修正模糊价差因子 (Corrected Ambiguity Spread)"""

    name = "CORRECTED_AMBIGUITY_SPREAD"
    category = "高频成交分布"
    description = "修正模糊价差，对传统买卖价差进行模糊区间修正后取滚动均值"

    def compute(
        self,
        daily_ambiguity_spread: pd.DataFrame,
        daily_return: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算修正模糊价差因子。

        公式:
            corrected_spread = daily_ambiguity_spread - abs(daily_return)
            factor = rolling_mean(corrected_spread, T)

        通过减去日收益率绝对值来修正价差中包含的价格变动成分。

        Args:
            daily_ambiguity_spread: 每日模糊价差 (买卖价差的日均值)，
                                    index=日期，columns=股票代码。
            daily_return: 日收益率，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 修正模糊价差因子值 (T日滚动均值)。
        """
        corrected = daily_ambiguity_spread - daily_return.abs()
        result = corrected.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 修正模糊价差因子在传统买卖价差的基础上，扣除日收益率绝对值来修正
# 价差中混入的价格变动成分。传统价差在价格大幅波动时会被高估，
# 修正后的价差更纯粹地反映流动性成本和信息不对称程度。
#
# 【使用示例】
#
#   from factors.corrected_ambiguity_spread import CorrectedAmbiguitySpreadFactor
#   factor = CorrectedAmbiguitySpreadFactor()
#   result = factor.compute(
#       daily_ambiguity_spread=spread_df, daily_return=ret_df, T=20
#   )
