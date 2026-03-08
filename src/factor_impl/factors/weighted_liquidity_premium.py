import pandas as pd

from factors.base import BaseFactor


class WeightedLiquidityPremiumFactor(BaseFactor):
    """流动性溢价因子-加权 (Weighted Liquidity Premium)"""

    name = "WEIGHTED_LIQUIDITY_PREMIUM"
    category = "高频流动性"
    description = "成交额加权的流动性溢价因子，衡量需求方定价与实际成交的相对差距"

    def compute(
        self,
        daily_cap_need: pd.DataFrame,
        daily_cap_actual: pd.DataFrame,
        daily_amount: pd.DataFrame,
        T: int = 21,
        **kwargs,
    ) -> pd.DataFrame:
        """计算流动性溢价因子-加权。

        公式:
            daily_premium = cap_need / cap_actual - 1
            weight = daily_amount / rolling_sum(daily_amount, T)
            factor = rolling_sum(weight * daily_premium, T)

        Args:
            daily_cap_need: 需求方定价下的每日市值，index=日期，columns=股票代码。
            daily_cap_actual: 实际平均交易的每日市值，index=日期，columns=股票代码。
            daily_amount: 每日成交额，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 21。

        Returns:
            pd.DataFrame: 加权流动性溢价因子值。
        """
        daily_premium = daily_cap_need / daily_cap_actual - 1
        amt_sum = daily_amount.rolling(window=T, min_periods=1).sum()
        weight = daily_amount / amt_sum
        weighted_premium = weight * daily_premium
        result = weighted_premium.rolling(window=T, min_periods=1).sum()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 流动性溢价因子-加权版本在等权版本的基础上，以成交额作为权重，
# 赋予成交活跃日更高的权重。成交额越大的日期，其流动性溢价信息
# 越具有代表性。因子值越大，说明流动性风险溢价越高。
#
# 【使用示例】
#
#   from factors.weighted_liquidity_premium import WeightedLiquidityPremiumFactor
#   factor = WeightedLiquidityPremiumFactor()
#   result = factor.compute(
#       daily_cap_need=need_df, daily_cap_actual=actual_df,
#       daily_amount=amount_df, T=21
#   )
