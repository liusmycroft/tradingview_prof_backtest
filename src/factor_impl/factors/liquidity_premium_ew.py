"""流动性溢价因子-等权 (Equal-Weighted Liquidity Premium)

等权加权下的流动性溢价因子，衡量需求方定价与实际成交的相对差距。
"""

import pandas as pd

from factors.base import BaseFactor


class LiquidityPremiumEWFactor(BaseFactor):
    """流动性溢价因子-等权"""

    name = "LIQUIDITY_PREMIUM_EW"
    category = "高频流动性"
    description = "流动性溢价因子-等权：需求方定价与实际成交的相对差距，取T日等权滚动均值"

    def compute(
        self,
        daily_cap_need: pd.DataFrame,
        daily_cap_actual: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算等权流动性溢价因子。

        公式: f = mean_T(cap_need / cap_actual) - 1

        Args:
            daily_cap_need: 每日需求方定价下的总市值
                (index=日期, columns=股票代码)
            daily_cap_actual: 每日实际平均交易下的总市值
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值
        """
        # 日度溢价比
        daily_ratio = daily_cap_need / daily_cap_actual

        # T 日等权滚动均值 - 1
        result = daily_ratio.rolling(window=T, min_periods=T).mean() - 1
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 流动性溢价因子衡量需求方定价下交易股票数量与实际平均交易股票数量的
# 相对差距。市场的平均成交价通常高于买单价格，高出部分即为流动性风险
# 溢价。等权版本对每日溢价赋予相同权重，取T日均值平滑波动。
# 溢价越高，说明流动性越差，投资者要求的补偿越高。
