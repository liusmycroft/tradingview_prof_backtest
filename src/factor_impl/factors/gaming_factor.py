import pandas as pd

from .base import BaseFactor


class GamingFactor(BaseFactor):
    """博弈因子 (Stren)"""

    name = "STREN"
    category = "高频资金流"
    description = "博弈因子，加权主买量与加权主卖量之比，衡量多空博弈的相对力量"

    def compute(
        self,
        daily_weighted_buy_vol: pd.DataFrame,
        daily_weighted_sell_vol: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算博弈因子。

        公式:
            Stren = sum(weight_t * Vol_Buy_t) / sum(weight_t * Vol_Sell_t)

        Args:
            daily_weighted_buy_vol: 预计算的加权主买量累计值
                sum(weight_t * Vol_Buy_t) (index=日期, columns=股票代码)
            daily_weighted_sell_vol: 预计算的加权主卖量累计值
                sum(weight_t * Vol_Sell_t) (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 博弈因子值
        """
        import numpy as np

        result = daily_weighted_buy_vol / daily_weighted_sell_vol.replace(0, np.nan)
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 博弈因子衡量多空双方博弈的相对力量，与未来收益负相关。
# 当个股过去一段时间主买量大于主卖量时，买方力量强于卖方力量，
# 多头筹码量增价，个股价格容易被高估；反之价格容易被低估。
#
# weight_t 可以等权或使用 21 天、42 天半衰期等指数加权。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.gaming_factor import GamingFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_weighted_buy_vol = pd.DataFrame(
#       np.random.rand(30, 2) * 1e6 + 1e5,
#       index=dates, columns=stocks,
#   )
#   daily_weighted_sell_vol = pd.DataFrame(
#       np.random.rand(30, 2) * 1e6 + 1e5,
#       index=dates, columns=stocks,
#   )
#
#   factor = GamingFactor()
#   result = factor.compute(
#       daily_weighted_buy_vol=daily_weighted_buy_vol,
#       daily_weighted_sell_vol=daily_weighted_sell_vol,
#   )
#   print(result.tail())
