"""
毒流动性因子 (Toxic Liquidity Factor)

衡量短期撤单与中期撤单的比值，反映市场中有毒流动性的程度。
"""

import pandas as pd
from factors.base import BaseFactor


class ToxicLiquidityFactor(BaseFactor):
    """毒流动性因子

    公式: rolling_mean(cancel_5s / cancel_30s, T)
    """

    name = "TOXIC_LIQUIDITY"
    category = "高频流动性"
    description = "毒流动性因子，衡量5秒撤单与30秒撤单的比值"

    def compute(
        self,
        daily_toxic: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算毒流动性因子。

        Args:
            daily_toxic: 预计算的每日毒流动性比值(cancel_5s/cancel_30s)，
                         index为日期，columns为股票代码。
            T: 滚动窗口天数，默认20。

        Returns:
            pd.DataFrame: 毒流动性因子值。
        """
        result = daily_toxic.rolling(window=T, min_periods=1).mean()
        return result


# 使用示例
if __name__ == "__main__":
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=30)
    stocks = ["000001", "000002"]
    daily_toxic = pd.DataFrame(np.random.uniform(0.3, 0.8, (30, 2)), index=dates, columns=stocks)

    factor = ToxicLiquidityFactor()
    print(factor)
    print(factor.compute(daily_toxic=daily_toxic))
