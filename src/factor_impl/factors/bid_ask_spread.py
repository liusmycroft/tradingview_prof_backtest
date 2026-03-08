import numpy as np
import pandas as pd

from factors.base import BaseFactor


class BidAskSpreadFactor(BaseFactor):
    """盘口价差因子 (Bid-Ask Spread)"""

    name = "BID_ASK_SPREAD"
    category = "高频因子-流动性类"
    description = "卖一价与买一价的相对距离，衡量市场流动性宽度"

    def compute(
        self,
        daily_spread: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算盘口价差因子。

        公式:
            spread = 2 * (a1 - b1) / (a1 + b1)
            月度因子 = EMA(日度均值, T)

        Args:
            daily_spread: 日内盘口价差均值 (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 盘口价差因子的 T 日 EMA
        """
        result = daily_spread.ewm(span=T, min_periods=1).mean()
        return result
