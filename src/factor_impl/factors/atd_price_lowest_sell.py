import pandas as pd

from factors.base import BaseFactor


class ATDPriceLowestSellFactor(BaseFactor):
    """股价最低时刻-主卖笔均成交金额因子 (ATD Price Lowest Sell)。"""

    name = "ATD_PRICE_LOWEST_SELL"
    category = "高频成交分布"
    description = "股价最低10%时刻主卖笔均成交金额与全天笔均成交金额之比"

    def compute(
        self,
        daily_atd_lowest_sell: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算股价最低时刻-主卖笔均成交金额因子。

        公式: SATD = ATD_{PriceLowest10%_Sell} / ATD_T
        因子值为 T 日滚动均值。

        Args:
            daily_atd_lowest_sell: 预计算的每日标准化主卖笔均成交金额
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_atd_lowest_sell.rolling(window=T, min_periods=T).mean()
        return result
