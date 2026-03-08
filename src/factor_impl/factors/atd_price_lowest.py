import pandas as pd

from factors.base import BaseFactor


class ATDPriceLowestFactor(BaseFactor):
    """股价最低时刻笔均成交金额因子 (ATD Price Lowest 10%)。"""

    name = "ATD_PRICE_LOWEST"
    category = "高频成交分布"
    description = "股价最低10%时刻的笔均成交金额与全天笔均成交金额之比"

    def compute(
        self,
        daily_atd_price_lowest: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算股价最低时刻笔均成交金额因子。

        公式: SATD = ATD_{PriceLowest10%} / ATD_T
        因子值为 T 日滚动均值。

        Args:
            daily_atd_price_lowest: 预计算的每日标准化笔均成交金额
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_atd_price_lowest.rolling(window=T, min_periods=T).mean()
        return result
