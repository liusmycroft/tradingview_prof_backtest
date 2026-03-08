"""
小买单主动成交度因子 (Small Buy Order Active Execution Ratio Factor)

衡量小买单中主动成交的占比，反映散户主动买入的积极程度。
"""

import pandas as pd
from factors.base import BaseFactor


class SmallBuyActiveFactor(BaseFactor):
    """小买单主动成交度因子

    公式: (1/T) * sum(small_active_buy / small_buy_total)
    """

    name = "SMALL_BUY_ACTIVE"
    category = "高频资金流"
    description = "小买单主动成交度因子，衡量小买单中主动成交的占比"

    def compute(
        self,
        small_active_buy: pd.DataFrame,
        small_buy_total: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算小买单主动成交度因子。

        Args:
            small_active_buy: 小买单主动成交量，index为日期，columns为股票代码。
            small_buy_total: 小买单总成交量，index为日期，columns为股票代码。
            T: 滚动窗口天数，默认20。

        Returns:
            pd.DataFrame: 因子值（T日滚动均值）。
        """
        daily_ratio = small_active_buy / small_buy_total
        result = daily_ratio.rolling(window=T, min_periods=1).mean()
        return result


# 使用示例
if __name__ == "__main__":
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=30)
    stocks = ["000001", "000002"]
    small_active_buy = pd.DataFrame(
        np.random.uniform(100, 500, (30, 2)), index=dates, columns=stocks
    )
    small_buy_total = pd.DataFrame(
        np.random.uniform(500, 1000, (30, 2)), index=dates, columns=stocks
    )

    factor = SmallBuyActiveFactor()
    print(factor)
    print(factor.compute(small_active_buy=small_active_buy, small_buy_total=small_buy_total))
