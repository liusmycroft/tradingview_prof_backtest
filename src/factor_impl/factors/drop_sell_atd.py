"""
跌幅最大时刻-主卖笔均成交金额因子 (Largest Drop Moment Main Sell Per-Transaction Amount Factor)

衡量跌幅最大时刻主卖方向的笔均成交金额占比，反映下跌时主动卖出的集中度。
"""

import pandas as pd
from factors.base import BaseFactor


class DropSellATDFactor(BaseFactor):
    """跌幅最大时刻-主卖笔均成交金额因子

    公式: SATD = rolling_mean(ATD_DownTop10%_Sell / ATD_Total, T)
    """

    name = "DROP_SELL_ATD"
    category = "高频成交分布"
    description = "跌幅最大时刻-主卖笔均成交金额因子，衡量下跌时主动卖出的集中度"

    def compute(
        self,
        daily_satd: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算跌幅最大时刻-主卖笔均成交金额因子。

        Args:
            daily_satd: 预计算的每日SATD值（ATD_DownTop10%_Sell / ATD_Total），
                        index为日期，columns为股票代码。
            T: 滚动窗口天数，默认20。

        Returns:
            pd.DataFrame: 因子值（T日滚动均值）。
        """
        result = daily_satd.rolling(window=T, min_periods=1).mean()
        return result


# 使用示例
if __name__ == "__main__":
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=30)
    stocks = ["000001", "000002"]
    daily_satd = pd.DataFrame(
        np.random.uniform(0.05, 0.3, (30, 2)), index=dates, columns=stocks
    )

    factor = DropSellATDFactor()
    print(factor)
    print(factor.compute(daily_satd=daily_satd))
