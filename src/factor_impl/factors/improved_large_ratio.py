"""
改进大单交易占比因子 (Improved Large Order Trading Ratio Factor)

改进的大单资金流因子，综合考虑大单与非大单之间的交叉交易。
"""

import pandas as pd
from factors.base import BaseFactor


class ImprovedLargeRatioFactor(BaseFactor):
    """改进大单交易占比因子

    公式: rolling_mean((-lb_nls - nlb_ls + lb_ls) / total_volume, T)
    """

    name = "IMPROVED_LARGE_RATIO"
    category = "高频资金流"
    description = "改进大单交易占比因子，综合考虑大单与非大单之间的交叉交易"

    def compute(
        self,
        lb_nls: pd.DataFrame,
        nlb_ls: pd.DataFrame,
        lb_ls: pd.DataFrame,
        total_volume: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算改进大单交易占比因子。

        Args:
            lb_nls: 大买非大卖成交量，index为日期，columns为股票代码。
            nlb_ls: 非大买大卖成交量，index为日期，columns为股票代码。
            lb_ls: 大买大卖成交量，index为日期，columns为股票代码。
            total_volume: 总成交量，index为日期，columns为股票代码。
            T: 滚动窗口天数，默认20。

        Returns:
            pd.DataFrame: 因子值（T日滚动均值）。
        """
        daily_ratio = (-1 * lb_nls - 1 * nlb_ls + lb_ls) / total_volume
        result = daily_ratio.rolling(window=T, min_periods=1).mean()
        return result


# 使用示例
if __name__ == "__main__":
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=30)
    stocks = ["000001", "000002"]
    lb_nls = pd.DataFrame(np.random.uniform(100, 500, (30, 2)), index=dates, columns=stocks)
    nlb_ls = pd.DataFrame(np.random.uniform(100, 500, (30, 2)), index=dates, columns=stocks)
    lb_ls = pd.DataFrame(np.random.uniform(200, 800, (30, 2)), index=dates, columns=stocks)
    total_volume = pd.DataFrame(np.random.uniform(1000, 5000, (30, 2)), index=dates, columns=stocks)

    factor = ImprovedLargeRatioFactor()
    print(factor)
    print(factor.compute(lb_nls=lb_nls, nlb_ls=nlb_ls, lb_ls=lb_ls, total_volume=total_volume))
