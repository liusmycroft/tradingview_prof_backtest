"""
谷岭加权价格比因子 (Valley-Ridge Weighted Price Ratio Factor)

衡量谷值与岭值成交量加权价格的比值，反映价格在波谷与波峰的相对强弱。
"""

import pandas as pd
from factors.base import BaseFactor


class ValleyRidgeRatioFactor(BaseFactor):
    """谷岭加权价格比因子

    公式: rolling_mean(valley_vwap / ridge_vwap, T)
    """

    name = "VALLEY_RIDGE_RATIO"
    category = "高频量价"
    description = "谷岭加权价格比因子，衡量波谷与波峰成交量加权价格的比值"

    def compute(
        self,
        valley_vwap: pd.DataFrame,
        ridge_vwap: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算谷岭加权价格比因子。

        Args:
            valley_vwap: 每日波谷成交量加权价格，index为日期，columns为股票代码。
            ridge_vwap: 每日波峰成交量加权价格，index为日期，columns为股票代码。
            T: 滚动窗口天数，默认20。

        Returns:
            pd.DataFrame: 谷岭加权价格比因子值。
        """
        daily_ratio = valley_vwap / ridge_vwap
        result = daily_ratio.rolling(window=T, min_periods=1).mean()
        return result


# 使用示例
if __name__ == "__main__":
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=30)
    stocks = ["000001", "000002"]
    valley_vwap = pd.DataFrame(np.random.uniform(9, 11, (30, 2)), index=dates, columns=stocks)
    ridge_vwap = pd.DataFrame(np.random.uniform(10, 12, (30, 2)), index=dates, columns=stocks)

    factor = ValleyRidgeRatioFactor()
    print(factor)
    print(factor.compute(valley_vwap=valley_vwap, ridge_vwap=ridge_vwap))
