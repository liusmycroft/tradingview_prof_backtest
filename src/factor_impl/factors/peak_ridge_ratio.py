"""
峰岭成交比因子 (Peak-Ridge Transaction Ratio Factor)

衡量波峰与波谷成交量的比值，反映价格高位与低位的成交活跃度差异。
"""

import pandas as pd
from factors.base import BaseFactor


class PeakRidgeRatioFactor(BaseFactor):
    """峰岭成交比因子

    公式: rolling_sum(peak_volume, T) / rolling_sum(valley_volume, T)
    """

    name = "PEAK_RIDGE_RATIO"
    category = "高频成交分布"
    description = "峰岭成交比因子，衡量波峰与波谷成交量的比值"

    def compute(
        self,
        peak_volume: pd.DataFrame,
        valley_volume: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算峰岭成交比因子。

        Args:
            peak_volume: 每日波峰成交量，index为日期，columns为股票代码。
            valley_volume: 每日波谷成交量，index为日期，columns为股票代码。
            T: 滚动窗口天数，默认20。

        Returns:
            pd.DataFrame: 因子值（T日滚动峰成交量之和 / T日滚动谷成交量之和）。
        """
        peak_sum = peak_volume.rolling(window=T, min_periods=1).sum()
        valley_sum = valley_volume.rolling(window=T, min_periods=1).sum()
        result = peak_sum / valley_sum
        return result


# 使用示例
if __name__ == "__main__":
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=30)
    stocks = ["000001", "000002"]
    peak_volume = pd.DataFrame(np.random.uniform(1000, 5000, (30, 2)), index=dates, columns=stocks)
    valley_volume = pd.DataFrame(np.random.uniform(500, 3000, (30, 2)), index=dates, columns=stocks)

    factor = PeakRidgeRatioFactor()
    print(factor)
    print(factor.compute(peak_volume=peak_volume, valley_volume=valley_volume))
