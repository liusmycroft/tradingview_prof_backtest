import numpy as np
import pandas as pd

from factors.base import BaseFactor


class PeakIntervalKurtosisFactor(BaseFactor):
    """量峰间隔峰度因子 (Peak Interval Kurtosis)"""

    name = "PEAK_INTERVAL_KURTOSIS"
    category = "高频因子-成交分布类"
    description = "量峰之间时间间隔分布的峰度，衡量知情交易者参与交易的时间间隔分布"

    def compute(
        self,
        peak_intervals: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算量峰间隔峰度因子。

        公式:
            1. 同时点成交量 > 均值+1倍标准差 => 喷发成交量
            2. 孤立喷发 = 量峰
            3. 计算同日前后两个量峰之间的时间间隔分布的峰度

        Args:
            peak_intervals: 预计算的量峰间隔峰度 (index=日期, columns=股票代码)
                           每个值为当日量峰间隔序列的峰度

        Returns:
            pd.DataFrame: 量峰间隔峰度因子值 (20日均值)
        """
        T = kwargs.get("T", 20)
        result = peak_intervals.rolling(window=T, min_periods=1).mean()
        return result
