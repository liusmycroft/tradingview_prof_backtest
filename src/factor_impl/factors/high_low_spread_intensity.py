import numpy as np
import pandas as pd

from factors.base import BaseFactor


class HighLowSpreadIntensityFactor(BaseFactor):
    """基于高低价的买卖价差因子 (High-Low Spread Intensity, HLI)"""

    name = "HLI"
    category = "高频流动性"
    description = "基于高低价估计的买卖价差除以成交额，衡量流动性强度"

    def compute(
        self,
        daily_hli: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算基于高低价的买卖价差因子。

        HLI 的日度值由高频数据预计算（涉及两日滚动窗口的 beta/gamma），
        此处接收预计算的日度 HLI 值，取 T 日滚动均值。

        Args:
            daily_hli: 预计算的每日 HLI 值 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: HLI 的 T 日滚动均值
        """
        result = daily_hli.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 每日的最高价几乎总是由买方发起的交易，而每日的最低价几乎总是由
# 卖方发起的交易。高低价格比反映了股票价格的真实方差和买卖价差，
# 所以可以此估计买卖价差。
#
# HLI = HL / $vol，其中 HL 由 Corwin-Schultz (2012) 方法从两日
# 高低价估计得到，$vol 为当日成交额。买卖价差越大、成交额越小，
# HLI 越大，流动性越弱，未来可获得流动性风险溢价。
