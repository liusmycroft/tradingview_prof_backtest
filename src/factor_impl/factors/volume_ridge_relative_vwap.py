import numpy as np
import pandas as pd

from factors.base import BaseFactor


class VolumeRidgeRelativeVWAPFactor(BaseFactor):
    """量岭相对加权价格因子 (Volume Ridge Relative VWAP)"""

    name = "VOLUME_RIDGE_RELATIVE_VWAP"
    category = "高频量价相关性"
    description = "量岭的成交量加权价格相对于总VWAP的比值，衡量个人投资者交易的价格水平"

    def compute(
        self,
        daily_ridge_vwap_ratio: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算量岭相对加权价格因子。

        日内预计算逻辑:
          1. 对过去20日同时点成交量按1倍标准差划分喷发/温和成交量
          2. 连续喷发成交量为"量岭"
          3. 量岭相对加权价格 = 量岭VWAP / 全日VWAP

        Args:
            daily_ridge_vwap_ratio: 预计算的每日量岭相对VWAP
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_ridge_vwap_ratio.rolling(window=T, min_periods=1).mean()
        return result
