import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .base import BaseFactor


class RidgeGapSkewFactor(BaseFactor):
    """量岭间隔偏度因子 (Volume Ridge Gap Skewness)。"""

    name = "RIDGE_GAP_SKEW"
    category = "高频成交分布"
    description = "量岭间隔偏度因子，衡量日内量岭之间时间间隔分布的偏度"

    def compute(
        self,
        daily_ridge_gap_skew: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算量岭间隔偏度因子。

        Args:
            daily_ridge_gap_skew: 预计算的每日量岭间隔偏度，
                                   index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期，columns=股票代码。
                          T 日滚动均值。
        """
        result = daily_ridge_gap_skew.rolling(window=T, min_periods=T).mean()

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 量岭间隔偏度因子衡量了日内量岭（连续喷发成交量）之间时间间隔分布的偏度。
# 核心逻辑：
#   1. 识别日内"量岭"（连续喷发成交量）时刻。
#   2. 计算同日前后两个量岭之间的时间间隔。
#   3. 对间隔分布计算偏度，取 T 日均值。
#
# 经济直觉：个人投资者通常是跟风交易，交易频率高，间隔时间短，
# 时间间隔分布的异常值相对较少，主要集中在左侧的短间隔区域，
# 因此量岭间隔偏度与未来收益负相关。
