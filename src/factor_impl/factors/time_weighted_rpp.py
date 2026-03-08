import numpy as np
import pandas as pd

from factors.base import BaseFactor


class TimeWeightedRelativePricePositionFactor(BaseFactor):
    """时间加权平均的股票相对价格位置因子 (ARPP)"""

    name = "TIME_WEIGHTED_RPP"
    category = "高频因子-收益分布类"
    description = "股票在价格相对高位停留的时间长短，TWAP相对区间最高最低价的分数"

    def compute(
        self,
        twap: pd.DataFrame,
        high_price: pd.DataFrame,
        low_price: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算时间加权平均的股票相对价格位置因子。

        公式:
            RPP_{i,t} = (P_{i,t} - L_i) / (H_i - L_i)
            ARPP_i = (TWAP_i - L_i) / (H_i - L_i)

        Args:
            twap: 时间加权平均价格 (index=日期, columns=股票代码)
            high_price: 区间最高价
            low_price: 区间最低价

        Returns:
            pd.DataFrame: ARPP 因子值
        """
        range_width = high_price - low_price
        result = (twap - low_price) / range_width.replace(0, np.nan)
        return result
