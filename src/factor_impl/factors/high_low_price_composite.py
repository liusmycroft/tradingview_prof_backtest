"""高低价复合交易占比因子 (High-Low Price Composite Trading Ratio)

从订单委托价格高低维度对逐笔订单进行划分和归整。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class HighLowPriceCompositeRatioFactor(BaseFactor):
    """高低价复合交易占比因子"""

    name = "HIGH_LOW_PRICE_COMPOSITE_RATIO"
    category = "高频资金流"
    description = "高低价复合交易占比：非高低价买卖成交量与高价买卖成交量之差占总成交量比"

    def compute(
        self,
        non_extreme_volume: pd.DataFrame,
        high_price_volume: pd.DataFrame,
        total_volume: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算高低价复合交易占比因子。

        公式:
            高低价复合交易占比 = (非高低价买单&非高低价卖单成交量
                                - 高价买单&高价卖单成交量) / 总成交量

        Args:
            non_extreme_volume: 非高低价买单&非高低价卖单的成交量
                                (index=日期, columns=股票代码)
            high_price_volume: 高价买单&高价卖单的成交量
                               (index=日期, columns=股票代码)
            total_volume: 总成交量 (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 因子值
        """
        result = (non_extreme_volume - high_price_volume) / total_volume
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 高低价复合交易占比因子从订单委托价格高低维度对逐笔订单进行划分。
# "高价买&高价卖"成交量占比越高，未来反转效应越明显；
# "非高低价买+非高低价卖"成交量占比越高，未来动量效应越强。
# 复合因子为两者的复合，与未来收益正相关。
