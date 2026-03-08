import numpy as np
import pandas as pd

from factors.base import BaseFactor


class ValleyWeightedPriceQuantileFactor(BaseFactor):
    """量谷加权价格分位点因子 (Valley Weighted Price Quantile)"""

    name = "VALLEY_WEIGHTED_PRICE_QUANTILE"
    category = "高频因子-成交分布类"
    description = "量谷成交量加权价格的相对分位点，衡量交易低迷时点的价格水平"

    def compute(
        self,
        valley_vwap: pd.DataFrame,
        high_price: pd.DataFrame,
        low_price: pd.DataFrame,
        prev_close: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算量谷加权价格分位点因子。

        公式:
            分位点_d = (VWAP_valley - min(Low, PrevClose)) /
                       (max(High, PrevClose) - min(Low, PrevClose))
            因子 = mean(分位点, T日)

        Args:
            valley_vwap: 量谷成交量加权价格 (index=日期, columns=股票代码)
            high_price: 日内最高价
            low_price: 日内最低价
            prev_close: 昨日收盘价
            T: 均值窗口，默认 20

        Returns:
            pd.DataFrame: 量谷加权价格分位点因子值
        """
        range_low = pd.DataFrame(
            np.minimum(low_price.values, prev_close.values),
            index=low_price.index, columns=low_price.columns,
        )
        range_high = pd.DataFrame(
            np.maximum(high_price.values, prev_close.values),
            index=high_price.index, columns=high_price.columns,
        )
        range_width = range_high - range_low
        quantile = (valley_vwap - range_low) / range_width.replace(0, np.nan)

        result = quantile.rolling(window=T, min_periods=1).mean()
        return result
