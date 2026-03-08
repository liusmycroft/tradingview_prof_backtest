import numpy as np
import pandas as pd

from factors.base import BaseFactor


class WeightedCloseRatioFactor(BaseFactor):
    """加权收盘价比因子 (Weighted Close Price Ratio)。"""

    name = "WEIGHTED_CLOSE_RATIO"
    category = "高频量价"
    description = "成交量加权收盘价与简单均价之比的滚动均值，捕捉量价关系"

    def compute(
        self,
        vol_weighted_close: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算加权收盘价比因子。

        Args:
            vol_weighted_close: 每日成交量加权收盘价 / 简单均价，
                index=日期, columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        result = vol_weighted_close.rolling(window=T, min_periods=T).mean()
        return result


# ==============================================================================
# 核心思想与原理说明
# ==============================================================================
#
# 加权收盘价比因子的核心思想：
#
# 1. 对于每个交易日，计算分钟级别的成交量加权收盘价与简单均价之比：
#    ratio = sum(vol_t / VOL * close_t) / mean(close_t)
#    其中 vol_t 是分钟成交量，VOL 是全天总成交量，close_t 是分钟收盘价。
#
# 2. 该比值反映了大资金在价格高位还是低位集中交易。如果比值 > 1，说明
#    成交量集中在价格较高的时段，可能暗示主力资金在高位出货。
#
# 3. 对日度比值取 T 日滚动均值以平滑噪声，得到最终因子值。
#
# ==============================================================================
# 简单用法示例
# ==============================================================================
#
# import pandas as pd
# import numpy as np
# from factors.weighted_close_ratio import WeightedCloseRatioFactor
#
# dates = pd.date_range("2024-01-01", periods=30, freq="B")
# stocks = ["000001.SZ", "000002.SZ"]
#
# np.random.seed(42)
# vol_wc = pd.DataFrame(
#     np.random.uniform(0.98, 1.02, (30, 2)), index=dates, columns=stocks
# )
#
# factor = WeightedCloseRatioFactor()
# result = factor.compute(vol_weighted_close=vol_wc, T=20)
# print(result.tail())
