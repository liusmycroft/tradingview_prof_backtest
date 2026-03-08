import numpy as np
import pandas as pd

from factors.base import BaseFactor


class WeightedUpperShadowFreqFactor(BaseFactor):
    """加权上引线频率因子 (Weighted Upper Shadow Frequency)"""

    name = "WEIGHTED_UPPER_SHADOW_FREQ"
    category = "量价因子改进"
    description = "加权上影线频率，衡量过去一段时间上影线长度大于阈值的天数加权之和"

    def compute(
        self,
        high: pd.DataFrame,
        open_price: pd.DataFrame,
        close: pd.DataFrame,
        prev_close: pd.DataFrame,
        M: int = 40,
        u: float = 0.01,
        half_life: float = 10.0,
        **kwargs,
    ) -> pd.DataFrame:
        """计算加权上影线频率因子。

        公式:
            上影线_{i,j} = (High - max(Open, Close)) / PrevClose
            w_j = 0.5^((t-j)/lambda)
            factor = sum(w_j * I(上影线 > u)) / M

        Args:
            high: 最高价 (index=日期, columns=股票代码)
            open_price: 开盘价
            close: 收盘价
            prev_close: 前收盘价
            M: 回望期天数，默认 40
            u: 上影线长度阈值，默认 0.01 (1%)
            half_life: 衰减半衰期，默认 10

        Returns:
            pd.DataFrame: 加权上影线频率因子值
        """
        # 计算上影线
        upper_shadow = (high - np.maximum(open_price, close)) / prev_close

        # 判断是否超过阈值
        indicator = (upper_shadow > u).astype(float)

        # 构建衰减权重并滚动加权求和
        weights = np.array([0.5 ** ((M - 1 - j) / half_life) for j in range(M)])

        def _weighted_freq(arr):
            valid = arr[~np.isnan(arr)]
            n = len(valid)
            if n == 0:
                return np.nan
            w = weights[-n:]
            return np.dot(w, valid) / n

        result = indicator.rolling(window=M, min_periods=1).apply(
            _weighted_freq, raw=True
        )
        return result
