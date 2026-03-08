import numpy as np
import pandas as pd

from factors.base import BaseFactor


class WeightedLowerShadowFreqFactor(BaseFactor):
    """加权下引线频率因子 (Weighted Lower Shadow Frequency)"""

    name = "WEIGHTED_LOWER_SHADOW_FREQ"
    category = "量价因子改进"
    description = "过去一段时间下影线长度大于阈值的天数加权之和，衡量下影线出现频率"

    def compute(
        self,
        open_price: pd.DataFrame,
        close_price: pd.DataFrame,
        low_price: pd.DataFrame,
        prev_close: pd.DataFrame,
        M: int = 40,
        u: float = 0.01,
        half_life: float = 20.0,
        **kwargs,
    ) -> pd.DataFrame:
        """计算加权下引线频率因子。

        公式:
            下影线 = (min(Open, Close) - Low) / PrevClose
            w_j = 0.5^((t-j)/lambda)
            WeightedLowerShadowFreq = sum(w_j * I(下影线 > u)) / M

        Args:
            open_price: 开盘价 (index=日期, columns=股票代码)
            close_price: 收盘价
            low_price: 最低价
            prev_close: 前收盘价
            M: 回望期天数，默认 40
            u: 下影线长度阈值，默认 0.01
            half_life: 半衰期，默认 20

        Returns:
            pd.DataFrame: 加权下引线频率因子值
        """
        body_low = pd.DataFrame(
            np.minimum(open_price.values, close_price.values),
            index=open_price.index,
            columns=open_price.columns,
        )
        lower_shadow = (body_low - low_price) / prev_close
        indicator = (lower_shadow > u).astype(float)

        result = pd.DataFrame(np.nan, index=open_price.index, columns=open_price.columns)

        for i in range(M - 1, len(open_price)):
            window = indicator.iloc[i - M + 1 : i + 1]
            # 时间衰退权重: 最近的 j=M-1 权重最大
            distances = np.arange(M - 1, -1, -1, dtype=float)
            weights = np.power(0.5, distances / half_life)
            weighted_sum = window.multiply(weights, axis=0).sum()
            result.iloc[i] = weighted_sum / M

        return result
