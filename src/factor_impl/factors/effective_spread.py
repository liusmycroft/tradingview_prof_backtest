import pandas as pd

from factors.base import BaseFactor


class EffectiveSpreadFactor(BaseFactor):
    """有效价差因子 (Effective Spread - ES)。"""

    name = "EFFECTIVE_SPREAD"
    category = "高频流动性"
    description = "成交价相对于中间价的相对偏差，反映交易成本"

    def compute(
        self,
        daily_effective_spread: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算有效价差因子。

        公式: ES = 2 * D * (P - Mid) / Mid
              Mid = (Ask + Bid) / 2
        因子值为 T 日 EMA。

        Args:
            daily_effective_spread: 预计算的每日有效价差均值
                (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 有效价差的 T 日 EMA
        """
        result = daily_effective_spread.ewm(span=T, min_periods=1).mean()
        return result
