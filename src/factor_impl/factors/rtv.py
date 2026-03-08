import numpy as np
import pandas as pd

from factors.base import BaseFactor


class RTVFactor(BaseFactor):
    """已实现三幂次变差 (RTV) — Realized Tripower Variation"""

    name = "RTV"
    category = "高频波动跳跃"
    description = "基于日内收益率绝对值三阶乘积的已实现三幂次变差，对跳跃具有稳健性"

    def compute(
        self,
        daily_rtv: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算已实现三幂次变差因子。

        公式: RTV_t = mu_{2/3}^{-3} * (n/(n-2)) *
               sum(|r_{j-2}|^{2/3} * |r_{j-1}|^{2/3} * |r_j|^{2/3})
        其中 mu_{2/3} = E[|Z|^{2/3}], Z ~ N(0,1)
        因子值为 T 日 EMA。

        Args:
            daily_rtv: 预计算的每日已实现三幂次变差 (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 已实现三幂次变差的 T 日 EMA
        """
        result = daily_rtv.ewm(span=T, min_periods=1).mean()
        return result
