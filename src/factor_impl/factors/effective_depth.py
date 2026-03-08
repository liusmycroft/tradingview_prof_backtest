import numpy as np
import pandas as pd

from factors.base import BaseFactor


class EffectiveDepthFactor(BaseFactor):
    """有效深度因子 (Effective Depth)。"""

    name = "EFFECTIVE_DEPTH"
    category = "高频流动性"
    description = "买一量与卖一量的最小值，衡量市场实际有效深度"

    def compute(
        self,
        daily_effective_depth: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算有效深度因子。

        公式: effective_depth = min(av1, bv1)
        因子值为 T 日 EMA。

        Args:
            daily_effective_depth: 预计算的每日有效深度均值
                (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 有效深度的 T 日 EMA
        """
        result = daily_effective_depth.ewm(span=T, min_periods=1).mean()
        return result
