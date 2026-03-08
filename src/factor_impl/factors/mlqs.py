import numpy as np
import pandas as pd

from factors.base import BaseFactor


class MLQSFactor(BaseFactor):
    """多层订单斜率 (MLQS) — Multi-Level Log Quote Slope"""

    name = "MLQS"
    category = "高频流动性"
    description = "多层订单簿对数报价斜率，衡量订单簿深度方向上的价格梯度"

    def compute(
        self,
        daily_mlqs: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算多层订单斜率因子。

        公式: MLQS = mean(log(ask_i/bid_i) / (i)) 对多档报价求均值
        因子值为 T 日 EMA。

        Args:
            daily_mlqs: 预计算的每日多层对数报价斜率 (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 多层订单斜率的 T 日 EMA
        """
        result = daily_mlqs.ewm(span=T, min_periods=1).mean()
        return result
