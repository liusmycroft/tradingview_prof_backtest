import numpy as np
import pandas as pd

from factors.base import BaseFactor


class LagAbsRetAdjAmountCorrFactor(BaseFactor):
    """滞后绝对收益与调整后成交量相关性因子 (Lag Abs Return Adjusted Amount Correlation)"""

    name = "LAG_ABS_RET_ADJ_AMOUNT_CORR"
    category = "高频因子-量价相关性类"
    description = "滞后一期绝对收益率与标准化成交额的相关性，捕捉量价异动"

    def compute(
        self,
        daily_corr: pd.DataFrame,
        T: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """计算滞后绝对收益与调整后成交量相关性因子。

        公式:
            adj_CORA_R = corr(|Ret_{t-1}|, adjAmount_t), Ret_{t-1} != 0
            adjAmount_t = (Amount_t - mu_t) / sigma_t
            月度因子 = mean(最近 T 日的日度因子)

        Args:
            daily_corr: 每日的 corr(|Ret_{t-1}|, adjAmount_t) (index=日期, columns=股票代码)
            T: 均值窗口天数，默认 10

        Returns:
            pd.DataFrame: 月度因子值
        """
        result = daily_corr.rolling(window=T, min_periods=1).mean()
        return result
