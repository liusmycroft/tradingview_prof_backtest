import numpy as np
import pandas as pd
from scipy.stats import norm

from factors.base import BaseFactor


class ConfidenceNormalActiveBuyFactor(BaseFactor):
    """置信正态分布主动占比因子 (Confidence Normal Distribution Active Buy Ratio)"""

    name = "CONFIDENCE_NORMAL_ACTIVE_BUY"
    category = "高频资金流"
    description = "基于正态分布CDF估计主动买入金额占比，提高主买卖额估计准确度"

    def compute(
        self,
        minute_returns: pd.DataFrame,
        minute_amount: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算置信正态分布主动占比因子。

        公式:
            主动买入金额_i = Amount_i * N(ret_i / 0.1 * 1.96)
            因子 = sum(主动买入金额_i) / sum(Amount_i)

        Args:
            minute_returns: 分钟收益率 (index=日期, columns=分钟编号)
                每行为一天的分钟收益率序列
            minute_amount: 分钟成交额 (index=日期, columns=分钟编号)

        Returns:
            pd.DataFrame: 因子值 (index=日期, columns=股票代码)
                单股票时 columns 为单列
        """
        # N(ret / 0.1 * 1.96)
        z_scores = minute_returns / 0.1 * 1.96
        cdf_values = pd.DataFrame(
            norm.cdf(z_scores.values),
            index=minute_returns.index,
            columns=minute_returns.columns,
        )

        active_buy_amount = minute_amount * cdf_values
        total_amount = minute_amount.sum(axis=1)
        active_buy_total = active_buy_amount.sum(axis=1)

        result = active_buy_total / total_amount
        return result.to_frame(name="factor")
