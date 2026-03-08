import numpy as np
import pandas as pd

from factors.base import BaseFactor


class AbnormalReturnAvgFactor(BaseFactor):
    """平均异常日收益率 ABNRETAVG 因子"""

    name = "ABNRETAVG"
    category = "行为金融-投资者注意力"
    description = "平均异常日收益率的平方均值，衡量异常收益引发的投资者注意力"

    def compute(
        self,
        daily_return: pd.DataFrame,
        market_return: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算平均异常日收益率 ABNRETAVG 因子。

        公式:
            abnormal_return = daily_return - market_return
            ABNRETAVG = rolling_mean(abnormal_return^2, T)

        Args:
            daily_return: 个股日收益率，index=日期，columns=股票代码。
            market_return: 市场日收益率，index=日期，columns=股票代码
                           (每列值相同，或广播兼容)。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: ABNRETAVG 因子值 (T日滚动均值)。
        """
        abnormal_return = daily_return - market_return
        squared = abnormal_return ** 2
        result = squared.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# ABNRETAVG 因子计算异常收益率平方的滚动均值 (mean squared abnormal return)。
# 异常收益率平方值越大，说明个股偏离市场的程度越大，越容易吸引投资者注意力。
# 与简单的异常收益绝对值不同，平方形式对极端异常收益赋予更高权重。
#
# 【使用示例】
#
#   from factors.abnormal_return_avg import AbnormalReturnAvgFactor
#   factor = AbnormalReturnAvgFactor()
#   result = factor.compute(
#       daily_return=ret_df, market_return=mkt_df, T=20
#   )
