"""反向日内逆转的异常频率因子 (Abnormal Frequency of Negative Reversal, AB_NR)

AB_NR_{i,t} = NR_{i,t} / ((1/12) * sum_{k=0}^{11} NR_{i,t-k})
NR_{i,t} = sum(I(RET_CO > 0) * I(RET_OC < 0)) / T
"""

import pandas as pd

from factors.base import BaseFactor


class AbnormalNegativeReversalFactor(BaseFactor):
    """反向日内逆转的异常频率因子 (AB_NR)"""

    name = "AB_NR"
    category = "高频收益分布"
    description = "反向日内逆转的异常频率，当月频率相对过去 12 个月均值的比值"

    def compute(
        self,
        monthly_nr: pd.DataFrame,
        K: int = 12,
        **kwargs,
    ) -> pd.DataFrame:
        """计算反向日内逆转的异常频率因子。

        Args:
            monthly_nr: 预计算的月度反向日内逆转频率 (index=月末日期, columns=股票代码)。
                NR_{i,t} = 月内隔夜收益>0 且日内收益<0 的交易日占比。
            K: 历史均值回看月数，默认 12。

        Returns:
            pd.DataFrame: 因子值 AB_NR (index=月末日期, columns=股票代码)。
        """
        # 过去 K 个月（含当月）的 NR 均值
        nr_mean = monthly_nr.rolling(window=K, min_periods=K).mean()

        # AB_NR = 当月 NR / 过去 K 月均值
        result = monthly_nr / nr_mean
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 当正的隔夜收益后伴随负的日内收益，即出现了反向日内逆转。
# AB_NR 衡量当月反向日内逆转频率相对过去 12 个月的异常程度。
# 取值越高，隔夜交易者和盘中交易者间"拉锯战"强度增高，价格修正越过度。
