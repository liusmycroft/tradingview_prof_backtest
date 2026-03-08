"""模糊金额比因子 (Fog Amount Ratio)

从金额维度统计投资者在模糊性较大时的成交程度。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class FogAmountRatioFactor(BaseFactor):
    """模糊金额比因子"""

    name = "FOG_AMOUNT_RATIO"
    category = "高频成交分布"
    description = "模糊金额比：起雾时刻成交额均值与总体成交额均值之比"

    def compute(
        self,
        daily_fog_amt_mean: pd.DataFrame,
        daily_fog_amt_std: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算模糊金额比因子。

        每月末对最近20天的日模糊金额比求均值和标准差并等权合并。

        公式:
            日模糊金额比 = 雾中金额 / 总体金额
            月度因子 = 0.5 * rolling_mean + 0.5 * rolling_std

        本方法接收预计算的滚动均值和滚动标准差，输出等权合并结果。

        Args:
            daily_fog_amt_mean: 日模糊金额比的滚动均值 (index=日期, columns=股票代码)
            daily_fog_amt_std: 日模糊金额比的滚动标准差 (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 模糊金额比因子值
        """
        result = 0.5 * daily_fog_amt_mean + 0.5 * daily_fog_amt_std
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 模糊金额比因子从金额维度统计了投资者在模糊性较大时的成交程度，
# 衡量了投资者对模糊性的厌恶程度，与未来收益负相关。
# 投资者对波动率模糊性的厌恶心理会使得投资者急于卖出而产生过度反应，
# 未来很可能会补涨。
