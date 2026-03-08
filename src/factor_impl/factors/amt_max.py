"""个股收益最高时的领先成交量异动因子 (AMT_MAX)

衡量个股收益率最高的分钟对应的前一分钟成交量相对于日均成交量的异动程度。
"""

import pandas as pd

from factors.base import BaseFactor


class AmtMaxFactor(BaseFactor):
    """个股收益最高时的领先成交量异动因子 (AMT_MAX)"""

    name = "AMT_MAX"
    category = "高频动量反转"
    description = "个股收益最高时的领先成交量异动：收益率最高分钟的前一分钟成交量与日均成交量之比，取T日滚动均值"

    def compute(
        self,
        daily_amt_max: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算领先成交量异动因子。

        公式: factor = (1/T) * sum(amt_max)
        其中 amt_max = volume_{t_max - 1} / avg_volume

        Args:
            daily_amt_max: 每日预计算的领先成交量异动值
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        result = daily_amt_max.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# AMT_MAX 因子捕捉了在个股收益率最高的分钟之前，成交量是否出现异常
# 放大。如果在价格大幅上涨前一分钟成交量显著放大，可能暗示知情交易者
# 提前布局。该因子值越高，说明价格上涨前的成交量异动越明显，可能存在
# 信息提前泄露或知情交易行为，未来收益可能出现反转。
