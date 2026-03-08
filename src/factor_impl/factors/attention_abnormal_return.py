import numpy as np
import pandas as pd

from factors.base import BaseFactor


class AttentionAbnormalReturnFactor(BaseFactor):
    """基于异常收益的注意力捕捉因子 (Attention Capture via Abnormal Returns)"""

    name = "ATTENTION_ABNORMAL_RETURN"
    category = "行为金融-投资者注意力"
    description = "基于异常收益的注意力捕捉，衡量异常收益吸引投资者注意力的程度"

    def compute(
        self,
        daily_return: pd.DataFrame,
        market_return: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算基于异常收益的注意力捕捉因子。

        公式:
            abnormal_return = daily_return - market_return
            attention = rolling_mean(|abnormal_return|, T)

        异常收益的绝对值越大，越容易吸引投资者注意力。

        Args:
            daily_return: 个股日收益率，index=日期，columns=股票代码。
            market_return: 市场日收益率，index=日期，columns=股票代码
                           (每列值相同，或广播兼容)。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 注意力捕捉因子值 (T日滚动均值)。
        """
        abnormal_return = daily_return - market_return
        abs_abnormal = abnormal_return.abs()
        result = abs_abnormal.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 基于异常收益的注意力捕捉因子衡量个股异常收益（相对于市场收益的偏离）
# 吸引投资者注意力的程度。异常收益绝对值越大，越容易引起投资者关注，
# 从而产生注意力驱动的交易行为。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.attention_abnormal_return import AttentionAbnormalReturnFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#   daily_return = pd.DataFrame(
#       np.random.randn(30, 2) * 0.02, index=dates, columns=stocks
#   )
#   market_return = pd.DataFrame(
#       np.random.randn(30, 1) * 0.01, index=dates, columns=["mkt"]
#   ).values  # broadcast
#   factor = AttentionAbnormalReturnFactor()
#   result = factor.compute(daily_return=daily_return, market_return=market_return)
