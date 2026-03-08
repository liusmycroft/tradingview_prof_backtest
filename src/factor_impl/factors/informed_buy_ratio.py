import numpy as np
import pandas as pd

from .base import BaseFactor


class InformedBuyRatioFactor(BaseFactor):
    """知情主买占比因子 (Informed Main Buy Ratio)。"""

    name = "INFORMED_BUY_RATIO"
    category = "高频资金流"
    description = "知情主买占比因子，基于知情交易者主动买入金额占总成交金额的比例"

    def compute(
        self,
        informed_buy_amount: pd.DataFrame,
        total_amount: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算知情主买占比因子。

        Args:
            informed_buy_amount: 知情交易者主动买入金额，
                                 index=日期，columns=股票代码。
            total_amount: 总成交金额，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期，columns=股票代码。
                          滚动 T 日均值。
        """
        # 每日知情主买占比
        ratio = informed_buy_amount / total_amount

        # 滚动 T 日均值: (1/T) * sum(ratio)
        result = ratio.rolling(window=T, min_periods=T).mean()

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 知情主买占比因子衡量知情交易者（通常是机构投资者）主动买入行为的强度。
# 核心公式：
#   ratio_t = informed_buy_amount_t / total_amount_t
#   factor = (1/T) * sum(ratio_{t-T+1:t})
#
# 经济直觉：知情交易者拥有信息优势，其主动买入行为往往预示着未来股价上涨。
# 通过 T 日滚动均值平滑，过滤掉单日噪声，得到稳定的知情资金流入信号。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.informed_buy_ratio import InformedBuyRatioFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=25)
#   informed_buy = pd.DataFrame(
#       np.random.uniform(1e6, 5e6, (25, 2)),
#       index=dates,
#       columns=["000001.SZ", "600000.SH"],
#   )
#   total = pd.DataFrame(
#       np.random.uniform(1e7, 5e7, (25, 2)),
#       index=dates,
#       columns=["000001.SZ", "600000.SH"],
#   )
#
#   factor = InformedBuyRatioFactor()
#   result = factor.compute(
#       informed_buy_amount=informed_buy, total_amount=total, T=20
#   )
#   print(result)
