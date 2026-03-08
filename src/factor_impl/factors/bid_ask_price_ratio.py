import pandas as pd

from .base import BaseFactor


class BidAskPriceRatioFactor(BaseFactor):
    """买卖委托价格比率因子 (PIR)"""

    name = "PIR"
    category = "高频流动性"
    description = "买卖委托价格比率，从价格角度描述买盘压力"

    def compute(
        self,
        daily_pir: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算买卖委托价格比率因子。

        公式:
            PIR_t = (P_WB - P_WA) / (P_WB + P_WA)
            P_WB(A) = sum(w_i * P_i^B(A)) / sum(w_i)
            w_i = 1 - (i-1)/5, i=1..5
            因子 = T 日 EMA

        Args:
            daily_pir: 预计算的每日 PIR 均值
                (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: PIR 因子值，T 日 EMA
        """
        result = daily_pir.ewm(span=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# PIR 为衰减加权买卖委托价差与其和的比值。权重 w_i = 1-(i-1)/5，
# 第一档权重最大，越远档位权重越小。PIR > 0 表示买盘加权价格高于
# 卖盘加权价格，买盘压力较大。取 T 日 EMA 平滑日间波动。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.bid_ask_price_ratio import BidAskPriceRatioFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_pir = pd.DataFrame(
#       np.random.uniform(-0.01, 0.01, (30, 2)),
#       index=dates, columns=stocks,
#   )
#
#   factor = BidAskPriceRatioFactor()
#   result = factor.compute(daily_pir=daily_pir, T=20)
#   print(result.tail())
