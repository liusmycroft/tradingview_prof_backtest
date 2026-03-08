import pandas as pd

from factors.base import BaseFactor


class CPQSIFactor(BaseFactor):
    """收盘委托价差因子 (Closing Price Quoted Spread Intensity - CPQSI)"""

    name = "CPQSI"
    category = "高频流动性"
    description = "收盘委托价差强度，衡量收盘前的交易成本与流动性"

    def compute(
        self,
        closing_ask: pd.DataFrame,
        closing_bid: pd.DataFrame,
        dollar_volume: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算收盘委托价差因子。

        公式:
            CPQS = (Closing_Ask - Closing_Bid) / ((Closing_Ask + Closing_Bid) / 2)
            CPQSI = CPQS / dollar_volume

        Args:
            closing_ask: 收盘卖1价 (index=日期, columns=股票代码)
            closing_bid: 收盘买1价 (index=日期, columns=股票代码)
            dollar_volume: 当日成交额 (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: CPQSI 因子值
        """
        mid = (closing_ask + closing_bid) / 2
        cpqs = (closing_ask - closing_bid) / mid
        result = cpqs / dollar_volume
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 收盘委托价差衡量了收盘前的交易成本。因子值越大，买卖价差越大，
# 流动性越差，未来可获得流动性风险溢价。
#
# CPQS 为相对价差，CPQSI 进一步除以成交额进行标准化，使得因子在
# 不同成交额量级的股票之间具有可比性。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.cpqsi import CPQSIFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=10)
#   stocks = ["000001.SZ", "000002.SZ"]
#   closing_ask = pd.DataFrame(
#       np.random.uniform(10, 20, (10, 2)), index=dates, columns=stocks,
#   )
#   closing_bid = closing_ask - np.random.uniform(0.01, 0.1, (10, 2))
#   dollar_volume = pd.DataFrame(
#       np.random.uniform(1e7, 1e9, (10, 2)), index=dates, columns=stocks,
#   )
#
#   factor = CPQSIFactor()
#   result = factor.compute(
#       closing_ask=closing_ask, closing_bid=closing_bid, dollar_volume=dollar_volume,
#   )
#   print(result.tail())
