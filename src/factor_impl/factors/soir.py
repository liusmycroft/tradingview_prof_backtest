import pandas as pd

from .base import BaseFactor


class SOIRFactor(BaseFactor):
    """逐档订单失衡率因子"""

    name = "SOIR"
    category = "高频流动性"
    description = "逐档订单失衡率因子，加权衡量盘口各档买卖委托量的不均衡程度"

    def compute(
        self,
        daily_soir: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算逐档订单失衡率因子。

        公式:
            SOIR_i = (V_i^B - V_i^A) / (V_i^B + V_i^A)
            w_i = 1 - (i-1)/5,  i=1,...,5
            SOIR = sum(w_i * SOIR_i) / sum(w_i)
            因子 = T 日 EMA

        Args:
            daily_soir: 预计算的每日 SOIR 值
                (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 逐档订单失衡率因子值
        """
        result = daily_soir.ewm(span=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# SOIR 是对各档订单失衡率的加权：
#   SOIR_i = (V_i^B - V_i^A) / (V_i^B + V_i^A)
#   w_i = 1 - (i-1)/5，越靠近盘口的档位权重越大
#   SOIR = sum(w_i * SOIR_i) / sum(w_i)
#
# 加权方式可以避免某一档订单量过大对总体比率的影响。
# SOIR 为正说明市场买压大于卖压，未来价格趋向上涨，
# 且 SOIR 值越大上涨概率越高，反之亦然。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.soir import SOIRFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_soir = pd.DataFrame(
#       np.random.randn(30, 2) * 0.3,
#       index=dates, columns=stocks,
#   )
#
#   factor = SOIRFactor()
#   result = factor.compute(daily_soir=daily_soir, T=20)
#   print(result.tail())
