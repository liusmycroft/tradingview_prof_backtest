import pandas as pd

from factors.base import BaseFactor


class LCPSFactor(BaseFactor):
    """卖出反弹偏离小单因子 (Low-volume Cost Price Sell deviation)"""

    name = "LCPS"
    category = "行为金融-遗憾规避"
    description = "卖出反弹偏离小单因子：小单卖出成交价相对收盘价的偏离程度，取T日滚动均值"

    def compute(
        self,
        daily_lcps: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算卖出反弹偏离小单因子。

        公式: LCPS = rolling_mean_T( sum(price_buy * I{price<close} * I{vol<vol_mean}) / close - 1 )
        此处输入已为预计算的每日 LCPS 值。

        Args:
            daily_lcps: 预计算的每日 LCPS 值 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        result = daily_lcps.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 遗憾规避理论认为非理性投资者在做决策时，会倾向于避免产生后悔情绪并
# 追求自满感。对于卖出后价格反弹（当天卖出的股票收盘价高于卖出成本价）
# 的投资者，会避免承认决策失误而存在惜售心理。卖出反弹偏离因子从成交
# 价格偏离维度来量化上述心理，仅统计小单（成交量低于当日均值的交易）。
# 因子值越高，卖出价格越低，成交价格偏离程度越大，未来买回动力较弱，
# 预期收益较低。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.lcps import LCPSFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_lcps = pd.DataFrame(
#       np.random.randn(30, 2) * 0.01, index=dates, columns=stocks,
#   )
#
#   factor = LCPSFactor()
#   result = factor.compute(daily_lcps=daily_lcps, T=20)
#   print(result.tail())
