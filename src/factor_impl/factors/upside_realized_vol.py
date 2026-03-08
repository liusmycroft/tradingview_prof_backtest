import pandas as pd

from factors.base import BaseFactor


class UpsideRealizedVolFactor(BaseFactor):
    """上行已实现波动率因子 (Upside Realized Volatility - RS+)"""

    name = "RS_PLUS"
    category = "高频波动跳跃"
    description = "上行已实现波动率：正收益率平方和，取T日滚动均值"

    def compute(
        self,
        daily_rs_plus: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算上行已实现波动率因子。

        公式:
            RS+_t = sum(r_{t,i}^2 * I(r_{t,i} >= 0))
            因子值为 T 日滚动均值。

        Args:
            daily_rs_plus: 预计算的每日上行已实现波动率 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 上行已实现波动率的 T 日滚动均值
        """
        result = daily_rs_plus.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 上行已实现波动率对应资产价格波动中的上行波动部分。
# RS+_t = sum(r^2 * I(r>=0))，仅累加非负收益率的平方。
#
# 上行波动率较高的股票短期股价出现大幅拉升，未来更有可能补跌，
# 收益出现反转。因子取T日滚动均值以平滑日间波动。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.upside_realized_vol import UpsideRealizedVolFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#   np.random.seed(42)
#   daily_rs_plus = pd.DataFrame(
#       np.random.uniform(0.0001, 0.005, (30, 2)),
#       index=dates, columns=stocks,
#   )
#
#   factor = UpsideRealizedVolFactor()
#   result = factor.compute(daily_rs_plus=daily_rs_plus, T=20)
#   print(result.tail())
