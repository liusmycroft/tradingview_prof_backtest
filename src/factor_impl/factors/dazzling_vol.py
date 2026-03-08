import pandas as pd

from factors.base import BaseFactor


class DazzlingVolFactor(BaseFactor):
    """耀眼波动率因子 (Dazzling Volatility)"""

    name = "DAZZLING_VOL"
    category = "高频波动"
    description = "耀眼波动率：日度耀眼波动率的T日滚动均值与标准差的等权平均"

    def compute(
        self,
        daily_dazzling: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算耀眼波动率因子。

        公式: 0.5 * rolling_mean(daily_dazzling, T) + 0.5 * rolling_std(daily_dazzling, T)

        Args:
            daily_dazzling: 每日预计算的耀眼波动率 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值
        """
        roll_mean = daily_dazzling.rolling(window=T, min_periods=T).mean()
        roll_std = daily_dazzling.rolling(window=T, min_periods=T).std(ddof=1)

        result = 0.5 * roll_mean + 0.5 * roll_std
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 耀眼波动率（Dazzling Volatility）衡量日内价格波动中"引人注目"的部分。
# 日度耀眼波动率由高频数据预计算得到，本因子对其取 T 日滚动均值和标准差
# 的等权平均。均值捕捉波动率水平，标准差捕捉波动率的波动性（vol of vol），
# 两者结合提供更全面的波动率特征。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.dazzling_vol import DazzlingVolFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_dazzling = pd.DataFrame(
#       np.random.uniform(0.01, 0.05, (30, 2)), index=dates, columns=stocks
#   )
#
#   factor = DazzlingVolFactor()
#   result = factor.compute(daily_dazzling=daily_dazzling, T=20)
#   print(result.tail())
