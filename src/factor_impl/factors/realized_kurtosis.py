import numpy as np
import pandas as pd

from .base import BaseFactor


class RealizedKurtosisFactor(BaseFactor):
    """高频已实现峰度因子 (Realized Kurtosis)。"""

    name = "REALIZED_KURTOSIS"
    category = "高频收益分布"
    description = "高频已实现峰度因子，基于日内收益率的四阶矩衡量尾部风险"

    def compute(
        self,
        daily_rkurt: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算已实现峰度因子。

        Args:
            daily_rkurt: 预计算的每日已实现峰度，index=日期，columns=股票代码。
                         Rkurt = N * sum((r_ij - r_bar)^4) / RVar^2。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期，columns=股票代码。
                          滚动 T 日均值。
        """
        result = daily_rkurt.rolling(window=T, min_periods=T).mean()

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 已实现峰度（Realized Kurtosis）衡量日内收益率分布的尾部厚度。核心公式：
#   Rkurt = N * sum((r_ij - r_bar)^4) / RVar^2
# 其中：
#   - N: 日内收益率观测数
#   - r_ij: 第 j 个日内收益率
#   - r_bar: 日内收益率均值
#   - RVar: 已实现方差
#
# 峰度越高，说明日内收益率分布的尾部越厚，极端收益出现的概率越大。
# 通过 T 日滚动均值平滑，得到稳定的尾部风险度量。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.realized_kurtosis import RealizedKurtosisFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=25)
#   daily_rkurt = pd.DataFrame(
#       np.random.uniform(2, 6, (25, 3)),
#       index=dates,
#       columns=["000001.SZ", "600000.SH", "000002.SZ"],
#   )
#
#   factor = RealizedKurtosisFactor()
#   result = factor.compute(daily_rkurt=daily_rkurt, T=20)
#   print(result)
