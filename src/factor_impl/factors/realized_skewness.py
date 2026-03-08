import numpy as np
import pandas as pd

from factors.base import BaseFactor


class RealizedSkewnessFactor(BaseFactor):
    """高频已实现偏度 (Realized Skewness) 因子。"""

    name = "REALIZED_SKEWNESS"
    category = "高频收益分布"
    description = "高频已实现偏度的滚动均值，衡量日内收益率分布的不对称性"

    def compute(
        self,
        daily_rskew: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算高频已实现偏度因子。

        Args:
            daily_rskew: 每日已实现偏度，index=日期, columns=股票代码。
                RSkew = sqrt(N) * sum((r_j - r_bar)^3) / RVar^(3/2)
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        result = daily_rskew.rolling(window=T, min_periods=T).mean()
        return result


# ==============================================================================
# 核心思想与原理说明
# ==============================================================================
#
# 高频已实现偏度因子的核心思想：
#
# 1. 对于每个交易日，利用分钟级收益率计算已实现偏度：
#    RSkew = sqrt(N) * sum((r_j - r_bar)^3) / RVar^(3/2)
#    其中 RVar = sum((r_j - r_bar)^2)，N 为分钟数。
#
# 2. 已实现偏度衡量日内收益率分布的不对称性。正偏度意味着存在较多的
#    正向极端收益（右尾厚），负偏度意味着存在较多的负向极端收益。
#
# 3. 对日度已实现偏度取 T 日滚动均值以平滑噪声。研究表明，已实现偏度
#    与未来收益率存在负相关关系：高偏度股票未来收益偏低（彩票效应）。
#
# ==============================================================================
# 简单用法示例
# ==============================================================================
#
# import pandas as pd
# import numpy as np
# from factors.realized_skewness import RealizedSkewnessFactor
#
# dates = pd.date_range("2024-01-01", periods=30, freq="B")
# stocks = ["000001.SZ", "000002.SZ"]
#
# np.random.seed(42)
# daily_rskew = pd.DataFrame(
#     np.random.normal(0, 1, (30, 2)), index=dates, columns=stocks
# )
#
# factor = RealizedSkewnessFactor()
# result = factor.compute(daily_rskew=daily_rskew, T=20)
# print(result.tail())
