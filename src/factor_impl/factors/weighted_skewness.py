import numpy as np
import pandas as pd

from factors.base import BaseFactor


class WeightedSkewnessFactor(BaseFactor):
    """加权偏度因子 (Weighted Skewness)。"""

    name = "WEIGHTED_SKEWNESS"
    category = "高频收益分布"
    description = "成交量加权偏度，刻画日内收益分布的非对称性"

    def compute(
        self,
        daily_weighted_skew: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算加权偏度因子。

        公式: sum(w_t * (close_t - close_bar)^3) / close_sigma^3
              其中 w_t = vol_t / VOL

        本方法接收预计算的每日加权偏度，输出滚动 T 日均值。

        Args:
            daily_weighted_skew: 预计算的每日加权偏度，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 滚动 T 日均值，index=日期，columns=股票代码。
        """
        result = daily_weighted_skew.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 加权偏度因子利用日内成交量对收益率偏度进行加权，捕捉高频收益分布的非对称性。
# 成交量大的时段对偏度的贡献更大，反映了"放量时段"的价格行为特征。
#
# 正偏度意味着收益分布右尾更厚（大涨概率高于大跌），负偏度则相反。
# 实证中，负偏度股票往往具有更高的风险溢价。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.weighted_skewness import WeightedSkewnessFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=30)
#   daily_skew = pd.DataFrame(
#       np.random.randn(30, 2) * 0.5,
#       index=dates,
#       columns=["000001.SZ", "600000.SH"],
#   )
#
#   factor = WeightedSkewnessFactor()
#   result = factor.compute(daily_weighted_skew=daily_skew, T=20)
#   print(result.tail())
