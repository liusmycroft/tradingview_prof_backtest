import pandas as pd

from .base import BaseFactor


class MinuteAmountVarianceFactor(BaseFactor):
    """分钟成交额方差因子 (VMA)"""

    name = "VMA"
    category = "高频成交分布"
    description = "分钟成交额方差，刻画日内分钟成交额的离散程度"

    def compute(
        self,
        daily_vma: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算分钟成交额方差因子。

        公式:
            VMA_d = (1/N) * sum((Amount_t - mu)^2)
            mu = (1/N) * sum(Amount_t)
            因子 = T 日滚动均值

        Args:
            daily_vma: 预计算的每日分钟成交额方差
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: VMA 因子值，T 日滚动均值
        """
        result = daily_vma.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 分钟成交额方差刻画了日内分钟成交额的高阶矩特征。方差越大说明
# 日内成交额分布越不均匀，可能存在集中交易时段。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.minute_amount_variance import MinuteAmountVarianceFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_vma = pd.DataFrame(
#       np.random.rand(30, 2) * 1e12,
#       index=dates, columns=stocks,
#   )
#
#   factor = MinuteAmountVarianceFactor()
#   result = factor.compute(daily_vma=daily_vma, T=20)
#   print(result.tail())
