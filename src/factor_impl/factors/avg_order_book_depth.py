import pandas as pd

from factors.base import BaseFactor


class AvgOrderBookDepthFactor(BaseFactor):
    """盘口平均深度因子 (Average Order Book Depth)。"""

    name = "AVG_ORDER_BOOK_DEPTH"
    category = "高频流动性"
    description = "买一量与卖一量的均值，反映市场整体挂单深度"

    def compute(
        self,
        daily_avg_depth: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算盘口平均深度因子。

        公式: avg_depth = (av1 + bv1) / 2
        因子值为 T 日 EMA。

        Args:
            daily_avg_depth: 预计算的每日盘口平均深度均值
                (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 盘口平均深度的 T 日 EMA
        """
        result = daily_avg_depth.ewm(span=T, min_periods=1).mean()
        return result
