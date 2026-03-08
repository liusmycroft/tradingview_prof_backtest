import pandas as pd

from factors.base import BaseFactor


class BidDepthFactor(BaseFactor):
    """买盘深度因子 (Bid Depth)"""

    name = "BID_DEPTH"
    category = "高频流动性"
    description = "买盘深度的指数移动平均，衡量买方流动性支撑强度"

    def compute(
        self,
        daily_bid_depth: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算买盘深度因子。

        公式: bid_depth = sum(|bv_i * mid_price / (b_i - last_mid + epsilon)|)
        因子值为 T 日 EMA。

        Args:
            daily_bid_depth: 预计算的每日买盘深度均值 (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 买盘深度的 T 日 EMA
        """
        result = daily_bid_depth.ewm(span=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 买盘深度因子衡量买方挂单的流动性支撑强度。对于每个买档价位 b_i，
# 计算其挂单量 bv_i 乘以中间价 mid_price，再除以该档位与最新中间价
# 的距离（加上极小值 epsilon 避免除零），然后对所有买档求和。
#
# 买盘深度越大，说明买方在接近当前价格处有大量挂单，流动性支撑强。
# 因子取 T 日 EMA 以平滑日间波动，同时赋予近期数据更高权重。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.bid_depth import BidDepthFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_bid_depth = pd.DataFrame(
#       np.random.rand(30, 2) * 1e8 + 1e7,
#       index=dates, columns=stocks,
#   )
#
#   factor = BidDepthFactor()
#   result = factor.compute(daily_bid_depth=daily_bid_depth, T=20)
#   print(result.tail())
