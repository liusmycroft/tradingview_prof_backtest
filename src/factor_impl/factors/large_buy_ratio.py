import pandas as pd

from factors.base import BaseFactor


class LargeBuyRatioFactor(BaseFactor):
    """大单买入占比因子 (Large Buy Order Ratio)"""

    name = "LARGE_BUY_RATIO"
    category = "高频资金流"
    description = "大单买入占比：大单买入成交额占总成交额的比例，取T日滚动均值"

    def compute(
        self,
        large_buy_amount: pd.DataFrame,
        total_amount: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算大单买入占比因子。

        公式: (1/T) * sum( large_buy_amount / total_amount )

        Args:
            large_buy_amount: 每日大单买入成交额 (index=日期, columns=股票代码)
            total_amount: 每日总成交额 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        daily_ratio = large_buy_amount / total_amount
        result = daily_ratio.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 大单类因子刻画了大资金的交易行为。大资金往往具有信息优势，一般受到
# 大资金关注的股票，未来通常具有更好的表现。因子计算每日大单买入成交额
# 占总成交额的比例，然后取 T 日滚动均值以平滑日间波动。大单的界定基于
# 逐笔成交数据中买卖单成交额对数调整后的"均值+1倍标准差"作为阈值。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.large_buy_ratio import LargeBuyRatioFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   large_buy = pd.DataFrame(
#       np.random.rand(30, 2) * 1e6, index=dates, columns=stocks,
#   )
#   total = pd.DataFrame(
#       np.random.rand(30, 2) * 1e7 + 1e6, index=dates, columns=stocks,
#   )
#
#   factor = LargeBuyRatioFactor()
#   result = factor.compute(large_buy_amount=large_buy, total_amount=total, T=20)
#   print(result.tail())
