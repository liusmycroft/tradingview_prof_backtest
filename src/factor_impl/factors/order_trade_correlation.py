import pandas as pd

from factors.base import BaseFactor


class OrderTradeCorrelationFactor(BaseFactor):
    """委托成交相关性因子 (Order-Trade Correlation)。"""

    name = "ORDER_TRADE_CORRELATION"
    category = "高频量价"
    description = "高频收益与净委买变化率的相关系数，刻画买入意愿与股价走势的关系"

    def compute(
        self,
        daily_order_trade_corr: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算委托成交相关性因子。

        公式: corr(r_{T,t}, netBid_{T,t})
        因子值为 T 日滚动均值。

        Args:
            daily_order_trade_corr: 预计算的每日委托成交相关性
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_order_trade_corr.rolling(window=T, min_periods=T).mean()
        return result
