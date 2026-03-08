import numpy as np
import pandas as pd

from factors.base import BaseFactor


class ConsistentSellTradeFactor(BaseFactor):
    """一致卖出交易因子 (Consistent Sell Trade - NCV)"""

    name = "CONSISTENT_SELL_TRADE"
    category = "高频因子-成交分布类"
    description = "下跌实体K线成交量占比的移动平均，捕捉集体一致卖出行为"

    def compute(
        self,
        daily_consistent_sell_ratio: pd.DataFrame,
        d: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算一致卖出交易因子。

        公式:
            实体K线: |close - open| <= alpha * |high - low|
            NCV_t = (1/d) * sum(ConsistentVolume_fall / Volume)

        Args:
            daily_consistent_sell_ratio: 每日下跌实体K线成交量占总成交量比
                                        (index=日期, columns=股票代码)
            d: 移动平均周期，默认 20

        Returns:
            pd.DataFrame: 一致卖出交易因子值
        """
        result = daily_consistent_sell_ratio.rolling(window=d, min_periods=1).mean()
        return result
