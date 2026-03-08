import pandas as pd

from factors.base import BaseFactor


class AvgOutflowPerTradeRatioFactor(BaseFactor):
    """平均单笔流出金额占比因子 (Average Outflow Per Trade Ratio)。"""

    name = "AVG_OUTFLOW_PER_TRADE_RATIO"
    category = "高频资金流"
    description = "下跌分钟的笔均成交金额与全天笔均成交金额之比的均值"

    def compute(
        self,
        daily_outflow_ratio: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算平均单笔流出金额占比因子。

        公式: (sum(Amt*I(r<0)) / sum(TrdNum*I(r<0))) / (sum(Amt) / sum(TrdNum))
        因子值为 T 日滚动均值。

        Args:
            daily_outflow_ratio: 预计算的每日平均单笔流出金额占比
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_outflow_ratio.rolling(window=T, min_periods=T).mean()
        return result
