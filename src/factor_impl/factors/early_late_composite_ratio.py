import pandas as pd

from factors.base import BaseFactor


class EarlyLateCompositeRatioFactor(BaseFactor):
    """早尾盘复合交易占比因子 (Early-Late Composite Trading Ratio)。"""

    name = "EARLY_LATE_COMPOSITE_RATIO"
    category = "高频资金流"
    description = "非早尾盘双边成交量与早盘双边成交量之差占总成交量的比例"

    def compute(
        self,
        daily_composite_ratio: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算早尾盘复合交易占比因子。

        公式: (非早尾盘买卖成交量 - 早盘买卖成交量) / 总成交量
        因子值为 T 日滚动均值。

        Args:
            daily_composite_ratio: 预计算的每日早尾盘复合交易占比
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_composite_ratio.rolling(window=T, min_periods=T).mean()
        return result
