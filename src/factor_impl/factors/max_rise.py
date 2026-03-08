import numpy as np
import pandas as pd

from factors.base import BaseFactor


class MaxRiseFactor(BaseFactor):
    """最大涨幅因子 (MAX Rise)"""

    name = "MAX_RISE"
    category = "高频动量反转"
    description = "当天前百分之十涨幅推动的股票具体涨幅，捕捉彩票型股票特征"

    def compute(
        self,
        daily_max_rise: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算最大涨幅因子。

        日内预计算逻辑:
          1. 取t日所有分钟涨跌幅 r_i
          2. 取涨幅最大的前10%分钟涨幅集合 maxprod
          3. MAX_t = prod(1 + r_i) for i in maxprod

        Args:
            daily_max_rise: 预计算的每日最大涨幅 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_max_rise.rolling(window=T, min_periods=1).mean()
        return result

    @staticmethod
    def compute_daily(minute_returns: pd.Series, quantile: float = 0.9) -> float:
        """从单日分钟收益率序列计算最大涨幅。

        Args:
            minute_returns: 单日分钟涨跌幅序列
            quantile: 分位数阈值，默认0.9(前10%)

        Returns:
            float: 最大涨幅 MAX_t
        """
        threshold = minute_returns.quantile(quantile)
        top_returns = minute_returns[minute_returns >= threshold]
        return (1 + top_returns).prod()
