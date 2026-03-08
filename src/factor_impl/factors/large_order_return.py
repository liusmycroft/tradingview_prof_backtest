"""
大单推动涨幅因子 (Large Order Driven Return Factor)

衡量大单交易推动的累计收益，反映大资金对价格的推动力。
"""

import numpy as np
import pandas as pd
from factors.base import BaseFactor


class LargeOrderReturnFactor(BaseFactor):
    """大单推动涨幅因子

    公式: rolling_product(1 + daily_large_return, T) - 1
    """

    name = "LARGE_ORDER_RETURN"
    category = "高频动量反转"
    description = "大单推动涨幅因子，衡量大单交易推动的累计收益"

    def compute(
        self,
        daily_large_return: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算大单推动涨幅因子。

        Args:
            daily_large_return: 预计算的每日大单收益率（每笔成交金额在前30%的分钟收益率），
                                index为日期，columns为股票代码。
            T: 滚动窗口天数，默认20。

        Returns:
            pd.DataFrame: 大单推动涨幅因子值（T日滚动累计乘积）。
        """
        # 滚动T日累计乘积: exp(sum(log(1+r)))
        log_ret = np.log(1 + daily_large_return)
        result = log_ret.rolling(window=T, min_periods=1).sum().apply(np.exp)
        return result


# 使用示例
if __name__ == "__main__":
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=30)
    stocks = ["000001", "000002"]
    daily_large_return = pd.DataFrame(
        np.random.uniform(-0.02, 0.03, (30, 2)), index=dates, columns=stocks
    )

    factor = LargeOrderReturnFactor()
    print(factor)
    print(factor.compute(daily_large_return=daily_large_return))
