import pandas as pd

from factors.base import BaseFactor


class EndOfDayAmountFactor(BaseFactor):
    """尾盘成交额占比因子 (End-of-Day Amount Proportion - APL)"""

    name = "END_OF_DAY_AMOUNT"
    category = "高频成交分布"
    description = "尾盘成交额占全天成交额的比例，N日均值"

    def compute(
        self,
        daily_apl: pd.DataFrame,
        N: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算尾盘成交额占比因子。

        公式:
            APL_{t,d} = sum(Amount_{i,d}, i=240-t..240) / TotalAmount_d
            APL_t = mean(APL_{t,d}, d=1..N)

        Args:
            daily_apl: 预计算的每日尾盘成交额占比 (index=日期, columns=股票代码)
            N: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 尾盘成交额占比的 N 日滚动均值
        """
        result = daily_apl.rolling(window=N, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 尾盘成交额占比衡量尾盘（如最后30分钟）成交额占全天成交额的比例。
# 与未来收益负相关：尾盘成交额占比高的股票存在知情交易者比例更多，
# 资金获得利好消息后选择在噪声交易者更少的尾盘买入。
# 取值高可识别为交易驱动型股票，反之为基本面驱动型股票。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.end_of_day_amount import EndOfDayAmountFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_apl = pd.DataFrame(
#       np.random.uniform(0.05, 0.25, (30, 2)),
#       index=dates, columns=stocks,
#   )
#
#   factor = EndOfDayAmountFactor()
#   result = factor.compute(daily_apl=daily_apl, N=20)
#   print(result.tail())
