import numpy as np
import pandas as pd

from factors.base import BaseFactor


class ClosingReturnFactor(BaseFactor):
    """尾盘半小时收益率 (Closing Half-Hour Return) 因子。"""

    name = "CLOSING_RETURN"
    category = "高频动量反转"
    description = "尾盘半小时收益率的滚动累计值，捕捉尾盘交易的动量或反转效应"

    def compute(
        self,
        close_1430: pd.DataFrame,
        close_1500: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算尾盘半小时收益率因子。

        Args:
            close_1430: 14:30 收盘价，index=日期, columns=股票代码。
            close_1500: 15:00 收盘价，形状同 close_1430。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        # 尾盘半小时收益率
        daily_ret = close_1500 / close_1430 - 1.0

        # 滚动 T 日求和
        result = daily_ret.rolling(window=T, min_periods=T).sum()
        return result


# ==============================================================================
# 核心思想与原理说明
# ==============================================================================
#
# 尾盘半小时收益率因子的核心思想：
#
# 1. 尾盘（14:30-15:00）是A股市场信息密度最高的时段之一。机构投资者
#    常在尾盘进行调仓操作，尾盘价格变动包含重要的信息信号。
#
# 2. 每日尾盘收益率 = close_15:00 / close_14:30 - 1
#
# 3. 对过去 T 天的尾盘收益率求和，得到累计尾盘动量。
#    - 正值：尾盘持续上涨，可能暗示机构持续买入；
#    - 负值：尾盘持续下跌，可能暗示机构持续卖出。
#
# 4. 该因子可用于捕捉尾盘交易的动量或反转效应。
#
# ==============================================================================
# 简单用法示例
# ==============================================================================
#
# import pandas as pd
# import numpy as np
# from factors.closing_return import ClosingReturnFactor
#
# dates = pd.date_range("2024-01-01", periods=30, freq="B")
# stocks = ["000001.SZ", "000002.SZ"]
#
# np.random.seed(42)
# close_1430 = pd.DataFrame(
#     np.random.uniform(10, 50, (30, 2)), index=dates, columns=stocks
# )
# close_1500 = close_1430 * (1 + np.random.normal(0, 0.005, (30, 2)))
#
# factor = ClosingReturnFactor()
# result = factor.compute(close_1430=close_1430, close_1500=close_1500, T=20)
# print(result.tail())
