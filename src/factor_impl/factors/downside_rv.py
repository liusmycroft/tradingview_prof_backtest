import pandas as pd

from .base import BaseFactor


class DownsideRVFactor(BaseFactor):
    """下行已实现波动率"""

    name = "DOWNSIDE_RV"
    category = "高频波动跳跃"
    description = "下行已实现波动率，衡量资产价格的下行波动风险"

    def compute(
        self,
        daily_downside_rv: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算下行已实现波动率因子。

        公式:
            RS_minus_t = sum(r_{t,i}^2 * I(r_{t,i} <= 0))
            因子 = T 日滚动均值

        Args:
            daily_downside_rv: 预计算的每日下行已实现波动率 RS_minus
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 下行已实现波动率因子值
        """
        result = daily_downside_rv.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 下行已实现波动率 RS_minus = sum(r^2 * I(r<=0))，仅对负收益率的平方求和，
# 对应资产价格波动中的下行波动部分。
#
# 下行已实现波动率较高的股票伴随着较高的下行风险，投资者因承担更高的
# 下行风险可以期望未来获得更高的风险补偿。因子取 T 日滚动均值以平滑
# 日间波动。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.downside_rv import DownsideRVFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_downside_rv = pd.DataFrame(
#       np.random.rand(30, 2) * 0.001,
#       index=dates, columns=stocks,
#   )
#
#   factor = DownsideRVFactor()
#   result = factor.compute(daily_downside_rv=daily_downside_rv, T=20)
#   print(result.tail())
