import pandas as pd

from .base import BaseFactor


class HerdingBehaviorFactor(BaseFactor):
    """羊群行为因子"""

    name = "HERDING_BEHAVIOR"
    category = "高频资金流"
    description = "羊群行为因子，衡量日内买卖压力的羊群效应强度"

    def compute(
        self,
        daily_herding: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算羊群行为因子。

        公式:
            H(i,T) = |B/(B+S) - p_T| - AF(i,T)
            AF 为独立交易假设下的调整项
            因子 = T 日 EMA

        Args:
            daily_herding: 预计算的每日羊群行为指标 H(i,T)
                (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 羊群行为因子值
        """
        result = daily_herding.ewm(span=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 羊群行为 H(i,T) = |B/(B+S) - p_T| - AF(i,T)
# 其中 B、S 分别为买方和卖方驱动单数量，p_T 为横截面买单占比均值，
# AF 为独立交易假设下的调整项（基于二项分布）。
#
# 可根据买单占比相对均值的高低，划分为买入羊群行为 HB 和卖出羊群行为 HS。
# 短期羊群行为会伴随价格反转：出现买入（卖出）羊群行为时，未来股价
# 可能下跌（上涨），且羊群行为越剧烈，未来价格变动幅度越大。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.herding_behavior import HerdingBehaviorFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_herding = pd.DataFrame(
#       np.random.randn(30, 2) * 0.05,
#       index=dates, columns=stocks,
#   )
#
#   factor = HerdingBehaviorFactor()
#   result = factor.compute(daily_herding=daily_herding, T=20)
#   print(result.tail())
