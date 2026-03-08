import numpy as np
import pandas as pd

from factors.base import BaseFactor


class MorningMistFactor(BaseFactor):
    """"朝没晨雾"因子 (Morning Mist Factor)。"""

    name = "MORNING_MIST"
    category = "高频量价"
    description = "滞后成交量差分回归t值的标准差，捕捉量价关系的不稳定性"

    def compute(
        self,
        daily_morning_mist: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算朝没晨雾因子。

        公式: 对滞后1-5期的成交量差分做回归，取各滞后项t值的标准差。
        本方法接收预计算的每日t值标准差，输出滚动 T 日均值。

        Args:
            daily_morning_mist: 预计算的每日t值标准差，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 滚动 T 日均值，index=日期，columns=股票代码。
        """
        result = daily_morning_mist.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# "朝没晨雾"因子通过对成交量差分的滞后回归，考察各滞后项系数t值的离散程度。
# t值标准差越大，说明不同滞后期的量价关系差异越大，市场微观结构越不稳定。
#
# 该因子可用于识别成交量模式异常的股票，这类股票往往存在信息不对称或
# 流动性风险。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.morning_mist import MorningMistFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=30)
#   daily_mist = pd.DataFrame(
#       np.random.uniform(0.5, 2.0, (30, 2)),
#       index=dates,
#       columns=["000001.SZ", "600000.SH"],
#   )
#
#   factor = MorningMistFactor()
#   result = factor.compute(daily_morning_mist=daily_mist, T=20)
#   print(result.tail())
