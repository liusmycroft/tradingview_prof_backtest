import numpy as np
import pandas as pd

from factors.base import BaseFactor


class VSACloseDiffFactor(BaseFactor):
    """成交量支撑区域下限与收盘价差异因子 (VSA Lower Bound vs Close)。"""

    name = "VSA_CLOSE_DIFF"
    category = "高频成交分布"
    description = "成交量支撑区域下限与收盘价的差异，差异越大反弹概率越高"

    def compute(
        self,
        vsa_low: pd.DataFrame,
        close: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 VSA 下限与收盘价差异因子。

        公式: VSA_Low - Close，取滚动 T 日均值。
              差值越大（正值越大），说明收盘价远低于成交密集区下限，反弹概率越高。

        Args:
            vsa_low: 成交量支撑区域下限价格，index=日期，columns=股票代码。
            close: 收盘价，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 滚动 T 日均值，index=日期，columns=股票代码。
        """
        daily_diff = vsa_low - close
        result = daily_diff.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# VSA (Volume Spread Analysis) 成交量支撑区域下限代表了成交密集区的底部价格。
# 当收盘价低于该下限时，意味着当前价格已跌破成交密集区，存在较强的支撑力量，
# 反弹概率较高。
#
# VSA_Low - Close 的值越大，说明收盘价偏离支撑区域越远，反弹动力越强。
# 取滚动均值可以平滑日间波动，得到更稳定的信号。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.vsa_close_diff import VSACloseDiffFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=30)
#   vsa_low = pd.DataFrame(
#       np.random.uniform(9, 11, (30, 2)),
#       index=dates,
#       columns=["000001.SZ", "600000.SH"],
#   )
#   close = pd.DataFrame(
#       np.random.uniform(9, 11, (30, 2)),
#       index=dates,
#       columns=["000001.SZ", "600000.SH"],
#   )
#
#   factor = VSACloseDiffFactor()
#   result = factor.compute(vsa_low=vsa_low, close=close, T=20)
#   print(result.tail())
