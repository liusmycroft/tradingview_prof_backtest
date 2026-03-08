import numpy as np
import pandas as pd

from factors.base import BaseFactor


class OvernightGapFactor(BaseFactor):
    """隔夜跳空因子 (Overnight Gap)"""

    name = "OVERNIGHT_GAP"
    category = "高频动量反转"
    description = "隔夜对数收益率绝对值的滚动求和，衡量隔夜信息冲击强度"

    def compute(
        self,
        open_price: pd.DataFrame,
        prev_close: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算隔夜跳空因子。

        公式: factor = sum over T days of |ln(Open_t / Close_{t-1})|

        Args:
            open_price: 开盘价 (index=日期, columns=股票代码)
            prev_close: 前一日收盘价 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 隔夜对数收益率绝对值的 T 日滚动求和
        """
        # 避免除零和负值
        ratio = open_price / prev_close.replace(0, np.nan)
        overnight_log_ret = np.log(ratio).abs()
        result = overnight_log_ret.rolling(window=T, min_periods=1).sum()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 隔夜跳空因子衡量股票在非交易时段（隔夜）的信息冲击强度。
# 通过计算开盘价与前一日收盘价的对数收益率绝对值，并在 T 日窗口内
# 求和，可以捕捉近期隔夜信息冲击的累积程度。
#
# 隔夜跳空大的股票通常受到较多的盘后信息影响（如公告、新闻等），
# 可能存在动量或反转效应。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.overnight_gap import OvernightGapFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   prev_close = pd.DataFrame(
#       np.random.rand(30, 2) * 10 + 10,
#       index=dates, columns=stocks,
#   )
#   open_price = prev_close * (1 + np.random.randn(30, 2) * 0.02)
#
#   factor = OvernightGapFactor()
#   result = factor.compute(open_price=open_price, prev_close=prev_close, T=20)
#   print(result.tail())
