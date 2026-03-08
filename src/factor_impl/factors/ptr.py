import pandas as pd

from factors.base import BaseFactor


class PTRFactor(BaseFactor):
    """筹码穿透率因子 (Chip Penetration Rate)"""

    name = "PTR"
    category = "行为金融-筹码分布"
    description = "筹码穿透率：新增解套筹码比例除以换手率，取T日滚动均值"

    def compute(
        self,
        winner: pd.DataFrame,
        turnover: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算筹码穿透率因子。

        公式: PTR = rolling_mean_T( (winner_t - winner_{t-1}) / turnover_t )

        Args:
            winner: 每日盈利筹码占总筹码的比例 (index=日期, columns=股票代码)
            turnover: 每日换手率 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        winner_diff = winner - winner.shift(1)
        daily_ptr = winner_diff / turnover
        result = daily_ptr.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 筹码穿透率的计算方式为当天新增解套筹码比例除以当天换手率，用于反映
# 穿透筹码的能力。因子值越大，说明相同换手率使套牢筹码解套的效率越高
# （少量换手就能让大量历史筹码解套），股票筹码的通透性越好，趋势延续
# 性强。取 T 日滚动均值以平滑日间波动。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.ptr import PTRFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   winner = pd.DataFrame(
#       np.random.rand(30, 2) * 0.5 + 0.3, index=dates, columns=stocks,
#   )
#   turnover = pd.DataFrame(
#       np.random.rand(30, 2) * 0.05 + 0.01, index=dates, columns=stocks,
#   )
#
#   factor = PTRFactor()
#   result = factor.compute(winner=winner, turnover=turnover, T=20)
#   print(result.tail())
