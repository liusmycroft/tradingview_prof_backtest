import numpy as np
import pandas as pd

from .base import BaseFactor


class IDMagFactor(BaseFactor):
    """基于收益规模的信息离散度"""

    name = "ID_MAG"
    category = "量价因子改进"
    description = "基于收益规模的信息离散度因子，考虑日度收益规模对信息离散度的影响"

    def compute(
        self,
        daily_returns: pd.DataFrame,
        pret: pd.DataFrame,
        daily_magnitude_weights: pd.DataFrame,
        N: int = 220,
        **kwargs,
    ) -> pd.DataFrame:
        """计算基于收益规模的信息离散度因子。

        公式:
            ID_MAG = -(1/N) * sign(PRET) * sum(sign(Return_i) * w_i)
            PRET: 过去12个月累计收益率（剔除最近1个月）
            w_i: 基于日度收益绝对值横截面排序的权重（小收益高权重）

        Args:
            daily_returns: 每日收益率 (index=日期, columns=股票代码)
            pret: 预计算的 PRET（过去12个月剔除最近1个月的累计收益率）
                (index=日期, columns=股票代码)
            daily_magnitude_weights: 预计算的每日规模权重 w_i
                (index=日期, columns=股票代码)
            N: 回溯天数，默认 220（约11个月交易日）

        Returns:
            pd.DataFrame: 基于收益规模的信息离散度因子值
        """
        signed_weighted = np.sign(daily_returns) * daily_magnitude_weights
        rolling_sum = signed_weighted.rolling(window=N, min_periods=N).sum()
        result = -(1.0 / N) * np.sign(pret) * rolling_sum
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 基于收益规模的信息离散度 ID_MAG = -(1/N) * sign(PRET) * sum(sign(Return_i) * w_i)
# 其中 PRET 为过去12个月累计收益率（剔除最近1个月），
# w_i 为基于日度收益绝对值横截面排序的权重（分5组，小收益高权重）。
#
# 作者认为"一系列频繁但微小的变化对于人的吸引力远不如少数却显著的变化，
# 投资者对于连续信息造成的股价变化是反应不足的"。信息离散度越低
# （信息连续性越强）越好。该计算方式考虑了日度收益规模对信息离散度的影响，
# 给予小幅收益更高的权重。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.id_mag import IDMagFactor
#
#   dates = pd.date_range("2024-01-01", periods=250, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_returns = pd.DataFrame(
#       np.random.randn(250, 2) * 0.02,
#       index=dates, columns=stocks,
#   )
#   pret = pd.DataFrame(
#       np.random.randn(250, 2) * 0.1,
#       index=dates, columns=stocks,
#   )
#   weights = [5/15, 4/15, 3/15, 2/15, 1/15]
#   daily_magnitude_weights = pd.DataFrame(
#       np.random.choice(weights, (250, 2)),
#       index=dates, columns=stocks,
#   )
#
#   factor = IDMagFactor()
#   result = factor.compute(
#       daily_returns=daily_returns,
#       pret=pret,
#       daily_magnitude_weights=daily_magnitude_weights,
#       N=220,
#   )
#   print(result.tail())
