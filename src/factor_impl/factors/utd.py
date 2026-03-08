import pandas as pd

from .base import BaseFactor


class UTDFactor(BaseFactor):
    """换手率分布均匀度因子 (Uniformity of Turnover Distribution)。"""

    name = "UTD"
    category = "流动性"
    description = "换手率分布均匀度，衡量日内换手率波动在时间序列上的稳定性"

    def compute(self, turn_vol_daily: pd.DataFrame, T: int = 20) -> pd.DataFrame:
        """计算 UTD 因子。

        Args:
            turn_vol_daily: 日内分钟换手率的标准差，index=日期，columns=股票代码。
            T: 回看窗口天数，默认 20。

        Returns:
            pd.DataFrame: UTD 因子值（变异系数），index=日期，columns=股票代码。
        """
        rolling_std = turn_vol_daily.rolling(window=T, min_periods=T).std(ddof=1)
        rolling_mean = turn_vol_daily.rolling(window=T, min_periods=T).mean()
        utd = rolling_std / rolling_mean
        return utd


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# UTD（Uniformity of Turnover Distribution）衡量的是"日内换手率波动"在时间
# 序列上的稳定程度。具体来说：
#   1. 每个交易日，先用分钟级成交量除以流通股本得到分钟换手率，再取标准差，
#      得到当日的日内换手率波动 TurnVolDaily。
#   2. 在过去 T（默认 20）个交易日的窗口内，计算 TurnVolDaily 的变异系数
#      （CV = std / mean）。
#
# 变异系数越低，说明日内换手率波动在不同交易日之间越稳定、分布越均匀，
# 通常意味着交易行为更理性、筹码分布更健康。反之，UTD 偏高则暗示存在
# 异常放量或缩量的交易日，可能伴随投机性资金进出。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.utd import UTDFactor
#
#   # turn_vol_daily: 预先计算好的日内分钟换手率标准差
#   # index 为交易日期，columns 为股票代码
#   turn_vol_daily = pd.DataFrame(
#       {
#           "000001.SZ": [0.12, 0.15, 0.11, 0.13, 0.14] * 5,
#           "600000.SH": [0.08, 0.09, 0.07, 0.10, 0.08] * 5,
#       },
#       index=pd.bdate_range("2025-01-01", periods=25),
#   )
#
#   factor = UTDFactor()
#   utd = factor.compute(turn_vol_daily, T=20)
#   print(utd.dropna())
