import pandas as pd

from .base import BaseFactor


class LockedChipsRatioFactor(BaseFactor):
    """上方/下方锁定筹码占比因子"""

    name = "LOCKED_CHIPS_RATIO"
    category = "行为金融-筹码分布"
    description = "上方/下方锁定筹码占比因子，衡量涨跌停价格以外的筹码分布"

    def compute(
        self,
        daily_locked_above: pd.DataFrame,
        daily_locked_below: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算锁定筹码占比因子。

        公式:
            ASR_above = 高位锁定筹码量 / 总筹码量
            ASR_below = 低位锁定筹码量 / 总筹码量
            因子 = (ASR_above - ASR_below) 的 T 日 EMA

        Args:
            daily_locked_above: 预计算的每日上方锁定筹码占比
                (index=日期, columns=股票代码)
            daily_locked_below: 预计算的每日下方锁定筹码占比
                (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 锁定筹码占比因子值
        """
        combined = daily_locked_above - daily_locked_below
        result = combined.ewm(span=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 锁定筹码指涨跌停价格以外的筹码，该部分筹码不易交易，通常较为稳定。
# 上方锁定筹码占比 ASR_above 衡量高位套牢程度，
# 下方锁定筹码占比 ASR_below 衡量低价筹码集中程度。
#
# 高位价格 = close * (1 + volatility)，低位价格 = close * (1 - volatility)，
# 其中 volatility 为股价最大涨跌幅（主板 10%，创业板/科创板 20%）。
#
# 因子取上方占比减去下方占比，再做 T 日 EMA 平滑。
# 上方锁定筹码越多说明套牢盘越重，下方锁定筹码越多说明低位支撑越强。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.locked_chips_ratio import LockedChipsRatioFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_locked_above = pd.DataFrame(
#       np.random.rand(30, 2) * 0.3,
#       index=dates, columns=stocks,
#   )
#   daily_locked_below = pd.DataFrame(
#       np.random.rand(30, 2) * 0.3,
#       index=dates, columns=stocks,
#   )
#
#   factor = LockedChipsRatioFactor()
#   result = factor.compute(
#       daily_locked_above=daily_locked_above,
#       daily_locked_below=daily_locked_below,
#       T=20,
#   )
#   print(result.tail())
