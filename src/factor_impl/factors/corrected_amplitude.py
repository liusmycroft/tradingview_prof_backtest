import numpy as np
import pandas as pd

from .base import BaseFactor


class CorrectedAmplitudeFactor(BaseFactor):
    """修正的振幅因子 (Corrected Amplitude)。"""

    name = "CORRECTED_AMPLITUDE"
    category = "高频波动"
    description = "修正的振幅因子，根据跳空方向修正日内振幅后取滚动均值"

    def compute(
        self,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        daily_jump_drop: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算修正的振幅因子。

        Args:
            high: 最高价，index=日期，columns=股票代码。
            low: 最低价，index=日期，columns=股票代码。
            close: 收盘价，index=日期，columns=股票代码。
            daily_jump_drop: 每日跳空/跌空幅度，index=日期，columns=股票代码。
                             正值表示跳空高开，负值表示跌空低开。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期，columns=股票代码。
        """
        # 前一日收盘价
        prev_close = close.shift(1)

        # 日内振幅
        amplitude = (high - low) / prev_close

        # 截面均值：每日所有股票的 jump_drop 均值
        cross_mean = daily_jump_drop.mean(axis=1)

        # 如果 jump_drop < 截面均值，翻转振幅符号
        flip_mask = daily_jump_drop.lt(cross_mean, axis=0)
        corrected = amplitude.copy()
        corrected[flip_mask] = -corrected[flip_mask]

        # 滚动 T 日均值
        result = corrected.rolling(window=T, min_periods=T).mean()

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 修正的振幅因子在传统振幅（(High-Low)/PrevClose）的基础上，根据跳空方向
# 进行符号修正。核心逻辑：
#   1. 计算日内振幅 = (High - Low) / PrevClose。
#   2. 计算每日跳空/跌空幅度的截面均值。
#   3. 如果个股的跳空幅度低于截面均值（相对弱势），则翻转振幅符号为负。
#   4. 对修正后的振幅取 T 日滚动均值。
#
# 经济直觉：单纯的振幅无法区分上涨波动和下跌波动。通过跳空方向修正，
# 正振幅代表强势波动，负振幅代表弱势波动，从而更好地预测未来收益。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.corrected_amplitude import CorrectedAmplitudeFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=25)
#   high = pd.DataFrame({"A": np.random.uniform(10, 12, 25)}, index=dates)
#   low = pd.DataFrame({"A": np.random.uniform(9, 10, 25)}, index=dates)
#   close = pd.DataFrame({"A": np.random.uniform(9.5, 11, 25)}, index=dates)
#   jump_drop = pd.DataFrame({"A": np.random.randn(25) * 0.01}, index=dates)
#
#   factor = CorrectedAmplitudeFactor()
#   result = factor.compute(
#       high=high, low=low, close=close, daily_jump_drop=jump_drop, T=20
#   )
#   print(result)
