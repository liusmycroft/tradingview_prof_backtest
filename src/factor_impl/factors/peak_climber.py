import pandas as pd

from factors.base import BaseFactor


class PeakClimberFactor(BaseFactor):
    """异常高波动下的收益波动比因子 (PeakClimber)"""

    name = "PEAK_CLIMBER"
    category = "高频波动跳跃"
    description = "异常高波动下收益波动比与更优波动率的协方差，衡量股价对异常高波动的风险补偿程度"

    def compute(
        self,
        daily_peak_climber: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 PeakClimber 因子。

        日内计算逻辑（预计算阶段）：
        1. BetterVol_t = (std(P_{t-4}..P_t) / mean(P_{t-4}..P_t))^2
        2. RVR_t = r_t / BetterVol_t
        3. 筛选 BetterVol >= mu + sigma 的异常高波动时刻
        4. daily_peak_climber = cov(RVR, BetterVol) 在异常高波动时刻

        本方法对预计算的日度因子取 T 日滚动均值。

        Args:
            daily_peak_climber: 预计算的每日 PeakClimber 值，
                index=日期, columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        result = daily_peak_climber.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# "勇攀高峰"因子衡量了股价对异常高波动提供的风险补偿程度。
# 对于每只股票，先计算分钟级别的"更优波动率"BetterVol（5分钟窗口内
# OHLC 20个价格的标准差/均值的平方），再计算收益波动比 RVR = r / BetterVol。
# 筛选 BetterVol >= mu + sigma 的异常高波动时刻，对该子集的 RVR 和
# BetterVol 求协方差，即为日度因子值。
#
# 因子与未来收益正相关：能对异常高波动及时提供风险补偿的股票，
# 往往是持续利好的实力股。
