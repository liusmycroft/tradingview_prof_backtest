import pandas as pd

from factors.base import BaseFactor


class VolumePeakMinutesFactor(BaseFactor):
    """量峰分钟数因子 (Volume Peak Minutes)"""

    name = "VOLUME_PEAK_MINUTES"
    category = "高频成交分布"
    description = "量峰分钟数，统计孤立喷发成交量的分钟数量，衡量知情交易者参与频率"

    def compute(
        self,
        daily_peak_minutes: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算量峰分钟数因子。

        日内预计算逻辑:
          1. 对过去20日同时点成交量计算均值+1倍标准差作为阈值
          2. 高于阈值为"喷发成交量"，低于为"温和成交量"
          3. 喷发成交量且前后1分钟均为温和成交量 -> "量峰"
          4. 统计量峰的分钟数

        Args:
            daily_peak_minutes: 预计算的每日量峰分钟数 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_peak_minutes.rolling(window=T, min_periods=1).mean()
        return result
