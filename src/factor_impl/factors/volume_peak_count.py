import numpy as np
import pandas as pd

from factors.base import BaseFactor


class VolumePeakCountFactor(BaseFactor):
    """成交量波峰计数因子 (Volume Peak Count)"""

    name = "VOLUME_PEAK_COUNT"
    category = "高频成交分布"
    description = "成交量波峰计数，统计日内放量现象的次数，衡量趋势或知情交易者参与度"

    def compute(
        self,
        daily_peak_count: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算成交量波峰计数因子。

        日内预计算逻辑:
          1. 计算240根K线的均值与标准差
          2. 筛选大于(均值+1倍标准差)的分钟数据
          3. 筛选与前一分钟时间差>1分钟的数据(孤立峰值)
          4. 剩余数据个数即为当日波峰计数

        Args:
            daily_peak_count: 预计算的每日波峰计数 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_peak_count.rolling(window=T, min_periods=1).mean()
        return result

    @staticmethod
    def compute_daily_peak_count(minute_volume: pd.Series) -> int:
        """从单日分钟成交量序列计算波峰计数。

        Args:
            minute_volume: 单日分钟成交量 (index=分钟编号, 长度约240)

        Returns:
            int: 波峰计数
        """
        mean_vol = minute_volume.mean()
        std_vol = minute_volume.std()
        threshold = mean_vol + std_vol

        # 筛选超过阈值的分钟
        above = minute_volume[minute_volume > threshold]
        if len(above) == 0:
            return 0

        # 筛选与前一分钟间隔>1的(孤立峰值)
        indices = above.index.values
        if len(indices) <= 1:
            return len(indices)

        # 第一个总是保留，后续只保留与前一个间隔>1的
        peaks = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] - indices[i - 1] > 1:
                peaks.append(indices[i])

        return len(peaks)
