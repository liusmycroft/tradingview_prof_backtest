import numpy as np
import pandas as pd

from factors.base import BaseFactor


class VolumeKurtosisFactor(BaseFactor):
    """成交量占比峰度 (Volume Ratio Kurtosis) 因子。"""

    name = "VOLUME_KURTOSIS"
    category = "高频成交分布"
    description = "分钟成交量占比的峰度滚动均值，衡量日内成交分布的尖峰程度"

    def compute(
        self,
        daily_volume_kurtosis: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算成交量占比峰度因子。

        Args:
            daily_volume_kurtosis: 每日分钟成交量占比的峰度，
                index=日期, columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        result = daily_volume_kurtosis.rolling(window=T, min_periods=T).mean()
        return result


# ==============================================================================
# 核心思想与原理说明
# ==============================================================================
#
# 成交量占比峰度因子的核心思想：
#
# 1. 对于每个交易日，计算分钟级别成交量占比 (volume_i / sum(volume_i))
#    的峰度 (kurtosis)。
#
# 2. 峰度衡量分布的尖峰程度。高峰度意味着成交量在少数几个时段高度集中，
#    可能暗示知情交易者的集中交易行为。
#
# 3. 对日度峰度取 T 日滚动均值以平滑噪声。高因子值的股票可能存在更多
#    的信息不对称，未来收益可能有特定模式。
#
# ==============================================================================
# 简单用法示例
# ==============================================================================
#
# import pandas as pd
# import numpy as np
# from factors.volume_kurtosis import VolumeKurtosisFactor
#
# dates = pd.date_range("2024-01-01", periods=30, freq="B")
# stocks = ["000001.SZ", "000002.SZ"]
#
# np.random.seed(42)
# daily_kurt = pd.DataFrame(
#     np.random.uniform(2.0, 5.0, (30, 2)), index=dates, columns=stocks
# )
#
# factor = VolumeKurtosisFactor()
# result = factor.compute(daily_volume_kurtosis=daily_kurt, T=20)
# print(result.tail())
