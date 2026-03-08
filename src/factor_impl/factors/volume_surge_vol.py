import numpy as np
import pandas as pd

from .base import BaseFactor


class VolumeSurgeVolFactor(BaseFactor):
    """量涌波动率因子 (Volume Surge Volatility)。"""

    name = "VOLUME_SURGE_VOL"
    category = "高频波动"
    description = "量涌波动率因子，基于量涌分段收益率标准差的滚动标准差"

    def compute(
        self,
        daily_segment_vol: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算量涌波动率因子。

        Args:
            daily_segment_vol: 预计算的每日分段收益率标准差，
                               index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期，columns=股票代码。
                          先截面标准化，再取滚动 T 日标准差。
        """
        # 截面标准化：每日减去截面均值，除以截面标准差
        cross_mean = daily_segment_vol.mean(axis=1)
        cross_std = daily_segment_vol.std(axis=1, ddof=1)
        standardized = daily_segment_vol.sub(cross_mean, axis=0).div(cross_std, axis=0)

        # 滚动 T 日标准差
        result = standardized.rolling(window=T, min_periods=T).std(ddof=1)

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 量涌波动率因子衡量的是在成交量突增（量涌）时段内收益率波动的稳定性。
# 核心逻辑：
#   1. 在日内识别量涌时点（成交量突然放大的时段）。
#   2. 计算量涌时段内分段收益率的标准差，作为每日的量涌波动率。
#   3. 先对日度量涌波动率进行截面标准化，消除量纲差异。
#   4. 再取 T 日滚动标准差，衡量量涌波动率本身的波动性。
#
# 经济直觉：量涌波动率的波动性越大，说明该股票在放量时段的价格行为越不稳定，
# 可能反映了信息不对称程度较高或市场微观结构噪声较大。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.volume_surge_vol import VolumeSurgeVolFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=25)
#   daily_segment_vol = pd.DataFrame(
#       np.random.rand(25, 3) * 0.05,
#       index=dates,
#       columns=["000001.SZ", "600000.SH", "000002.SZ"],
#   )
#
#   factor = VolumeSurgeVolFactor()
#   result = factor.compute(daily_segment_vol=daily_segment_vol, T=20)
#   print(result)
