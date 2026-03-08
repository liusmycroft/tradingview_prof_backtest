"""成交量比值因子 (Volume Ratio, VR)

VR_t = EWM of (Vol_morning / Vol_afternoon)
使用指数加权移动平均，alpha = 2/(1+d)。
"""

import pandas as pd

from factors.base import BaseFactor


class VolumeRatioFactor(BaseFactor):
    """成交量比值因子 (VR)"""

    name = "VR"
    category = "高频成交分布"
    description = "成交量比值，上午与下午开盘前 30 分钟成交量比值的指数加权移动平均"

    def compute(
        self,
        daily_volume_ratio: pd.DataFrame,
        d: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算成交量比值因子。

        Args:
            daily_volume_ratio: 预计算的每日上午/下午成交量比值
                (index=日期, columns=股票代码)。
                每日值 = Vol_morning_30min / Vol_afternoon_30min。
            d: EWM 窗口天数，默认 20。alpha = 2/(1+d)。

        Returns:
            pd.DataFrame: 因子值，d 日 EWM。
        """
        result = daily_volume_ratio.ewm(span=d, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 成交量比值因子衡量上午与下午开盘前 30 分钟的成交量分布差异。
# 上午开盘 30 分钟成交量越小或下午开盘 30 分钟成交量越大（比值越小），
# 对次月收益预测的有效性越强。
#
# 使用指数加权移动平均赋予近期数据更高权重，alpha = 2/(1+d)。
