import pandas as pd

from factors.base import BaseFactor


class FogVolumeRatioFactor(BaseFactor):
    """模糊数量比因子 (Fog Volume Ratio)"""

    name = "FOG_VOLUME_RATIO"
    category = "高频成交分布"
    description = "起雾时刻成交量均值与总体成交量均值之比，衡量投资者对模糊性的厌恶程度"

    def compute(
        self,
        daily_fog_ratio_mean: pd.DataFrame,
        daily_fog_ratio_std: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算模糊数量比因子。

        每月末对最近 T 天的日模糊数量比求均值和标准差并等权合并。

        公式:
            日模糊数量比 = 雾中数量 / 总体数量
            月度因子 = rolling_mean(日模糊数量比, T) + rolling_std(日模糊数量比, T)

        本方法接收预计算的每日滚动均值和滚动标准差，输出等权合并结果。

        Args:
            daily_fog_ratio_mean: 日模糊数量比的滚动 T 日均值 (index=日期, columns=股票代码)
            daily_fog_ratio_std: 日模糊数量比的滚动 T 日标准差 (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 模糊数量比因子值（均值 + 标准差等权合并）
        """
        result = 0.5 * daily_fog_ratio_mean + 0.5 * daily_fog_ratio_std
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 模糊数量比因子从量维度统计了投资者在模糊性较大时的成交程度，衡量了
# 投资者对模糊性的厌恶程度，与未来收益负相关。
#
# 模糊性定义：分钟波动率的标准差（即波动率的波动率）。
# 起雾时刻：日内模糊性大于日内模糊性均值的时间段。
# 雾中数量：起雾时刻的分钟成交量均值。
# 总体数量：日内所有时刻的分钟成交量均值。
#
# 投资者对波动率模糊性的厌恶心理会使得投资者急于卖出而产生过度反应，
# 未来很可能会补涨。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.fog_volume_ratio import FogVolumeRatioFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=30)
#   stocks = ["000001.SZ", "000002.SZ"]
#   fog_mean = pd.DataFrame(
#       np.random.uniform(0.8, 1.5, (30, 2)), index=dates, columns=stocks,
#   )
#   fog_std = pd.DataFrame(
#       np.random.uniform(0.1, 0.5, (30, 2)), index=dates, columns=stocks,
#   )
#
#   factor = FogVolumeRatioFactor()
#   result = factor.compute(daily_fog_ratio_mean=fog_mean, daily_fog_ratio_std=fog_std)
#   print(result.tail())
