import numpy as np
import pandas as pd

from factors.base import BaseFactor


class MinuteIdealAmplitudeFactor(BaseFactor):
    """分钟理想振幅因子 (Minute Ideal Amplitude)"""

    name = "MINUTE_IDEAL_AMPLITUDE"
    category = "高频波动跳跃"
    description = "高价分钟振幅均值与低价分钟振幅均值之差，衡量不同价格区间的波动差异"

    def compute(
        self,
        daily_v_high: pd.DataFrame,
        daily_v_low: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算分钟理想振幅因子。

        公式: V(lambda) = V_high(lambda) - V_low(lambda)

        V_high 和 V_low 由分钟级数据预计算：
        - 取最近 N 个交易日的 1 分钟 K 线
        - 按分钟收盘价排序，取最高 lambda 分位的分钟计算平均振幅 -> V_high
        - 取最低 lambda 分位的分钟计算平均振幅 -> V_low

        Args:
            daily_v_high: 预计算的高价分钟振幅均值 (index=日期, columns=股票代码)
            daily_v_low: 预计算的低价分钟振幅均值 (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 分钟理想振幅因子值
        """
        result = daily_v_high - daily_v_low
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 基于股价将振幅因子进行切割得到的理想振幅因子，考虑了不同价格区间的
# 振幅分布信息差异。高价振幅因子具有更强的负向选股能力，可用于刻画
# 不同价格位置的资金多空博弈情况。
# 将日频数据提升至分钟级，对交易信息提取更充分，选股效果更优。
