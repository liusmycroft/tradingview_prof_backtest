"""日内振幅切割因子 (Intraday Amplitude Cut Factor)

V(lambda) = V_high(lambda) - V_low(lambda)
最终因子 = 过去 N 日 V(lambda) 的均值和标准差的标准化合成。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class IntradayAmplitudeCutFactor(BaseFactor):
    """日内振幅切割因子"""

    name = "INTRADAY_AMPLITUDE_CUT"
    category = "高频波动跳跃"
    description = "日内振幅切割因子，高价与低价分钟振幅之差的标准化合成"

    def compute(
        self,
        daily_amplitude_cut: pd.DataFrame,
        N: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """计算日内振幅切割因子。

        Args:
            daily_amplitude_cut: 预计算的每日振幅切割值 V(lambda)
                (index=日期, columns=股票代码)。
                每日值 = V_high(lambda) - V_low(lambda)。
            N: 回看窗口天数，默认 10。

        Returns:
            pd.DataFrame: 因子值，V_mean 和 V_std 等权合成。
        """
        v_mean = daily_amplitude_cut.rolling(window=N, min_periods=N).mean()
        v_std = daily_amplitude_cut.rolling(window=N, min_periods=N).std()

        # 横截面标准化后等权合成
        # z-score 标准化: (x - cross_mean) / cross_std
        def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
            row_mean = df.mean(axis=1)
            row_std = df.std(axis=1)
            # 避免除零
            row_std = row_std.replace(0, np.nan)
            return df.sub(row_mean, axis=0).div(row_std, axis=0)

        z_mean = cross_sectional_zscore(v_mean)
        z_std = cross_sectional_zscore(v_std)

        # 等权合成
        result = (z_mean + z_std) / 2.0
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 日内振幅切割因子的构造步骤：
# 1. 计算每分钟振幅 (high/low - 1)；
# 2. 按 1 分钟涨跌幅排序，取最高/最低 lambda 比例的分钟，
#    分别计算振幅均值 V_high 和 V_low；
# 3. V(lambda) = V_high - V_low；
# 4. 回看 N 日，计算 V_mean 和 V_std，横截面标准化后等权合成。
