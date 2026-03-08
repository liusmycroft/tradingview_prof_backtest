"""每日最大异常成交量因子 (Max Daily Abnormal Volume, ABNVOLD)

ABNVOLD = max_t(VOL_{i,t} / VOL_bar_{i,t})
其中 VOL_bar 为滚动 1 年平均成交量。
"""

import pandas as pd

from factors.base import BaseFactor


class AbnormalVolumeDailyFactor(BaseFactor):
    """每日最大异常成交量因子 (ABNVOLD)"""

    name = "ABNVOLD"
    category = "行为金融-投资者注意力"
    description = "每日最大异常成交量，日频成交量除以过去一年平均成交量的滚动最大值"

    def compute(
        self,
        volume: pd.DataFrame,
        Y: int = 252,
        M: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算每日最大异常成交量因子。

        Args:
            volume: 每日成交量 (index=日期, columns=股票代码)。
            Y: 滚动平均窗口（约 1 年交易日），默认 252。
            M: 取最大值的窗口（约 1 月交易日），默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        # 滚动 1 年平均成交量
        vol_mean = volume.rolling(window=Y, min_periods=Y).mean()

        # 异常成交量比值
        abnormal_vol = volume / vol_mean

        # 过去 M 日内的最大异常成交量
        result = abnormal_vol.rolling(window=M, min_periods=M).max()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 每日最大异常成交量 ABNVOLD 衡量投资者注意力。异常成交量 = 当日成交量 /
# 过去一年平均成交量，取月度窗口内的最大值。因子值越高，说明该股票在近期
# 出现了极端的交易量放大，投资者关注度越高。
#
# A 股市场做多做空不对称，高关注度股票因投资者净买入而存在溢价，
# 未来更可能出现反转。
