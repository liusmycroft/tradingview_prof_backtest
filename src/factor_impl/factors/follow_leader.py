"""「待著而救」因子 (Follow-the-Leader Factor)

衡量股票日内普通投资者跟随优势投资者的反应程度。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class FollowLeaderFactor(BaseFactor):
    """「待著而救」因子"""

    name = "FOLLOW_LEADER"
    category = "高频量价相关性"
    description = "待著而救因子：日跟随系数的均值与标准差等权合成，衡量跟随效应强度"

    def compute(
        self,
        daily_follow_coeff: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算「待著而救」因子。

        步骤:
        1. 输入为预计算的每日跟随系数
        2. 过去T天的日跟随系数均值和标准差等权合成

        Args:
            daily_follow_coeff: 每日跟随系数 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值
        """
        rolling_mean = daily_follow_coeff.rolling(window=T, min_periods=T).mean()
        rolling_std = daily_follow_coeff.rolling(window=T, min_periods=T).std(ddof=1)
        result = 0.5 * rolling_mean + 0.5 * rolling_std
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 「待著而救」因子衡量了股票日内普通投资者跟随优势投资者的反应程度，
# 与未来收益负相关。因子值越大，表明优势投资者短时间内大量买入造成的
# 成交量激增将导致普通投资者的明显跟随买入，使得股价短期内反应过度，
# 股价未来更可能出现较大回落。
