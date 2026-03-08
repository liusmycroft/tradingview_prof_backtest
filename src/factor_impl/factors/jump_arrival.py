"""跳跃强度因子 (Jump Arrival Rate, JArr)

一段时间内出现价格跳跃的天数占比，衡量股价异常波动的持续性。
"""

import numpy as np
import pandas as pd
from scipy import stats

from factors.base import BaseFactor


class JumpArrivalFactor(BaseFactor):
    """跳跃强度因子"""

    name = "JUMP_ARRIVAL"
    category = "高频波动跳跃"
    description = "跳跃强度：观察期内日内价格存在跳跃的天数占比"

    def compute(
        self,
        daily_jump_indicator: pd.DataFrame,
        D: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算跳跃强度因子。

        公式:
            JArr = (1/D) * sum(I_jump_t) for t=1..D

        其中 I_jump_t 为日内是否存在跳跃的示性函数（预计算）。

        Args:
            daily_jump_indicator: 每日跳跃示性函数 0/1
                                  (index=日期, columns=股票代码)
            D: 观察期天数，默认 20

        Returns:
            pd.DataFrame: 跳跃强度因子值
        """
        result = daily_jump_indicator.rolling(window=D, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 跳跃强度为一段时间内出现价格跳跃的天数占比，衡量了股价异常波动的
# 持续性，可以用于刻画信息冲击的强度。基于 SwV 检验统计量判断日内
# 是否存在跳跃，然后在观察期内统计跳跃天数占比。
