import numpy as np
import pandas as pd

from factors.base import BaseFactor


class UniformActiveRatioFactor(BaseFactor):
    """均匀分布主动占比因子 (Uniform Distribution Active Ratio)"""

    name = "UNIFORM_ACTIVE_RATIO"
    category = "高频因子-资金流类"
    description = "基于均匀分布映射的主动买入金额占比，提高主买卖额估计准确度"

    def compute(
        self,
        minute_amount: pd.DataFrame,
        minute_return: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算均匀分布主动占比因子。

        公式:
            均匀分布主动买入金额_i = Amount_i * (ret_i - 0.1) / 0.2
            因子 = sum(主动买入金额) / sum(Amount)

        注意: 这里 ret 做了从 [-0.1, 0.1] 到 [0, 1] 的线性变换

        Args:
            minute_amount: 分钟成交额 (index=时间, columns=股票代码)
                          每行为一个时间段
            minute_return: 分钟收益率 (同结构)

        Returns:
            pd.DataFrame: 均匀分布主动占比因子值 (单行)
        """
        # 线性变换: (ret - (-0.1)) / 0.2 = (ret + 0.1) / 0.2
        # 原文公式: Amount_i * (ret_i - 0.1) / 0.2
        # 但根据说明，映射是从 [-0.1, 0.1] -> [0, 1]
        # 即 active_ratio = (ret + 0.1) / 0.2
        # 原文公式写的是 (ret_i - 0.1) / 0.2，但结合上下文应为 (ret_i - (-0.1)) / 0.2
        active_buy = minute_amount * (minute_return + 0.1) / 0.2
        total_amount = minute_amount.sum(axis=0)
        active_buy_sum = active_buy.sum(axis=0)

        result = active_buy_sum / total_amount.replace(0, np.nan)
        return result.to_frame().T
