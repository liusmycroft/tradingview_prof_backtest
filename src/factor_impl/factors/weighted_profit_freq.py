"""加权盈利频率因子 (Weighted Profit Frequency)

衰退指数加权的超额收益超过阈值的天数占比。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class WeightedProfitFreqFactor(BaseFactor):
    """加权盈利频率因子"""

    name = "WEIGHTED_PROFIT_FREQ"
    category = "量价因子改进"
    description = "衰退指数加权的超额收益超过阈值的天数占比，衡量投资者关注度与投机性需求"

    def compute(
        self,
        excess_return: pd.DataFrame,
        M: int = 40,
        u: float = 0.02,
        lam: float = 10.0,
        **kwargs,
    ) -> pd.DataFrame:
        """计算加权盈利频率因子。

        公式:
          f_w = sum(w_j * I(r_j > u)) / M
          w_decay_j = 0.5^((t-j)/lambda)

        Args:
            excess_return: 每日超额收益率 (index=日期, columns=股票代码)
            M: 回望期天数，默认 40
            u: 收益阈值，默认 0.02
            lam: 衰退半衰期参数，默认 10.0

        Returns:
            pd.DataFrame: 加权盈利频率因子值
        """
        dates = excess_return.index
        stocks = excess_return.columns
        n_dates = len(dates)
        result = np.full((n_dates, len(stocks)), np.nan)

        for t in range(n_dates):
            start = max(0, t - M + 1)
            window = excess_return.iloc[start : t + 1]
            window_len = len(window)

            # 衰退权重: 最近的权重最大
            # j 从 start 到 t, 距离 = t - j
            distances = np.arange(window_len - 1, -1, -1, dtype=float)
            weights = np.power(0.5, distances / lam)

            indicator = (window.values > u).astype(float)
            weighted_sum = (weights[:, None] * indicator).sum(axis=0)
            result[t, :] = weighted_sum / M

        return pd.DataFrame(result, index=dates, columns=stocks)


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 加权盈利频率为股票过去一段时间收益率大于一定阈值的天数加权之和。
# 因子值越大表示股票过去一段时间日超额收益大于一定阈值的天数越多
# 并且出现的时间越近，越容易受到投资者关注，投机性需求较高，
# 定价相对较高，未来预期回报较低（反向因子）。
