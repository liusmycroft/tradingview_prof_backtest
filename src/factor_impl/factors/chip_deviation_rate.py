import numpy as np
import pandas as pd

from factors.base import BaseFactor


class ChipDeviationRateFactor(BaseFactor):
    """筹码乖离率因子 (Chip Deviation Rate, BIAS)"""

    name = "CHIP_DEVIATION_RATE"
    category = "行为金融-筹码分布"
    description = "动态加权的盈利筹码占比，刻画股价与筹码持仓成本间的偏离"

    def compute(
        self,
        winner: pd.DataFrame,
        turnover: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算筹码乖离率因子。

        公式: BIAS_t = winner_t * turnover_t + BIAS_{t-1} * (1 - turnover_t)

        Args:
            winner: 每日盈利筹码占比 (0-1), index=日期, columns=股票代码
            turnover: 每日换手率 (0-1), index=日期, columns=股票代码

        Returns:
            pd.DataFrame: 筹码乖离率因子值
        """
        dates = winner.index
        stocks = winner.columns

        winner_vals = winner.values
        turnover_vals = turnover.values
        num_dates, num_stocks = winner_vals.shape

        bias = np.full((num_dates, num_stocks), np.nan)

        # 初始值: BIAS_0 = winner_0 (相当于 BIAS_{-1} = 0 或直接用第一天)
        bias[0] = winner_vals[0]

        for t in range(1, num_dates):
            bias[t] = (
                winner_vals[t] * turnover_vals[t]
                + bias[t - 1] * (1.0 - turnover_vals[t])
            )

        return pd.DataFrame(bias, index=dates, columns=stocks)


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 筹码乖离率采用动态加权平均的方式，用于刻画盈利筹码的持续性变化，
# 即股价与筹码持仓成本间的偏离与回归情况。
# 一般取值越高，盈利程度越高，趋势稳定，投资者情绪相对乐观。
#
# 递推公式中，turnover 控制了新旧信息的混合比例：
# 换手率高时，当日盈利筹码占比权重大；换手率低时，历史值权重大。
