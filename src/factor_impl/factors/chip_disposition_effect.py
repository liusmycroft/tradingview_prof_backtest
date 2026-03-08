import numpy as np
import pandas as pd

from factors.base import BaseFactor


class ChipDispositionEffectFactor(BaseFactor):
    """基于筹码分布的处置效应 CDE 因子"""

    name = "CHIP_DISPOSITION_EFFECT"
    category = "行为金融-处置效应"
    description = "基于筹码分布的处置效应，衡量盈利筹码与亏损筹码的卖出倾向差异"

    def compute(
        self,
        profit_chip_ratio: pd.DataFrame,
        loss_chip_ratio: pd.DataFrame,
        turnover: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算基于筹码分布的处置效应因子。

        公式:
            PGR = profit_chip_ratio * turnover  (盈利筹码卖出比例)
            PLR = loss_chip_ratio * turnover    (亏损筹码卖出比例)
            CDE = rolling_mean(PGR - PLR, T)

        Args:
            profit_chip_ratio: 盈利筹码占比，index=日期，columns=股票代码。
            loss_chip_ratio: 亏损筹码占比，index=日期，columns=股票代码。
            turnover: 换手率，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 处置效应因子值 (T日滚动均值)。
        """
        pgr = profit_chip_ratio * turnover
        plr = loss_chip_ratio * turnover
        daily_cde = pgr - plr
        result = daily_cde.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 基于筹码分布的处置效应 (CDE) 因子利用筹码分布数据量化处置效应。
# 处置效应指投资者倾向于卖出盈利股票而持有亏损股票的行为偏差。
# PGR (盈利卖出比例) 与 PLR (亏损卖出比例) 的差值越大，
# 说明处置效应越强，投资者越倾向于卖出盈利筹码。
#
# 【使用示例】
#
#   from factors.chip_disposition_effect import ChipDispositionEffectFactor
#   factor = ChipDispositionEffectFactor()
#   result = factor.compute(
#       profit_chip_ratio=profit_df, loss_chip_ratio=loss_df,
#       turnover=turnover_df, T=20
#   )
