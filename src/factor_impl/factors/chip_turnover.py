"""筹码换手率因子 (Chip Turnover Rate)

基于筹码分布计算的换手率，衡量筹码流动性。
"""

import pandas as pd

from factors.base import BaseFactor


class ChipTurnoverFactor(BaseFactor):
    """筹码换手率因子"""

    name = "CHIP_TURNOVER"
    category = "行为金融-筹码分布"
    description = "筹码换手率：基于筹码分布的换手率，取T日滚动均值"

    def compute(
        self,
        daily_chip_turnover: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算筹码换手率因子。

        公式: factor = (1/T) * sum(daily_chip_turnover)

        Args:
            daily_chip_turnover: 每日筹码换手率 (index=日期, columns=股票代码)
                即当日成交量中实际发生筹码转移的比例
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        result = daily_chip_turnover.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 筹码换手率衡量基于筹码分布的实际换手情况。与传统换手率不同，筹码
# 换手率考虑了不同成本价位的筹码转移情况，更能反映真实的持仓变动。
# 筹码换手率越高，说明筹码流动性越好，持仓结构变化越快。
