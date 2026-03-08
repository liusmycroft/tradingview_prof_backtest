import pandas as pd

from factors.base import BaseFactor


class ActiveChipRatioFactor(BaseFactor):
    """活动筹码占比因子 (Active Chip Ratio - ASR)。"""

    name = "ACTIVE_CHIP_RATIO"
    category = "行为金融筹码"
    description = "涨跌停价格区间内的筹码占比，衡量交易活跃程度"

    def compute(
        self,
        daily_active_chip_ratio: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算活动筹码占比因子。

        公式: ASR_t = sum(vol_{price in [low, high]}) / sum(vol)
              其中 high = close*(1+volatility), low = close*(1-volatility)
        因子值为 T 日 EMA。

        Args:
            daily_active_chip_ratio: 预计算的每日活动筹码占比 (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 活动筹码占比的 T 日 EMA
        """
        result = daily_active_chip_ratio.ewm(span=T, min_periods=1).mean()
        return result
