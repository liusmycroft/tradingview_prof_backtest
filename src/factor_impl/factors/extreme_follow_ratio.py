import pandas as pd

from factors.base import BaseFactor


class ExtremeFollowRatioFactor(BaseFactor):
    """极端跟随行为_比值因子 — Extreme Following Behavior Ratio"""

    name = "EXTREME_FOLLOW_RATIO"
    category = "量价因子改进"
    description = "极端涨跌时跟随交易量与反向交易量的比值，衡量投资者跟风程度"

    def compute(
        self,
        daily_follow_ratio: pd.DataFrame,
        daily_reverse_ratio: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算极端跟随行为比值因子。

        公式: ExtremeFollowRatio = follow_volume_ratio / reverse_volume_ratio
        因子值为 T 日均值。

        Args:
            daily_follow_ratio: 预计算的每日跟随交易量占比 (index=日期, columns=股票代码)
            daily_reverse_ratio: 预计算的每日反向交易量占比 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 极端跟随行为比值的 T 日均值
        """
        ratio = daily_follow_ratio / daily_reverse_ratio.replace(0, float("nan"))
        result = ratio.rolling(window=T, min_periods=1).mean()
        return result
