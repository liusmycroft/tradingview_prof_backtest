import pandas as pd

from factors.base import BaseFactor


class LCVOLFactor(BaseFactor):
    """卖出反弹占比因子 (LCVOL) — Sell Bounce Volume Ratio"""

    name = "LCVOL"
    category = "高频成交分布"
    description = "卖出后价格反弹时段的成交量占比，衡量卖压释放后的反弹力度"

    def compute(
        self,
        daily_lcvol: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算卖出反弹占比因子。

        公式: LCVOL = 卖出后反弹时段成交量 / 总成交量
        因子值为 T 日 EMA。

        Args:
            daily_lcvol: 预计算的每日卖出反弹成交量占比 (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 卖出反弹占比的 T 日 EMA
        """
        result = daily_lcvol.ewm(span=T, min_periods=1).mean()
        return result
