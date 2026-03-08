import pandas as pd

from factors.base import BaseFactor


class TLRatioFactor(BaseFactor):
    """当日新增筹码亏损占比 (TLRatio) — Today's New Chip Loss Ratio"""

    name = "TLRatio"
    category = "行为金融-筹码分布"
    description = "当日新增筹码亏损额占账面总亏损额的比例，衡量短期筹码亏损压力"

    def compute(
        self,
        daily_tl_ratio: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算当日新增筹码亏损占比因子。

        公式:
            TL = sum(IF(p > close, Vol * turnover * (p - close), 0))
            PL = sum(IF(p > close, Chip * (p - close), 0))
            TLRatio = TL / PL
        因子值为 T 日均值。

        Args:
            daily_tl_ratio: 预计算的每日新增筹码亏损占比 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 当日新增筹码亏损占比的 T 日均值
        """
        result = daily_tl_ratio.rolling(window=T, min_periods=1).mean()
        return result
