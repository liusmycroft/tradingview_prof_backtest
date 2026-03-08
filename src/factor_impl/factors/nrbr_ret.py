import pandas as pd

from factors.base import BaseFactor


class NRBRRetFactor(BaseFactor):
    """"自信溢出"因子 (NRBR_ret) — Confidence Spillover via Adjacent Stock Returns"""

    name = "NRBR_RET"
    category = "高频动量反转"
    description = "通过相邻股票收益率衡量自信溢出效应，捕捉投资者过度自信导致的动量溢出"

    def compute(
        self,
        daily_neighbor_ret: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算自信溢出因子。

        公式: NRBR_ret = 相邻股票收益率加权均值的 T 日 EMA
        daily_neighbor_ret 为预计算的每日相邻股票加权收益率。

        Args:
            daily_neighbor_ret: 预计算的每日相邻股票加权收益率 (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 自信溢出因子的 T 日 EMA
        """
        result = daily_neighbor_ret.ewm(span=T, min_periods=1).mean()
        return result
