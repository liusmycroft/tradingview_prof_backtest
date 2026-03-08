import pandas as pd

from factors.base import BaseFactor


class SMMFactor(BaseFactor):
    """最大供应商动量 (smm) — Largest Supplier Momentum"""

    name = "SMM"
    category = "图谱网络-动量溢出"
    description = "最大供应商过去一段时间的收益率动量，捕捉供应链上游的动量溢出效应"

    def compute(
        self,
        daily_supplier_ret: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算最大供应商动量因子。

        公式: smm = 最大供应商过去 T 日累计收益率
        daily_supplier_ret 为预计算的每日最大供应商收益率。

        Args:
            daily_supplier_ret: 预计算的每日最大供应商收益率 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 最大供应商 T 日累计收益率
        """
        result = daily_supplier_ret.rolling(window=T, min_periods=1).sum()
        return result
