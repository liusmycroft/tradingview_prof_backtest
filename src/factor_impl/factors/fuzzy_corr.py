import numpy as np
import pandas as pd

from factors.base import BaseFactor


class FuzzyCorrFactor(BaseFactor):
    """模糊关联度因子 (Fuzzy Correlation)"""

    name = "FUZZY_CORR"
    category = "高频因子-量价相关性类"
    description = "分钟模糊性序列与成交额序列的相关系数，衡量投资者对模糊性的厌恶程度"

    def compute(
        self,
        daily_fuzzy_corr: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算模糊关联度因子。

        公式:
            FuzzyCorr_d = corr(Ambiguity_t, Amount_t)
            月度因子 = 0.5 * mean(FuzzyCorr, T日) + 0.5 * std(FuzzyCorr, T日)

        Args:
            daily_fuzzy_corr: 每日模糊关联度 (index=日期, columns=股票代码)
            T: 月度合成窗口，默认 20

        Returns:
            pd.DataFrame: 月度模糊关联度因子值
        """
        rolling_mean = daily_fuzzy_corr.rolling(window=T, min_periods=1).mean()
        rolling_std = daily_fuzzy_corr.rolling(window=T, min_periods=1).std().fillna(0)
        result = 0.5 * rolling_mean + 0.5 * rolling_std
        return result
