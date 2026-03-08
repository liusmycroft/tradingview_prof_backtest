import pandas as pd

from factors.base import BaseFactor


class HFDownsideVolRatioFactor(BaseFactor):
    """高频下行波动占比因子 (High-Frequency Downside Volatility Ratio)。"""

    name = "HF_DOWNSIDE_VOL_RATIO"
    category = "高频波动跳跃"
    description = "负收益平方和占总收益平方和的比例，衡量下行波动特征"

    def compute(
        self,
        daily_downside_vol_ratio: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算高频下行波动占比因子。

        公式: sum(r_i^2 * I(r_i<0)) / sum(r_i^2)
        因子值为 T 日滚动均值。

        Args:
            daily_downside_vol_ratio: 预计算的每日下行波动占比
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_downside_vol_ratio.rolling(window=T, min_periods=T).mean()
        return result
