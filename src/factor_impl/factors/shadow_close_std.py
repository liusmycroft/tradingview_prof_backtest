import pandas as pd

from factors.base import BaseFactor


class ShadowCloseStdFactor(BaseFactor):
    """上下影线/收盘价的标准差因子 (Shadow-to-Close Std)。"""

    name = "SHADOW_CLOSE_STD"
    category = "量价因子改进"
    description = "每日K线上下影线总长度与收盘价之比的标准差，衡量多空博弈稳定性"

    def compute(
        self,
        daily_shadow_ratio: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算上下影线/收盘价的标准差因子。

        公式: std((上影线长度 + 下影线长度) / 收盘价)
        因子值为 T 日滚动标准差。

        Args:
            daily_shadow_ratio: 预计算的每日 (上影线+下影线)/收盘价
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动标准差
        """
        result = daily_shadow_ratio.rolling(window=T, min_periods=T).std()
        return result
