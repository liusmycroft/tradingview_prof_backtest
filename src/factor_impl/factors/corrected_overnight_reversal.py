import numpy as np
import pandas as pd

from factors.base import BaseFactor


class CorrectedOvernightReversalFactor(BaseFactor):
    """修正的隔夜反转因子 (Corrected Overnight Reversal)。"""

    name = "CORRECTED_OVERNIGHT_REVERSAL"
    category = "高频动量反转"
    description = "根据隔夜距离波动率修正的隔夜反转因子，低波动时翻转隔夜距离方向"

    def compute(
        self,
        overnight_distance: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算修正的隔夜反转因子。

        公式:
            1. overnight_distance = abs(overnight_return - cross_section_mean)
            2. vol = rolling_std(overnight_distance, T)
            3. 若 vol < 截面均值，则翻转 overnight_distance 符号
            4. 取 T 日滚动均值

        Args:
            overnight_distance: 隔夜距离 abs(隔夜收益-截面均值)
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 修正后的隔夜反转因子
        """
        roll_vol = overnight_distance.rolling(window=T, min_periods=T).std()

        cross_section_mean = roll_vol.mean(axis=1)

        low_vol_mask = roll_vol.lt(cross_section_mean, axis=0)

        corrected = overnight_distance.copy()
        corrected[low_vol_mask] = -corrected[low_vol_mask]

        result = corrected.rolling(window=T, min_periods=T).mean()
        return result
