import numpy as np
import pandas as pd

from factors.base import BaseFactor


class KLinePatternFactor(BaseFactor):
    """K线形态因子 (K-Line Pattern)"""

    name = "K_LINE_PATTERN"
    category = "量价因子改进"
    description = "基于2K形态胜率的时间衰退加权得分，度量K线形态表现"

    def compute(
        self,
        pattern_win_rate: pd.DataFrame,
        lookback: int = 40,
        half_life: float = 20.0,
        **kwargs,
    ) -> pd.DataFrame:
        """计算K线形态因子。

        公式:
            KP_{i,T} = sum(w_t * wr_{i,t})
            w_t = 0.5^((T-t)/lambda)

        Args:
            pattern_win_rate: 每日K线形态胜率 wr_{i,t} (index=日期, columns=股票代码)
            lookback: 回望期天数，默认 40
            half_life: 半衰期，默认 20

        Returns:
            pd.DataFrame: K线形态因子值
        """
        result = pd.DataFrame(np.nan, index=pattern_win_rate.index, columns=pattern_win_rate.columns)

        for i in range(lookback - 1, len(pattern_win_rate)):
            window = pattern_win_rate.iloc[i - lookback + 1 : i + 1]
            distances = np.arange(lookback - 1, -1, -1, dtype=float)
            weights = np.power(0.5, distances / half_life)
            weighted_sum = window.multiply(weights, axis=0).sum()
            result.iloc[i] = weighted_sum

        return result
