import pandas as pd

from factors.base import BaseFactor


class TrendRatioFactor(BaseFactor):
    """趋势占比因子 (Trend Ratio)"""

    name = "TREND_RATIO"
    category = "高频动量反转"
    description = "日内价格位移与路程之比，衡量日内股价的趋势强度"

    def compute(
        self,
        daily_trend_ratio: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算趋势占比因子。

        日内计算逻辑（预计算阶段）：
        trend_ratio = (P_T - P_1) / sum(|P_t - P_{t-1}|, t=2..T)
        即日内价格位移除以路程，取值范围 [-1, 1]。

        本方法对预计算的日度因子取 T 日滚动均值。

        Args:
            daily_trend_ratio: 预计算的每日趋势占比，
                index=日期, columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        result = daily_trend_ratio.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 趋势占比可以看作日内价格位移与路程之比，衡量日内股价的趋势强度。
# 取值范围为 [-1, 1]，通常与未来收益负相关。
# 趋势占比绝对值越大，说明日内价格走势越单边；
# 趋势占比绝对值越小，说明日内价格走势越震荡。
