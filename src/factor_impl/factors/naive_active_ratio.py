import pandas as pd

from factors.base import BaseFactor


class NaiveActiveRatioFactor(BaseFactor):
    """朴素主动占比因子"""

    name = "NAIVE_ACTIVE_RATIO"
    category = "高频资金流"
    description = "基于 t 分布 CDF 估计的主动买入金额占比，衡量买方驱动力量"

    def compute(
        self,
        daily_naive_active_ratio: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算朴素主动占比因子。

        日内计算逻辑（预计算阶段）：
        1. 主动买入金额_i = Amount_i * t_cdf((Close_i - Close_{i-1}) / sigma, df)
           其中 t_cdf 为 t 分布累计分布函数，sigma 为区间内价格变动标准差
        2. 朴素主动占比 = sum(主动买入金额) / sum(Amount)

        本方法对预计算的日度因子取 T 日 EMA。

        Args:
            daily_naive_active_ratio: 预计算的每日朴素主动占比，
                index=日期, columns=股票代码。
            T: EMA 窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        result = daily_naive_active_ratio.ewm(span=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 朴素主动占比因子采用 t 分布 CDF 作为连续值系数来估计主动买入金额，
# 相比传统的 0/1 划分更精确。价格正向变动以买方驱动为主，
# 价格负向变动以卖方驱动为主；价格变动越大，主要驱动力量占比越大。
