import pandas as pd

from factors.base import BaseFactor


class VolumeValleyPriceFactor(BaseFactor):
    """量谷相对加权价格因子"""

    name = "VOLUME_VALLEY_PRICE"
    category = "高频量价相关性"
    description = "量谷时段成交量加权价格相对于全天VWAP的比值，衡量低成交量时段的价格水平"

    def compute(
        self,
        valley_vwap: pd.DataFrame,
        daily_vwap: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算量谷相对加权价格因子。

        公式:
            daily_ratio = valley_vwap / daily_vwap
            factor = rolling_mean(daily_ratio, T)

        量谷: 日内成交量处于低谷的时段（如成交量低于日均的时段）。
        valley_vwap: 量谷时段的成交量加权平均价。
        daily_vwap: 全天的成交量加权平均价。

        Args:
            valley_vwap: 每日量谷时段VWAP (index=日期, columns=股票代码)
            daily_vwap: 每日全天VWAP (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 量谷相对加权价格的 T 日滚动均值
        """
        daily_ratio = valley_vwap / daily_vwap
        result = daily_ratio.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 量谷相对加权价格衡量成交量低谷时段的价格水平相对于全天均价的偏离。
# 若量谷时段价格偏高（ratio > 1），说明低成交量时段价格被推高，
# 可能存在拉抬行为；若偏低（ratio < 1），说明低量时段价格承压。
# 因子取 T 日滚动均值以平滑日间波动。
