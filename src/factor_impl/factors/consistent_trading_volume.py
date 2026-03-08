import pandas as pd

from factors.base import BaseFactor


class ConsistentTradingVolumeFactor(BaseFactor):
    """一致交易因子 (Consistent Trading Volume, TCV)"""

    name = "TCV"
    category = "高频成交分布"
    description = "一致交易量占比的T日滚动均值，衡量买卖方向一致的成交量占总成交量的比例"

    def compute(
        self,
        daily_consistent_volume_ratio: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算一致交易因子。

        一致交易定义: 在某个时间段内，若主动买入量与主动卖出量的方向一致
        （即净买入或净卖出占比超过阈值），则该时段的成交量视为一致交易量。
        TCV = 一致交易量 / 总成交量

        Args:
            daily_consistent_volume_ratio: 预计算的每日一致交易量占比
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 一致交易量占比的 T 日滚动均值
        """
        result = daily_consistent_volume_ratio.rolling(
            window=T, min_periods=1
        ).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 一致交易 TCV 衡量市场参与者交易方向的一致性程度。
# 当一致交易占比较高时，说明市场参与者对股票的看法趋于一致，
# 可能预示着趋势的延续或反转。
# 因子取 T 日滚动均值以平滑日间波动。
