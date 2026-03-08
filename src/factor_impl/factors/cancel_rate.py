import pandas as pd

from .base import BaseFactor


class CancelRateFactor(BaseFactor):
    """三小将-撤单率因子 (Three Soldiers Cancel Rate)。"""

    name = "CANCEL_RATE"
    category = "高频流动性"
    description = "三小将-撤单率因子，基于集合竞价阶段卖方全撤、部撤和废单撤单率等权合成"

    def compute(
        self,
        full_cancel_rate: pd.DataFrame,
        partial_cancel_rate: pd.DataFrame,
        invalid_cancel_rate: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算三小将-撤单率因子。

        Args:
            full_cancel_rate: 全撤撤单率（撤单量/自由流通股本），
                              index=日期，columns=股票代码。
            partial_cancel_rate: 部撤撤单率，index=日期，columns=股票代码。
            invalid_cancel_rate: 废单撤单率，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期，columns=股票代码。
                          三类撤单率等权合成后取 T 日均值。
        """
        # 等权合成
        composite = (full_cancel_rate + partial_cancel_rate + invalid_cancel_rate) / 3.0

        # T 日均值低频化
        result = composite.rolling(window=T, min_periods=T).mean()

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 三小将-撤单率因子基于早盘集合竞价阶段（9:15-9:20）的卖方撤单数据，
# 将全撤、部撤和废单三类撤单率等权合成。核心公式：
#   撤单率 = 撤单量 / 自由流通股本
#   factor = rolling_mean(T, (full + partial + invalid) / 3)
#
# 经济直觉：撤单率因子取值越高，股票当天的撤单越多，交易越不活跃，
# 而市场会给予低流动性标的更高的收益补偿，是一个正向因子。
