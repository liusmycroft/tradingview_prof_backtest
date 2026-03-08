import pandas as pd

from factors.base import BaseFactor


class VolumeProportionCompositeFactor(BaseFactor):
    """成交量占比复合因子 (Opening/Closing Volume Proportion Composite, OBCVP)"""

    name = "OBCVP"
    category = "高频成交分布"
    description = "开盘与收盘集合竞价成交量占比的加权复合因子"

    def compute(
        self,
        opening_auction_volume: pd.DataFrame,
        closing_auction_volume: pd.DataFrame,
        daily_volume: pd.DataFrame,
        T: int = 20,
        alpha: float = 0.5,
        **kwargs,
    ) -> pd.DataFrame:
        """计算成交量占比复合因子。

        公式:
            OCVP = rolling_mean(opening_auction_volume / daily_volume, T)
            BCVP = rolling_mean(closing_auction_volume / daily_volume, T)
            OBCVP = alpha * OCVP + (1 - alpha) * BCVP

        Args:
            opening_auction_volume: 开盘集合竞价成交量 (index=日期, columns=股票代码)
            closing_auction_volume: 收盘前5分钟成交量 (index=日期, columns=股票代码)
            daily_volume: 日内总成交量 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20
            alpha: OCVP 权重，默认 0.5

        Returns:
            pd.DataFrame: 成交量占比复合因子值
        """
        ocvp = (opening_auction_volume / daily_volume).rolling(
            window=T, min_periods=T
        ).mean()
        bcvp = (closing_auction_volume / daily_volume).rolling(
            window=T, min_periods=T
        ).mean()
        result = alpha * ocvp + (1 - alpha) * bcvp
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 集合竞价阶段是反映投资者行为信息的重要时点。开盘集合竞价成交量占比
# (OCVP) 和收盘集合竞价成交量占比 (BCVP) 分别衡量开盘和收盘时段的
# 成交集中度。复合因子 OBCVP 将两者加权组合。
# 集合竞价成交量占比越低，股票次月的收益率越高。
