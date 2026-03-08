import pandas as pd

from factors.base import BaseFactor


class MidPriceChangeFactor(BaseFactor):
    """中间价变化率因子 (Mid-Price Change Rate, MPC)"""

    name = "MPC"
    category = "高频流动性"
    description = "中间价变化率的T日滚动均值，衡量中间价的平均变动幅度"

    def compute(
        self,
        daily_mid_price_change: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算中间价变化率因子。

        公式: MPC = rolling_mean(daily_mid_price_change, T)

        mid_price = (best_ask + best_bid) / 2
        mid_price_change = |mid_price_t - mid_price_{t-1}| / mid_price_{t-1}
        daily_mid_price_change 为日内中间价变化率的均值。

        Args:
            daily_mid_price_change: 预计算的每日中间价变化率均值
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 中间价变化率的 T 日滚动均值
        """
        result = daily_mid_price_change.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 中间价变化率 MPC 衡量买卖最优报价中间价的平均变动幅度。
# 中间价变化率越大，说明市场微观价格波动越剧烈，流动性可能越差。
# 因子取 T 日滚动均值以平滑日间噪声。
