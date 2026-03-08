import pandas as pd

from .base import BaseFactor


class PriceResiliencyFactor(BaseFactor):
    """价格弹性因子 (Price Resiliency)。"""

    name = "PRICE_RESILIENCY"
    category = "高频流动性"
    description = "价格弹性因子，衡量单位成交额下股价的波动幅度及其近期变化趋势"

    def compute(
        self,
        daily_resiliency: pd.DataFrame,
        T_short: int = 20,
        T_long: int = 120,
    ) -> pd.DataFrame:
        """计算价格弹性因子。

        公式: resiliency = (high - low) / turnover
        因子值 = (EMA_short - MA_long) / STD_long

        Args:
            daily_resiliency: 预计算的每日价格弹性均值 (high-low)/turnover，
                              index=日期，columns=股票代码。
            T_short: 短期 EMA 窗口天数，默认 20。
            T_long: 长期滚动窗口天数，默认 120。

        Returns:
            pd.DataFrame: 因子值，index=日期，columns=股票代码。
        """
        ema_short = daily_resiliency.ewm(span=T_short, min_periods=T_short).mean()
        ma_long = daily_resiliency.rolling(window=T_long, min_periods=T_long).mean()
        std_long = daily_resiliency.rolling(window=T_long, min_periods=T_long).std(ddof=1)

        result = (ema_short - ma_long) / std_long

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 价格弹性因子反映了单位成交额下股价的波动幅度，与未来收益正相关。
# 核心公式：
#   resiliency = (high - low) / turnover
#   factor = (EMA_20 - MA_120) / STD_120
#
# 弹性越大说明单位成交额对价格冲击越大，市场流动性越弱，
# 而流动性较低的个股未来收益表现相对较好。
