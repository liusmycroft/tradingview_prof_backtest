import pandas as pd

from factors.base import BaseFactor


class TrendCapitalNetSupportFactor(BaseFactor):
    """趋势资金净支撑量因子"""

    name = "TREND_CAPITAL_NET_SUPPORT"
    category = "量价因子改进"
    description = "趋势资金支撑成交量与阻力成交量之差除以流通股本，衡量趋势资金的净支撑力度"

    def compute(
        self,
        daily_trend_net_support: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算趋势资金净支撑量因子。

        日内计算逻辑（预计算阶段）：
        1. 回看过去 k 日分钟成交量的 m% 分位数作为阈值 (k=5, m=90)
        2. 当日分钟成交量 > 阈值的分钟定义为趋势资金交易
        3. 趋势资金分钟收盘价均值为分界线：
           - 收盘价 < 均值的分钟成交量加总 = 支撑成交量
           - 收盘价 > 均值的分钟成交量加总 = 阻力成交量
        4. daily = (支撑成交量 - 阻力成交量) / 流通股本

        本方法对预计算的日度因子取 T 日滚动均值。

        Args:
            daily_trend_net_support: 预计算的每日趋势资金净支撑量，
                index=日期, columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        result = daily_trend_net_support.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 趋势资金净支撑量因子只考虑趋势资金（成交量超过近期高分位数阈值的
# 分钟）的净支撑情况。因子值较大表明支撑价格不破位的力量较强，
# 后市上涨概率较高；因子值较小则表明阻碍价格上涨的力量较强，
# 后市相对偏空。
