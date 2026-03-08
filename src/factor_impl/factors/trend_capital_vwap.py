import pandas as pd

from factors.base import BaseFactor


class TrendCapitalVWAPFactor(BaseFactor):
    """趋势资金相对均价因子 (Trend Capital Relative VWAP)"""

    name = "TREND_CAPITAL_VWAP"
    category = "量价因子改进"
    description = "趋势资金VWAP与全量VWAP的比值减1，取T日滚动均值"

    def compute(
        self,
        daily_trend_vwap_ratio: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算趋势资金相对均价因子。

        公式:
            daily = 趋势资金VWAP / 全量VWAP - 1
            因子值为 T 日滚动均值。

        趋势资金定义：当日分钟成交量超过过去k日分钟成交量m%分位数的交易。

        Args:
            daily_trend_vwap_ratio: 预计算的每日趋势资金相对均价
                (趋势资金VWAP/全量VWAP - 1) (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 趋势资金相对均价因子的 T 日滚动均值
        """
        result = daily_trend_vwap_ratio.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 趋势资金相对均价因子衡量趋势资金（成交量超过阈值的分钟交易）的
# 成交量加权平均价格相对于全量VWAP的偏离程度。
#
# 因子值较大表明趋势资金交易价格相对较高，出货可能性更大，后市看空；
# 因子值较小表明趋势资金交易价格较低，更可能是逢低买入，后市看涨。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.trend_capital_vwap import TrendCapitalVWAPFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#   np.random.seed(42)
#   daily_trend_vwap_ratio = pd.DataFrame(
#       np.random.uniform(-0.02, 0.02, (30, 2)),
#       index=dates, columns=stocks,
#   )
#
#   factor = TrendCapitalVWAPFactor()
#   result = factor.compute(daily_trend_vwap_ratio=daily_trend_vwap_ratio, T=20)
#   print(result.tail())
