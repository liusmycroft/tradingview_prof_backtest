import numpy as np
import pandas as pd

from .base import BaseFactor


class InformedSellRatioFactor(BaseFactor):
    """知情主卖占比因子 (Informed Main Sell Ratio)。"""

    name = "INFORMED_SELL_RATIO"
    category = "高频资金流"
    description = "知情主卖占比因子，基于知情交易者主动卖出金额占总成交金额的比例"

    def compute(
        self,
        informed_sell_amount: pd.DataFrame,
        total_amount: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算知情主卖占比因子。

        Args:
            informed_sell_amount: 知情交易者主动卖出金额，
                                  index=日期，columns=股票代码。
            total_amount: 总成交金额，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期，columns=股票代码。
                          滚动 T 日均值。
        """
        # 每日知情主卖占比
        ratio = informed_sell_amount / total_amount

        # 滚动 T 日均值
        result = ratio.rolling(window=T, min_periods=T).mean()

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 知情主卖占比因子衡量知情交易者主动卖出行为的强度。核心公式：
#   ratio_t = informed_sell_amount_t / total_amount_t
#   factor = (1/T) * sum(ratio_{t-T+1:t})
#
# 经济直觉：知情交易者拥有信息优势，其主动卖出行为往往预示着未来股价下跌。
# 知情主卖占比越高，股票未来收益越差，因为具有信息优势的知情交易者
# 不看好股票未来表现。
