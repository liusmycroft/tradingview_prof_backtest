"""主买占比因子 (Active Buy Ratio)

factor = (1/T) * sum_{n=t-T+1}^{t} (主动买入成交额 / 成交额)
"""

import pandas as pd

from factors.base import BaseFactor


class ActiveBuyRatioFactor(BaseFactor):
    """主买占比因子"""

    name = "ACTIVE_BUY_RATIO"
    category = "高频资金流"
    description = "主买占比，主动买入成交额占总成交额的 T 日滚动均值"

    def compute(
        self,
        daily_active_buy_ratio: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算主买占比因子。

        Args:
            daily_active_buy_ratio: 预计算的每日主买占比
                (index=日期, columns=股票代码)。
                每日值 = sum(主动买入成交额) / sum(成交额)，已剔除涨跌停分钟。
            T: 滚动窗口天数，默认 20（月度选股）。

        Returns:
            pd.DataFrame: 因子值，T 日滚动均值。
        """
        result = daily_active_buy_ratio.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 主买占比因子基于逐笔成交数据中的 BS 标志，将成交数据合成为分钟级别的
# 主动买卖金额。B 为主动买入，S 为主动卖出。剔除涨跌停分钟后，计算
# 主动买入成交额占总成交额的比例，取 T 日滚动均值。
#
# 除收盘时段外，主买占比呈现动量效应：前期主买占比越高，未来表现越好。
