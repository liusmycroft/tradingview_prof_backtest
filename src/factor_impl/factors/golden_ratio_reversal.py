import numpy as np
import pandas as pd

from .base import BaseFactor


class GoldenRatioReversalFactor(BaseFactor):
    """日内黄金分割反转因子 (Intraday Golden Ratio Reversal)。"""

    name = "GOLDEN_RATIO_REVERSAL"
    category = "高频动量反转"
    description = "日内黄金分割反转，基于10:00价格到收盘价的对数收益率滚动求和"

    def compute(
        self,
        close: pd.DataFrame,
        price_1000: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算日内黄金分割反转因子。

        Args:
            close: 收盘价，index=日期，columns=股票代码。
            price_1000: 10:00 AM 价格，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期，columns=股票代码。
                          滚动 T 日 ln(Close / Price_10:00) 之和。
        """
        # 计算每日从 10:00 到收盘的对数收益率
        log_return = np.log(close / price_1000)

        # 滚动 T 日求和
        result = log_return.rolling(window=T, min_periods=T).sum()

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 日内黄金分割反转因子捕捉的是日内从 10:00 到收盘这段时间的价格动量信号。
# 10:00 是开盘后约 30 分钟，此时早盘的噪声交易已基本消化，价格趋于稳定，
# 被视为日内的"黄金分割"时点。
#
# 核心逻辑：
#   1. 每日计算 ln(Close_t / Price_10:00_t)，即 10:00 到收盘的对数收益率。
#   2. 对过去 T 天（默认 20 天）的对数收益率求和，得到累积动量信号。
#   3. 该因子通常呈现反转特征——累积动量过高的股票未来倾向于回调。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.golden_ratio_reversal import GoldenRatioReversalFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=25)
#   close = pd.DataFrame(
#       {"000001.SZ": [10 + 0.1 * i for i in range(25)]},
#       index=dates,
#   )
#   price_1000 = pd.DataFrame(
#       {"000001.SZ": [10 + 0.05 * i for i in range(25)]},
#       index=dates,
#   )
#
#   factor = GoldenRatioReversalFactor()
#   result = factor.compute(close=close, price_1000=price_1000)
#   print(result)
