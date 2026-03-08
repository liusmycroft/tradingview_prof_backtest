import pandas as pd

from factors.base import BaseFactor


class ABNRETDFactor(BaseFactor):
    """最大异常日收益率因子 (Max Abnormal Daily Return - ABNRETD)"""

    name = "ABNRETD"
    category = "行为金融-投资者注意力"
    description = "月内日频收益率与市场收益率绝对差值的最大值，衡量投资者关注度"

    def compute(
        self,
        stock_returns: pd.DataFrame,
        market_returns: pd.Series,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算最大异常日收益率因子。

        公式: ABNRETD = max_t |r_{i,t} - r_mkt_t|  (over rolling T days)

        Args:
            stock_returns: 个股日收益率 (index=日期, columns=股票代码)
            market_returns: 全市场日收益率 (index=日期)
            T: 滚动窗口天数，默认 20（月度）

        Returns:
            pd.DataFrame: ABNRETD 因子值
        """
        # 计算每日异常收益的绝对值
        abnormal = stock_returns.sub(market_returns, axis=0).abs()
        # 滚动窗口取最大值
        result = abnormal.rolling(window=T, min_periods=T).max()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# ABNRETD 为每月日频收益率与市场收益率绝对差值的最大值，属于极端日收益
# 类因子。极端收益（过高或过低）的股票更能吸引投资者注意，因子值越高，
# 投资者对该股票的关注度越高。
#
# A股市场做多与做空不对称，使得投资者非理性买入高于非理性卖出，进而
# 导致关注度高的股票由于投资者净买入而存在溢价，未来更可能出现反转。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.abnretd import ABNRETDFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=30)
#   stocks = ["000001.SZ", "000002.SZ"]
#   stock_returns = pd.DataFrame(
#       np.random.randn(30, 2) * 0.02, index=dates, columns=stocks,
#   )
#   market_returns = pd.Series(np.random.randn(30) * 0.01, index=dates)
#
#   factor = ABNRETDFactor()
#   result = factor.compute(stock_returns=stock_returns, market_returns=market_returns)
#   print(result.tail())
