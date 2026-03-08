import pandas as pd

from factors.base import BaseFactor


class PvolRetCorrFactor(BaseFactor):
    """每笔成交量收益率相关性因子 (Per-Transaction Volume Return Correlation)"""

    name = "PVOL_RET_CORR"
    category = "高频量价"
    description = "每笔成交量与收益率的相关性：日度相关性的T日滚动均值"

    def compute(
        self,
        daily_pvol_corr: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算每笔成交量收益率相关性因子。

        公式: (1/T) * sum(corr(pvol, ret)) where pvol = volume / trade_count

        Args:
            daily_pvol_corr: 每日预计算的每笔成交量与收益率相关性
                             (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        result = daily_pvol_corr.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 每笔成交量 (pvol = volume / trade_count) 反映单笔交易的平均规模，
# 大单交易通常对应更高的每笔成交量。本因子计算每笔成交量与收益率
# 的相关性，正相关意味着大单推动价格上涨（买入主导），负相关意味着
# 大单推动价格下跌（卖出主导）。T日均值平滑短期噪声。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.pvol_ret_corr import PvolRetCorrFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_pvol_corr = pd.DataFrame(
#       np.random.uniform(-0.5, 0.5, (30, 2)), index=dates, columns=stocks
#   )
#
#   factor = PvolRetCorrFactor()
#   result = factor.compute(daily_pvol_corr=daily_pvol_corr, T=20)
#   print(result.tail())
