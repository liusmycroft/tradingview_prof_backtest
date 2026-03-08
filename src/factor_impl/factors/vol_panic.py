import numpy as np
import pandas as pd

from .base import BaseFactor


class VolPanicFactor(BaseFactor):
    """波动率加剧-惊恐因子"""

    name = "VOL_PANIC"
    category = "高频动量反转"
    description = "波动率加剧-惊恐因子，结合惊恐度与日内波动率加权调整收益率"

    def compute(
        self,
        daily_vol_panic: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算波动率加剧-惊恐因子。

        公式:
            惊恐度 = abs(stock_ret - mkt_ret) / (abs(stock_ret) + abs(mkt_ret) + 0.1)
            加权调整收益率 = 惊恐度 * 日收益率 * 日内波动率
            惊恐收益 = mean(加权调整收益率, T日)
            惊恐波动 = std(加权调整收益率, T日)
            因子 = (惊恐收益 + 惊恐波动) / 2

        Args:
            daily_vol_panic: 预计算的每日加权调整收益率
                (惊恐度 * 日收益率 * 日内波动率)
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 波动率加剧-惊恐因子值
        """
        panic_ret = daily_vol_panic.rolling(window=T, min_periods=T).mean()
        panic_vol = daily_vol_panic.rolling(window=T, min_periods=T).std()
        result = (panic_ret + panic_vol) / 2
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 波动率加剧-惊恐因子在原始惊恐因子上新增了日内波动率权重。
# 惊恐度 = abs(个股收益率 - 市场收益率) / (abs(个股收益率) + abs(市场收益率) + 0.1)
# 衡量个股收益率对市场收益率的偏离程度。
#
# 将惊恐度、日收益率、日内波动率三者相乘得到加权调整收益率，
# 再计算过去 T 日的均值（惊恐收益）和标准差（惊恐波动），
# 等权合成为最终因子。
#
# 显著效应在波动率加剧时更加强烈，投资者会被具有显著收益的股票所吸引，
# 导致它们被错误定价。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.vol_panic import VolPanicFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_vol_panic = pd.DataFrame(
#       np.random.randn(30, 2) * 0.01,
#       index=dates, columns=stocks,
#   )
#
#   factor = VolPanicFactor()
#   result = factor.compute(daily_vol_panic=daily_vol_panic, T=20)
#   print(result.tail())
