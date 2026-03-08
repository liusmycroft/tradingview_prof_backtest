import numpy as np
import pandas as pd

from factors.base import BaseFactor


class CorrectedIntradayReversalFactor(BaseFactor):
    """修正的日内反转因子 (Corrected Intraday Reversal)。"""

    name = "CORRECTED_INTRADAY_REVERSAL"
    category = "高频动量反转"
    description = "根据日内波动率修正的日内反转因子，低波动时翻转日内收益方向"

    def compute(
        self,
        intraday_return: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算修正的日内反转因子。

        公式: 若日内波动率 < 截面均值，则翻转日内收益的符号；否则保持原值。
              取滚动 T 日均值。

        Args:
            intraday_return: 日内收益率 (close/open - 1)，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 修正后的日内反转因子，index=日期，columns=股票代码。
        """
        # 用日内收益率的绝对值作为日内波动率的代理
        intraday_vol = intraday_return.abs()

        # 截面均值 (每日所有股票的均值)
        cross_section_mean = intraday_vol.mean(axis=1)

        # 低波动标记: 日内波动率 < 截面均值
        low_vol_mask = intraday_vol.lt(cross_section_mean, axis=0)

        # 修正: 低波动时翻转符号
        corrected = intraday_return.copy()
        corrected[low_vol_mask] = -corrected[low_vol_mask]

        # 滚动 T 日均值
        result = corrected.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 修正的日内反转因子基于以下观察：
# 1. 日内收益率通常具有反转效应（今天日内涨，明天倾向于跌）。
# 2. 但这种反转效应在低波动环境下更为显著。
#
# 因此，对于日内波动率低于截面均值的股票，翻转其日内收益的符号，
# 增强反转信号；对于高波动股票，保持原始信号。
#
# 取滚动均值可以平滑噪声，得到更稳定的因子值。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.corrected_intraday_reversal import CorrectedIntradayReversalFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=30)
#   intraday_ret = pd.DataFrame(
#       np.random.randn(30, 3) * 0.01,
#       index=dates,
#       columns=["000001.SZ", "000002.SZ", "600000.SH"],
#   )
#
#   factor = CorrectedIntradayReversalFactor()
#   result = factor.compute(intraday_return=intraday_ret, T=20)
#   print(result.tail())
