import pandas as pd

from factors.base import BaseFactor


class ExtremeReturnReversalFactor(BaseFactor):
    """极端收益反转因子 (Extreme Return Reversal)"""

    name = "EXTREME_RETURN_REVERSAL"
    category = "高频动量反转"
    description = "日内最极端收益bar及其前一分钟收益的截面排序等权合成，捕捉极端收益反转效应"

    def compute(
        self,
        daily_extreme_ret: pd.DataFrame,
        daily_pre_extreme_ret: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算极端收益反转因子。

        公式:
            rank(mean(r_{extreme}, T)) + rank(mean(r_{pre_extreme}, T))

        本方法接收预计算的每日最极端bar收益率和前一分钟收益率，
        输出滚动 T 日均值的截面排序等权合成。

        Args:
            daily_extreme_ret: 每日最极端bar的收益率 (index=日期, columns=股票代码)
            daily_pre_extreme_ret: 每日最极端bar前一分钟的收益率 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 极端收益反转因子值
        """
        # 滚动 T 日均值
        mean_extreme = daily_extreme_ret.rolling(window=T, min_periods=T).mean()
        mean_pre = daily_pre_extreme_ret.rolling(window=T, min_periods=T).mean()

        # 截面排序（每行rank）并等权合成
        rank_extreme = mean_extreme.rank(axis=1, pct=True)
        rank_pre = mean_pre.rank(axis=1, pct=True)

        result = rank_extreme + rank_pre
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 极端收益反转因子基于日内最极端收益bar的反转特性构建。每日寻找日内
# 收益最极端的那根bar（偏离中位数最大），取其收益率和前一分钟收益率，
# 滚动 T 日求均值后截面排序等权合成。
#
# 日内最极端收益前呈反转特性，最极端收益后呈动量特性；切割出的极端
# 收益反转因子具有强反转效应。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.extreme_return_reversal import ExtremeReturnReversalFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=30)
#   stocks = ["000001.SZ", "000002.SZ", "600000.SH"]
#   daily_extreme = pd.DataFrame(
#       np.random.randn(30, 3) * 0.02, index=dates, columns=stocks,
#   )
#   daily_pre = pd.DataFrame(
#       np.random.randn(30, 3) * 0.01, index=dates, columns=stocks,
#   )
#
#   factor = ExtremeReturnReversalFactor()
#   result = factor.compute(
#       daily_extreme_ret=daily_extreme, daily_pre_extreme_ret=daily_pre, T=20,
#   )
#   print(result.tail())
