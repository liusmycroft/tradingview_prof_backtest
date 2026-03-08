import pandas as pd

from factors.base import BaseFactor


class UpsideVolRatioFactor(BaseFactor):
    """高频上行波动占比因子 (HF Upside Volatility Ratio)"""

    name = "UPSIDE_VOL_RATIO"
    category = "高频波动"
    description = "上行波动占总波动的比例滚动均值，衡量波动的方向性特征"

    def compute(
        self,
        daily_upside_ratio: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算高频上行波动占比因子。

        公式: ratio = sum(r_i^2 * I(r_i>0)) / sum(r_i^2)
        因子值为 T 日滚动均值。

        Args:
            daily_upside_ratio: 预计算的每日上行波动占比 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 上行波动占比的 T 日滚动均值
        """
        result = daily_upside_ratio.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 高频上行波动占比因子将日内收益率的平方分为上行（r>0）和下行（r<0）
# 两部分，计算上行波动占总波动的比例。
#
# 该比例反映了波动的方向性：比例高于 0.5 说明上行波动占主导，
# 比例低于 0.5 说明下行波动占主导。因子取 T 日滚动均值以平滑噪声。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.upside_vol_ratio import UpsideVolRatioFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_upside_ratio = pd.DataFrame(
#       np.random.uniform(0.3, 0.7, (30, 2)),
#       index=dates, columns=stocks,
#   )
#
#   factor = UpsideVolRatioFactor()
#   result = factor.compute(daily_upside_ratio=daily_upside_ratio, T=20)
#   print(result.tail())
