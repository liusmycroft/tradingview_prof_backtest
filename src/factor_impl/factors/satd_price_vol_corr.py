import pandas as pd

from factors.base import BaseFactor


class SatdPriceVolCorrFactor(BaseFactor):
    """量价背离时刻笔均成交金额因子 (SATD_PriceVolCorr)"""

    name = "SATD_PRICE_VOL_CORR"
    category = "高频成交分布"
    description = "量价背离时刻笔均成交金额与全天笔均成交金额之比，衡量主力资金在量价背离时的交易强度"

    def compute(
        self,
        daily_satd: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算量价背离时刻笔均成交金额因子。

        日内计算逻辑（已预计算为 daily_satd）:
          1. 将日内240分钟按价格与成交量相关系数排序，取最小50%为量价背离时刻集合P
          2. ATD_P = sum(Amt_t for t in P) / sum(DealNum_t for t in P)
          3. ATD_T = sum(Amt_t for all t) / sum(DealNum_t for all t)
          4. SATD = ATD_P / ATD_T

        因子值为 T 日 EMA。

        Args:
            daily_satd: 预计算的每日 SATD 值 (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: SATD 的 T 日 EMA
        """
        result = daily_satd.ewm(span=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 量价背离时刻笔均成交金额因子通过汇总"量价背离"这一特殊时刻的成交金额
# 情况来刻画主力资金的行为动向。量价背离时刻指日内价格与成交量相关系数
# 最小的50%时刻，在这些时刻笔均成交金额越大，说明主力资金优势越明显。
# 因子取 T 日 EMA 以平滑日间波动。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.satd_price_vol_corr import SatdPriceVolCorrFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_satd = pd.DataFrame(
#       np.random.uniform(0.8, 1.5, (30, 2)),
#       index=dates, columns=stocks,
#   )
#
#   factor = SatdPriceVolCorrFactor()
#   result = factor.compute(daily_satd=daily_satd, T=20)
#   print(result.tail())
