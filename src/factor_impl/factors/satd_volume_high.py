import pandas as pd

from factors.base import BaseFactor


class SATDVolumeHighFactor(BaseFactor):
    """成交量最高时刻笔均成交金额因子 (Standardized Avg Trade amount at peak volume moments)"""

    name = "SATD_VolumeHigh"
    category = "高频成交分布"
    description = "成交量最高时刻笔均成交金额：成交量最高10%时刻的笔均成交金额除以全天笔均成交金额，取T日滚动均值"

    def compute(
        self,
        daily_atd_high: pd.DataFrame,
        daily_atd_all: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算成交量最高时刻笔均成交金额因子。

        公式: SATD = rolling_mean_T( ATD_VolumeHigh10% / ATD_T )

        Args:
            daily_atd_high: 预计算的每日成交量最高10%时刻的笔均成交金额
                           (index=日期, columns=股票代码)
            daily_atd_all: 预计算的每日全天笔均成交金额
                          (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        daily_satd = daily_atd_high / daily_atd_all
        result = daily_satd.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 笔均成交额类因子通过汇总"特殊的、具有较多信息含量"时刻的成交金额
# 情况来刻画主力资金的行为动向。将日内240个1分钟数据中成交量最高的
# 10%时刻标记出来，计算这些时刻的笔均成交金额，再除以全天的笔均成交
# 金额进行标准化。笔均成交金额越大，说明该特殊时间内资金优势越明显，
# 属于主力资金的可能性越大。取 T 日滚动均值以平滑日间波动。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.satd_volume_high import SATDVolumeHighFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_atd_high = pd.DataFrame(
#       np.random.rand(30, 2) * 5000 + 3000, index=dates, columns=stocks,
#   )
#   daily_atd_all = pd.DataFrame(
#       np.random.rand(30, 2) * 3000 + 2000, index=dates, columns=stocks,
#   )
#
#   factor = SATDVolumeHighFactor()
#   result = factor.compute(daily_atd_high=daily_atd_high, daily_atd_all=daily_atd_all, T=20)
#   print(result.tail())
