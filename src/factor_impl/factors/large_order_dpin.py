import pandas as pd

from factors.base import BaseFactor


class LargeOrderDpinFactor(BaseFactor):
    """大单的动态知情交易概率因子 (Large Order DPIN)"""

    name = "LARGE_ORDER_DPIN"
    category = "高频资金流"
    description = "大单下的动态知情交易概率，筛选大单区间的知情交易占比"

    def compute(
        self,
        daily_large_dpin: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算大单的动态知情交易概率因子。

        日内预计算逻辑:
          1. 构建自回归模型: R_{i,j} = gamma_0 + sum(gamma_1*D_Day) +
             sum(gamma_2*D_Int) + sum(gamma_3*R_{i,j-k}) + eps
          2. DPIN = [NB/NT * I(eps<0) + NS/NT * I(eps>0)] * LT
             其中 LT=1 当区间成交量超过当日各区间中位数
          3. daily_large_dpin 为日内各区间 DPIN 的均值

        Args:
            daily_large_dpin: 预计算的每日大单DPIN均值 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_large_dpin.rolling(window=T, min_periods=1).mean()
        return result
