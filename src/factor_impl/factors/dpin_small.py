"""小单的动态知情交易概率因子 (DPIN_SMALL)

在DPIN基础上加入小单虚拟变量，筛选出小订单下的知情交易概率。
"""

import pandas as pd

from factors.base import BaseFactor


class DpinSmallFactor(BaseFactor):
    """小单的动态知情交易概率因子"""

    name = "DPIN_SMALL"
    category = "高频资金流"
    description = "小单条件下的动态知情交易概率，捕捉知情交易者拆单行为"

    def compute(
        self,
        daily_dpin_small_mean: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算小单的动态知情交易概率因子。

        日内计算逻辑（已预计算为 daily_dpin_small_mean）:
          1. 构建自回归模型计算非预期收益率 eps_{i,j}
          2. DPIN_SMALL = [NB/NT * I(eps<0) + NS/NT * I(eps>0)] * ST
             其中 ST 为小单虚拟变量（区间总交易量 < 当日各区间中位数时为1）
          3. daily_dpin_small_mean 为日内各区间 DPIN_SMALL 的均值

        因子值为 T 日滚动均值。

        Args:
            daily_dpin_small_mean: 预计算的每日小单DPIN均值
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_dpin_small_mean.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 在市场交易不活跃的时段，知情交易者为不暴露自身的信息优势，更有可能
# 选择将大订单拆成小订单进行交易，所以可以进一步筛选出小订单下的知情
# 交易概率。日内交易不活跃时段的小单知情交易概率通常更高，日内因子
# 走势整体呈倒"U"型。与未来收益正相关。
