import numpy as np
import pandas as pd

from factors.base import BaseFactor


class DpinFactor(BaseFactor):
    """动态知情交易概率因子 (Dynamic Probability of Informed Trading)"""

    name = "DPIN"
    category = "高频资金流"
    description = "基于日内非预期收益方向的知情交易占比，衡量信息不对称程度"

    def compute(
        self,
        daily_dpin_mean: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算动态知情交易概率因子。

        日内计算逻辑（已预计算为 daily_dpin_mean）:
          1. 构建自回归模型计算非预期收益率 eps_{i,j}
          2. DPIN_BASE = (NB/NT)*I(eps<0) + (NS/NT)*I(eps>0)
          3. daily_dpin_mean 为日内各区间 DPIN_BASE 的均值

        因子值为 T 日滚动均值。

        Args:
            daily_dpin_mean: 预计算的每日 DPIN 均值 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_dpin_mean.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# DPIN 基于日内交易活动将得知新信息后的反转交易占比作为知情交易概率的
# 衡量指标。当非预期收益为负时，知情买入占比较高；当非预期收益为正时，
# 知情卖出占比较高。DPIN 越大，信息不对称程度越大，投资者要求的风险
# 回报越高，通常与未来收益正相关。
