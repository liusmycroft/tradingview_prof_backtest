import numpy as np
import pandas as pd

from factors.base import BaseFactor


class BusinessComplexityLinkageFactor(BaseFactor):
    """结合业务复杂度的相似业务收益联动因子 (Business Complexity Weighted Linkage)"""

    name = "BIZ_COMPLEXITY_LINKAGE"
    category = "行为金融-投资者注意力"
    description = "业务复杂度加权的相似公司超额收益联动，衡量动量溢出效应"

    def compute(
        self,
        daily_complexity: pd.DataFrame,
        daily_sim_excess_ret: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算结合业务复杂度的相似业务收益联动因子。

        预计算逻辑:
          SIM_{i,j} = cosine(P_i, P_j)  (业务关键词向量余弦相似度)
          complexity_i = mean(SIM跨行业) - mean(SIM同行业)
          sim_excess_ret_i = sum(SIM*Ret)/sum(SIM) - Ret_i

        因子值: Linkage = complexity * sim_excess_ret，取 T 日 EMA。

        Args:
            daily_complexity: 预计算的业务复杂度 (index=日期, columns=股票代码)
            daily_sim_excess_ret: 预计算的相似公司超额收益
                                  (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 联动因子的 T 日 EMA
        """
        linkage = daily_complexity * daily_sim_excess_ret
        result = linkage.ewm(span=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 相似业务收益联动因子度量了与公司i相似的公司在当月超过i公司的收益率，
# 该值越大，说明i公司越有可能在下一期出现补涨行情。公司业务复杂度越高，
# 投资者可能更难把握和判断市场信息对其的影响，带来认知资源的限制，
# 进而对应更强的动量溢出效应。
