import numpy as np
import pandas as pd

from factors.base import BaseFactor


class ModifiedBusinessLinkageFactor(BaseFactor):
    """修正的相似业务收益联动因子"""

    name = "MODIFIED_BUSINESS_LINKAGE"
    category = "图谱网络-动量溢出"
    description = "修正的相似业务收益联动因子，基于业务相似度加权的关联公司收益"

    def compute(
        self,
        similarity_weighted_return: pd.DataFrame,
        own_return: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算修正的相似业务收益联动因子。

        公式:
            linkage = similarity_weighted_return - own_return
            factor = rolling_mean(linkage, T)

        用业务相似度加权的关联公司收益减去自身收益，衡量动量溢出。

        Args:
            similarity_weighted_return: 业务相似度加权的关联公司收益率，
                                        index=日期，columns=股票代码。
            own_return: 个股自身收益率，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 修正的相似业务收益联动因子值 (T日滚动均值)。
        """
        linkage = similarity_weighted_return - own_return
        result = linkage.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 修正的相似业务收益联动因子基于公司间的业务相似度构建关联网络，
# 计算关联公司的加权收益率与自身收益率的差值。当关联公司表现优于
# 自身时，存在动量溢出的可能，即自身未来可能跟随上涨。
# 修正版本在原始联动因子基础上扣除了自身收益的影响。
#
# 【使用示例】
#
#   from factors.modified_business_linkage import ModifiedBusinessLinkageFactor
#   factor = ModifiedBusinessLinkageFactor()
#   result = factor.compute(
#       similarity_weighted_return=sim_ret_df,
#       own_return=own_ret_df, T=20
#   )
