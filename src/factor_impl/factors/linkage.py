import numpy as np
import pandas as pd

from factors.base import BaseFactor


class LinkageFactor(BaseFactor):
    """相似业务收益联动因子 (Business Similarity Return Linkage)"""

    name = "LINKAGE"
    category = "图谱网络-动量溢出"
    description = "基于业务相似度加权的关联公司超额收益，捕捉动量溢出效应"

    def compute(
        self,
        similarity: pd.DataFrame,
        returns: pd.Series,
        **kwargs,
    ) -> pd.Series:
        """计算相似业务收益联动因子。

        公式:
            Linkage_i = sum(SIM_{i,j} * Ret_j) / sum(SIM_{i,j}) - Ret_i

        Args:
            similarity: 业务相似度矩阵 (index=股票代码, columns=股票代码)
                SIM_{i,j} = cos(P_i, P_j)，基于业务关键词向量的余弦相似度。
            returns: 当期各股票收益率 (index=股票代码)

        Returns:
            pd.Series: Linkage 因子值 (index=股票代码)
        """
        # 对齐：只保留 similarity 和 returns 共有的股票
        common = similarity.index.intersection(returns.index)
        sim = similarity.loc[common, common].copy()
        ret = returns.reindex(common, fill_value=np.nan)

        # 排除自身：对角线置零
        np.fill_diagonal(sim.values, 0.0)

        # 加权平均收益
        sim_arr = sim.values
        ret_arr = ret.values
        weighted = sim_arr * ret_arr[np.newaxis, :]
        numerator = np.nansum(weighted, axis=1)
        denominator = sim.sum(axis=1).values

        # 权重和为0时置NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            weighted_ret = np.where(denominator != 0, numerator / denominator, np.nan)

        result = pd.Series(weighted_ret, index=common) - ret
        result.name = self.name
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 相似业务收益联动因子度量了当期与公司 i 业务相似的公司超过 i 公司的
# 收益率。该值越大，说明 i 公司越有可能在下一期出现补涨行情。
#
# 业务相似度 SIM_{i,j} 基于公司年报中业务描述的关键词向量余弦相似度。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.linkage import LinkageFactor
#
#   stocks = ["A", "B", "C"]
#   similarity = pd.DataFrame(
#       [[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]],
#       index=stocks, columns=stocks,
#   )
#   returns = pd.Series({"A": 0.05, "B": -0.02, "C": 0.10})
#
#   factor = LinkageFactor()
#   result = factor.compute(similarity=similarity, returns=returns)
#   print(result)
