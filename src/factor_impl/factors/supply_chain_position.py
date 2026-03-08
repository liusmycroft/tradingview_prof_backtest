import numpy as np
import pandas as pd

from factors.base import BaseFactor


class SupplyChainPositionFactor(BaseFactor):
    """产业链地位优势因子 (Supply Chain Position Advantage)。"""

    name = "SUPPLY_CHAIN_POSITION"
    category = "网络结构"
    description = "产业链关联权重之和，衡量公司在供应链网络中的地位优势"

    def compute(
        self,
        weights: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        """计算产业链地位优势因子。

        Args:
            weights: 关联权重矩阵，index=公司, columns=公司。
                weights.loc[A, B] 表示公司 A 与公司 B 的关联权重。

        Returns:
            pd.Series: 地位优势得分，index=公司。
        """
        # 将对角线设为 0（排除自身）
        w = weights.values.copy()
        np.fill_diagonal(w, 0.0)

        # 对每个公司，求其与所有关联公司的权重之和
        scores = np.nansum(w, axis=1)

        return pd.Series(scores, index=weights.index)


# ==============================================================================
# 核心思想与原理说明
# ==============================================================================
#
# 产业链地位优势因子的核心思想：
#
# 1. 构建产业链关联网络，其中节点为公司，边权重为上下游关联强度。
#
# 2. 对于公司 A，其地位优势得分 = sum(w(A, j))，即与所有关联公司的
#    权重之和（包括上游供应商和下游客户）。
#
# 3. 得分越高，说明该公司在产业链中的连接越广泛、地位越重要。
#    这类公司通常具有更强的议价能力和更稳定的经营。
#
# ==============================================================================
# 简单用法示例
# ==============================================================================
#
# import pandas as pd
# import numpy as np
# from factors.supply_chain_position import SupplyChainPositionFactor
#
# companies = ["A", "B", "C", "D"]
# weights = pd.DataFrame(
#     [[0.0, 0.3, 0.2, 0.0],
#      [0.3, 0.0, 0.0, 0.5],
#      [0.2, 0.0, 0.0, 0.1],
#      [0.0, 0.5, 0.1, 0.0]],
#     index=companies, columns=companies,
# )
#
# factor = SupplyChainPositionFactor()
# result = factor.compute(weights=weights)
# print(result)
