import numpy as np
import pandas as pd

from factors.base import BaseFactor


class SearchRatioDownstreamFactor(BaseFactor):
    """年度搜索比-downstream 因子 (Annual Search Ratio Downstream)。"""

    name = "SEARCH_RATIO_DOWNSTREAM"
    category = "动量溢出"
    description = "基于搜索关联度加权的下游公司收益率，捕捉动量溢出效应"

    def compute(
        self,
        search_weights: pd.DataFrame,
        returns: pd.Series,
        **kwargs,
    ) -> pd.Series:
        """计算年度搜索比-downstream 因子。

        公式: R_i = sum(f_ij * Ret_j) / sum(f_ij)
              其中 f_ij 为公司 i 到公司 j 的搜索关联度权重。

        Args:
            search_weights: 搜索关联度矩阵，index=公司i，columns=公司j，
                           值为搜索比 f_ij。
            returns: 各公司收益率，index=公司代码。

        Returns:
            pd.Series: 加权收益率，index=公司代码。
        """
        # 确保 returns 的 index 与 search_weights 的 columns 对齐
        common = search_weights.columns.intersection(returns.index)
        weights = search_weights[common]
        ret = returns[common]

        # 加权收益: sum(f_ij * Ret_j) / sum(f_ij)
        weight_sum = weights.sum(axis=1)
        weight_sum[weight_sum == 0] = np.nan

        weighted_ret = weights.dot(ret) / weight_sum

        return weighted_ret


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 年度搜索比-downstream 因子利用投资者在搜索引擎上的联合搜索行为，
# 构建公司间的关联度矩阵。当投资者搜索公司 i 后又搜索公司 j，
# 说明两者在投资者认知中存在关联。
#
# 该因子计算的是：以搜索关联度为权重，对下游关联公司的收益率进行加权平均。
# 核心逻辑是"动量溢出"——关联公司的收益率变动会逐步传导到目标公司。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.search_ratio_downstream import SearchRatioDownstreamFactor
#
#   stocks = ["A", "B", "C", "D"]
#   weights = pd.DataFrame(
#       [[0, 0.3, 0.5, 0.2],
#        [0.4, 0, 0.1, 0.5],
#        [0.2, 0.3, 0, 0.5],
#        [0.1, 0.6, 0.3, 0]],
#       index=stocks, columns=stocks,
#   )
#   returns = pd.Series([0.02, -0.01, 0.03, 0.01], index=stocks)
#
#   factor = SearchRatioDownstreamFactor()
#   result = factor.compute(search_weights=weights, returns=returns)
#   print(result)
