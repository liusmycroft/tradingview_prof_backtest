import numpy as np
import pandas as pd

from .base import BaseFactor


class UpstreamMomentumFactor(BaseFactor):
    """下游传导动量因子 (Upstream/Downstream Momentum)。"""

    name = "UpstreamMomentum"
    category = "动量"
    description = "下游传导动量，基于供应链关联权重计算上游公司的下游加权收益"

    def compute(
        self,
        weights: pd.DataFrame,
        returns: pd.Series,
    ) -> pd.Series:
        """计算下游传导动量因子。

        Args:
            weights: 供应链关联权重矩阵，index=上游股票，columns=下游股票，
                     values=关联权重 wgt_{A,B,t}。
            returns: 当月所有股票的收益率，index=股票代码。

        Returns:
            pd.Series: 每个上游公司的下游传导动量因子值，index=股票代码。
        """
        # 对齐：只保留 weights 列中存在收益数据的下游股票
        common_downstream = weights.columns.intersection(returns.index)
        w = weights[common_downstream].copy()
        r = returns.reindex(common_downstream, fill_value=np.nan)

        # 排除自身：对角线置零（仅当行列有交集时）
        common = w.index.intersection(w.columns)
        for stock in common:
            w.loc[stock, stock] = 0.0

        # 加权求和：upstream_moment_A = sum(wgt_AB * rtn_B) / sum(wgt_AB)
        # 逐元素相乘，让 NaN 只在非零权重处传播
        w_arr = w.values
        r_arr = r.values
        weighted = w_arr * r_arr[np.newaxis, :]  # (n_upstream, n_downstream)
        numerator = pd.Series(np.nansum(weighted, axis=1), index=w.index)

        # 如果某行存在 wgt>0 但 rtn=NaN 的情况，结果应为 NaN
        has_nan_contribution = np.any((w_arr != 0) & np.isnan(r_arr[np.newaxis, :]), axis=1)
        numerator[has_nan_contribution] = np.nan

        denominator = w.sum(axis=1)

        result = numerator / denominator
        # 权重和为 0 的公司无法计算，置为 NaN
        result[denominator == 0] = np.nan

        result.name = self.name
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 下游传导动量（Upstream Momentum）捕捉的是供应链中"下游需求变化向上游
# 传导"的价格信号。核心逻辑如下：
#   1. 通过产品级营收数据构建上下游公司之间的供应链关联权重 wgt_{A,B,t}，
#      权重越大说明 A、B 之间的供应链联系越紧密。
#   2. 对于上游公司 A，用其所有下游公司 B 的月度收益率按关联权重加权平均，
#      得到 A 的"下游传导动量"。
#
# 经济直觉：当下游公司整体表现强劲时，上游供应商往往也会受益于需求增长，
# 但由于信息传导存在时滞，上游公司的股价反应可能滞后。因此该因子具有一定
# 的收益预测能力——下游传导动量高的上游公司，未来收益倾向于更好。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.upstream_momentum import UpstreamMomentumFactor
#
#   # 供应链关联权重矩阵（行=上游，列=下游）
#   weights = pd.DataFrame(
#       {
#           "A": [0.0, 0.3, 0.1],
#           "B": [0.4, 0.0, 0.2],
#           "C": [0.2, 0.5, 0.0],
#       },
#       index=["A", "B", "C"],
#   )
#
#   # 当月各股票收益率
#   returns = pd.Series({"A": 0.05, "B": -0.02, "C": 0.10})
#
#   factor = UpstreamMomentumFactor()
#   result = factor.compute(weights=weights, returns=returns)
#   print(result)
