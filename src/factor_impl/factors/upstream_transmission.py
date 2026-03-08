import numpy as np
import pandas as pd

from .base import BaseFactor


class UpstreamTransmissionFactor(BaseFactor):
    """上游传导动量因子 (Upstream Transmission Momentum)。"""

    name = "UPSTREAM_TRANSMISSION"
    category = "动量溢出"
    description = "上游传导动量，从下游公司视角计算上游公司收益率的加权平均"

    def compute(
        self,
        weights: pd.DataFrame,
        returns: pd.Series,
    ) -> pd.Series:
        """计算上游传导动量因子。

        与 UpstreamMomentumFactor 方向相反：对于下游公司 A，加权平均其
        上游公司的收益率。

        Args:
            weights: 供应链关联权重矩阵，index=下游股票，columns=上游股票，
                     values=关联权重 wgt_{A,B,t}。
            returns: 当月所有股票的收益率，index=股票代码。

        Returns:
            pd.Series: 每个下游公司的上游传导动量因子值，index=股票代码。
        """
        # 对齐：只保留 weights 列中存在收益数据的上游股票
        common_upstream = weights.columns.intersection(returns.index)
        w = weights[common_upstream].copy()
        r = returns.reindex(common_upstream, fill_value=np.nan)

        # 排除自身：对角线置零（仅当行列有交集时）
        common = w.index.intersection(w.columns)
        for stock in common:
            w.loc[stock, stock] = 0.0

        # 加权求和：transmission_A = sum(wgt_AB * rtn_B) / sum(wgt_AB)
        w_arr = w.values
        r_arr = r.values
        weighted = w_arr * r_arr[np.newaxis, :]  # (n_downstream, n_upstream)
        numerator = pd.Series(np.nansum(weighted, axis=1), index=w.index)

        # 如果某行存在 wgt>0 但 rtn=NaN 的情况，结果应为 NaN
        has_nan_contribution = np.any(
            (w_arr != 0) & np.isnan(r_arr[np.newaxis, :]), axis=1
        )
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
# 上游传导动量（Upstream Transmission Momentum）捕捉的是供应链中"上游供给
# 变化向下游传导"的价格信号。与 UpstreamMomentumFactor 方向相反：
#   - UpstreamMomentumFactor: 对于上游公司，加权平均下游公司的收益率。
#   - UpstreamTransmissionFactor: 对于下游公司，加权平均上游公司的收益率。
#
# 经济直觉：当上游供应商整体表现强劲时，下游公司可能面临成本上升压力，
# 但也可能受益于上游景气度的传导。该因子捕捉这种跨产业链的动量溢出效应。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.upstream_transmission import UpstreamTransmissionFactor
#
#   # 供应链关联权重矩阵（行=下游，列=上游）
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
#   factor = UpstreamTransmissionFactor()
#   result = factor.compute(weights=weights, returns=returns)
#   print(result)
