import pandas as pd

from factors.base import BaseFactor


class ComplexMomentumFactor(BaseFactor):
    """复杂公司动量因子 (Complex Company Momentum - COMRET)"""

    name = "COMPLEX_MOMENTUM"
    category = "动量溢出"
    description = "基于行业营收权重的加权行业收益率，捕捉复杂公司的动量溢出效应"

    def compute(
        self,
        industry_weights: pd.DataFrame,
        industry_returns: pd.Series,
        **kwargs,
    ) -> pd.Series:
        """计算复杂公司动量因子。

        公式: COMRET_i = sum(W_ij * RET_j) / sum(W_ij)
        其中 W_ij 为公司 i 在行业 j 的营收占比权重。

        Args:
            industry_weights: 公司×行业营收权重矩阵
                (index=股票代码, columns=行业代码)
            industry_returns: 各行业纯净收益率 (index=行业代码)

        Returns:
            pd.Series: 各公司的 COMRET 因子值 (index=股票代码)
        """
        # 对齐行业维度
        common_industries = industry_weights.columns.intersection(industry_returns.index)
        weights = industry_weights[common_industries]
        returns = industry_returns[common_industries]

        # 加权收益
        weighted_ret = weights.mul(returns, axis=1).sum(axis=1)
        weight_sum = weights.sum(axis=1)

        # 避免除零
        weight_sum = weight_sum.replace(0, float("nan"))
        result = weighted_ret / weight_sum
        result.name = self.name
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 复杂公司动量因子 (COMRET) 的核心思想来自 Cohen & Lou (2012)：
# 业务跨多个行业的"复杂公司"，其股价对行业信息的反应较慢。
#
# 通过将公司在各行业的营收占比作为权重，加权计算行业纯净收益率，
# 可以构造出该公司"应有"的收益率。当实际收益率滞后于 COMRET 时，
# 存在动量溢出效应，可用于预测未来收益。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.complex_momentum import ComplexMomentumFactor
#
#   stocks = ["000001.SZ", "000002.SZ", "600000.SH"]
#   industries = ["银行", "地产", "科技"]
#
#   industry_weights = pd.DataFrame(
#       [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.0, 0.1, 0.9]],
#       index=stocks, columns=industries,
#   )
#   industry_returns = pd.Series([0.02, -0.01, 0.03], index=industries)
#
#   factor = ComplexMomentumFactor()
#   result = factor.compute(
#       industry_weights=industry_weights,
#       industry_returns=industry_returns,
#   )
#   print(result)
