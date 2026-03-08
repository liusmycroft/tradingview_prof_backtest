"""客户端中心性变化因子 (Customer Centrality Change)

当期标准化后的客户接近中心性减去上期标准化中心性，
衡量供应商公司核心地位的变化程度。
"""

import pandas as pd

from factors.base import BaseFactor


class CustomerCentralityChangeFactor(BaseFactor):
    """客户端中心性变化因子"""

    name = "CUSTOMER_CENTRALITY_CHANGE"
    category = "图谱网络-网络结构"
    description = "当期与上期标准化客户接近中心性之差，衡量供应商公司核心地位的变化"

    def compute(
        self,
        current_centrality: pd.DataFrame,
        previous_centrality: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算客户端中心性变化因子。

        公式:
        1. C(u) = (n-1)/(N-1) * (n-1) / sum(D(u,v))
           其中 D(u,v) = exp(-s(u,v)), s(u,v) = sales_{u,v} / Total_Sales_u
        2. C_st(u) = (C(u) - C_min) / (C_max - C_min)  标准化
        3. factor = C_st_current - C_st_previous

        Args:
            current_centrality: 当期标准化后的客户接近中心性
                (index=日期, columns=股票代码)
            previous_centrality: 上期标准化后的客户接近中心性
                (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 因子值
        """
        result = current_centrality - previous_centrality
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 客户端中心性变化为当期标准化后的客户接近中心性减去上期标准化中心性，
# 衡量了供应商公司核心地位的变化程度。中心性变强的组为top组，收益更高。
# 客户端中心性变化的选股效果优于供应端中心性变化。
#
# 接近中心性基于客户关联度 s(u,v) = sales_{u,v} / Total_Sales_u
# 计算公司间距离 D(u,v) = exp(-s(u,v))，再通过调和平均距离的倒数得到。
