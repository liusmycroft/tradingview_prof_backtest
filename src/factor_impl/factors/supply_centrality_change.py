import pandas as pd

from factors.base import BaseFactor


class SupplyCentralityChangeFactor(BaseFactor):
    """供给端中心性变化因子"""

    name = "SUPPLY_CENTRALITY_CHANGE"
    category = "图谱网络-网络结构"
    description = "当期标准化供应商接近中心性减去上期值，衡量客户公司核心地位的变化程度"

    def compute(
        self,
        current_centrality: pd.DataFrame,
        previous_centrality: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算供给端中心性变化因子。

        公式:
        1. C(u) = (n-1)/(N-1) * (n-1) / sum(D(u,v))
           其中 D(u,v) = exp(-s(u,v)), s(u,v) = sales_{u,v} / Total_Procurement_u
        2. C_st(u) = (C(u) - C_min) / (C_max - C_min)  标准化
        3. factor = C_st_current - C_st_previous

        Args:
            current_centrality: 当期标准化后的供应商接近中心性，
                index=日期, columns=股票代码。
            previous_centrality: 上期标准化后的供应商接近中心性，
                index=日期, columns=股票代码。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        result = current_centrality - previous_centrality
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 供给端中心性变化为当期标准化后的供应商接近中心性减去上期标准化中心性，
# 衡量客户公司核心地位的变化程度。中心性变强的组收益更高。
# 接近中心性基于供应关联度 s(u,v) 计算公司间距离 D(u,v) = exp(-s(u,v))，
# 再通过调和平均距离的倒数得到。
