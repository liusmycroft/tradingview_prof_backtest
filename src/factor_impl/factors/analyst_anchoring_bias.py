"""基于 PE 的分析师锚效益偏差因子 (Analyst Anchoring Bias via PE, CAF_EP)

CAF_EP_{i,t} = (FEP_{i,t} - IFEP_t) / IFEP_t
FEP_{i,t} = 1 / FPE_{i,t}
IFEP_t = median(FEP) within industry
"""

import pandas as pd

from factors.base import BaseFactor


class AnalystAnchoringBiasFactor(BaseFactor):
    """基于 PE 的分析师锚效益偏差因子 (CAF_EP)"""

    name = "CAF_EP"
    category = "行为金融-锚定效应"
    description = "基于 PE 的分析师锚效益偏差，衡量分析师预测相对行业中位数的偏离程度"

    def compute(
        self,
        forecast_pe: pd.DataFrame,
        industry: pd.Series,
        **kwargs,
    ) -> pd.DataFrame:
        """计算分析师锚效益偏差因子。

        Args:
            forecast_pe: 分析师预测 PE (index=日期, columns=股票代码)。
            industry: 股票所属行业 (index=股票代码, values=行业代码)。

        Returns:
            pd.DataFrame: 因子值 CAF_EP (index=日期, columns=股票代码)。
        """
        # FEP = 1 / FPE
        fep = 1.0 / forecast_pe

        # 按行业计算 FEP 中位数 IFEP
        result = pd.DataFrame(index=fep.index, columns=fep.columns, dtype=float)

        for date in fep.index:
            row = fep.loc[date]
            for ind in industry.unique():
                members = industry[industry == ind].index
                members_in_row = members.intersection(row.index)
                if len(members_in_row) == 0:
                    continue
                ifep = row[members_in_row].median()
                if ifep == 0 or pd.isna(ifep):
                    result.loc[date, members_in_row] = float("nan")
                else:
                    result.loc[date, members_in_row] = (
                        (row[members_in_row] - ifep) / ifep
                    )

        return result.astype(float)


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 锚定效应指分析师以公司所在行业作为定价的"锚"。当预测的 EP 超出行业
# 中位数时，分析师会做出悲观预测以接近行业中位数，财报公布后实际值将
# 高于预测值，股价表现更优。反之亦然。
#
# CAF_EP > 0 表示分析师预测的 EP 高于行业中位数，可能存在锚定偏差。
