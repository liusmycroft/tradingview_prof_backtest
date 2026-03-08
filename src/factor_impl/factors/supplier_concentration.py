import pandas as pd

from factors.base import BaseFactor


class SupplierConcentrationFactor(BaseFactor):
    """供应商行业集中度因子 (Supplier Industry Concentration - SH)"""

    name = "SUPPLIER_CONCENTRATION"
    category = "网络结构"
    description = "供应商行业集中度：基于采购权重与行业赫芬达尔指数的加权求和"

    def compute(
        self,
        procurement_weights: pd.DataFrame,
        herfindahl: pd.Series,
        **kwargs,
    ) -> pd.Series:
        """计算供应商行业集中度因子。

        公式: SH_i = sum_j(w_ji * H_j)
        其中 w_ji 为公司 i 对供应商行业 j 的采购权重，
        H_j 为行业 j 的赫芬达尔指数。

        Args:
            procurement_weights: 采购权重矩阵 (index=公司, columns=供应商行业)
                                 每行之和应为 1（或接近 1）。
            herfindahl: 各行业赫芬达尔指数 (index=行业代码)

        Returns:
            pd.Series: 每家公司的 SH 值 (index=公司)
        """
        # 对齐行业维度：只取两者共有的行业
        common_industries = procurement_weights.columns.intersection(herfindahl.index)
        weights_aligned = procurement_weights[common_industries]
        herfindahl_aligned = herfindahl[common_industries]

        # SH = W @ H (矩阵乘向量)
        result = weights_aligned.dot(herfindahl_aligned)
        result.name = self.name

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 供应商行业集中度 (SH) 衡量一家公司的供应链在行业层面的集中程度。
# 赫芬达尔指数 H_j = sum(s_ij^2) 度量行业 j 内部的集中度（s_ij 为行业 j
# 中各公司的市场份额）。将公司 i 对各供应商行业的采购权重 w_ji 与对应行业
# 的赫芬达尔指数 H_j 加权求和，即得到 SH_i。
#
# SH 越高，说明公司的供应商集中在少数高集中度行业中，供应链风险越大；
# SH 越低，说明供应链分散在多个竞争性行业中，抗风险能力更强。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.supplier_concentration import SupplierConcentrationFactor
#
#   companies = ["公司A", "公司B"]
#   industries = ["行业1", "行业2", "行业3"]
#
#   weights = pd.DataFrame(
#       [[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]],
#       index=companies, columns=industries,
#   )
#   herfindahl = pd.Series([0.25, 0.10, 0.40], index=industries)
#
#   factor = SupplierConcentrationFactor()
#   result = factor.compute(procurement_weights=weights, herfindahl=herfindahl)
#   print(result)
