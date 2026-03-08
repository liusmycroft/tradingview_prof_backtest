import pandas as pd

from .base import BaseFactor


class SupplyChainDegreeFactor(BaseFactor):
    """供应链节点度"""

    name = "SUPPLY_CHAIN_DEGREE"
    category = "图谱网络-网络结构"
    description = "供应链节点度因子，衡量公司供应链关系的复杂程度"

    def compute(
        self,
        daily_in_degree: pd.DataFrame,
        daily_out_degree: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算供应链节点度因子。

        公式:
            d_i = d_i^(in) + d_i^(out)

        Args:
            daily_in_degree: 每日入度（供应商数量）
                (index=日期, columns=股票代码)
            daily_out_degree: 每日出度（客户数量）
                (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 供应链节点度因子值
        """
        result = daily_in_degree + daily_out_degree
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 供应链节点度 d_i = d_in + d_out，其中 d_in 为入度（供应商数量），
# d_out 为出度（客户数量）。供应链图谱网络边的方向为供应商指向客户。
#
# 节点度较高的公司供应链关系更复杂，可以通过多样化供应链或建立冗余系统
# 来减少潜在的供应链中断风险（物流对冲）；供应链简单的公司由于缺乏
# 物流对冲会面临更高的风险，但投资者因承担了更高的风险可获得更高的
# 风险溢价。建议对因子做市值行业中心化处理。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.supply_chain_degree import SupplyChainDegreeFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_in_degree = pd.DataFrame(
#       np.random.randint(0, 10, (30, 2)).astype(float),
#       index=dates, columns=stocks,
#   )
#   daily_out_degree = pd.DataFrame(
#       np.random.randint(0, 10, (30, 2)).astype(float),
#       index=dates, columns=stocks,
#   )
#
#   factor = SupplyChainDegreeFactor()
#   result = factor.compute(
#       daily_in_degree=daily_in_degree,
#       daily_out_degree=daily_out_degree,
#   )
#   print(result.tail())
