import numpy as np
import pandas as pd

from factors.base import BaseFactor


class CustomerClosenessFactor(BaseFactor):
    """客户接近中心性因子 (Customer Closeness Centrality)"""

    name = "CUSTOMER_CLOSENESS"
    category = "图谱网络-网络结构"
    description = "供应链网络中客户节点的接近中心性，衡量信息传播效率"

    def compute(
        self,
        daily_closeness: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算客户接近中心性因子。

        公式:
            c_i = 1 / ( (1/(N-1)) * sum_{j!=i} D_ij )
            即所有供应商到客户i的平均最短路径的倒数。

        Args:
            daily_closeness: 预计算的每期客户接近中心性
                (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 客户接近中心性因子值
        """
        return daily_closeness.copy()


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 接近中心性衡量供应链网络中能达到某个客户节点的所有供应商节点的
# 平均最短距离的倒数。接近中心性越高，说明该企业与网络中其他企业
# 建立连接的路径长度越短。
#
# 供应链网络中占据中心位置的企业更能吸引投资者注意力，市场对其
# 信息反应更快，有较低的股票回报可预测性；中心性较低的企业由于
# 信息传播较慢，有较高的股票回报可预测性。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.customer_closeness import CustomerClosenessFactor
#
#   dates = pd.date_range("2024-01-01", periods=4, freq="Q")
#   stocks = ["000001.SZ", "000002.SZ", "600000.SH"]
#   np.random.seed(42)
#   daily_closeness = pd.DataFrame(
#       np.random.uniform(0.1, 1.0, (4, 3)),
#       index=dates, columns=stocks,
#   )
#
#   factor = CustomerClosenessFactor()
#   result = factor.compute(daily_closeness=daily_closeness)
#   print(result)
