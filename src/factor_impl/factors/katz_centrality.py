import numpy as np
import pandas as pd

from .base import BaseFactor


class KatzCentralityFactor(BaseFactor):
    """客户Katz-Bonacich中心性因子"""

    name = "KATZ_CENTRALITY"
    category = "图谱网络-网络结构"
    description = "供应链网络中的Katz中心性，衡量节点在供应链中的综合重要性"

    def compute(
        self,
        adjacency: pd.DataFrame,
        lam: float = 0.1,
        beta: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
        **kwargs,
    ) -> pd.DataFrame:
        """计算Katz-Bonacich中心性因子。

        公式:
            k_i = lambda * sum(x_ij * k_j) + beta
            迭代直到收敛

        Args:
            adjacency: 供应链邻接矩阵 (index=columns=股票代码, 值为连接权重)
            lam: 衰减因子，通常小于邻接矩阵最大特征值的倒数，默认 0.1
            beta: 基础中心度，默认 1.0
            max_iter: 最大迭代次数，默认 100
            tol: 收敛阈值，默认 1e-6

        Returns:
            pd.DataFrame: Katz中心性因子值 (单行 DataFrame, columns=股票代码)
        """
        nodes = adjacency.columns
        n = len(nodes)
        A = adjacency.values.astype(float)

        # 迭代计算 k = lam * A @ k + beta
        k = np.full(n, beta, dtype=float)
        for _ in range(max_iter):
            k_new = lam * A @ k + beta
            if np.max(np.abs(k_new - k)) < tol:
                k = k_new
                break
            k = k_new

        result = pd.DataFrame([k], columns=nodes)
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# Katz 中心性在衡量供应链节点重要性时考虑了节点间的所有路径长度，
# 可以捕捉节点间较长路径的影响。供应链网络中占据中心位置的企业
# 更能吸引投资者的注意力，市场对其信息反应更快，有较低的股票回报
# 可预测性；中心性较低的企业由于信息传播较慢，有较高的股票回报
# 可预测性。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.katz_centrality import KatzCentralityFactor
#
#   stocks = ["000001.SZ", "000002.SZ", "600000.SH"]
#   adj = pd.DataFrame(
#       [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
#       index=stocks, columns=stocks, dtype=float,
#   )
#
#   factor = KatzCentralityFactor()
#   result = factor.compute(adjacency=adj, lam=0.1, beta=1.0)
#   print(result)
