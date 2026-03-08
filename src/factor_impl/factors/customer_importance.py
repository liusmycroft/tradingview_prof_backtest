import numpy as np
import pandas as pd

from factors.base import BaseFactor


class CustomerImportanceFactor(BaseFactor):
    """客户重要性因子 (Customer Importance - PageRank)"""

    name = "CUSTOMER_IMPORTANCE"
    category = "图谱网络-网络结构"
    description = "利用 PageRank 算法计算客户重要性，考虑供应商本身的重要性"

    def compute(
        self,
        adjacency_matrix: np.ndarray,
        stock_list: list,
        q: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-8,
        **kwargs,
    ) -> pd.DataFrame:
        """计算客户重要性因子 (PageRank)。

        公式:
            p(i) = q/V + (1-q) * sum_{j->i} p(j) / k_j^out

        Args:
            adjacency_matrix: 邻接矩阵 (V x V)，A[j][i]=1 表示 j->i
            stock_list: 股票代码列表，长度 V
            q: 阻尼因子，默认 0.85
            max_iter: 最大迭代次数
            tol: 收敛阈值

        Returns:
            pd.DataFrame: PageRank 得分 (index=股票代码, columns=["pagerank"])
        """
        V = len(stock_list)
        A = np.array(adjacency_matrix, dtype=float)

        out_degree = A.sum(axis=1)
        out_degree[out_degree == 0] = 1  # 避免除零

        p = np.ones(V) / V

        for _ in range(max_iter):
            p_new = np.zeros(V)
            for i in range(V):
                incoming = np.where(A[:, i] > 0)[0]
                s = 0.0
                for j in incoming:
                    s += p[j] / out_degree[j]
                p_new[i] = q / V + (1 - q) * s

            if np.max(np.abs(p_new - p)) < tol:
                p = p_new
                break
            p = p_new

        result = pd.DataFrame(p, index=stock_list, columns=["pagerank"])
        return result
