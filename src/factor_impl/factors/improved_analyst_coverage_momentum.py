import numpy as np
import pandas as pd

from factors.base import BaseFactor


class ImprovedAnalystCoverageMomentumFactor(BaseFactor):
    """改进的分析师共同覆盖间接关联动量因子"""

    name = "IMPROVED_ANALYST_COVERAGE_MOMENTUM"
    category = "图谱网络-动量溢出"
    description = "以直接关联数量改进的间接关联强度加权关联股票收益率"

    def compute(
        self,
        direct_coverage: pd.DataFrame,
        ret_20d: pd.DataFrame,
        num_direct: pd.Series,
        **kwargs,
    ) -> pd.DataFrame:
        """计算改进的分析师共同覆盖间接关联动量因子。

        公式:
            m_ij = sum_k(log(n_ik+1)*log(n_kj+1)) / log(num_i+1)
            ADJ_CF2_RET_i = sum(m_ij * Ret_j) / sum(m_ij)

        Args:
            direct_coverage: 直接关联强度矩阵 n_ij (N x N DataFrame)
            ret_20d: 过去20日收益率 (Series, index=股票代码)
            num_direct: 每只股票的直接关联公司数量 (Series)

        Returns:
            pd.DataFrame: 因子值 (index=股票代码, columns=["factor"])
        """
        stocks = direct_coverage.index.tolist()
        N = len(stocks)
        n_mat = direct_coverage.values.astype(float)
        log_n = np.log(n_mat + 1)

        result_vals = np.full(N, np.nan)

        for i in range(N):
            stock_i = stocks[i]
            num_i = num_direct.get(stock_i, 0)
            denom_i = np.log(num_i + 1)
            if denom_i == 0:
                continue

            m_ij = np.zeros(N)
            for j in range(N):
                if i == j:
                    continue
                s = 0.0
                for k in range(N):
                    s += log_n[i, k] * log_n[k, j]
                m_ij[j] = s / denom_i

            total_m = np.sum(m_ij)
            if total_m == 0:
                continue

            ret_vals = ret_20d.reindex(stocks).values.astype(float)
            result_vals[i] = np.nansum(m_ij * ret_vals) / total_m

        return pd.DataFrame(result_vals, index=stocks, columns=["factor"])
