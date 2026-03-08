import numpy as np
import pandas as pd

from factors.base import BaseFactor


class AnalystCoCoverageIndirectMomentumFactor(BaseFactor):
    """分析师共同覆盖间接关联动量因子 (Analyst Co-Coverage Indirect Linkage Momentum)"""

    name = "ANALYST_CO_COVERAGE_INDIRECT_MOMENTUM"
    category = "图谱网络"
    description = "通过分析师共同覆盖的间接关联强度加权关联股票收益率，捕捉隐蔽的领先滞后效应"

    def compute(
        self,
        indirect_strength: pd.DataFrame,
        peer_returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算分析师共同覆盖间接关联动量因子。

        公式:
            m_{ij} = sum_k log(n_{ik}+1) * log(n_{kj}+1)
            CF2_RET_i = sum_j(m_{ij} * Ret_j) / sum_j(m_{ij})

        本因子接收预计算的间接关联加权收益率。

        Args:
            indirect_strength: 间接关联强度加权的关联股票收益率
                (index=日期, columns=股票代码)，即 CF2_RET
            peer_returns: 自身过去20日收益率 (index=日期, columns=股票代码)，
                用于回归取残差

        Returns:
            pd.DataFrame: 间接关联动量因子值(残差)
        """
        # 对每只股票做横截面回归取残差: CF2_RET ~ Ret
        # 简化为逐列回归
        result = pd.DataFrame(index=indirect_strength.index,
                              columns=indirect_strength.columns, dtype=float)

        for col in indirect_strength.columns:
            y = indirect_strength[col]
            x = peer_returns[col]
            mask = y.notna() & x.notna()
            if mask.sum() < 2:
                result[col] = np.nan
                continue
            y_valid = y[mask]
            x_valid = x[mask]
            x_mean = x_valid.mean()
            y_mean = y_valid.mean()
            beta = ((x_valid - x_mean) * (y_valid - y_mean)).sum() / (
                (x_valid - x_mean) ** 2
            ).sum()
            alpha = y_mean - beta * x_mean
            residual = y - (alpha + beta * x)
            result[col] = residual

        return result.astype(float)
