import numpy as np
import pandas as pd

from factors.base import BaseFactor


class BottleneckCompanyFactor(BaseFactor):
    """瓶颈公司因子 (Bottleneck Company - Betweenness Centrality)"""

    name = "BOTTLENECK_COMPANY"
    category = "图谱网络"
    description = "供应链网络中的中介中心性，衡量公司在网络中的桥梁地位"

    def compute(
        self,
        betweenness_centrality: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算瓶颈公司因子。

        公式:
            C_B(i) = sum_{s!=t!=i} sigma_{st}(i) / sigma_{st}

        其中 sigma_{st} 为从s到t的所有最短路径数量，
        sigma_{st}(i) 为经过i的最短路径数量。

        Args:
            betweenness_centrality: 预计算的中介中心性
                (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 中介中心性因子值
        """
        return betweenness_centrality.copy()
