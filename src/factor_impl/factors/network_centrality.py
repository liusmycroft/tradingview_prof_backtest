import pandas as pd

from factors.base import BaseFactor


class NetworkCentralityFactor(BaseFactor):
    """股票网络中心度因子 (Stock Network Centrality - CC)"""

    name = "NETWORK_CENTRALITY"
    category = "网络结构"
    description = "空间网络中心度与时间网络中心度的等权组合，衡量股票在网络中的综合核心程度"

    def compute(
        self,
        scc: pd.DataFrame,
        tcc: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算股票网络中心度因子。

        公式: CC = 0.5 * SCC + 0.5 * TCC

        Args:
            scc: 空间网络中心度 (index=日期, columns=股票代码)
            tcc: 时间网络中心度 (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 综合网络中心度
        """
        result = 0.5 * scc + 0.5 * tcc
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 股票网络中心度因子将空间网络中心度 (SCC) 和时间网络中心度 (TCC)
# 进行等权组合，综合衡量股票在市场网络中的核心程度。
#
# SCC 基于收益率的横截面相关性构建空间网络，衡量股票与其他股票的
# 同期联动程度；TCC 基于收益率的时序领先-滞后关系构建时间网络，
# 衡量股票对其他股票的领先/跟随程度。
#
# 综合中心度高的股票通常是市场的风向标，对系统性风险更敏感。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.network_centrality import NetworkCentralityFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ", "600000.SH"]
#
#   np.random.seed(42)
#   scc = pd.DataFrame(
#       np.random.rand(30, 3), index=dates, columns=stocks,
#   )
#   tcc = pd.DataFrame(
#       np.random.rand(30, 3), index=dates, columns=stocks,
#   )
#
#   factor = NetworkCentralityFactor()
#   result = factor.compute(scc=scc, tcc=tcc)
#   print(result.tail())
