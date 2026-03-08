import numpy as np
import pandas as pd

from .base import BaseFactor


class TCCFactor(BaseFactor):
    """时间网络相对中心度因子 (Temporal Network Centrality - TCC)。"""

    name = "TCC"
    category = "网络结构"
    description = "时间网络相对中心度，基于个股收益率偏离市场均值的标准化程度衡量网络中心性"

    def compute(
        self,
        returns: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算 TCC 因子。

        Args:
            returns: 个股收益率，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: TCC 因子值，index=日期，columns=股票代码。
        """
        # 截面均值和标准差
        r_hat_m = returns.mean(axis=1)  # 每日截面均值
        sigma_m = returns.std(axis=1, ddof=1)  # 每日截面标准差

        # 标准化得分 z_i,t = (r_i,t - r_hat_m,t) / sigma_m,t
        z = returns.sub(r_hat_m, axis=0).div(sigma_m, axis=0)

        # z^2
        z_sq = z ** 2

        # 滚动 T 日均值的平方根: z_bar_i = sqrt(1/T * sum(z_i^2))
        z_bar_sq = z_sq.rolling(window=T, min_periods=T).mean()
        # z_bar = np.sqrt(z_bar_sq)  # 不需要单独计算

        # TCC = 1 / z_bar^2 = 1 / z_bar_sq
        tcc = 1.0 / z_bar_sq

        return tcc


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# TCC（Temporal Network Centrality，时间网络相对中心度）衡量个股在市场网络
# 中的中心性程度。核心逻辑：
#   1. 对每日截面收益率进行标准化，得到 z-score。
#   2. 对 z^2 在时间维度上取滚动均值再开方，得到 z_bar。
#   3. TCC = 1/z_bar^2，z_bar 越小说明该股票收益率越接近市场均值，
#      即在网络中越"中心"，TCC 值越大。
#
# 经济直觉：中心度高的股票与市场整体走势高度同步，可能反映了系统性风险暴露。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.tcc import TCCFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=25)
#   returns = pd.DataFrame(
#       np.random.randn(25, 5) * 0.02,
#       index=dates,
#       columns=["A", "B", "C", "D", "E"],
#   )
#
#   factor = TCCFactor()
#   result = factor.compute(returns=returns, T=20)
#   print(result)
