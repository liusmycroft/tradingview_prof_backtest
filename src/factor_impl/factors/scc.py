import numpy as np
import pandas as pd

from factors.base import BaseFactor


class SCCFactor(BaseFactor):
    """空间网络相对中心度 (Spatial Network Centrality - SCC) 因子。"""

    name = "SCC"
    category = "网络结构"
    description = "基于收益率相关性的空间网络中心度，衡量股票在网络中的核心程度"

    def compute(
        self,
        returns: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 SCC 因子。

        Args:
            returns: 日收益率，index=日期, columns=股票代码。
            T: 回看窗口天数，默认 20。

        Returns:
            pd.DataFrame: SCC 因子值，index=日期, columns=股票代码。
        """
        dates = returns.index
        stocks = returns.columns
        vals = returns.values
        num_dates, num_stocks = vals.shape

        scc = np.full((num_dates, num_stocks), np.nan)

        for t in range(T - 1, num_dates):
            window = vals[t - T + 1 : t + 1]  # (T, num_stocks)

            # 检查是否有足够的非NaN数据
            valid_mask = ~np.isnan(window).any(axis=0)
            if valid_mask.sum() < 2:
                continue

            # 计算相关系数矩阵
            valid_cols = np.where(valid_mask)[0]
            sub = window[:, valid_cols]
            # 标准化
            means = np.nanmean(sub, axis=0)
            stds = np.nanstd(sub, axis=0, ddof=0)
            stds[stds == 0] = np.nan
            normed = (sub - means) / stds

            corr_matrix = np.dot(normed.T, normed) / T  # (n, n)

            n = len(valid_cols)
            for idx_i in range(n):
                # 平均相关系数 (排除自身)
                corr_sum = 0.0
                count = 0
                for idx_j in range(n):
                    if idx_i == idx_j:
                        continue
                    c = corr_matrix[idx_i, idx_j]
                    if not np.isnan(c):
                        corr_sum += c
                        count += 1

                if count == 0:
                    continue

                p_bar = corr_sum / count
                d_bar_sq = 2.0 * (1.0 - p_bar)

                if d_bar_sq <= 0:
                    scc[t, valid_cols[idx_i]] = np.nan
                else:
                    scc[t, valid_cols[idx_i]] = 1.0 / d_bar_sq

        return pd.DataFrame(scc, index=dates, columns=stocks)


# ==============================================================================
# 核心思想与原理说明
# ==============================================================================
#
# 空间网络相对中心度 (SCC) 因子的核心思想：
#
# 1. 将股票市场视为一个空间网络，股票之间的"距离"由收益率相关性决定：
#    d_ij = sqrt(2 * (1 - corr(r_i, r_j)))
#
# 2. 对于每只股票，计算其与所有其他股票的平均相关系数 p_bar_i，
#    进而得到平均距离 d_bar_i = sqrt(2 * (1 - p_bar_i))。
#
# 3. SCC_i = 1 / d_bar_i^2 = 1 / (2 * (1 - p_bar_i))
#    SCC 值越高，说明该股票与其他股票的平均相关性越强，在网络中越"中心"。
#
# 4. 中心度高的股票通常是市场的风向标，对系统性风险更敏感。
#
# ==============================================================================
# 简单用法示例
# ==============================================================================
#
# import pandas as pd
# import numpy as np
# from factors.scc import SCCFactor
#
# dates = pd.date_range("2024-01-01", periods=30, freq="B")
# stocks = ["000001.SZ", "000002.SZ", "600000.SH"]
#
# np.random.seed(42)
# returns = pd.DataFrame(
#     np.random.normal(0, 0.02, (30, 3)), index=dates, columns=stocks
# )
#
# factor = SCCFactor()
# result = factor.compute(returns=returns, T=20)
# print(result.tail())
