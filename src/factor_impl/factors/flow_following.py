import numpy as np
import pandas as pd
from scipy import stats

from factors.base import BaseFactor


class FlowFollowingFactor(BaseFactor):
    """"随波逐流"因子 (Going with the Flow)。"""

    name = "FLOW_FOLLOWING"
    category = "高频量价"
    description = "个股高低位成交额差异与其他股票的平均绝对Spearman相关性"

    def compute(
        self,
        high_low_diff: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算随波逐流因子。

        Args:
            high_low_diff: 每日 (高位成交额 - 低位成交额) / 流通市值，
                index=日期, columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        dates = high_low_diff.index
        stocks = high_low_diff.columns
        vals = high_low_diff.values
        num_dates, num_stocks = vals.shape

        result = np.full((num_dates, num_stocks), np.nan)

        for t in range(T - 1, num_dates):
            window = vals[t - T + 1 : t + 1]  # (T, num_stocks)

            # 对每只股票，计算与其他所有股票的 |Spearman 相关系数| 的均值
            for s in range(num_stocks):
                col_s = window[:, s]
                if np.any(np.isnan(col_s)):
                    continue

                abs_corrs = []
                for j in range(num_stocks):
                    if j == s:
                        continue
                    col_j = window[:, j]
                    if np.any(np.isnan(col_j)):
                        continue
                    # 检查常数序列
                    if np.std(col_s) == 0 or np.std(col_j) == 0:
                        continue
                    corr, _ = stats.spearmanr(col_s, col_j)
                    if not np.isnan(corr):
                        abs_corrs.append(abs(corr))

                if len(abs_corrs) > 0:
                    result[t, s] = np.mean(abs_corrs)

        return pd.DataFrame(result, index=dates, columns=stocks)


# ==============================================================================
# 核心思想与原理说明
# ==============================================================================
#
# "随波逐流"因子的核心思想：
#
# 1. 对于每只股票，计算日内高位成交额与低位成交额之差，除以流通市值，
#    得到标准化的"高低位成交额差异"指标。
#
# 2. 在过去 T 天内，计算该指标与所有其他股票同指标的 Spearman 秩相关系数
#    的绝对值的均值。
#
# 3. 因子值高意味着该股票的高低位成交模式与市场整体高度同步，即"随波逐流"。
#    这类股票可能缺乏独立的信息驱动，更多受市场情绪影响。
#
# ==============================================================================
# 简单用法示例
# ==============================================================================
#
# import pandas as pd
# import numpy as np
# from factors.flow_following import FlowFollowingFactor
#
# dates = pd.date_range("2024-01-01", periods=30, freq="B")
# stocks = ["000001.SZ", "000002.SZ", "600000.SH"]
#
# np.random.seed(42)
# hld = pd.DataFrame(
#     np.random.normal(0, 0.01, (30, 3)), index=dates, columns=stocks
# )
#
# factor = FlowFollowingFactor()
# result = factor.compute(high_low_diff=hld, T=20)
# print(result.tail())
