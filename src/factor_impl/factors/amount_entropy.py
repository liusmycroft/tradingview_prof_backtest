import pandas as pd

from factors.base import BaseFactor


class AmountEntropyFactor(BaseFactor):
    """成交额占比熵因子 (Transaction Amount Entropy)"""

    name = "AMOUNT_ENTROPY"
    category = "高频量价"
    description = "成交额占比熵的滚动均值：衡量成交额在时间段内的分布均匀程度"

    def compute(
        self,
        daily_entropy: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算成交额占比熵因子。

        公式: H = -sum(p_k * ln(p_k)) where p_k = amount_k / total_amount
        因子值为 T 日滚动均值。

        Args:
            daily_entropy: 预计算的每日熵值 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 熵值的 T 日滚动均值
        """
        result = daily_entropy.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 成交额占比熵将每日的成交额按时间段（如分钟）划分，计算各时间段成交额
# 占总成交额的比例 p_k，然后用信息熵 H = -sum(p_k * ln(p_k)) 来度量
# 成交额分布的均匀程度。熵越高，说明成交额在各时间段分布越均匀；
# 熵越低，说明成交额集中在少数时间段。
#
# 因子取 T 日滚动均值，平滑日间波动，反映近期成交额分布的平均特征。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.amount_entropy import AmountEntropyFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_entropy = pd.DataFrame(
#       np.random.rand(30, 2) * 2 + 1,
#       index=dates, columns=stocks,
#   )
#
#   factor = AmountEntropyFactor()
#   result = factor.compute(daily_entropy=daily_entropy, T=20)
#   print(result.tail())
