import pandas as pd

from factors.base import BaseFactor


class VolumeEntropyFactor(BaseFactor):
    """成交量分桶熵因子 (Volume Bucketing Entropy)"""

    name = "VOLUME_ENTROPY"
    category = "高频成交分布"
    description = "成交量分桶熵的滚动标准差：衡量成交量分布不确定性的波动"

    def compute(
        self,
        daily_entropy: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算成交量分桶熵因子。

        公式: vol_entropy = -sum(p_k * ln(p_k)) for p_k > 0
        因子值为 T 日滚动标准差。

        Args:
            daily_entropy: 预计算的每日熵值 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 熵值的 T 日滚动标准差
        """
        result = daily_entropy.rolling(window=T, min_periods=1).std()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 成交量分桶熵将每日的成交量按金额大小分成若干桶（bucket），计算各桶的
# 概率分布，然后用信息熵 H = -sum(p_k * ln(p_k)) 来度量成交量分布的
# 不确定性。熵越高，说明成交量在各桶中分布越均匀；熵越低，说明成交量
# 集中在少数桶中。
#
# 因子取 T 日滚动标准差，衡量熵值的波动程度。熵值波动大说明成交量分布
# 结构不稳定，可能反映市场参与者结构的变化。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.volume_entropy import VolumeEntropyFactor
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
#   factor = VolumeEntropyFactor()
#   result = factor.compute(daily_entropy=daily_entropy, T=20)
#   print(result.tail())
