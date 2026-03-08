import pandas as pd

from factors.base import BaseFactor


class MOFIFactor(BaseFactor):
    """多层次订单失衡因子 (Multi-level Order Flow Imbalance - MOFI)"""

    name = "MOFI"
    category = "高频流动性"
    description = "五档订单失衡的加权平均，权重为档位序号/5，取T日滚动均值"

    def compute(
        self,
        daily_ofi_1: pd.DataFrame,
        daily_ofi_2: pd.DataFrame,
        daily_ofi_3: pd.DataFrame,
        daily_ofi_4: pd.DataFrame,
        daily_ofi_5: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算多层次订单失衡因子。

        公式:
            MOFI = sum(w_i * OFI_i) / sum(w_i), w_i = i/5, i=1..5
            因子值为 T 日滚动均值。

        Args:
            daily_ofi_1 ~ daily_ofi_5: 第1~5档预计算的每日OFI值
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: MOFI 的 T 日滚动均值
        """
        # w_i = i/5, sum(w_i) = (1+2+3+4+5)/5 = 3
        w = [1 / 5, 2 / 5, 3 / 5, 4 / 5, 5 / 5]
        ofi_list = [daily_ofi_1, daily_ofi_2, daily_ofi_3, daily_ofi_4, daily_ofi_5]
        w_sum = sum(w)

        weighted = sum(wi * ofi for wi, ofi in zip(w, ofi_list)) / w_sum
        result = weighted.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# MOFI 衡量了不同档位订单失衡的加权累积影响。第i档权重为 i/5，
# 即越深档位权重越高，能更准确量化订单失衡对股价的影响。
# 短期内订单失衡通常与未来收益正相关；中长期随着买卖压力失衡消失，
# 超额收益出现均值回复。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.mofi import MOFIFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#   np.random.seed(42)
#   ofi = {f"daily_ofi_{i}": pd.DataFrame(
#       np.random.randn(30, 2) * 100, index=dates, columns=stocks
#   ) for i in range(1, 6)}
#
#   factor = MOFIFactor()
#   result = factor.compute(**ofi, T=20)
#   print(result.tail())
