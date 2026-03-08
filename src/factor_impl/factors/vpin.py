import pandas as pd

from factors.base import BaseFactor


class VPINFactor(BaseFactor):
    """交易量同步的知情交易概率因子 (Volume-Synchronized PIN - VPIN)"""

    name = "VPIN"
    category = "高频资金流"
    description = "交易量同步的知情交易概率，衡量等交易量时间段内的交易量不平衡性"

    def compute(
        self,
        daily_vpin: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 VPIN 因子。

        公式: VPIN = sum(|V_tau^S - V_tau^B|) / (n * V)

        本方法接收预计算的每日 VPIN 值，输出滚动 T 日均值。

        Args:
            daily_vpin: 预计算的每日 VPIN 值 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 滚动 T 日均值
        """
        result = daily_vpin.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# VPIN 是知情交易概率的一种非参数估计方法，衡量了等交易量时间段内
# (volume time) 的交易量不平衡性。知情交易概率指在一段时间内，拥有
# 信息优势的知情交易者提交的订单数量占总委托单数量的比例。
#
# 用以刻画该段时间内的信息不对称程度，通常与未来收益正相关，取值越大，
# 信息不对称程度越大，投资者要求的风险回报越高。
#
# 买卖方向判定使用正态CDF方法：
#   V_tau^B = sum(V_i * Z((P_i - P_{i-1}) / sigma_dP))
#   V_tau^S = V - V_tau^B
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.vpin import VPINFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=30)
#   stocks = ["000001.SZ", "000002.SZ"]
#   daily_vpin = pd.DataFrame(
#       np.random.uniform(0.1, 0.5, (30, 2)), index=dates, columns=stocks,
#   )
#
#   factor = VPINFactor()
#   result = factor.compute(daily_vpin=daily_vpin, T=20)
#   print(result.tail())
