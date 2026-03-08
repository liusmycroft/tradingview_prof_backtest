import numpy as np
import pandas as pd

from .base import BaseFactor


class CustomerMomentumIDFactor(BaseFactor):
    """改进的客户动量因子（基于信息离散度）"""

    name = "CUSTOMER_MOMENTUM_ID"
    category = "图谱网络-动量溢出"
    description = "基于供应链客户动量与信息离散度的改进因子，信息连续时动量效应更强"

    def compute(
        self,
        daily_customer_momentum: pd.DataFrame,
        daily_info_discreteness: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算改进的客户动量因子。

        公式:
            comm_i = sum(w_ij * mom_j)  (客户动量)
            ID = sign(CR) * (%neg - %pos)  (信息离散度)
            因子 = comm_i * (-ID) 的 T 日 EMA

        Args:
            daily_customer_momentum: 预计算的每日客户动量 comm_i
                (index=日期, columns=股票代码)
            daily_info_discreteness: 预计算的每日信息离散度 ID
                (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 改进的客户动量因子值
        """
        # 信息离散度为负向指标，取负号后与客户动量相乘
        combined = daily_customer_momentum * (-daily_info_discreteness)
        result = combined.ewm(span=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 改进的客户动量因子利用供应链数据计算客户动量 comm_i = sum(w_ij * mom_j)，
# 并结合信息离散度 ID = sign(CR) * (%neg - %pos) 进行改进。
#
# 投资者对经济相关公司提供的连续信息反应不足，而对离散信息反应迅速。
# 信息离散度越低（信息越连续），客户动量效应越强。因此用 -ID 加权
# 客户动量，可以增强因子的预测能力。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.customer_momentum_id import CustomerMomentumIDFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_customer_momentum = pd.DataFrame(
#       np.random.randn(30, 2) * 0.05,
#       index=dates, columns=stocks,
#   )
#   daily_info_discreteness = pd.DataFrame(
#       np.random.randn(30, 2) * 0.3,
#       index=dates, columns=stocks,
#   )
#
#   factor = CustomerMomentumIDFactor()
#   result = factor.compute(
#       daily_customer_momentum=daily_customer_momentum,
#       daily_info_discreteness=daily_info_discreteness,
#       T=20,
#   )
#   print(result.tail())
