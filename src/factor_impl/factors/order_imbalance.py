import pandas as pd

from factors.base import BaseFactor


class OrderImbalanceFactor(BaseFactor):
    """订单失衡因子 (Order Imbalance - VOI)"""

    name = "ORDER_IMBALANCE"
    category = "高频流动性"
    description = "订单失衡：基于买卖盘挂单量变化的VOI指标，取T日滚动均值"

    def compute(
        self,
        daily_voi: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算订单失衡因子。

        公式: (1/T) * sum(VOI) where VOI = delta_V_WB - delta_V_WA

        Args:
            daily_voi: 每日预计算的VOI值 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        result = daily_voi.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# VOI (Volume Order Imbalance) 衡量买卖盘挂单量的不对称变化。
# VOI = delta_V_WB - delta_V_WA，其中 delta_V_WB 为加权买盘量变化，
# delta_V_WA 为加权卖盘量变化。正值表示买盘力量增强，负值表示
# 卖盘力量增强。T日均值平滑短期噪声，捕捉持续的订单流方向。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.order_imbalance import OrderImbalanceFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   daily_voi = pd.DataFrame(np.random.randn(30, 2) * 100, index=dates, columns=stocks)
#
#   factor = OrderImbalanceFactor()
#   result = factor.compute(daily_voi=daily_voi, T=20)
#   print(result.tail())
