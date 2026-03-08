import numpy as np
import pandas as pd

from factors.base import BaseFactor


class MorningInflowStabilityFactor(BaseFactor):
    """早盘主动净流入率稳定性因子 (Morning Active Net Inflow Stability)。"""

    name = "MORNING_INFLOW_STABILITY"
    category = "量价改进"
    description = "早盘主动净流入率的均值与标准差之比，衡量资金流入的稳定性"

    def compute(
        self,
        morning_net_inflow_rate: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算早盘主动净流入率稳定性因子。

        公式: mean(morning_net_inflow_rate) / std(morning_net_inflow_rate)
              在滚动 T 日窗口内计算。

        Args:
            morning_net_inflow_rate: 每日早盘主动净流入率，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 滚动均值/标准差比值，index=日期，columns=股票代码。
        """
        rolling_mean = morning_net_inflow_rate.rolling(window=T, min_periods=T).mean()
        rolling_std = morning_net_inflow_rate.rolling(window=T, min_periods=T).std()
        # 标准差为0时结果为 NaN
        result = rolling_mean / rolling_std
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 早盘主动净流入率稳定性因子衡量的是早盘时段主动买入资金净流入的稳定程度。
# 均值/标准差的比值类似于信息比率 (Information Ratio)：
#   - 比值高：资金持续稳定流入，说明有坚定的买方力量。
#   - 比值低或为负：资金流入不稳定或持续流出。
#
# 稳定的资金流入往往预示着机构投资者的持续建仓行为，对未来收益有正向预测力。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.morning_inflow_stability import MorningInflowStabilityFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=30)
#   inflow = pd.DataFrame(
#       np.random.randn(30, 2) * 0.02,
#       index=dates,
#       columns=["000001.SZ", "600000.SH"],
#   )
#
#   factor = MorningInflowStabilityFactor()
#   result = factor.compute(morning_net_inflow_rate=inflow, T=20)
#   print(result.tail())
