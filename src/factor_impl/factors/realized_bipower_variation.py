"""已实现双幂次变差因子 (Realized Bipower Variation, RBV)

RBV_t = mu_1^{-2} * sum_{i=2}^{n} |r_{t_i}| * |r_{t_{i-1}}|
mu_q = 2^{q/2} * Gamma((q+1)/2) / Gamma(1/2)

用于估计积分波动率（连续分量），对跳跃具有鲁棒性。
"""

from math import gamma

import pandas as pd

from factors.base import BaseFactor

# mu_1 = E(|Z|) = 2^{1/2} * Gamma(1) / Gamma(1/2) = sqrt(2/pi)
MU_1 = 2 ** 0.5 * gamma(1.0) / gamma(0.5)


class RealizedBipowerVariationFactor(BaseFactor):
    """已实现双幂次变差因子 (RBV)"""

    name = "RBV"
    category = "高频波动跳跃"
    description = "已实现双幂次变差，估计积分波动率中的连续分量，对跳跃具有鲁棒性"

    def compute(
        self,
        daily_rbv: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算已实现双幂次变差因子。

        Args:
            daily_rbv: 预计算的每日 RBV 值 (index=日期, columns=股票代码)。
                       每日 RBV = mu_1^{-2} * sum_{i=2}^{n} |r_i| * |r_{i-1}|
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: T 日滚动均值。
        """
        result = daily_rbv.rolling(window=T, min_periods=T).mean()
        return result

    @staticmethod
    def compute_daily_rbv_from_intraday(intraday_returns: pd.DataFrame) -> float:
        """从日内分钟收益率序列计算单日 RBV（辅助方法）。

        Args:
            intraday_returns: 单日日内对数收益率，shape=(n,) 或 Series。

        Returns:
            float: 当日 RBV 值。
        """
        r = intraday_returns.values if hasattr(intraday_returns, "values") else intraday_returns
        abs_r = abs(r)
        # sum_{i=1}^{n-1} |r_i| * |r_{i+1}|
        bipower_sum = sum(abs_r[i] * abs_r[i + 1] for i in range(len(abs_r) - 1))
        return MU_1 ** (-2) * bipower_sum


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 已实现双幂次变差 (RBV) 是对积分波动率的稳健估计量。与已实现波动率 (RV)
# 不同，RBV 对价格跳跃具有鲁棒性，因此 RV - RBV 可以用来检测和度量跳跃。
#
# mu_1 = sqrt(2/pi) ≈ 0.7979，是标准正态分布的一阶绝对矩。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.realized_bipower_variation import RealizedBipowerVariationFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#   daily_rbv = pd.DataFrame(
#       np.random.uniform(0.001, 0.01, (30, 2)), index=dates, columns=stocks
#   )
#   factor = RealizedBipowerVariationFactor()
#   result = factor.compute(daily_rbv=daily_rbv, T=20)
#   print(result.tail())
