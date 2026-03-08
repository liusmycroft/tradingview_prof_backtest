import pandas as pd

from factors.base import BaseFactor


class TSRJVFactor(BaseFactor):
    """跳跃显著程度加权的上下行跳跃波动不对称性因子 (TSRJV)"""

    name = "TSRJV"
    category = "高频波动跳跃"
    description = "跳跃显著程度加权的上下行跳跃波动不对称性，提升SRVJ因子表现"

    def compute(
        self,
        daily_srjv: pd.DataFrame,
        daily_jump_t: pd.DataFrame,
        T: int = 20,
        alpha_quantile: float = 0.05,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 TSRJV 因子。

        公式:
            TSRJV = sum(T_stat / Phi_inv(1-alpha) * SRJV) / sum(T_stat / Phi_inv(1-alpha))

        本方法接收预计算的每日 SRJV 和跳跃检验统计量 T_stat，
        输出滚动 T 日加权均值。

        Args:
            daily_srjv: 每日上下行跳跃波动不对称性 SRJV = RVLJP - RVJN
                (index=日期, columns=股票代码)
            daily_jump_t: 每日跳跃检验统计量 T_stat
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20
            alpha_quantile: 显著性水平，默认 0.05

        Returns:
            pd.DataFrame: TSRJV 因子值
        """
        # 加权：权重为 T_stat（跳跃显著程度）
        # 滚动窗口内的加权均值
        weighted = daily_jump_t * daily_srjv
        numerator = weighted.rolling(window=T, min_periods=T).sum()
        denominator = daily_jump_t.rolling(window=T, min_periods=T).sum()
        result = numerator / denominator
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# TSRJV 通过跳跃显著程度加权来提升上下行跳跃波动不对称性因子的表现。
# 跳跃越显著的交易日，其 SRJV 值在因子计算中获得更高的权重。
#
# SRJV = RVLJP - RVJN，衡量上行跳跃波动与下行跳跃波动的不对称性。
# T_stat 为检验跳跃是否显著的统计量。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.tsrjv import TSRJVFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=30)
#   stocks = ["000001.SZ", "000002.SZ"]
#   daily_srjv = pd.DataFrame(
#       np.random.randn(30, 2) * 0.01, index=dates, columns=stocks,
#   )
#   daily_jump_t = pd.DataFrame(
#       np.random.uniform(0.5, 3.0, (30, 2)), index=dates, columns=stocks,
#   )
#
#   factor = TSRJVFactor()
#   result = factor.compute(daily_srjv=daily_srjv, daily_jump_t=daily_jump_t, T=20)
#   print(result.tail())
