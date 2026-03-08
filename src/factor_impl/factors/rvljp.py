import numpy as np
import pandas as pd

from factors.base import BaseFactor


class RVLJPFactor(BaseFactor):
    """大的上行跳跃波动率因子 (Large Upward Jump Volatility - RVLJP)。"""

    name = "RVLJP"
    category = "高频波动"
    description = "大的上行跳跃波动率，捕捉显著正向跳跃对已实现波动率的贡献"

    def compute(
        self,
        rvjp: pd.DataFrame,
        large_positive_jump: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 RVLJP 因子。

        公式: RVLJP = min(RVJP, sum(r_i^2 * I(r_i > gamma)))
              其中 gamma = alpha * delta_n^0.49 * sqrt(IV)

        本方法接收预计算的 RVJP 和大正跳跃平方和，取逐元素最小值。

        Args:
            rvjp: 上行跳跃波动率，index=日期，columns=股票代码。
            large_positive_jump: 预计算的超过阈值的正收益平方和，形状同 rvjp。

        Returns:
            pd.DataFrame: 逐元素最小值，index=日期，columns=股票代码。
        """
        result = pd.DataFrame(
            np.minimum(rvjp.values, large_positive_jump.values),
            index=rvjp.index,
            columns=rvjp.columns,
        )
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# RVLJP (Realized Large Jump Positive) 度量资产价格中由显著正向跳跃引起的
# 波动成分。与 RVJP（所有正向跳跃）不同，RVLJP 只关注超过阈值 gamma 的
# 大幅正向跳跃。
#
# 取 min(RVJP, large_positive_jump) 确保 RVLJP 不超过总的正向跳跃波动率，
# 同时也不超过大跳跃的贡献。
#
# 该因子在实证中常用于：
#   - 识别具有显著正向跳跃风险的标的
#   - 作为波动率模型的附加解释变量
#   - 横截面选股：RVLJP 较高的股票可能存在过度反应
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.rvljp import RVLJPFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=5)
#   rvjp = pd.DataFrame(
#       {"000001.SZ": [0.005, 0.008, 0.003, 0.006, 0.010],
#        "600000.SH": [0.003, 0.004, 0.002, 0.005, 0.007]},
#       index=dates,
#   )
#   large_pos = pd.DataFrame(
#       {"000001.SZ": [0.003, 0.010, 0.001, 0.004, 0.008],
#        "600000.SH": [0.004, 0.002, 0.003, 0.006, 0.005]},
#       index=dates,
#   )
#
#   factor = RVLJPFactor()
#   result = factor.compute(rvjp=rvjp, large_positive_jump=large_pos)
#   print(result)
