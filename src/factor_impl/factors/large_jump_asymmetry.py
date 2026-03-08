import numpy as np
import pandas as pd

from factors.base import BaseFactor


class LargeJumpAsymmetryFactor(BaseFactor):
    """大的上下行跳跃波动不对称性因子 (Large Jump Asymmetry - SRVLJ)。"""

    name = "LARGE_JUMP_ASYMMETRY"
    category = "高频波动跳跃"
    description = "大的上行跳跃波动率与大的下行跳跃波动率之差"

    def compute(
        self,
        rvljp: pd.DataFrame,
        rvljn: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算大的上下行跳跃波动不对称性因子。

        公式: SRVLJ = RVLJP - RVLJN

        Args:
            rvljp: 大的上行跳跃波动率 (index=日期, columns=股票代码)
            rvljn: 大的下行跳跃波动率 (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 大的上下行跳跃波动不对称性
        """
        result = rvljp - rvljn
        return result
