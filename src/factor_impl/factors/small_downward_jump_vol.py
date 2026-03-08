import numpy as np
import pandas as pd

from factors.base import BaseFactor


class SmallDownwardJumpVolFactor(BaseFactor):
    """小的下行跳跃波动率因子 (Small Downward Jump Volatility - RVSJN)。"""

    name = "SMALL_DOWNWARD_JUMP_VOL"
    category = "高频波动跳跃"
    description = "小的下行跳跃波动率，捕捉非信息冲击导致的小幅负向跳跃波动"

    def compute(
        self,
        rvjn: pd.DataFrame,
        rvljn: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算小的下行跳跃波动率因子。

        公式: RVSJN = RVJN - RVLJN
              RVJN: 下行跳跃波动率
              RVLJN: 大的下行跳跃波动率

        Args:
            rvjn: 下行跳跃波动率 (index=日期, columns=股票代码)
            rvljn: 大的下行跳跃波动率 (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 小的下行跳跃波动率
        """
        result = rvjn - rvljn
        return result
