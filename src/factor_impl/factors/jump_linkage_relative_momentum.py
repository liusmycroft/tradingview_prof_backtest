import numpy as np
import pandas as pd

from factors.base import BaseFactor


class JumpLinkageRelativeMomentumFactor(BaseFactor):
    """跳跃关联相对动量因子 (Jump Linkage Relative Momentum)"""

    name = "JUMP_LINKAGE_RELATIVE_MOMENTUM"
    category = "图谱网络"
    description = "基于股价跳跃关联度加权关联股票收益率，捕捉领先-滞后联动效应"

    def compute(
        self,
        peer_ret: pd.DataFrame,
        own_ret: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算跳跃关联相对动量因子。

        公式:
          1. 计算跳跃关联度 Corr_{i,j} = cojump_num_{i,j} / jump_num_i
          2. Peer_Ret_{i,t} = sum(Corr_{i,j} * Ret_{j,t}) / sum(Corr_{i,j})
          3. 对 Peer_Ret 关于 Ret 做回归取残差

        本因子接收预计算的 peer_ret 和 own_ret，做回归取残差。

        Args:
            peer_ret: 跳跃关联度加权的关联股票收益率
                (index=日期, columns=股票代码)
            own_ret: 自身过去20日收益率 (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 跳跃关联相对动量(残差)
        """
        result = pd.DataFrame(index=peer_ret.index,
                              columns=peer_ret.columns, dtype=float)

        for col in peer_ret.columns:
            y = peer_ret[col]
            x = own_ret[col]
            mask = y.notna() & x.notna()
            if mask.sum() < 2:
                result[col] = np.nan
                continue
            y_v = y[mask]
            x_v = x[mask]
            x_mean = x_v.mean()
            y_mean = y_v.mean()
            ss_xx = ((x_v - x_mean) ** 2).sum()
            if ss_xx == 0:
                result[col] = y - y_mean
                continue
            beta = ((x_v - x_mean) * (y_v - y_mean)).sum() / ss_xx
            alpha = y_mean - beta * x_mean
            result[col] = y - (alpha + beta * x)

        return result.astype(float)
