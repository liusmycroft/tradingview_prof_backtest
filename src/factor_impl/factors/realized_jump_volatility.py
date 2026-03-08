import numpy as np
import pandas as pd
from math import gamma as math_gamma

from factors.base import BaseFactor


class RealizedJumpVolatilityFactor(BaseFactor):
    """已实现跳跃波动率因子 (Realized Jump Volatility)"""

    name = "REALIZED_JUMP_VOLATILITY"
    category = "高频波动跳跃"
    description = "已实现跳跃波动率，刻画股价波动率中跳跃部分的波动水平"

    # mu_q = 2^{q/2} * Gamma((q+1)/2) / sqrt(pi)
    # 对 q=2/3: mu_{2/3}
    MU_2_3 = 2 ** (1 / 3) * math_gamma(5 / 6) / math_gamma(1 / 2)

    def compute(
        self,
        rv: pd.DataFrame,
        iv_hat: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算已实现跳跃波动率因子。

        公式:
            RVJ_t = max(RV_t - IV_hat_t, 0)

        Args:
            rv: 已实现波动率 RV_t = sum(r_i^2)，index=日期, columns=股票代码
            iv_hat: 积分方差估计量，index=日期, columns=股票代码

        Returns:
            pd.DataFrame: 已实现跳跃波动率
        """
        rvj = (rv - iv_hat).clip(lower=0)
        return rvj

    @staticmethod
    def compute_iv_hat(intraday_returns: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        """从日内收益率计算积分方差估计量。

        公式:
            IV_hat = mu_{2/3}^{-k} * sum_{i=k}^{n} prod_{j=0}^{k-1} |r_{i-j}|^{2/3}

        Args:
            intraday_returns: 日内对数收益率 (index=日期, columns=时段编号)
            k: 滑动窗口长度，默认 3

        Returns:
            pd.DataFrame: 积分方差估计量 (index=日期, 单列)
        """
        mu_2_3 = RealizedJumpVolatilityFactor.MU_2_3
        abs_r_23 = intraday_returns.abs() ** (2 / 3)
        cols = abs_r_23.values
        n_cols = cols.shape[1]

        if n_cols < k:
            return pd.Series(np.nan, index=intraday_returns.index)

        running = np.ones_like(cols)
        for j in range(k):
            shifted = np.roll(cols, j, axis=1)
            shifted[:, :j] = np.nan
            running = running * shifted

        row_sums = np.nansum(running[:, k - 1:], axis=1)
        iv_hat = pd.Series(
            mu_2_3 ** (-k) * row_sums,
            index=intraday_returns.index,
        )
        return iv_hat
