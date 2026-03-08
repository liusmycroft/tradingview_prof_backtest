import numpy as np
import pandas as pd

from factors.base import BaseFactor


class IdealReversalFactor(BaseFactor):
    """理想反转因子 (Ideal Reversal)"""

    name = "IDEAL_REVERSAL"
    category = "高频因子-动量反转类"
    description = "基于日内成交金额分布高分位值区分大单交易日，计算大单日与小单日涨跌幅之差"

    def compute(
        self,
        daily_return: pd.DataFrame,
        daily_amount_quantile: pd.DataFrame,
        N: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算理想反转因子。

        公式:
            1. 回溯 N 日，按日内逐笔成交金额分布的 13/16 分位值排序
            2. 分位值最高的 N/2 日涨跌幅之和为 M_high
            3. 分位值最低的 N/2 日涨跌幅之和为 M_low
            4. M = M_high - M_low

        Args:
            daily_return: 日涨跌幅 (index=日期, columns=股票代码)
            daily_amount_quantile: 日内成交金额 13/16 分位值 (index=日期, columns=股票代码)
            N: 回溯天数，默认 20

        Returns:
            pd.DataFrame: 理想反转因子值
        """
        result = pd.DataFrame(np.nan, index=daily_return.index, columns=daily_return.columns)
        half = N // 2

        for i in range(N - 1, len(daily_return)):
            window_ret = daily_return.iloc[i - N + 1 : i + 1]
            window_q = daily_amount_quantile.iloc[i - N + 1 : i + 1]

            for col in daily_return.columns:
                ret_vals = window_ret[col].values
                q_vals = window_q[col].values

                if np.any(np.isnan(q_vals)) or np.any(np.isnan(ret_vals)):
                    continue

                rank_idx = np.argsort(q_vals)
                low_idx = rank_idx[:half]
                high_idx = rank_idx[-half:]

                m_high = np.sum(ret_vals[high_idx])
                m_low = np.sum(ret_vals[low_idx])
                result.loc[result.index[i], col] = m_high - m_low

        return result
