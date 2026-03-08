import numpy as np
import pandas as pd
from scipy import stats

from factors.base import BaseFactor


class TDistActiveBuyFactor(BaseFactor):
    """T分布主动占比 — T-Distribution Active Buy Ratio Using Returns"""

    name = "T_DIST_ACTIVE_BUY"
    category = "高频资金流"
    description = "基于收益率 t 分布拟合的主动买入占比，区分知情与噪声交易"

    def compute(
        self,
        daily_active_buy_ratio: pd.DataFrame,
        daily_returns: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 T 分布主动占比因子。

        对每只股票过去 T 日收益率拟合 t 分布，以自由度作为权重
        调整主动买入占比：weight = 1 / (1 + df)，df 越小尾部越厚，
        权重越大，表示极端收益下的主动买入信号更强。

        因子值 = T 日滚动加权主动买入占比。

        Args:
            daily_active_buy_ratio: 预计算的每日主动买入占比 (index=日期, columns=股票代码)
            daily_returns: 每日收益率 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 分布加权主动买入占比
        """
        result = pd.DataFrame(
            np.nan, index=daily_active_buy_ratio.index, columns=daily_active_buy_ratio.columns
        )

        for col in daily_active_buy_ratio.columns:
            abr = daily_active_buy_ratio[col].values
            ret = daily_returns[col].values
            n = len(abr)
            out = np.full(n, np.nan)

            for i in range(n):
                start = max(0, i - T + 1)
                window_ret = ret[start : i + 1]
                window_abr = abr[start : i + 1]

                valid = ~(np.isnan(window_ret) | np.isnan(window_abr))
                if valid.sum() < 3:
                    continue

                wr = window_ret[valid]
                wa = window_abr[valid]

                try:
                    df_val, _, _ = stats.t.fit(wr)
                except Exception:
                    df_val = 30.0

                df_val = max(df_val, 1.0)
                weight = 1.0 / (1.0 + df_val)
                out[i] = weight * np.nanmean(wa)

            result[col] = out

        return result
