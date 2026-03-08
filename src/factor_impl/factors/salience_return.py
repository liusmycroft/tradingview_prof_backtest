import numpy as np
import pandas as pd

from factors.base import BaseFactor


class SalienceReturnFactor(BaseFactor):
    """凸显性收益因子 (Salience Return - STR)。"""

    name = "SALIENCE_RETURN"
    category = "高频动量反转"
    description = "凸显性权重与收益率的协方差，捕捉投资者对极端收益的过度关注"

    def compute(
        self,
        daily_return: pd.DataFrame,
        market_median_return: pd.DataFrame,
        T: int = 20,
        delta: float = 0.7,
        theta: float = 0.1,
        **kwargs,
    ) -> pd.DataFrame:
        """计算凸显性收益因子。

        公式:
            sigma(r, r_bar) = |r - r_bar| / (|r| + |r_bar| + theta)
            omega = delta^rank / sum(delta^rank * pi)
            STR = cov(omega, r)

        Args:
            daily_return: 日收益率 (index=日期, columns=股票代码)
            market_median_return: 全市场日收益率中位数 (index=日期, 单列或同形)
            T: 滚动窗口天数，默认 20
            delta: 认知衰减参数，默认 0.7
            theta: 零收益控制参数，默认 0.1

        Returns:
            pd.DataFrame: 凸显性收益因子值
        """
        if isinstance(market_median_return, pd.DataFrame) and market_median_return.shape[1] == 1:
            market_med = market_median_return.iloc[:, 0]
        else:
            market_med = market_median_return

        def _calc_str(ret_window, med_window):
            """计算单窗口的 STR。"""
            r = ret_window.values
            r_bar = med_window.values
            sigma = np.abs(r - r_bar) / (np.abs(r) + np.abs(r_bar) + theta)
            ranks = np.argsort(np.argsort(-sigma)) + 1
            weights_raw = delta ** ranks
            pi = 1.0 / len(r)
            weights = weights_raw / (np.sum(weights_raw) * pi)
            cov_val = np.mean((weights - np.mean(weights)) * (r - np.mean(r)))
            return cov_val

        results = {}
        for col in daily_return.columns:
            vals = []
            for i in range(len(daily_return)):
                if i < T - 1:
                    vals.append(np.nan)
                else:
                    ret_w = daily_return[col].iloc[i - T + 1: i + 1]
                    med_w = market_med.iloc[i - T + 1: i + 1]
                    if ret_w.isna().any() or med_w.isna().any():
                        vals.append(np.nan)
                    else:
                        vals.append(_calc_str(ret_w, med_w))
            results[col] = vals

        result = pd.DataFrame(results, index=daily_return.index)
        return result
