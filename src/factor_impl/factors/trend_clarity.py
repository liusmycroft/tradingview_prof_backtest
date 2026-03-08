"""趋势清晰度因子 (Trend Clarity, TC)

TC = R^2 of price ~ time 回归，衡量价格趋势的线性程度。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class TrendClarityFactor(BaseFactor):
    """趋势清晰度因子 (TC)"""

    name = "TREND_CLARITY"
    category = "量价因子改进"
    description = "趋势清晰度：价格对时间回归的R²，衡量价格趋势的线性程度"

    def compute(
        self,
        close: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算趋势清晰度因子。

        公式: TC = R² of (close ~ time) over rolling window T

        Args:
            close: 收盘价 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，滚动R²
        """
        dates = close.index
        stocks = close.columns
        n_dates = len(dates)

        result = pd.DataFrame(np.nan, index=dates, columns=stocks)

        # 时间序列 [0, 1, ..., T-1]
        t_vec = np.arange(T, dtype=np.float64)
        t_mean = t_vec.mean()
        ss_t = ((t_vec - t_mean) ** 2).sum()

        for i in range(T - 1, n_dates):
            window = close.iloc[i - T + 1 : i + 1]  # (T, n_stocks)

            for col in stocks:
                y = window[col].values.astype(np.float64)
                if np.any(np.isnan(y)):
                    continue

                y_mean = y.mean()
                ss_y = ((y - y_mean) ** 2).sum()
                if ss_y == 0:
                    # 价格恒定，完美拟合
                    result.loc[dates[i], col] = 1.0
                    continue

                cov_ty = ((t_vec - t_mean) * (y - y_mean)).sum()
                r_squared = (cov_ty ** 2) / (ss_t * ss_y)
                result.loc[dates[i], col] = r_squared

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 趋势清晰度 TC 通过价格对时间的线性回归 R² 来衡量价格走势的趋势性。
# R² 越接近 1，说明价格走势越接近线性趋势（无论涨跌），趋势越清晰；
# R² 越接近 0，说明价格走势越随机，缺乏明确方向。
# 趋势清晰的股票可能存在动量效应，趋势模糊的股票可能存在反转机会。
