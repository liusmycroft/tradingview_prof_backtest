import numpy as np
import pandas as pd

from .base import BaseFactor


class PeakPriceQuantileFactor(BaseFactor):
    """量峰加权价格分位点因子 (Volume Peak Weighted Price Quantile)。"""

    name = "PEAK_PRICE_QUANTILE"
    category = "高频量价"
    description = "量峰加权价格分位点因子，衡量知情交易价格的相对水平"

    def compute(
        self,
        daily_peak_quantile: pd.DataFrame,
        daily_ret_20: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算量峰加权价格分位点因子。

        Args:
            daily_peak_quantile: 预计算的每日量峰成交量加权价格分位点，
                                  index=日期，columns=股票代码。
            daily_ret_20: 过去 20 日收益率（反转因子），
                          index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期，columns=股票代码。
                          T 日均值后做反转因子中性化。
        """
        # T 日均值
        raw_factor = daily_peak_quantile.rolling(window=T, min_periods=T).mean()

        # 对反转因子做截面中性化（逐日回归取残差）
        result = pd.DataFrame(
            np.nan, index=raw_factor.index, columns=raw_factor.columns
        )

        for i in range(len(raw_factor)):
            y = raw_factor.iloc[i]
            x = daily_ret_20.iloc[i]

            valid = y.notna() & x.notna()
            if valid.sum() < 3:
                continue

            y_valid = y[valid].values
            x_valid = x[valid].values

            # 简单 OLS 回归取残差
            x_with_const = np.column_stack([np.ones(len(x_valid)), x_valid])
            try:
                beta = np.linalg.lstsq(x_with_const, y_valid, rcond=None)[0]
                residuals = y_valid - x_with_const @ beta
                result.iloc[i, valid.values] = residuals
            except np.linalg.LinAlgError:
                continue

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 量峰加权价格分位点因子衡量了知情交易价格的相对水平。核心逻辑：
#   1. 识别日内"量峰"（孤立喷发成交量）时刻。
#   2. 计算量峰成交量加权价格在当日价格区间中的相对分位点。
#   3. 取 T 日均值，并对 20 日反转因子做中性化处理。
#
# 经济直觉：因子值越高，知情交易者情绪越乐观，未来收益也可能越高。
