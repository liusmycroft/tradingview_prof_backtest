"""高频特异波动率因子 (High-Frequency Idiosyncratic Volatility)

在Fama三因子基础上加入流动性和反转因子，计算特异率，
再与高频波动率相乘得到高频特异波动率。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class HfIdioVolFactor(BaseFactor):
    """高频特异波动率因子"""

    name = "HF_IDIO_VOL"
    category = "高频波动跳跃"
    description = "高频波动率与增强特异率的乘积开方，捕捉个股异常波动信息"

    def compute(
        self,
        stock_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        hf_volatility: pd.DataFrame,
        T: int = 21,
        **kwargs,
    ) -> pd.DataFrame:
        """计算高频特异波动率因子。

        步骤:
        1. 对过去T天个股收益率做多因子时序回归:
           r_t = a + b_mkt*MKT + b_smb*SMB + b_hml*HML + b_ret*RET + b_liq*LIQ + eps
        2. 新特异率 = 1 - R^2
        3. 高频特异波动率 = sqrt(新特异率 * 高频波动率)

        Args:
            stock_returns: 个股日收益率 (index=日期, columns=股票代码)
            factor_returns: 因子收益率 (index=日期, columns=[MKT, SMB, HML, RET, LIQ])
            hf_volatility: 预计算的高频波动率（日内分钟收益率标准差）
                (index=日期, columns=股票代码)
            T: 回归窗口天数，默认 21

        Returns:
            pd.DataFrame: 高频特异波动率因子值
        """
        dates = stock_returns.index
        stocks = stock_returns.columns
        n_dates = len(dates)
        result = np.full((n_dates, len(stocks)), np.nan)

        factor_cols = factor_returns.columns
        n_factors = len(factor_cols)

        for t in range(T - 1, n_dates):
            start = t - T + 1
            y_window = stock_returns.iloc[start : t + 1]
            x_window = factor_returns.iloc[start : t + 1]

            X = np.column_stack([np.ones(T), x_window.values])

            for s_idx, stock in enumerate(stocks):
                y = y_window[stock].values
                mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
                if mask.sum() < n_factors + 2:
                    continue

                y_clean = y[mask]
                X_clean = X[mask]

                try:
                    beta, residuals, rank, sv = np.linalg.lstsq(
                        X_clean, y_clean, rcond=None
                    )
                    fitted = X_clean @ beta
                    ss_res = np.sum((y_clean - fitted) ** 2)
                    ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)
                    if ss_tot == 0:
                        continue
                    r_squared = 1.0 - ss_res / ss_tot
                    r_squared = max(0.0, min(1.0, r_squared))
                    idio_rate = 1.0 - r_squared
                except np.linalg.LinAlgError:
                    continue

                hf_vol = hf_volatility.iloc[t].get(stock, np.nan)
                if np.isnan(hf_vol):
                    continue

                result[t, s_idx] = np.sqrt(idio_rate * hf_vol)

        return pd.DataFrame(result, index=dates, columns=stocks)


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 在Fama三因子基础上加入流动性和反转因子，可以更纯粹地剥离价格变动
# 相对市场的偏离情况。特异波动率通过个股波动（基础波动率）和个股特异
# （特异率）的正向关系，杠杆性地加强了个股股价变动的异常信息。
# 个股波动部分用高频数据进行细化，有助于提升因子表现。
