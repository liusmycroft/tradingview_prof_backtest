"""特异质换手波动率因子 (Idiosyncratic Turnover Volatility - OLS version)

借鉴特质波动率构建理念，通过OLS回归剥离市场主流风格对个股换手率的影响，
取残差标准差作为特异质换手波动率。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class IdioTurnoverVolFactor(BaseFactor):
    """特异质换手波动率因子"""

    name = "IDIO_TURNOVER_VOL"
    category = "量价因子改进"
    description = "特异质换手波动率：OLS回归剥离换手率风格因子后残差的标准差"

    def compute(
        self,
        turnover: pd.DataFrame,
        tmkt: pd.Series,
        tsmb: pd.Series,
        thml: pd.Series,
        tmom: pd.Series,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算特异质换手波动率因子。

        对每只股票做时序OLS回归:
            turnover_i = alpha + b1*TMKT + b2*TSMB + b3*THML + b4*TMOM + eps
        Ivol_turnover = std(eps)

        Args:
            turnover: 个股换手率 (index=日期, columns=股票代码)
            tmkt: 全市场截面换手率均值 (index=日期)
            tsmb: 换手率规模因子 (index=日期)
            thml: 换手率价值因子 (index=日期)
            tmom: 换手率动量因子 (index=日期)
            T: 回归窗口天数，默认 20

        Returns:
            pd.DataFrame: 特异质换手波动率 (index=日期, columns=股票代码)
        """
        dates = turnover.index
        stocks = turnover.columns
        n_dates = len(dates)

        result = pd.DataFrame(np.nan, index=dates, columns=stocks)

        # 构建因子矩阵 X: (n_dates, 5) 含截距
        X_full = np.column_stack([
            np.ones(n_dates),
            tmkt.reindex(dates).values,
            tsmb.reindex(dates).values,
            thml.reindex(dates).values,
            tmom.reindex(dates).values,
        ])

        for t in range(T - 1, n_dates):
            X_win = X_full[t - T + 1: t + 1]  # (T, 5)

            # 检查 X 是否有 NaN
            if np.any(np.isnan(X_win)):
                continue

            for stock in stocks:
                y_win = turnover[stock].iloc[t - T + 1: t + 1].values.astype(float)

                if np.any(np.isnan(y_win)):
                    continue

                # OLS: beta = (X'X)^{-1} X'y
                try:
                    beta = np.linalg.lstsq(X_win, y_win, rcond=None)[0]
                    residuals = y_win - X_win @ beta
                    result.loc[dates[t], stock] = np.std(residuals, ddof=1)
                except np.linalg.LinAlgError:
                    continue

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 借鉴特质波动率的构建理念，特异质换手波动率剥离了市场主流风格
# （TMKT/TSMB/THML/TMOM）对于个股换手率的影响，可以更好地测算
# 市场对错误定价的修正效率。残差标准差越大，说明个股换手率中
# 不可被市场风格解释的部分波动越大。
