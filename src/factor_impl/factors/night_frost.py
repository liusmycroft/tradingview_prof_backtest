"""「夜眠霜路」因子 (Night Frost Path Factor)

通过日内收益率对增量成交量的滞后回归，提取截距项t值，
再计算其与全市场截面的相关性均值。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class NightFrostFactor(BaseFactor):
    """「夜眠霜路」因子"""

    name = "NIGHT_FROST"
    category = "高频量价相关性"
    description = "夜眠霜路因子：日内收益率对滞后增量成交量回归截距t值的截面相关性均值"

    def compute(
        self,
        daily_t_intercept: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算「夜眠霜路」因子。

        步骤:
        1. 输入为预计算的每日 t-intercept 值
        2. 对每只股票过去T天的 t-intercept 序列，计算与其他所有股票
           同期 t-intercept 序列的相关系数绝对值的均值

        Args:
            daily_t_intercept: 每日回归截距项的t值 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值 (index=日期, columns=股票代码)
        """
        dates = daily_t_intercept.index
        stocks = daily_t_intercept.columns
        n_dates = len(dates)
        n_stocks = len(stocks)

        result = pd.DataFrame(np.nan, index=dates, columns=stocks)

        for t in range(T - 1, n_dates):
            window = daily_t_intercept.iloc[t - T + 1: t + 1]  # (T, n_stocks)
            mat = window.values.astype(float)

            # 计算相关系数矩阵
            valid_mask = ~np.isnan(mat).any(axis=0)
            valid_idx = np.where(valid_mask)[0]

            if len(valid_idx) < 2:
                continue

            valid_mat = mat[:, valid_idx]
            corr_matrix = np.corrcoef(valid_mat.T)  # (n_valid, n_valid)

            for k, col_idx in enumerate(valid_idx):
                # 取该股票与其他所有股票的相关系数绝对值的均值
                abs_corr = np.abs(corr_matrix[k, :])
                # 排除自身（对角线=1）
                mask = np.ones(len(valid_idx), dtype=bool)
                mask[k] = False
                if mask.sum() == 0:
                    continue
                mean_abs_corr = abs_corr[mask].mean()
                result.iloc[t, col_idx] = mean_abs_corr

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 「夜眠霜路」因子剥离了 t-intercept 中的市场层面信息与个股中长期
# 基本面信息，与未来收益正相关。因子值越大，表明该股票的其他信息中
# 与其余所有股票共同的部分越多（即市场信息占比越大），个股中长期
# 基本面信息占比越小（投资者分歧较小），未来越容易产生高收益。
