import numpy as np
import pandas as pd

from factors.base import BaseFactor


class TrendClarityMomentumFactor(BaseFactor):
    """趋势清晰度动量因子 (Trend Clarity Momentum, TM)"""

    name = "TREND_CLARITY_MOMENTUM"
    category = "趋势清晰度动量"
    description = "动量与趋势清晰度标准化后差值的负绝对值，衡量动量与趋势的匹配程度"

    def compute(
        self,
        close: pd.DataFrame,
        daily_return: pd.DataFrame,
        lookback: int = 240,
        skip: int = 20,
        min_periods: int = 200,
        **kwargs,
    ) -> pd.DataFrame:
        """计算趋势清晰度动量因子。

        公式:
            Mom = sum(r_{t-skip} to r_{t-lookback})
            TC  = R^2 of regression P = beta0 + beta1 * t
            TM  = -|MOM' - TC'|  (MOM', TC' 为横截面标准化后的值)

        Args:
            close: 收盘价，index=日期, columns=股票代码
            daily_return: 日收益率，形状同 close
            lookback: 回看总天数，默认 240
            skip: 跳过最近天数，默认 20
            min_periods: 回归最少需要的交易日数，默认 200

        Returns:
            pd.DataFrame: 趋势清晰度动量因子值
        """
        dates = close.index
        stocks = close.columns
        close_vals = close.values
        ret_vals = daily_return.values
        num_dates, num_stocks = close_vals.shape

        result = np.full((num_dates, num_stocks), np.nan)

        for t in range(lookback, num_dates):
            # 动量: sum of returns from t-lookback to t-skip (inclusive)
            ret_window = ret_vals[t - lookback : t - skip + 1]  # (lookback-skip+1, S)
            mom = np.nansum(ret_window, axis=0)  # (S,)

            # 趋势清晰度: R^2 of price ~ time regression
            price_window = close_vals[t - lookback : t - skip + 1]  # (lookback-skip+1, S)
            n_obs = price_window.shape[0]

            tc = np.full(num_stocks, np.nan)
            x = np.arange(n_obs, dtype=float)
            x_mean = x.mean()
            ss_x = np.sum((x - x_mean) ** 2)

            for s in range(num_stocks):
                y = price_window[:, s]
                valid = ~np.isnan(y)
                if valid.sum() < min_periods:
                    continue
                y_v = y[valid]
                x_v = x[valid]
                x_v_mean = x_v.mean()
                y_v_mean = y_v.mean()
                ss_xx = np.sum((x_v - x_v_mean) ** 2)
                ss_yy = np.sum((y_v - y_v_mean) ** 2)
                ss_xy = np.sum((x_v - x_v_mean) * (y_v - y_v_mean))
                if ss_yy == 0 or ss_xx == 0:
                    tc[s] = np.nan
                else:
                    tc[s] = (ss_xy ** 2) / (ss_xx * ss_yy)

            # 横截面标准化
            valid_mom = ~np.isnan(mom)
            valid_tc = ~np.isnan(tc)
            valid_both = valid_mom & valid_tc

            if valid_both.sum() < 2:
                continue

            mom_mean = np.nanmean(mom[valid_both])
            mom_std = np.nanstd(mom[valid_both], ddof=0)
            tc_mean = np.nanmean(tc[valid_both])
            tc_std = np.nanstd(tc[valid_both], ddof=0)

            if mom_std == 0 or tc_std == 0:
                continue

            mom_z = (mom - mom_mean) / mom_std
            tc_z = (tc - tc_mean) / tc_std

            result[t] = -np.abs(mom_z - tc_z)

        return pd.DataFrame(result, index=dates, columns=stocks)


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 趋势清晰度动量因子结合了动量因子和趋势清晰度因子 (R^2)。
# 过去下跌且趋势清晰的股票未来更可能继续下跌；
# 过去上涨且趋势清晰的股票未来更可能继续上涨。
# TM = -|MOM' - TC'| 衡量动量与趋势清晰度的匹配程度，
# 值越接近 0（绝对值越小），说明动量与趋势越匹配。
