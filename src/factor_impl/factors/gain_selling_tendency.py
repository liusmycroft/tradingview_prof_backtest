import numpy as np
import pandas as pd

from factors.base import BaseFactor


class GainSellingTendencyFactor(BaseFactor):
    """盈利卖出倾向因子 (Gain Selling Tendency)"""

    name = "GAIN_SELLING_TENDENCY"
    category = "行为金融-处置效应"
    description = "换手率加权的历史盈利累积，衡量投资者盈利卖出的倾向"

    def compute(
        self,
        close: pd.DataFrame,
        vwap: pd.DataFrame,
        turnover: pd.DataFrame,
        T: int = 60,
        **kwargs,
    ) -> pd.DataFrame:
        """计算盈利卖出倾向因子。

        公式:
            gain_t = (Close_t - P_{t-n}) / Close_t * I{Close_t >= P_{t-n}}
            Gain_t = sum(w_{t-n} * gain_{t-n})
            w_{t-n} = (1/k) * V_{t-n} * prod_{s=1}^{n-1}(1 - V_{t-n+s})

        Args:
            close: 收盘价，index=日期, columns=股票代码
            vwap: 成交量加权平均价，形状同 close
            turnover: 换手率 (0-1)，形状同 close
            T: 回看窗口天数，默认 60

        Returns:
            pd.DataFrame: 盈利卖出倾向因子值
        """
        dates = close.index
        stocks = close.columns

        close_vals = close.values
        vwap_vals = vwap.values
        turnover_vals = turnover.values
        num_dates, num_stocks = close_vals.shape

        # 预计算每日盈利部分: max(0, (close - vwap) / close)
        gain_daily = np.where(
            close_vals >= vwap_vals,
            (close_vals - vwap_vals) / close_vals,
            0.0,
        )

        result = np.full((num_dates, num_stocks), np.nan)

        for t in range(T - 1, num_dates):
            # 窗口 [t-T+1, t]，翻转使 idx 0 = day t (n=0)
            tv_rev = turnover_vals[t - T + 1 : t + 1][::-1]  # (T, S)
            gain_rev = gain_daily[t - T + 1 : t + 1][::-1]   # (T, S)

            # 权重: w_n = V_{t-n} * prod_{m=1}^{n-1}(1 - V_{t-m})
            one_minus = 1.0 - tv_rev[1:]  # (T-1, S)
            cum_keep = np.vstack([
                np.ones((1, num_stocks)),
                np.cumprod(one_minus, axis=0),
            ])  # (T, S)
            shifted = np.vstack([np.ones((1, num_stocks)), cum_keep[:-1]])  # (T, S)
            weights = tv_rev * shifted

            k = np.nansum(weights, axis=0)
            k[k == 0] = np.nan

            result[t] = np.nansum(weights * gain_rev, axis=0) / k

        return pd.DataFrame(result, index=dates, columns=stocks)


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 盈利卖出倾向因子衡量了大部分投资者"未实现"的盈利。
# 因子值越大，投资者未实现收益越大，卖出意愿越强，抛压大，
# 股价上涨受阻，与未来收益负相关。
#
# 权重方案采用换手率加权：近期换手率高的日期权重更大，
# 反映了当前持有者的成本分布。
