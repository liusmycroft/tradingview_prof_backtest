import numpy as np
import pandas as pd

from factors.base import BaseFactor


class IdealAmplitudeFactor(BaseFactor):
    """理想振幅因子 (Ideal Amplitude)。"""

    name = "IDEAL_AMPLITUDE"
    category = "高频波动"
    description = "高价日与低价日振幅之差，衡量价格水平对波动率的非对称影响"

    def compute(
        self,
        close: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        N: int = 20,
        quantile: float = 0.25,
        **kwargs,
    ) -> pd.DataFrame:
        """计算理想振幅因子。

        Args:
            close: 收盘价，index=日期, columns=股票代码。
            high: 最高价，形状同 close。
            low: 最低价，形状同 close。
            N: 回看窗口天数，默认 20。
            quantile: 分位数阈值，默认 0.25。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        dates = close.index
        stocks = close.columns
        close_vals = close.values
        high_vals = high.values
        low_vals = low.values
        num_dates, num_stocks = close_vals.shape

        # 日振幅: high/low - 1
        amplitude = high_vals / low_vals - 1.0

        result = np.full((num_dates, num_stocks), np.nan)

        for t in range(N - 1, num_dates):
            close_win = close_vals[t - N + 1 : t + 1]  # (N, num_stocks)
            amp_win = amplitude[t - N + 1 : t + 1]

            for s in range(num_stocks):
                c = close_win[:, s]
                a = amp_win[:, s]

                valid = ~(np.isnan(c) | np.isnan(a))
                if valid.sum() < 4:  # 至少需要足够数据分组
                    continue

                c_valid = c[valid]
                a_valid = a[valid]

                low_thresh = np.quantile(c_valid, quantile)
                high_thresh = np.quantile(c_valid, 1.0 - quantile)

                low_mask = c_valid <= low_thresh
                high_mask = c_valid >= high_thresh

                if low_mask.sum() == 0 or high_mask.sum() == 0:
                    continue

                v_high = np.mean(a_valid[high_mask])
                v_low = np.mean(a_valid[low_mask])
                result[t, s] = v_high - v_low

        return pd.DataFrame(result, index=dates, columns=stocks)


# ==============================================================================
# 核心思想与原理说明
# ==============================================================================
#
# 理想振幅因子的核心思想：
#
# 1. 振幅 = high/low - 1，衡量日内价格波动幅度。
#
# 2. 将过去 N 天按收盘价排序，取最高 25% 和最低 25% 的交易日，
#    分别计算这两组的平均振幅 V_high 和 V_low。
#
# 3. V(lambda) = V_high - V_low：
#    - 若 > 0，说明股价高位时波动更大，可能暗示高位分歧加大；
#    - 若 < 0，说明股价低位时波动更大，可能暗示低位恐慌。
#
# 4. 该因子捕捉了价格水平对波动率的非对称影响，是一种条件波动率因子。
#
# ==============================================================================
# 简单用法示例
# ==============================================================================
#
# import pandas as pd
# import numpy as np
# from factors.ideal_amplitude import IdealAmplitudeFactor
#
# dates = pd.date_range("2024-01-01", periods=30, freq="B")
# stocks = ["000001.SZ", "000002.SZ"]
#
# np.random.seed(42)
# close = pd.DataFrame(
#     np.cumsum(np.random.normal(0, 1, (30, 2)), axis=0) + 50,
#     index=dates, columns=stocks,
# )
# high = close + np.random.uniform(0.5, 2.0, (30, 2))
# low = close - np.random.uniform(0.5, 2.0, (30, 2))
#
# factor = IdealAmplitudeFactor()
# result = factor.compute(close=close, high=high, low=low, N=20)
# print(result.tail())
