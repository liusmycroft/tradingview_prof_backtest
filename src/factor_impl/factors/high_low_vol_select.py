"""高低位放量选股因子 - 五分钟波动率版本 (High-Low Volatility Selection Factor)

比较股票在历史价格高位区间与低位区间的日内波动率相对强弱。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class HighLowVolSelectFactor(BaseFactor):
    """高低位放量选股因子（五分钟波动率版本）"""

    name = "HIGH_LOW_VOL_SELECT"
    category = "高频波动跳跃"
    description = "高低位放量选股因子：低位与高位波动率占比之差，衡量高低位波动率相对强弱"

    def compute(
        self,
        minute_close: pd.DataFrame,
        minute_ret_std: pd.DataFrame,
        T: int = 20,
        low_quantile: float = 0.2,
        high_quantile: float = 0.8,
        **kwargs,
    ) -> pd.DataFrame:
        """计算高低位放量选股因子。

        公式:
            Factor_Low = Avg_Std_Low / Avg_Std_Total
            Factor_High = Avg_Std_High / Avg_Std_Total
            Factor_Diff = Factor_Low - Factor_High

        Args:
            minute_close: 分钟收盘价 MultiIndex (date, minute) x stocks
                          或预计算的每日因子值 (index=日期, columns=股票代码)
            minute_ret_std: 5分钟滚动波动率，同上结构
            T: 回看期天数，默认 20
            low_quantile: 低位阈值分位数，默认 0.2
            high_quantile: 高位阈值分位数，默认 0.8

        Returns:
            pd.DataFrame: Factor_Diff 因子值
        """
        # 预计算模式
        if not isinstance(minute_close.index, pd.MultiIndex):
            return minute_close.copy()

        dates = minute_close.index.get_level_values(0).unique()
        stocks = minute_close.columns
        result = pd.DataFrame(np.nan, index=dates, columns=stocks)

        for i, end_date_idx in enumerate(range(len(dates))):
            start_idx = max(0, end_date_idx - T + 1)
            window_dates = dates[start_idx: end_date_idx + 1]

            if len(window_dates) < T:
                continue

            for stock in stocks:
                closes = []
                stds = []
                for d in window_dates:
                    c = minute_close.loc[d, stock].values.astype(float)
                    s = minute_ret_std.loc[d, stock].values.astype(float)
                    closes.extend(c)
                    stds.extend(s)

                closes = np.array(closes)
                stds = np.array(stds)

                valid = ~(np.isnan(closes) | np.isnan(stds))
                if valid.sum() < 10:
                    continue

                closes_v = closes[valid]
                stds_v = stds[valid]

                thd_low = np.quantile(closes_v, low_quantile)
                thd_high = np.quantile(closes_v, high_quantile)

                low_mask = closes_v <= thd_low
                high_mask = closes_v >= thd_high

                avg_std_total = np.mean(stds_v)
                if avg_std_total <= 0:
                    continue

                avg_std_low = np.mean(stds_v[low_mask]) if low_mask.sum() > 0 else 0
                avg_std_high = np.mean(stds_v[high_mask]) if high_mask.sum() > 0 else 0

                factor_low = avg_std_low / avg_std_total
                factor_high = avg_std_high / avg_std_total
                result.loc[dates[end_date_idx], stock] = factor_low - factor_high

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 通过比较股票在历史价格高位区间与低位区间的日内波动率相对强弱来选股。
# 低位波动率占比越高、高位波动率占比越低，说明低位时交易更活跃，
# 可能存在低位吸筹行为，未来收益更高。
