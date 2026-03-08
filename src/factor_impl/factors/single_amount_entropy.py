"""单一成交额占比熵因子 (Single Transaction Amount Proportion Entropy)

以单位一成交额占比作为概率，用信息熵刻画成交的混乱程度。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class SingleAmountEntropyFactor(BaseFactor):
    """单一成交额占比熵因子"""

    name = "SINGLE_AMOUNT_ENTROPY"
    category = "高频量价相关性"
    description = "单一成交额占比熵：以各时段成交额占比为概率计算信息熵，衡量成交集中度"

    def compute(
        self,
        minute_close: pd.DataFrame,
        minute_volume: pd.DataFrame,
        T: int = 48,
        **kwargs,
    ) -> pd.DataFrame:
        """计算单一成交额占比熵因子。

        公式:
            p_i = (vol_i * close_i) / (VOL * CLOSE)
            H = -sum(p_i * ln(p_i))

        其中 VOL = sum(vol_i), CLOSE = sum(close_i)

        Args:
            minute_close: 分钟收盘价，MultiIndex (date, minute) x stocks
                          或预计算的每日熵值 (index=日期, columns=股票代码)
            minute_volume: 分钟成交量，同上结构；若 minute_close 为预计算熵值则忽略
            T: 日内时间段个数（如5分钟频率=48），默认 48

        Returns:
            pd.DataFrame: 因子值 (index=日期, columns=股票代码)
        """
        # 如果输入已经是预计算的每日熵值（单层索引），直接返回
        if not isinstance(minute_close.index, pd.MultiIndex):
            return minute_close.copy()

        dates = minute_close.index.get_level_values(0).unique()
        stocks = minute_close.columns
        result = pd.DataFrame(np.nan, index=dates, columns=stocks)

        for date in dates:
            close_day = minute_close.loc[date]  # (T, n_stocks)
            vol_day = minute_volume.loc[date]

            for stock in stocks:
                c = close_day[stock].values.astype(float)
                v = vol_day[stock].values.astype(float)

                if np.any(np.isnan(c)) or np.any(np.isnan(v)):
                    continue

                total_vol = np.sum(v)
                total_close = np.sum(c)

                if total_vol <= 0 or total_close <= 0:
                    continue

                p = (v * c) / (total_vol * total_close)
                # 过滤掉 p<=0 的项
                mask = p > 0
                if mask.sum() == 0:
                    continue

                entropy = -np.sum(p[mask] * np.log(p[mask]))
                result.loc[date, stock] = entropy

        return result
