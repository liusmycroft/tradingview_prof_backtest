"""同时点峰岭数相关性因子 (Concurrent Peak-Ridge Count Correlation)

衡量知情交易者与个人投资者在同一交易时点的参与度一致性。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class PeakRidgeCorrFactor(BaseFactor):
    """同时点峰岭数相关性因子"""

    name = "PEAK_RIDGE_CORR"
    category = "高频成交分布"
    description = "同时点峰岭数相关性：过去20日同一时点量峰数与量岭数序列的相关系数"

    def compute(
        self,
        daily_peak_counts: pd.DataFrame,
        daily_ridge_counts: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算同时点峰岭数相关性因子。

        输入为预计算的每日240个分钟点对应的峰数和岭数（过去20日统计），
        每日输出一个相关系数。

        简化接口：接收预计算的每日相关系数值。

        Args:
            daily_peak_counts: 每日同时点量峰数序列的相关系数
                               (index=日期, columns=股票代码)
            daily_ridge_counts: 占位参数，若 daily_peak_counts 已是最终值则忽略

        Returns:
            pd.DataFrame: 因子值 (index=日期, columns=股票代码)
        """
        # 如果输入是单层索引（预计算的每日相关系数），直接返回
        if not isinstance(daily_peak_counts.index, pd.MultiIndex):
            return daily_peak_counts.copy()

        # 否则计算：daily_peak_counts 和 daily_ridge_counts 均为
        # MultiIndex (date, minute_idx) x stocks
        dates = daily_peak_counts.index.get_level_values(0).unique()
        stocks = daily_peak_counts.columns
        result = pd.DataFrame(np.nan, index=dates, columns=stocks)

        for date in dates:
            peaks = daily_peak_counts.loc[date].values.astype(float)
            ridges = daily_ridge_counts.loc[date].values.astype(float)

            for j, stock in enumerate(stocks):
                p = peaks[:, j]
                r = ridges[:, j]
                valid = ~(np.isnan(p) | np.isnan(r))
                if valid.sum() < 3:
                    continue
                corr = np.corrcoef(p[valid], r[valid])[0, 1]
                result.loc[date, stock] = corr

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 同时点峰岭数相关性衡量了知情交易者与个人投资者在同一交易时点的
# 参与度一致性。一致性越高，个人投资者对知情交易者跟随越紧密，
# 知情交易者的投资动向可能已被识别，个股未来表现会减弱（负向因子）。
