import numpy as np
import pandas as pd

from .base import BaseFactor


class DropTimeCentroidFactor(BaseFactor):
    """跌幅时间重心偏离因子 (Drop Time Centroid Deviation)。"""

    name = "DROP_TIME_CENTROID"
    category = "高频收益分布"
    description = "跌幅时间重心偏离因子，基于涨跌时间重心的截面回归残差"

    def compute(
        self,
        daily_g_up: pd.DataFrame,
        daily_g_down: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算跌幅时间重心偏离因子。

        Args:
            daily_g_up: 每日涨幅时间重心，index=日期，columns=股票代码。
            daily_g_down: 每日跌幅时间重心，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期，columns=股票代码。
                          截面回归残差的 T 日均值。
        """
        dates = daily_g_up.index
        stocks = daily_g_up.columns
        residuals = pd.DataFrame(np.nan, index=dates, columns=stocks)

        # 逐日截面回归: G_d = alpha + beta * G_u + epsilon
        for i in range(len(dates)):
            g_u = daily_g_up.iloc[i]
            g_d = daily_g_down.iloc[i]

            valid = g_u.notna() & g_d.notna()
            if valid.sum() < 3:
                continue

            x = g_u[valid].values
            y = g_d[valid].values

            x_with_const = np.column_stack([np.ones(len(x)), x])
            try:
                beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
                eps = y - x_with_const @ beta
                residuals.iloc[i, valid.values] = eps
            except np.linalg.LinAlgError:
                continue

        # T 日均值
        result = residuals.rolling(window=T, min_periods=T).mean()

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 跌幅时间重心偏离因子通过分钟涨跌幅加权的时间重心来捕捉日内交易行为特征。
# 核心逻辑：
#   1. 计算日内涨幅时间重心 G_u 和跌幅时间重心 G_d。
#   2. 逐日做截面回归 G_d = alpha + beta * G_u + epsilon。
#   3. 取残差 epsilon 的 T 日均值作为因子值。
#
# 经济直觉：涨跌时间重心的相对位置（时间差）是一个有效的 Alpha 因子，
# 与未来收益正相关；"时间差 Alpha"是收益率结构和"低波效应"的综合。
