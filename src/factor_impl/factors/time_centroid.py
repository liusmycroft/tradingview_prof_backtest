"""
时间重心偏离因子 (Time Centroid Deviation Factor)

通过截面回归分离上涨与下跌时间重心的关系，取残差均值作为因子。
"""

import numpy as np
import pandas as pd
from factors.base import BaseFactor


class TimeCentroidFactor(BaseFactor):
    """时间重心偏离因子

    公式: TGD = (1/T) * sum(epsilon)，其中 G_d = alpha + beta * G_u + epsilon
    """

    name = "TIME_CENTROID"
    category = "高频收益分布"
    description = "时间重心偏离因子，通过截面回归分离上涨与下跌时间重心的关系"

    def compute(
        self,
        g_up: pd.DataFrame,
        g_down: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算时间重心偏离因子。

        Args:
            g_up: 每日上涨时间重心，index为日期，columns为股票代码。
            g_down: 每日下跌时间重心，index为日期，columns为股票代码。
            T: 滚动窗口天数，默认20。

        Returns:
            pd.DataFrame: 因子值（T日滚动均值的截面回归残差）。
        """
        residuals = pd.DataFrame(np.nan, index=g_up.index, columns=g_up.columns)

        # 逐日做截面回归: g_down = alpha + beta * g_up + epsilon
        for date in g_up.index:
            x = g_up.loc[date].values
            y = g_down.loc[date].values

            # 过滤NaN
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < 3:
                continue

            x_valid = x[valid]
            y_valid = y[valid]

            # OLS回归: y = alpha + beta * x
            X = np.column_stack([np.ones(len(x_valid)), x_valid])
            try:
                beta_hat = np.linalg.lstsq(X, y_valid, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue

            # 计算所有股票的残差（包括有效的）
            y_pred = beta_hat[0] + beta_hat[1] * x
            eps = y - y_pred
            residuals.loc[date] = eps

        # T日滚动均值
        result = residuals.rolling(window=T, min_periods=1).mean()
        return result


# 使用示例
if __name__ == "__main__":
    dates = pd.date_range("2024-01-01", periods=30)
    stocks = ["000001", "000002", "000003", "000004", "000005"]
    g_up = pd.DataFrame(np.random.uniform(0.3, 0.7, (30, 5)), index=dates, columns=stocks)
    g_down = pd.DataFrame(np.random.uniform(0.3, 0.7, (30, 5)), index=dates, columns=stocks)

    factor = TimeCentroidFactor()
    print(factor)
    print(factor.compute(g_up=g_up, g_down=g_down))
