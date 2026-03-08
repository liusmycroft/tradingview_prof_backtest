import numpy as np
import pandas as pd

from .base import BaseFactor


class PVCorrTrendFactor(BaseFactor):
    """价量相关性趋势因子"""

    name = "PV_CORR_TREND"
    category = "高频量价相关性"
    description = "价量相关性趋势因子，对日度价量相关系数序列回归取斜率"

    def compute(
        self,
        daily_pv_corr: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算价量相关性趋势因子。

        公式:
            rho_t = corr(v_t, p_t)  (每日分钟价量相关系数)
            rho_t = beta * t + epsilon
            因子 = beta (回归斜率)

        Args:
            daily_pv_corr: 预计算的每日价量相关系数 rho_t
                (index=日期, columns=股票代码)
            T: 回归窗口天数，默认 20

        Returns:
            pd.DataFrame: 价量相关性趋势因子值（回归斜率 beta）
        """
        def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
            """对序列做滚动 OLS 回归，返回斜率。"""
            # t = 1, 2, ..., window
            t = np.arange(1, window + 1, dtype=float)
            t_mean = t.mean()
            t_var = ((t - t_mean) ** 2).sum()

            result = pd.Series(np.nan, index=series.index)
            values = series.values
            for i in range(window - 1, len(values)):
                y = values[i - window + 1 : i + 1]
                if np.isnan(y).sum() > window // 2:
                    continue
                # 用非 NaN 部分做回归
                mask = ~np.isnan(y)
                if mask.sum() < 2:
                    continue
                t_sub = t[mask]
                y_sub = y[mask]
                t_mean_sub = t_sub.mean()
                y_mean_sub = y_sub.mean()
                beta = ((t_sub - t_mean_sub) * (y_sub - y_mean_sub)).sum() / (
                    (t_sub - t_mean_sub) ** 2
                ).sum()
                result.iloc[i] = beta
            return result

        result = daily_pv_corr.apply(lambda col: _rolling_slope(col, T))
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 每日计算分钟收盘价与分钟成交量的相关系数 rho_t = corr(v_t, p_t)，
# 然后将过去 T 个交易日的 rho_t 对时间 t 做线性回归，取回归斜率 beta。
#
# PV_corr_trend 越小（即价量相关系数随时间推移变小的股票），
# 未来收益倾向于越高。建议在横截面上剔除市值、传统价量类因子后使用。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.pv_corr_trend import PVCorrTrendFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_pv_corr = pd.DataFrame(
#       np.random.randn(30, 2) * 0.3,
#       index=dates, columns=stocks,
#   )
#
#   factor = PVCorrTrendFactor()
#   result = factor.compute(daily_pv_corr=daily_pv_corr, T=20)
#   print(result.tail())
