import numpy as np
import pandas as pd

from factors.base import BaseFactor


class AttentionLimitFactor(BaseFactor):
    """基于涨跌停的注意力捕捉因子 (Attention Capture - Limit Moves)"""

    name = "ATTENTION_LIMIT"
    category = "行为金融"
    description = "基于涨跌停的注意力捕捉：个股收益率对市场涨跌停比例回归的绝对beta值"

    def compute(
        self,
        stock_returns: pd.DataFrame,
        limit_prop: pd.Series,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算注意力捕捉因子。

        公式: |beta_i| from rolling regression r_i,d = mu + beta * LIMIT_PROP + epsilon

        Args:
            stock_returns: 个股日收益率 (index=日期, columns=股票代码)
            limit_prop: 每日涨跌停股票占比 (index=日期)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，滚动回归绝对beta
        """
        result = pd.DataFrame(np.nan, index=stock_returns.index, columns=stock_returns.columns)

        limit_arr = limit_prop.reindex(stock_returns.index).values

        for col in stock_returns.columns:
            y_arr = stock_returns[col].values
            betas = np.full(len(y_arr), np.nan)

            for i in range(T - 1, len(y_arr)):
                y_win = y_arr[i - T + 1 : i + 1]
                x_win = limit_arr[i - T + 1 : i + 1]

                # 跳过含 NaN 的窗口
                mask = ~(np.isnan(y_win) | np.isnan(x_win))
                if mask.sum() < T:
                    continue

                y_w = y_win[mask]
                x_w = x_win[mask]

                # OLS: beta = cov(x, y) / var(x)
                x_mean = x_w.mean()
                y_mean = y_w.mean()
                x_demean = x_w - x_mean
                var_x = (x_demean ** 2).sum()
                if var_x == 0:
                    betas[i] = 0.0
                    continue
                cov_xy = (x_demean * (y_w - y_mean)).sum()
                betas[i] = abs(cov_xy / var_x)

            result[col] = betas

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 涨跌停是市场中最引人注目的事件之一，会吸引投资者的注意力。
# 本因子通过回归个股收益率对市场涨跌停比例，衡量个股对市场极端
# 事件的敏感度。|beta| 越大，说明该股票越容易受到市场注意力驱动
# 的影响，可能存在过度反应或动量效应。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.attention_limit import AttentionLimitFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   stock_returns = pd.DataFrame(np.random.randn(30, 2) * 0.02, index=dates, columns=stocks)
#   limit_prop = pd.Series(np.random.uniform(0.01, 0.05, 30), index=dates)
#
#   factor = AttentionLimitFactor()
#   result = factor.compute(stock_returns=stock_returns, limit_prop=limit_prop, T=20)
#   print(result.tail())
