import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from factors.base import BaseFactor


class RetailHerdFactor(BaseFactor):
    """散户羊群效应因子 (Retail Herd Behavior)。"""

    name = "RETAIL_HERD"
    category = "高频资金流"
    description = "收益率与次日小单净流入的秩相关系数，衡量散户跟风行为"

    def compute(
        self,
        returns: pd.DataFrame,
        s_net_inflow: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算散户羊群效应因子。

        公式: RankCorr(R_t, S_{t+1})
              即收益率与次日小单净流入的 Spearman 秩相关系数，滚动 T 日计算。

        Args:
            returns: 日收益率，index=日期，columns=股票代码。
            s_net_inflow: 小单净流入，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 滚动秩相关系数，index=日期，columns=股票代码。
        """
        # 将小单净流入前移一天 (S_{t+1} 对齐到 t)
        s_lead = s_net_inflow.shift(-1)

        dates = returns.index
        stocks = returns.columns
        num_dates = len(dates)
        num_stocks = len(stocks)

        ret_vals = returns.values
        s_lead_vals = s_lead.values

        result = np.full((num_dates, num_stocks), np.nan)

        for col in range(num_stocks):
            for t in range(T - 1, num_dates):
                r_window = ret_vals[t - T + 1 : t + 1, col]
                s_window = s_lead_vals[t - T + 1 : t + 1, col]

                # 跳过含 NaN 的窗口
                valid = ~(np.isnan(r_window) | np.isnan(s_window))
                if valid.sum() < 3:
                    continue

                corr, _ = spearmanr(r_window[valid], s_window[valid])
                result[t, col] = corr

        return pd.DataFrame(result, index=dates, columns=stocks)


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 散户羊群效应因子通过计算当日收益率与次日小单净流入的秩相关系数，
# 衡量散户投资者的跟风行为强度。
#
# - 正相关：今天涨，明天散户买入增加（追涨行为）
# - 负相关：今天涨，明天散户反而卖出（逆向行为）
#
# 实证中，散户羊群效应强的股票往往存在过度反应，未来收益可能出现反转。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.retail_herd import RetailHerdFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=30)
#   returns = pd.DataFrame(
#       np.random.randn(30, 2) * 0.02,
#       index=dates,
#       columns=["000001.SZ", "600000.SH"],
#   )
#   s_net_inflow = pd.DataFrame(
#       np.random.randn(30, 2) * 1e6,
#       index=dates,
#       columns=["000001.SZ", "600000.SH"],
#   )
#
#   factor = RetailHerdFactor()
#   result = factor.compute(returns=returns, s_net_inflow=s_net_inflow, T=20)
#   print(result.tail())
