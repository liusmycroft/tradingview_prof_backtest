import numpy as np
import pandas as pd

from factors.base import BaseFactor


class BuyIlliquidityFactor(BaseFactor):
    """买单非流动性因子 (Buy Order Illiquidity)。"""

    name = "BUY_ILLIQUIDITY"
    category = "高频流动性"
    description = "买单金额对收益率的回归系数，衡量买方冲击成本"

    def compute(
        self,
        returns: pd.DataFrame,
        sell_amount: pd.DataFrame,
        buy_amount: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算买单非流动性因子。

        公式: r = alpha + beta1*S + beta2*B + epsilon
              因子值 = beta2 (买单金额的回归系数)

        对每个截面日期，用时间序列回归（或直接用当日截面数据）估计 beta2。
        这里对每只股票在时间序列上做回归。

        Args:
            returns: 日收益率，index=日期，columns=股票代码。
            sell_amount: 卖单金额，形状同 returns。
            buy_amount: 买单金额，形状同 returns。

        Returns:
            pd.DataFrame: beta2 回归系数，index=日期，columns=股票代码。
        """
        dates = returns.index
        stocks = returns.columns
        num_dates = len(dates)
        num_stocks = len(stocks)

        result = np.full((num_dates, num_stocks), np.nan)

        # 对每个日期做截面回归: r_i = alpha + beta1*S_i + beta2*B_i
        for t in range(num_dates):
            r = returns.values[t, :]
            s = sell_amount.values[t, :]
            b = buy_amount.values[t, :]

            valid = ~(np.isnan(r) | np.isnan(s) | np.isnan(b))
            n_valid = valid.sum()
            if n_valid < 3:
                continue

            r_v = r[valid]
            s_v = s[valid]
            b_v = b[valid]

            # OLS: r = alpha + beta1*S + beta2*B
            X = np.column_stack([np.ones(n_valid), s_v, b_v])
            try:
                betas = np.linalg.lstsq(X, r_v, rcond=None)[0]
                # beta2 对应所有有效股票
                beta2 = betas[2]
                result[t, valid] = beta2
            except np.linalg.LinAlgError:
                continue

        return pd.DataFrame(result, index=dates, columns=stocks)


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 买单非流动性因子通过回归分析，分离买单和卖单对收益率的影响。
# beta2 衡量的是买单金额每增加一个单位对收益率的边际影响（冲击成本）。
#
# - beta2 较大：买单对价格的冲击较大，说明流动性较差（买方非流动性高）。
# - beta2 较小：买单对价格的冲击较小，说明流动性较好。
#
# 该因子可用于识别流动性风险，非流动性高的股票通常具有流动性溢价。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.buy_illiquidity import BuyIlliquidityFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=20)
#   stocks = ["A", "B", "C", "D", "E"]
#   returns = pd.DataFrame(
#       np.random.randn(20, 5) * 0.02,
#       index=dates, columns=stocks,
#   )
#   sell_amt = pd.DataFrame(
#       np.random.uniform(1e6, 1e7, (20, 5)),
#       index=dates, columns=stocks,
#   )
#   buy_amt = pd.DataFrame(
#       np.random.uniform(1e6, 1e7, (20, 5)),
#       index=dates, columns=stocks,
#   )
#
#   factor = BuyIlliquidityFactor()
#   result = factor.compute(returns=returns, sell_amount=sell_amt, buy_amount=buy_amt)
#   print(result.tail())
