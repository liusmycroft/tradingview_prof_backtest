import numpy as np
import pandas as pd

from factors.base import BaseFactor


class SellIlliquidityFactor(BaseFactor):
    """卖单非流动性因子 (Sell Order Illiquidity)"""

    name = "SELL_ILLIQUIDITY"
    category = "高频流动性"
    description = "卖单非流动性：收益率对卖出金额回归的系数，衡量卖方冲击成本"

    def compute(
        self,
        returns: pd.DataFrame,
        sell_amount: pd.DataFrame,
        buy_amount: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算卖单非流动性因子。

        对每个截面日期，做横截面回归:
            r_i = alpha + beta1 * S_i + beta2 * B_i + epsilon_i
        因子值为 beta1（卖单非流动性系数）。

        Args:
            returns: 股票收益率 (index=日期, columns=股票代码)
            sell_amount: 卖出金额 (index=日期, columns=股票代码)
            buy_amount: 买入金额 (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 每日每只股票的 beta1 系数（广播为与输入同形状）
        """
        beta1_series = pd.Series(index=returns.index, dtype=float)

        for date in returns.index:
            r = returns.loc[date].values.astype(float)
            s = sell_amount.loc[date].values.astype(float)
            b = buy_amount.loc[date].values.astype(float)

            # 去除 NaN
            valid = np.isfinite(r) & np.isfinite(s) & np.isfinite(b)
            if valid.sum() < 3:
                beta1_series[date] = np.nan
                continue

            r_v, s_v, b_v = r[valid], s[valid], b[valid]

            # OLS: r = alpha + beta1*s + beta2*b
            X = np.column_stack([np.ones(len(r_v)), s_v, b_v])
            try:
                betas = np.linalg.lstsq(X, r_v, rcond=None)[0]
                beta1_series[date] = betas[1]
            except np.linalg.LinAlgError:
                beta1_series[date] = np.nan

        # 将 beta1 广播到所有股票（截面回归系数对所有股票相同）
        result = pd.DataFrame(
            np.tile(beta1_series.values[:, None], (1, len(returns.columns))),
            index=returns.index,
            columns=returns.columns,
        )
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 卖单非流动性因子通过横截面回归 r = alpha + beta1*S + beta2*B + epsilon
# 来分离卖方和买方对价格的冲击效应。beta1 衡量卖出金额对收益率的边际影响，
# 即卖方冲击成本（sell-side price impact）。
#
# beta1 越负，说明卖出对价格的压力越大，市场流动性越差；
# beta1 接近零，说明卖出对价格影响较小，市场流动性较好。
#
# 该因子常用于：
#   - 流动性风险溢价研究：低流动性股票可能具有更高的预期收益。
#   - 交易成本建模：估计大额卖单的市场冲击。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.sell_illiquidity import SellIlliquidityFactor
#
#   dates = pd.date_range("2024-01-01", periods=20, freq="B")
#   stocks = ["A", "B", "C", "D", "E"]
#
#   np.random.seed(42)
#   returns     = pd.DataFrame(np.random.randn(20, 5) * 0.02, index=dates, columns=stocks)
#   sell_amount = pd.DataFrame(np.random.rand(20, 5) * 1e6, index=dates, columns=stocks)
#   buy_amount  = pd.DataFrame(np.random.rand(20, 5) * 1e6, index=dates, columns=stocks)
#
#   factor = SellIlliquidityFactor()
#   result = factor.compute(returns=returns, sell_amount=sell_amount, buy_amount=buy_amount)
#   print(result.tail())
