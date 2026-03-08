import numpy as np
import pandas as pd

from factors.base import BaseFactor


class LossSellingFactor(BaseFactor):
    """亏损卖出倾向因子 (Loss Selling Tendency)。"""

    name = "LOSS_SELLING"
    category = "行为金融"
    description = "换手率加权的历史亏损累积，衡量投资者亏损卖出的倾向"

    def compute(
        self,
        close: pd.DataFrame,
        vwap: pd.DataFrame,
        turnover: pd.DataFrame,
        T: int = 60,
        **kwargs,
    ) -> pd.DataFrame:
        """计算亏损卖出倾向因子。

        公式: Loss_t = sum(w_{t-n} * loss_{t-n})
              其中 loss = min(0, (close - vwap) / vwap)  (负收益部分)
              权重 w 采用与 CGO 相同的换手率加权方案:
                w_0 = V_t
                w_1 = V_{t-1}
                w_n = V_{t-n} * prod_{s=1}^{n-1}(1 - V_{t-s})  (n >= 2)

        Args:
            close: 收盘价，index=日期，columns=股票代码。
            vwap: 成交量加权平均价，形状同 close。
            turnover: 换手率 (0-1)，形状同 close。
            T: 回看窗口天数，默认 60。

        Returns:
            pd.DataFrame: 亏损卖出倾向因子值，index=日期，columns=股票代码。
        """
        dates = close.index
        stocks = close.columns

        # 每日亏损部分: min(0, (close - vwap) / vwap)
        daily_loss = ((close.values - vwap.values) / vwap.values).clip(max=0)

        turnover_vals = turnover.values
        num_dates, num_stocks = close.values.shape

        loss_factor = np.full((num_dates, num_stocks), np.nan)

        for t in range(T - 1, num_dates):
            # 窗口 [t-T+1, t]，翻转使 idx 0 = day t (n=0)
            tv_rev = turnover_vals[t - T + 1 : t + 1][::-1]  # (T, S)
            loss_rev = daily_loss[t - T + 1 : t + 1][::-1]   # (T, S)

            # 权重计算 (与 CGO 相同)
            one_minus = 1.0 - tv_rev[1:]  # (T-1, S)
            cum_keep = np.vstack([
                np.ones((1, num_stocks)),
                np.cumprod(one_minus, axis=0),
            ])  # (T, S)
            shifted = np.vstack([np.ones((1, num_stocks)), cum_keep[:-1]])  # (T, S)
            weights = tv_rev * shifted

            k = np.nansum(weights, axis=0)
            k[k == 0] = np.nan

            loss_factor[t] = np.nansum(weights * loss_rev, axis=0) / k

        return pd.DataFrame(loss_factor, index=dates, columns=stocks)


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 亏损卖出倾向因子 (Loss Selling Tendency) 衡量的是投资者在亏损状态下的
# 卖出压力。其核心思想是：
#
# 1. 处置效应的反面：投资者不仅倾向于卖出盈利股票，在亏损累积到一定程度时
#    也会出现"割肉"行为。
#
# 2. 权重方案：采用与 CGO 相同的换手率加权方案，近期换手率高的日期权重更大，
#    反映了当前持有者的成本分布。
#
# 3. 因子含义：
#    - Loss 值越负：持有者平均亏损越大，割肉卖出压力越大。
#    - Loss 值接近 0：持有者亏损较小或无亏损。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.loss_selling import LossSellingFactor
#
#   dates = pd.bdate_range("2024-01-01", periods=80)
#   stocks = ["000001.SZ", "600000.SH"]
#   close = pd.DataFrame(
#       np.random.uniform(10, 50, (80, 2)), index=dates, columns=stocks
#   )
#   vwap = close * np.random.uniform(0.99, 1.01, (80, 2))
#   turnover = pd.DataFrame(
#       np.random.uniform(0.01, 0.10, (80, 2)), index=dates, columns=stocks
#   )
#
#   factor = LossSellingFactor()
#   result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=60)
#   print(result.tail())
