import numpy as np
import pandas as pd

from factors.base import BaseFactor


class CGOFactor(BaseFactor):
    """资本利得突出量 (Capital Gains Overhang) 因子。"""

    name = "CGO"
    category = "行为金融"
    description = "资本利得突出量，衡量投资者未实现盈亏相对于当前价格的比例"

    def compute(
        self,
        close: pd.DataFrame,
        vwap: pd.DataFrame,
        turnover: pd.DataFrame,
        T: int = 60,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 CGO 因子。

        Args:
            close: 收盘价，index=日期, columns=股票代码。
            vwap: 成交量加权平均价，形状同 close。
            turnover: 换手率 (0-1)，形状同 close。
            T: 回看窗口天数，默认 60。

        Returns:
            pd.DataFrame: CGO 因子值，index=日期, columns=股票代码。
        """
        dates = close.index
        stocks = close.columns

        turnover_vals = turnover.values
        vwap_vals = vwap.values
        close_vals = close.values
        num_dates, num_stocks = close_vals.shape

        cgo = np.full((num_dates, num_stocks), np.nan)

        for t in range(T - 1, num_dates):
            # 取窗口 [t-T+1, t]，翻转使 idx 0 = day t (n=0)
            tv_rev = turnover_vals[t - T + 1 : t + 1][::-1]   # (T, num_stocks)
            vwap_rev = vwap_vals[t - T + 1 : t + 1][::-1]

            # 权重公式: w_n = V_{t-n} * prod_{s=1}^{n-1}(1 - V_{t-s})
            # 展开后:
            #   w_0 = V_t
            #   w_1 = V_{t-1}
            #   w_2 = V_{t-2} * (1 - V_{t-1})
            #   w_3 = V_{t-3} * (1 - V_{t-1}) * (1 - V_{t-2})
            #   ...
            # 即 w_n = tv_rev[n] * prod_{m=1}^{n-1}(1 - tv_rev[m])

            # cum_keep[j] = prod_{m=1}^{j}(1 - tv_rev[m]),  cum_keep[0] = 1
            one_minus = 1.0 - tv_rev[1:]                                    # (T-1, S)
            cum_keep = np.vstack([
                np.ones((1, num_stocks)),
                np.cumprod(one_minus, axis=0),
            ])                                                               # (T, S)

            # shift 使 idx n 对应 cum_keep[n-1]，idx 0 和 1 都映射到 1
            shifted = np.vstack([np.ones((1, num_stocks)), cum_keep[:-1]])   # (T, S)

            weights = tv_rev * shifted

            k = np.nansum(weights, axis=0)
            k[k == 0] = np.nan

            rp = np.nansum(weights * vwap_rev, axis=0) / k
            cgo[t] = (close_vals[t] - rp) / close_vals[t]

        return pd.DataFrame(cgo, index=dates, columns=stocks)


# ==============================================================================
# 核心思想与原理说明
# ==============================================================================
#
# 资本利得突出量 (Capital Gains Overhang, CGO) 是一个行为金融因子，源自
# Grinblatt & Han (2005) 的研究。其核心思想是：
#
# 1. 前景理论与处置效应：投资者倾向于过早卖出盈利股票（落袋为安），而过久
#    持有亏损股票（不愿割肉）。这种行为偏差被称为"处置效应"。
#
# 2. 参考价格 (Reference Price)：CGO 通过换手率加权的历史 VWAP 来估算市场
#    中持有者的平均成本。权重的含义是：某天的换手率代表当天买入的份额，而
#    此后每天未换手的概率累乘代表这些份额至今仍被持有的概率。
#
# 3. 因子含义：
#    - CGO > 0：当前价格高于参考价格，持有者整体浮盈，处置效应导致卖压增大，
#      未来收益可能偏低。
#    - CGO < 0：当前价格低于参考价格，持有者整体浮亏，惜售心理导致卖压减小，
#      未来收益可能偏高。
#
# 4. 因此 CGO 因子通常呈现负向预测能力：低 CGO 的股票未来表现更好。
#
# ==============================================================================
# 简单用法示例
# ==============================================================================
#
# import pandas as pd
# import numpy as np
# from factors.cgo import CGOFactor
#
# dates = pd.date_range("2024-01-01", periods=120, freq="B")
# stocks = ["000001.SZ", "000002.SZ", "600000.SH"]
#
# np.random.seed(42)
# close = pd.DataFrame(
#     np.random.uniform(10, 50, (120, 3)), index=dates, columns=stocks
# )
# vwap = close * np.random.uniform(0.99, 1.01, (120, 3))
# turnover = pd.DataFrame(
#     np.random.uniform(0.01, 0.10, (120, 3)), index=dates, columns=stocks
# )
#
# factor = CGOFactor()
# result = factor.compute(close=close, vwap=vwap, turnover=turnover, T=60)
# print(result.tail())
