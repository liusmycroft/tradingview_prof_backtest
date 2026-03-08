import numpy as np
import pandas as pd

from factors.base import BaseFactor


class ProspectTKFactor(BaseFactor):
    """基于PB的前景价值TK (Prospect Theory TK based on PB) 因子。"""

    name = "PROSPECT_TK"
    category = "行为金融"
    description = "基于前景理论对PB变化率计算TK值，衡量投资者对损益的非对称感知"

    def compute(
        self,
        pb_change_rates: pd.DataFrame,
        N: int = 20,
        alpha: float = 0.88,
        lam: float = 2.25,
        gamma: float = 0.61,
        delta: float = 0.69,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 PROSPECT_TK 因子。

        Args:
            pb_change_rates: PB变化率，index=日期, columns=股票代码。
            N: 回看窗口天数，默认 20。
            alpha: 价值函数幂参数，默认 0.88。
            lam: 损失厌恶系数，默认 2.25。
            gamma: 正面概率权重参数，默认 0.61。
            delta: 负面概率权重参数，默认 0.69。

        Returns:
            pd.DataFrame: TK 因子值，index=日期, columns=股票代码。
        """
        dates = pb_change_rates.index
        stocks = pb_change_rates.columns
        vals = pb_change_rates.values
        num_dates, num_stocks = vals.shape

        tk = np.full((num_dates, num_stocks), np.nan)

        for t in range(N - 1, num_dates):
            window = vals[t - N + 1 : t + 1]  # (N, num_stocks)
            for s in range(num_stocks):
                col = window[:, s]
                if np.any(np.isnan(col)):
                    continue
                sorted_r = np.sort(col)
                n = len(sorted_r)
                # 等概率分位: quantile_i = i / n, i = 0, 1, ..., n
                quantiles = np.arange(n + 1) / n  # length n+1

                tk_val = 0.0
                for i in range(n):
                    r = sorted_r[i]
                    q_upper = quantiles[i + 1]
                    q_lower = quantiles[i]

                    # 价值函数
                    if r >= 0:
                        v = r ** alpha
                    else:
                        v = -lam * ((-r) ** alpha)

                    # 概率权重
                    if r >= 0:
                        w_upper = _w_plus(q_upper, gamma)
                        w_lower = _w_plus(q_lower, gamma)
                    else:
                        w_upper = _w_minus(q_upper, delta)
                        w_lower = _w_minus(q_lower, delta)

                    tk_val += v * (w_upper - w_lower)

                tk[t, s] = tk_val

        return pd.DataFrame(tk, index=dates, columns=stocks)


def _w_plus(p: float, gamma: float) -> float:
    """正面概率权重函数 w+(p)。"""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    pg = p ** gamma
    return pg / (pg + (1.0 - p) ** gamma) ** (1.0 / gamma)


def _w_minus(p: float, delta: float) -> float:
    """负面概率权重函数 w-(p)。"""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    pd_ = p ** delta
    return pd_ / (pd_ + (1.0 - p) ** delta) ** (1.0 / delta)


# ==============================================================================
# 核心思想与原理说明
# ==============================================================================
#
# 前景理论 TK (Tversky-Kahneman) 因子源自行为金融学中的前景理论。核心思想：
#
# 1. 价值函数 v(x)：投资者对收益和损失的感知是非线性的。收益区间呈凹函数
#    (边际效用递减)，损失区间呈凸函数，且损失的痛苦大于等额收益的快乐
#    (损失厌恶，lambda > 1)。
#
# 2. 概率权重函数 w(p)：投资者会高估小概率事件、低估大概率事件。正面和
#    负面事件的概率扭曲程度不同 (gamma != delta)。
#
# 3. TK 值综合了价值函数和概率权重，反映投资者对该股票PB变化的主观评价。
#    TK 值高的股票被投资者主观高估，未来可能回调；TK 值低的股票被低估。
#
# ==============================================================================
# 简单用法示例
# ==============================================================================
#
# import pandas as pd
# import numpy as np
# from factors.prospect_tk import ProspectTKFactor
#
# dates = pd.date_range("2024-01-01", periods=30, freq="B")
# stocks = ["000001.SZ", "000002.SZ"]
#
# np.random.seed(42)
# pb_change = pd.DataFrame(
#     np.random.uniform(-0.1, 0.1, (30, 2)), index=dates, columns=stocks
# )
#
# factor = ProspectTKFactor()
# result = factor.compute(pb_change_rates=pb_change, N=20)
# print(result.tail())
