import numpy as np
import pandas as pd
from scipy.special import comb

from .base import BaseFactor


class SLSVHerdingFactor(BaseFactor):
    """SLSV模型（Signed LSV Herding Model）"""

    name = "SLSV_HERDING"
    category = "行为金融-羊群效应"
    description = "带符号的LSV羊群效应模型，区分买入从众与卖出从众"

    def compute(
        self,
        daily_buy_count: pd.DataFrame,
        daily_sell_count: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算SLSV羊群效应因子。

        公式:
            p_it = B(i,t) / (B(i,t) + S(i,t))
            p_t  = mean(p_it)  (全市场均值)
            AF(i,t) = sum_k C(N,k) * p_t^k * (1-p_t)^(N-k) * |k/N - p_t|
            LSV = |p_it - p_t| - AF(i,t)
            SLSV = LSV if p_it > p_t, -LSV if p_it < p_t

        Args:
            daily_buy_count: 每日买入基金数 B(i,t) (index=日期, columns=股票代码)
            daily_sell_count: 每日卖出基金数 S(i,t) (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: SLSV因子值
        """
        N_it = daily_buy_count + daily_sell_count
        p_it = daily_buy_count / N_it.replace(0, np.nan)

        # 全市场均值 p_t (每行均值)
        p_t = p_it.mean(axis=1)

        # p_it - p_t
        diff = p_it.sub(p_t, axis=0)
        abs_diff = diff.abs()

        # 计算调整因子 AF(i,t)
        dates = p_it.index
        stocks = p_it.columns
        af = pd.DataFrame(np.nan, index=dates, columns=stocks)

        for t_idx, date in enumerate(dates):
            pt = p_t.iloc[t_idx]
            if np.isnan(pt):
                continue
            for s_idx, stock in enumerate(stocks):
                n = N_it.iloc[t_idx, s_idx]
                if np.isnan(n) or n <= 0:
                    continue
                n = int(n)
                k_arr = np.arange(n + 1)
                binom_coeff = comb(n, k_arr, exact=False)
                probs = binom_coeff * (pt ** k_arr) * ((1 - pt) ** (n - k_arr))
                abs_dev = np.abs(k_arr / n - pt)
                af.iloc[t_idx, s_idx] = np.sum(probs * abs_dev)

        lsv = abs_diff - af

        # 带符号: p_it > p_t => +LSV, p_it < p_t => -LSV
        sign = np.sign(diff)
        result = sign * lsv
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# SLSV 在 LSV 基础上考虑了投资者的买卖方向。当 SLSV 显著为正时，
# 对应买入从众效应；当 SLSV 显著为负时，对应卖出从众效应；
# SLSV 越趋于 0 时，从众效应越弱。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.slsv_herding import SLSVHerdingFactor
#
#   dates = pd.date_range("2024-01-01", periods=4, freq="Q")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_buy_count = pd.DataFrame(
#       np.random.randint(50, 300, (4, 2)),
#       index=dates, columns=stocks,
#   )
#   daily_sell_count = pd.DataFrame(
#       np.random.randint(50, 300, (4, 2)),
#       index=dates, columns=stocks,
#   )
#
#   factor = SLSVHerdingFactor()
#   result = factor.compute(
#       daily_buy_count=daily_buy_count,
#       daily_sell_count=daily_sell_count,
#   )
#   print(result)
