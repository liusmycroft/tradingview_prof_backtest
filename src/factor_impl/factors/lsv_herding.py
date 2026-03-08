import numpy as np
import pandas as pd
from scipy.special import comb

from factors.base import BaseFactor


class LSVHerdingFactor(BaseFactor):
    """LSV羊群效应因子 (LSV Herding Model)"""

    name = "LSV_HERDING"
    category = "行为金融-羊群效应"
    description = "LSV模型：|p_it - p_t| - E|p_it - p_t|，衡量基金抱团程度"

    def compute(
        self,
        daily_buy_count: pd.DataFrame,
        daily_sell_count: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算LSV羊群效应因子。

        公式:
            p_it = B(i,t) / (B(i,t) + S(i,t))
            p_t  = mean(p_it)  (全市场均值)
            AF(i,t) = sum_k C(N,k) * p_t^k * (1-p_t)^(N-k) * |k/N - p_t|
            LSV = |p_it - p_t| - AF(i,t)

        Args:
            daily_buy_count: 每日买入基金数 B(i,t) (index=日期, columns=股票代码)
            daily_sell_count: 每日卖出基金数 S(i,t) (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: LSV因子值
        """
        N_it = daily_buy_count + daily_sell_count
        # 避免除零
        p_it = daily_buy_count / N_it.replace(0, np.nan)

        # 全市场均值 p_t (每行均值)
        p_t = p_it.mean(axis=1)

        # |p_it - p_t|
        abs_diff = p_it.sub(p_t, axis=0).abs()

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
                # AF = sum_{k=0}^{N} C(N,k) * p_t^k * (1-p_t)^(N-k) * |k/N - p_t|
                k_arr = np.arange(n + 1)
                binom_coeff = comb(n, k_arr, exact=False)
                probs = binom_coeff * (pt ** k_arr) * ((1 - pt) ** (n - k_arr))
                abs_dev = np.abs(k_arr / n - pt)
                af.iloc[t_idx, s_idx] = np.sum(probs * abs_dev)

        result = abs_diff - af
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# LSV模型从基金持股变动数据出发，判断一只股票是否被集中买入或卖出，
# 捕捉基金"抱团"现象。p_it为股票i在区间t内买入基金占比，p_t为全市场
# 均值。AF(i,t)为二项分布下的期望绝对偏差，用于修正随机波动。
# LSV > 0 表示该股票的买卖方向一致性超出随机预期，存在羊群效应。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.lsv_herding import LSVHerdingFactor
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
#   factor = LSVHerdingFactor()
#   result = factor.compute(
#       daily_buy_count=daily_buy_count,
#       daily_sell_count=daily_sell_count,
#   )
#   print(result)
