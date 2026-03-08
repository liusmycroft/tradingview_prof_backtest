"""买入浮盈系数修正后的积极灵活筹码因子 (AFH Close - Active Flexible Holdings adjusted by floating profit)

利用收盘价与成交价的比值作为浮盈系数，对小单主动买入量进行修正，
衡量过去 T 日内积极灵活资金的浮盈加权持仓强度。
"""

import pandas as pd

from .base import BaseFactor


class AFHCloseFactor(BaseFactor):
    name = "afh_close"
    category = "money_flow"
    description = "买入浮盈系数修正后的积极灵活筹码：Close_t * Σ(Vol_AFH_i/Price_i) / Σ(Volume_t)，T日滚动"

    def compute(
        self,
        close: pd.DataFrame,
        weighted_afh_volume: pd.DataFrame,
        total_volume: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 AFH_Close 因子。

        Args:
            close: 收盘价，index=日期, columns=股票代码
            weighted_afh_volume: 每日 Σ(Vol_AFH_i / TradePrice_i)，index=日期, columns=股票代码
            total_volume: 每日总成交量，index=日期, columns=股票代码
            T: 回看窗口天数，默认 20

        Returns:
            AFH_Close 因子值，index=日期, columns=股票代码
        """
        rolling_weighted_vol = weighted_afh_volume.rolling(window=T, min_periods=1).sum()
        rolling_total_vol = total_volume.rolling(window=T, min_periods=1).sum()

        ratio = rolling_weighted_vol / rolling_total_vol.replace(0, float("nan"))

        afh_close = close * ratio

        return afh_close


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【核心思想】
# AFH_Close（买入浮盈系数修正后的积极灵活筹码）因子关注的是小单（<5万元）
# 主动买入行为。其核心逻辑是：
#
# 1. 浮盈系数 a_i = Close_t / TradePrice_i
#    - 当 a_i > 1 时，该笔买入处于浮盈状态，说明买入时机较好
#    - 当 a_i < 1 时，该笔买入处于浮亏状态
#    - 用浮盈系数对买入量加权，使得"买对了"的资金获得更高权重
#
# 2. 将修正后的积极灵活筹码量除以区间总成交量，得到归一化的持仓强度
#
# 3. 该因子可以捕捉散户/灵活资金的"聪明钱"效应——如果小单买入后
#    股价上涨（浮盈系数 > 1），说明这些资金具有一定的择时能力，
#    因子值会相应放大，形成正向信号。
#
# 【计算公式】
# AFH_Close_t = Close_t * Σ_{d=t-T+1}^{t} weighted_afh_volume_d
#                         / Σ_{d=t-T+1}^{t} total_volume_d
#
# 其中 weighted_afh_volume_d = Σ_i (Vol_AFH_i / TradePrice_i)，
# 即当日所有符合条件的小单主动买入量除以各自成交价之和。
#
# 【使用示例】
#
# import pandas as pd
# from factors.afh_close import AFHCloseFactor
#
# close = pd.read_csv("close.csv", index_col=0, parse_dates=True)
# weighted_afh_volume = pd.read_csv("weighted_afh_volume.csv", index_col=0, parse_dates=True)
# total_volume = pd.read_csv("total_volume.csv", index_col=0, parse_dates=True)
#
# factor = AFHCloseFactor()
# result = factor.compute(
#     close=close,
#     weighted_afh_volume=weighted_afh_volume,
#     total_volume=total_volume,
#     T=20,
# )
# print(result.tail())
