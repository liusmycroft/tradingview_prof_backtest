"""非流动性因子 (Illiquidity - ILLIQ, K-line Shortcut Path)

基于日内 K 线捷径路径衡量个股非流动性水平，捷径路径越大、成交额越小，
说明价格变动所需的资金越少，流动性越差。
"""

import numpy as np
import pandas as pd

from .base import BaseFactor


class ILLIQFactor(BaseFactor):
    name = "illiq"
    category = "liquidity"
    description = "非流动性因子：基于日内K线捷径路径与成交额之比的滚动均值"

    def compute(
        self,
        daily_illiq: pd.DataFrame,
        d: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 ILLIQ 因子。

        Args:
            daily_illiq: 每日非流动性值（日内各 bar shortcut/value 之和），
                         index=日期, columns=股票代码
            d: 滚动平均天数，默认 20

        Returns:
            ILLIQ 因子值，index=日期, columns=股票代码
        """
        return daily_illiq.rolling(window=d, min_periods=1).mean()

    @staticmethod
    def compute_daily_illiq(
        open_prices: pd.Series | pd.DataFrame,
        high_prices: pd.Series | pd.DataFrame,
        low_prices: pd.Series | pd.DataFrame,
        close_prices: pd.Series | pd.DataFrame,
        values: pd.Series | pd.DataFrame,
    ) -> float | pd.Series:
        """从单日日内 K 线 bar 数据计算当日非流动性值。

        每个参数的行代表同一天内的各个 bar（例如 48 根 5 分钟 K 线）。
        如果传入 DataFrame，列代表不同股票，返回 pd.Series；
        如果传入 Series，返回标量 float。

        Args:
            open_prices: 各 bar 开盘价
            high_prices: 各 bar 最高价
            low_prices: 各 bar 最低价
            close_prices: 各 bar 收盘价
            values: 各 bar 成交额

        Returns:
            当日 daily_illiq 值
        """
        shortcut = 2 * (high_prices - low_prices) - (close_prices - open_prices).abs()

        # 成交额为 0 或负值的 bar 不参与计算
        safe_values = values.where(values > 0, other=np.nan)

        ratio = shortcut / safe_values
        return ratio.sum()


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【核心思想】
# 非流动性因子（ILLIQ）源自 Amihud (2002) 的思路，但此处使用 K 线捷径路径
# （Shortcut Path）替代简单的收益率绝对值来度量价格变动幅度。
#
# K 线捷径路径定义为：
#   ShortCut = 2 * (High - Low) - |Close - Open|
#
# 它刻画了一根 K 线中价格实际走过的"最短路径"（去除方向后的振幅），
# 比单纯的 |Close - Open| 更能反映日内真实波动。
#
# 将每根日内 bar 的 ShortCut 除以对应成交额 Value，再对全天所有 bar 求和，
# 得到当日非流动性值 daily_illiq。最后对 daily_illiq 取 d 日滚动均值，
# 即为最终的 ILLIQ 因子。
#
# - ILLIQ 越大：单位资金引起的价格变动越大，流动性越差
# - ILLIQ 越小：单位资金引起的价格变动越小，流动性越好
#
# 该因子常用于流动性风险溢价研究，也可作为选股因子的流动性过滤条件。
#
# 【计算公式】
# ShortCut_j = 2 * (High_j - Low_j) - |Close_j - Open_j|
# daily_illiq_t = Σ_{j=1}^{p} ShortCut_j / Value_j
# ILLIQ_t = (1/d) * Σ_{i=1}^{d} daily_illiq_{t-i}
#
# 【使用示例】
#
# import pandas as pd
# from factors.illiq import ILLIQFactor
#
# # 方式一：已有预计算的 daily_illiq
# daily_illiq = pd.read_csv("daily_illiq.csv", index_col=0, parse_dates=True)
# factor = ILLIQFactor()
# result = factor.compute(daily_illiq=daily_illiq, d=20)
# print(result.tail())
#
# # 方式二：从日内 bar 数据计算单日 daily_illiq
# intraday = pd.read_csv("intraday_bars.csv")
# daily_val = ILLIQFactor.compute_daily_illiq(
#     open_prices=intraday["open"],
#     high_prices=intraday["high"],
#     low_prices=intraday["low"],
#     close_prices=intraday["close"],
#     values=intraday["value"],
# )
# print(daily_val)
