import pandas as pd

from .base import BaseFactor


class PATVFactor(BaseFactor):
    """持续异常交易量因子"""

    name = "PATV"
    category = "高频成交分布"
    description = "持续异常交易量因子，衡量日内异常交易量的持续性"

    def compute(
        self,
        daily_patv: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算持续异常交易量因子。

        公式:
            ATV = 实际交易量 / 预期交易量
            rank_ATV: 各分钟全市场横截面排名百分位
            PATV = mean(rank_ATV) / std(rank_ATV) + kurt(rank_ATV)
            因子 = T 日 EMA

        Args:
            daily_patv: 预计算的每日 PATV 值
                (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 持续异常交易量因子值
        """
        result = daily_patv.ewm(span=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 持续异常交易量 PATV = mean(rank_ATV) / std(rank_ATV) + kurt(rank_ATV)
# 其中 rank_ATV 为日内各分钟异常交易量在全市场横截面上的排名百分位。
#
# mean 越高说明日内异常交易量相对更高；std 越小说明日内异常交易量更稳定；
# kurt 越大说明高峰点的数据分布更集中。PATV 衡量了股票日内异常交易量的
# 持续性，取值越高持续性越强，与未来收益负相关。
#
# 持续异常的高交易量代表了非理性的情绪化交易，当极端情绪持续越长，
# 累计的股价偏差越大，随后的错误定价修正幅度也越大。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.patv import PATVFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_patv = pd.DataFrame(
#       np.random.rand(30, 2) * 5 + 1,
#       index=dates, columns=stocks,
#   )
#
#   factor = PATVFactor()
#   result = factor.compute(daily_patv=daily_patv, T=20)
#   print(result.tail())
