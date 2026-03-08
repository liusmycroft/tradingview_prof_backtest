import pandas as pd

from factors.base import BaseFactor


class PRSIFactor(BaseFactor):
    """单位成交额的百分比实现价差因子 (Percentage Realized Spread per dollar volume Intensity)"""

    name = "PRSI"
    category = "高频流动性"
    description = "单位成交额的百分比实现价差：百分比实现价差除以成交额，取T日滚动均值"

    def compute(
        self,
        daily_prs: pd.DataFrame,
        dollar_volume: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算单位成交额的百分比实现价差因子。

        公式: PRSI = rolling_mean_T( PRS / $vol )
        其中 PRS = 2 * D * (ln(P) - ln((Ask+Bid)/2))

        Args:
            daily_prs: 预计算的每日百分比实现价差 (index=日期, columns=股票代码)
            dollar_volume: 每日成交额 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        daily_prsi = daily_prs / dollar_volume
        result = daily_prsi.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 单位成交额的百分比实现价差属于"每美元成本 cost-per-dollar-volume"
# 形式的流动性指标，代表交易中每单位交易额的边际交易成本，反映流动性
# 效率。股票的交易成本越高，流动性越差，未来会有正向的流动性风险溢价。
# PRS 通过买卖标识和成交价与最优报价中间价的偏离来衡量实现价差，
# 除以成交额后得到单位成交额的边际成本。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.prsi import PRSIFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_prs = pd.DataFrame(
#       np.random.rand(30, 2) * 0.001, index=dates, columns=stocks,
#   )
#   dollar_volume = pd.DataFrame(
#       np.random.rand(30, 2) * 1e8 + 1e6, index=dates, columns=stocks,
#   )
#
#   factor = PRSIFactor()
#   result = factor.compute(daily_prs=daily_prs, dollar_volume=dollar_volume, T=20)
#   print(result.tail())
