import pandas as pd

from factors.base import BaseFactor


class VCVaRFactor(BaseFactor):
    """日内条件在险价值因子 (Volume-weighted CVaR)"""

    name = "VCVaR"
    category = "高频收益分布"
    description = "日内条件在险价值：基于成交量加权收益率的CVaR，取T日EMA"

    def compute(
        self,
        daily_vcvar: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算日内条件在险价值因子。

        公式: VCVaR = EMA_T( CVaR_alpha(VWAR) )
        其中 VWAR = sum(Return_t * Volume_t) / sum(Volume_t)

        Args:
            daily_vcvar: 预计算的每日CVaR值 (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: CVaR 的 T 日 EMA
        """
        result = daily_vcvar.ewm(span=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 在分钟高频数据环境下，激烈成交的时段往往比稀疏成交的时段更具有价格发现功能。
# 利用成交量加权平均收益率（VWAR）可以淡化成交稀疏的时间段，强化真正由买卖
# 力量形成的收益率。在此基础上计算左侧5%的条件在险价值（CVaR），衡量极端
# 下行风险。因子值越小（越负），说明该股票在成交活跃时段的尾部风险越大。
# 取 T 日 EMA 以平滑日间波动，同时赋予近期数据更高权重。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.vcvar import VCVaRFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_vcvar = pd.DataFrame(
#       np.random.randn(30, 2) * 0.02 - 0.03,
#       index=dates, columns=stocks,
#   )
#
#   factor = VCVaRFactor()
#   result = factor.compute(daily_vcvar=daily_vcvar, T=20)
#   print(result.tail())
