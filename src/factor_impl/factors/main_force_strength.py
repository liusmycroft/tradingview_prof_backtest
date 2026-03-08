import pandas as pd

from factors.base import BaseFactor


class MainForceStrengthFactor(BaseFactor):
    """主力交易强度因子 (Main Force Trading Strength - MTS)"""

    name = "MAIN_FORCE_STRENGTH"
    category = "高频成交分布"
    description = "逐笔成交金额与分钟成交金额的秩相关系数滚动均值，衡量主力资金活跃度"

    def compute(
        self,
        daily_ts: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算主力交易强度因子。

        公式: MTS = RankCorr(A_order, A_minute)
        因子值为 T 日滚动均值。

        Args:
            daily_ts: 预计算的每日秩相关系数 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 秩相关系数的 T 日滚动均值
        """
        result = daily_ts.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 主力交易强度因子通过计算逐笔成交金额（A_order）与分钟成交金额
# （A_minute）之间的秩相关系数来衡量主力资金的活跃程度。
#
# 当大单成交集中在成交额较高的分钟时，秩相关系数较高，说明主力资金
# 在积极交易。因子取 T 日滚动均值以平滑日间波动。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.main_force_strength import MainForceStrengthFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_ts = pd.DataFrame(
#       np.random.uniform(-0.5, 0.8, (30, 2)),
#       index=dates, columns=stocks,
#   )
#
#   factor = MainForceStrengthFactor()
#   result = factor.compute(daily_ts=daily_ts, T=20)
#   print(result.tail())
