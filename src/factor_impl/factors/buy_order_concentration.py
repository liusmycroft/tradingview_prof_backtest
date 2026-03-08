import pandas as pd

from factors.base import BaseFactor


class BuyOrderConcentrationFactor(BaseFactor):
    """买单集中度因子 (Buy Order Concentration)"""

    name = "BUY_ORDER_CONCENTRATION"
    category = "高频资金流"
    description = "买单成交额平方和与总成交额平方的比值，衡量买单成交的集中程度"

    def compute(
        self,
        daily_buy_concentration: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算买单集中度因子。

        公式: (1/T) * sum_{n} (sum_j buy_amt_{i,j,n}^2) / (total_amt_{i,n})^2

        日度买单集中度由逐笔成交数据预计算，此处取 T 日滚动均值。

        Args:
            daily_buy_concentration: 预计算的每日买单集中度 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 买单集中度的 T 日滚动均值
        """
        result = daily_buy_concentration.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 买单集中度刻画了日内买单成交金额分布的均匀程度。
# 集中度越高（少数大单主导），未来超额收益表现越好。
# 基于逐笔成交数据中的叫买单号将逐笔成交合成为买单数据，
# 计算每笔买单成交额的平方和除以总成交额的平方。
