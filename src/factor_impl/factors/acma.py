import pandas as pd

from factors.base import BaseFactor


class ACMAFactor(BaseFactor):
    """分钟成交额自相关性因子 (Autocorrelation of Minute Amount - ACMA)"""

    name = "ACMA"
    category = "高频量价相关性"
    description = "日内分钟成交额序列的自相关系数，衡量成交额的持续性与羊群效应"

    def compute(
        self,
        daily_acma: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算分钟成交额自相关性因子。

        公式: ACMA = corr(Amount_{t-i}, Amount_t)

        本方法接收预计算的每日自相关系数，输出滚动 T 日均值。

        Args:
            daily_acma: 预计算的每日分钟成交额自相关系数 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 滚动 T 日均值
        """
        result = daily_acma.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 日内分钟成交额序列有显著的自相关性。自相关性越强的股票，微观上可
# 理解为股票受到事件刺激频率更高，对信息反应时容易羊群交易，羊群效应
# 显著，未来收益表现差。
#
# ACMA = corr(Amount_{t-i}, Amount_t)，其中 i 为滞后阶数。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.acma import ACMAFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=30)
#   stocks = ["000001.SZ", "000002.SZ"]
#   daily_acma = pd.DataFrame(
#       np.random.uniform(0.1, 0.9, (30, 2)), index=dates, columns=stocks,
#   )
#
#   factor = ACMAFactor()
#   result = factor.compute(daily_acma=daily_acma, T=20)
#   print(result.tail())
