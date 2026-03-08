import pandas as pd

from factors.base import BaseFactor


class QUAFactor(BaseFactor):
    """单笔成交金额分位数因子 (Single Transaction Amount Quantile)"""

    name = "QUA"
    category = "高频成交分布"
    description = "单笔成交金额分位数：衡量成交金额分布左尾厚度的滚动均值"

    def compute(
        self,
        daily_qua: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 QUA 因子。

        公式: S = (A_0.1 - A_min) / (A_max - A_min)，然后取 T 日滚动均值。

        Args:
            daily_qua: 预计算的每日 S 值 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: QUA 因子值
        """
        result = daily_qua.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# QUA（单笔成交金额分位数）通过计算每日成交金额的 10% 分位数相对于极差的
# 位置来衡量成交金额分布的左尾特征。
# S = (A_0.1 - A_min) / (A_max - A_min)
# S 越小，说明小额成交占比越高，散户交易越活跃；S 越大，说明成交金额
# 分布更均匀或大单占比更高。取 T 日滚动均值以平滑日间波动。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.qua import QUAFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   daily_qua = pd.DataFrame(0.3, index=dates, columns=stocks)
#
#   factor = QUAFactor()
#   result = factor.compute(daily_qua=daily_qua, T=20)
#   print(result.tail())
