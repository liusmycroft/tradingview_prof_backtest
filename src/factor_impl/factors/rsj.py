import pandas as pd

from .base import BaseFactor


class RSJFactor(BaseFactor):
    """上下行波动率不对称性因子 (RSJ)"""

    name = "RSJ"
    category = "高频波动跳跃"
    description = "上下行波动率不对称性，(RV_up - RV_down) / RV"

    def compute(
        self,
        daily_rsj: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算上下行波动率不对称性因子。

        公式:
            RV_d = sum(r_{d,i}^2)
            RV_up_d = sum(r_{d,i}^2 * I(r>0))
            RV_down_d = sum(r_{d,i}^2 * I(r<0))
            RSJ_d = (RV_up_d - RV_down_d) / RV_d
            因子 = T 日滚动均值

        Args:
            daily_rsj: 预计算的每日 RSJ 值
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: RSJ 因子值，T 日滚动均值
        """
        result = daily_rsj.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# RSJ 衡量上下行波动的不对称性。RSJ 通常有负风险溢价，短期日内
# 情绪不稳定的大幅上涨往往跟着未来的补跌（RSJ 越大），短期日内
# 情绪不稳定的大幅下跌也往往跟着未来的补涨。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.rsj import RSJFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_rsj = pd.DataFrame(
#       np.random.uniform(-0.5, 0.5, (30, 2)),
#       index=dates, columns=stocks,
#   )
#
#   factor = RSJFactor()
#   result = factor.compute(daily_rsj=daily_rsj, T=20)
#   print(result.tail())
