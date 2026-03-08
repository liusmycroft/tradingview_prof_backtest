import pandas as pd

from factors.base import BaseFactor


class COMinFactor(BaseFactor):
    """个股收益最低时的价变共振因子 (Co-movement at Minimum Returns)"""

    name = "CO_MIN"
    category = "高频动量反转"
    description = "个股收益最低时的价变共振：个股最差分钟对应的市场均值除以市场标准差，取T日滚动均值"

    def compute(
        self,
        daily_rm_min: pd.DataFrame,
        daily_std_min: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算个股收益最低时的价变共振因子。

        公式: CO_MIN = rolling_mean_T( rank_inv(RM_MIN) / STD_MIN )
        此处输入已为预计算的每日 RM_MIN 和 STD_MIN，
        截面逆序排名归一化在日频预处理中完成，此处直接做比值+滚动均值。

        Args:
            daily_rm_min: 预计算的每日 RM_MIN 值 (index=日期, columns=股票代码)
            daily_std_min: 预计算的每日 STD_MIN 值 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        daily_ratio = daily_rm_min / daily_std_min
        result = daily_ratio.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# RM_MIN 刻画了个股与市场趋势的一致性：在个股日内最差的M个分钟里，
# 全市场等权收益率的均值。因子值越小，说明个股在市场表现较差时同步
# 下跌，与全市场下跌趋势一致，属于恐慌性错杀，未来更容易被修正。
# STD_MIN 刻画了市场整体收益的分歧度，分歧度越小，市场趋势更具
# 代表性，个股与市场共振越有效。两者相除后取滚动均值得到最终因子。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.co_min import COMinFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_rm_min = pd.DataFrame(
#       np.random.randn(30, 2) * 0.001, index=dates, columns=stocks,
#   )
#   daily_std_min = pd.DataFrame(
#       np.random.rand(30, 2) * 0.01 + 0.001, index=dates, columns=stocks,
#   )
#
#   factor = COMinFactor()
#   result = factor.compute(daily_rm_min=daily_rm_min, daily_std_min=daily_std_min, T=20)
#   print(result.tail())
