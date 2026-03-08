import pandas as pd

from factors.base import BaseFactor


class MRRFactor(BaseFactor):
    """基于投资者交流的关联动量因子 (Mention-Related Return)"""

    name = "MRR"
    category = "图谱网络-动量溢出"
    description = "基于投资者交流的关联动量：投资者论坛中被提及最多的5只关联股票收益等权均值，取T日滚动均值"

    def compute(
        self,
        daily_mrr: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算基于投资者交流的关联动量因子。

        公式: MRR = rolling_mean_T( (1/5) * sum(Ret_j) )
        此处输入已为预计算的每日关联动量值。

        Args:
            daily_mrr: 预计算的每日关联动量值 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        result = daily_mrr.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 投资者交流过程中被提及的关联公司与目标公司间存在收益联动，而且关联
# 股票被投资者提及的越频繁，股票间收益联动越强。因子通过统计东方财富
# 论坛中目标股票论坛上被提及次数最多的5只关联股票的等权收益均值来
# 构建关联动量信号。取 T 日滚动均值以平滑日间波动。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.mrr import MRRFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_mrr = pd.DataFrame(
#       np.random.randn(30, 2) * 0.01, index=dates, columns=stocks,
#   )
#
#   factor = MRRFactor()
#   result = factor.compute(daily_mrr=daily_mrr, T=20)
#   print(result.tail())
