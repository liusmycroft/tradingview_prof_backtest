import numpy as np
import pandas as pd

from factors.base import BaseFactor


class SuperBigRetFactor(BaseFactor):
    """超大单涨跌幅 SuperBigRet 因子"""

    name = "SUPER_BIG_RET"
    category = "高频动量反转"
    description = "超大单涨跌幅，衡量超大单成交推动的价格变动幅度"

    def compute(
        self,
        daily_super_big_return: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算超大单涨跌幅因子。

        公式:
            SuperBigRet = rolling_sum(log(1 + daily_super_big_return), T)

        对超大单推动的日收益率取对数后滚动求和，得到T日累计涨跌幅。

        Args:
            daily_super_big_return: 每日超大单推动的收益率，
                                     index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 超大单涨跌幅因子值 (T日滚动累计)。
        """
        log_ret = np.log1p(daily_super_big_return)
        result = log_ret.rolling(window=T, min_periods=1).sum()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 超大单涨跌幅 (SuperBigRet) 因子衡量超大单成交推动的累计价格变动。
# 超大单通常代表机构或大资金的交易行为，其推动的涨跌幅反映了
# 大资金对股价方向的判断。因子值为正表示大资金推动上涨，
# 为负表示大资金推动下跌。可用于动量或反转策略。
#
# 【使用示例】
#
#   from factors.super_big_ret import SuperBigRetFactor
#   factor = SuperBigRetFactor()
#   result = factor.compute(daily_super_big_return=ret_df, T=20)
