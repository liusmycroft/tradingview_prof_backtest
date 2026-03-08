"""横盘时刻-主买笔均成交金额因子 (Sideways Active Buy ATD)

横盘时刻下主动买入的笔均成交金额与全天笔均成交金额之比。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class SidewaysBuyATDFactor(BaseFactor):
    """横盘时刻-主买笔均成交金额因子"""

    name = "SIDEWAYS_BUY_ATD"
    category = "高频成交分布"
    description = "横盘时刻主买笔均成交金额：横盘时刻主买笔均成交金额与全天笔均成交金额之比"

    def compute(
        self,
        zero_buy_amt: pd.DataFrame,
        zero_buy_deal_num: pd.DataFrame,
        total_amt: pd.DataFrame,
        total_deal_num: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算横盘时刻-主买笔均成交金额因子。

        公式:
            ATD_Zero_Buy = sum(Amt_t for t in P_Buy) / sum(DealNum_t for t in P_Buy)
            ATD_T = sum(Amt_t for all t) / sum(DealNum_t for all t)
            SATD_Zero_Buy = ATD_Zero_Buy / ATD_T

        Args:
            zero_buy_amt: 横盘时刻主买成交额合计 (index=日期, columns=股票代码)
            zero_buy_deal_num: 横盘时刻主买成交笔数合计 (index=日期, columns=股票代码)
            total_amt: 全天成交额合计 (index=日期, columns=股票代码)
            total_deal_num: 全天成交笔数合计 (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: SATD_Zero_Buy 因子值
        """
        atd_zero_buy = zero_buy_amt / zero_buy_deal_num
        atd_total = total_amt / total_deal_num
        result = atd_zero_buy / atd_total
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 笔均成交额类因子通过汇总"特殊的、具有较多信息含量"时刻的成交金额
# 来刻画主力资金的行为动向。笔均成交金额越大，说明该特殊时间内资金
# 优势越明显，属于主力资金的可能性越大。
# 买方主导的横盘时刻，成交越多，未来收益越低。
