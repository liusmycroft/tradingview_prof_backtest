import numpy as np
import pandas as pd

from factors.base import BaseFactor


class DropMomentATDFactor(BaseFactor):
    """下跌时刻笔均成交金额因子 (Standardized ATD Down)"""

    name = "DROP_MOMENT_ATD"
    category = "高频因子-成交分布类"
    description = "下跌时刻笔均成交金额与全天笔均成交金额之比，刻画主力资金抄底行为"

    def compute(
        self,
        down_amount: pd.DataFrame,
        down_deal_num: pd.DataFrame,
        total_amount: pd.DataFrame,
        total_deal_num: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算下跌时刻笔均成交金额因子。

        公式:
            ATD_Down = sum(Amt_t, t in P) / sum(DealNum_t, t in P)
            ATD_T = sum(Amt_t, t in T) / sum(DealNum_t, t in T)
            SATD_Down = ATD_Down / ATD_T

        Args:
            down_amount: 下跌时刻成交额之和 (index=日期, columns=股票代码)
            down_deal_num: 下跌时刻成交笔数之和
            total_amount: 全天成交额之和
            total_deal_num: 全天成交笔数之和

        Returns:
            pd.DataFrame: 标准化下跌时刻笔均成交金额因子值
        """
        atd_down = down_amount / down_deal_num.replace(0, np.nan)
        atd_total = total_amount / total_deal_num.replace(0, np.nan)
        result = atd_down / atd_total.replace(0, np.nan)
        return result
