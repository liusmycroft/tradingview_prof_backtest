"""成交量最高时刻-主卖笔均成交金额因子 (Avg Sell Trade Amount at Peak Volume)

衡量成交量最高时刻的主卖方向笔均成交金额。
"""

import pandas as pd

from factors.base import BaseFactor


class PeakVolSellAmtFactor(BaseFactor):
    """成交量最高时刻-主卖笔均成交金额因子"""

    name = "PEAK_VOL_SELL_AMT"
    category = "高频成交分布"
    description = "成交量最高时刻的主卖笔均成交金额，取T日滚动均值"

    def compute(
        self,
        daily_peak_sell_amt: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算成交量最高时刻-主卖笔均成交金额因子。

        公式: factor = (1/T) * sum(peak_sell_avg_amount)

        Args:
            daily_peak_sell_amt: 每日预计算的成交量最高时刻主卖笔均成交金额
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        result = daily_peak_sell_amt.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 成交量最高时刻通常是市场最活跃、信息交易最密集的时段。在该时刻的
# 主卖方向笔均成交金额反映了卖方的交易规模特征。笔均成交金额越大，
# 说明大单卖出越集中，可能暗示机构或知情交易者在集中出货。
# 该因子与未来收益通常呈负相关。
