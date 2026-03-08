"""反向日内逆转的频率因子 (Negative Reversal Frequency, NR)

统计隔夜收益为负且日内收益也为负（反向日内逆转）的交易日占比。
"""

import pandas as pd

from factors.base import BaseFactor


class NegativeReversalFreqFactor(BaseFactor):
    """反向日内逆转的频率因子 (NR)"""

    name = "NEGATIVE_REVERSAL_FREQ"
    category = "高频收益分布"
    description = "反向日内逆转的频率：隔夜收益为负且日内收益也为负的交易日占比，取T日滚动均值"

    def compute(
        self,
        overnight_ret: pd.DataFrame,
        intraday_ret: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算反向日内逆转的频率因子。

        公式: NR = (1/T) * sum(I(overnight_ret < 0 & intraday_ret < 0))

        Args:
            overnight_ret: 每日隔夜收益率 (index=日期, columns=股票代码)
            intraday_ret: 每日日内收益率 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        # 隔夜为负且日内也为负的指示变量
        indicator = ((overnight_ret < 0) & (intraday_ret < 0)).astype(float)

        # 将原始 NaN 位置保持为 NaN
        mask = overnight_ret.isna() | intraday_ret.isna()
        indicator[mask] = float("nan")

        # T 日滚动均值即为频率
        result = indicator.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 反向日内逆转的频率 NR 衡量隔夜收益为负且日内收益也为负的交易日占比。
# 当隔夜收益为负时，如果日内收益继续为负（未发生逆转），说明市场对负面
# 信息的消化不充分，存在持续的卖压。该频率越高，表明股票的负面信息冲击
# 越持久，未来可能继续承压。
