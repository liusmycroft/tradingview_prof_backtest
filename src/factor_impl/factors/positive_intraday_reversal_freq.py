import pandas as pd

from factors.base import BaseFactor


class PositiveIntradayReversalFreqFactor(BaseFactor):
    """正向日内逆转的频率因子 (Positive Intraday Reversal Frequency, PR)"""

    name = "POSITIVE_INTRADAY_REVERSAL_FREQ"
    category = "高频收益分布"
    description = "隔夜收益为负且日内收益为正的交易日占比，衡量正向日内逆转频率"

    def compute(
        self,
        ret_co: pd.DataFrame,
        ret_oc: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算正向日内逆转频率因子。

        公式: PR = (1/T) * sum(I{RET_CO < 0} * I{RET_OC > 0})

        Args:
            ret_co: 隔夜收益率 (close-to-open), index=日期, columns=股票代码
            ret_oc: 日内收益率 (open-to-close), index=日期, columns=股票代码
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 正向日内逆转频率
        """
        # 隔夜收益为负 且 日内收益为正
        indicator = ((ret_co < 0) & (ret_oc > 0)).astype(float)

        # T 日滚动均值即为频率
        result = indicator.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 正向日内逆转频率衡量隔夜交易者与盘中交易者之间"拉锯战"的强度。
# 当负的隔夜收益后伴随正的日内收益，即出现了正向日内逆转。
# 该频率越高，说明日内价格被过度修正拉高，未来越有可能补跌，
# 因此与未来收益负相关。
