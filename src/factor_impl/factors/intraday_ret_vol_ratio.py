import numpy as np
import pandas as pd

from factors.base import BaseFactor


class IntradayReturnVolRatioFactor(BaseFactor):
    """日内收益波动比因子 (Intraday Return-Volatility Ratio)"""

    name = "INTRADAY_RET_VOL_RATIO"
    category = "高频波动跳跃"
    description = "日内收益波动比，衡量收益率与波动率的比值，用于捕捉灾后重建效应"

    def compute(
        self,
        daily_intraday_return: pd.DataFrame,
        daily_intraday_volatility: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算日内收益波动比因子。

        公式:
            daily_ratio = intraday_return / intraday_volatility
            factor = rolling_mean(daily_ratio, T)

        Args:
            daily_intraday_return: 每日日内收益率，index=日期，columns=股票代码。
            daily_intraday_volatility: 每日日内波动率，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 日内收益波动比因子值 (T日滚动均值)。
        """
        daily_ratio = daily_intraday_return / daily_intraday_volatility
        result = daily_ratio.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 日内收益波动比因子衡量日内收益率与日内波动率的比值。
# 该因子可用于捕捉"灾后重建"效应：当股票经历大幅下跌（高波动、负收益）后，
# 收益波动比极低，未来可能出现反弹。反之，高收益波动比可能预示过热。
#
# 【使用示例】
#
#   from factors.intraday_ret_vol_ratio import IntradayReturnVolRatioFactor
#   factor = IntradayReturnVolRatioFactor()
#   result = factor.compute(
#       daily_intraday_return=ret_df,
#       daily_intraday_volatility=vol_df, T=20
#   )
