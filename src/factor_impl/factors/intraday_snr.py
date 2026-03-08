import pandas as pd

from factors.base import BaseFactor


class IntradaySNRFactor(BaseFactor):
    """日内信噪比因子 (Intraday Signal-to-Noise Ratio)。"""

    name = "INTRADAY_SNR"
    category = "高频收益分布"
    description = "EMD分解后信号标准差与噪声标准差之比的对数，衡量价格信号质量"

    def compute(
        self,
        daily_snr: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算日内信噪比因子。

        公式: SNR = log(std(r_n(t)) / std(P(t) - r_n(t)))
              r_n(t) 为 EMD 分解后的信号序列
        因子值为 T 日滚动均值。

        Args:
            daily_snr: 预计算的每日信噪比 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_snr.rolling(window=T, min_periods=T).mean()
        return result
