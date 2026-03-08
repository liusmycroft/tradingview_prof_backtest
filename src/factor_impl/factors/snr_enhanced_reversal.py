import pandas as pd

from factors.base import BaseFactor


class SNREnhancedReversalFactor(BaseFactor):
    """信噪比增强反转因子 (SNR Enhanced Reversal)。"""

    name = "SNR_ENHANCED_REVERSAL"
    category = "高频动量反转"
    description = "以信噪比归一化权重乘以反转因子，增强高信号股票的反转信号"

    def compute(
        self,
        snr: pd.DataFrame,
        reversal: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算信噪比增强反转因子。

        公式: enhanced_reversal = Weight_i * reversal_i
              Weight_i = (SNR_i - min(SNR)) / (max(SNR) - min(SNR))

        Args:
            snr: 日度信噪比因子 (index=日期, columns=股票代码)
            reversal: 20日反转因子 (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 信噪比增强反转因子值
        """
        snr_min = snr.min(axis=1)
        snr_max = snr.max(axis=1)
        snr_range = snr_max - snr_min
        snr_range = snr_range.replace(0, float("nan"))

        weight = snr.sub(snr_min, axis=0).div(snr_range, axis=0)

        result = weight * reversal
        return result
