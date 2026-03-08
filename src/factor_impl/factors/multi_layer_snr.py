"""多层加权日内信噪比因子 (Multi-Layer Weighted Intraday SNR)

以日内价格波动率为权重，对第二层和第三层信噪比因子加权合成。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class MultiLayerSNRFactor(BaseFactor):
    """多层加权日内信噪比因子"""

    name = "MULTI_LAYER_SNR"
    category = "高频收益分布"
    description = "多层加权日内信噪比：以日内波动率为权重对不同层级信噪比加权合成"

    def compute(
        self,
        snr_layer2: pd.DataFrame,
        snr_layer3: pd.DataFrame,
        intraday_vol: pd.DataFrame,
        delta: float = 2.0,
        **kwargs,
    ) -> pd.DataFrame:
        """计算多层加权日内信噪比因子。

        公式:
            Vol_std = (Vol - min(Vol)) / (max(Vol) - min(Vol))
            Weight_vol = 0.5 + (Vol_std - 0.5) / delta
            新SNR = Weight_vol * SNR_layer2 + (1 - Weight_vol) * SNR_layer3

        Args:
            snr_layer2: 第二层信噪比 (index=日期, columns=股票代码)
            snr_layer3: 第三层信噪比 (index=日期, columns=股票代码)
            intraday_vol: 日内价格波动率 (index=日期, columns=股票代码)
            delta: 权重缩放参数，默认 2.0

        Returns:
            pd.DataFrame: 多层加权日内信噪比因子值
        """
        # 横截面归一化波动率
        vol_min = intraday_vol.min(axis=1)
        vol_max = intraday_vol.max(axis=1)
        vol_range = vol_max - vol_min
        vol_range = vol_range.replace(0, np.nan)

        vol_std = intraday_vol.sub(vol_min, axis=0).div(vol_range, axis=0)

        # 计算权重
        weight_vol = 0.5 + (vol_std - 0.5) / delta

        # 裁剪权重到 [0, 1]
        weight_vol = weight_vol.clip(0, 1)

        # 加权合成
        result = weight_vol * snr_layer2 + (1 - weight_vol) * snr_layer3
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 日内价格波动越大时，需要剥离更多噪声，赋予第三层更大的权重；
# 加权后的复合信噪比综合了不同分解层级的信息，表现更稳健。
