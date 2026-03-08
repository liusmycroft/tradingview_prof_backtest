import numpy as np
import pandas as pd

from factors.base import BaseFactor


class ChipPnlAdjustedAFHFactor(BaseFactor):
    """筹码盈亏系数修正后的积极灵活筹码因子 (Chip PnL Adjusted AFH)"""

    name = "CHIP_PNL_ADJUSTED_AFH"
    category = "行为金融因子-筹码分布"
    description = "基于筹码盈亏系数修正积极灵活筹码，捕捉处置效应下的投资者行为"

    def compute(
        self,
        close_price: pd.DataFrame,
        trade_price_vol_afh: pd.DataFrame,
        total_volume_20d: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算筹码盈亏系数修正后的积极灵活筹码因子。

        公式:
            alpha_i = Close_t / TradePrice_i
            AFH_Return = sum(alpha_i * Vol_AFH_i) / sum(Volume_k, k=T-20..T)

        Args:
            close_price: 截止日收盘价 (index=日期, columns=股票代码)
            trade_price_vol_afh: 预计算的 sum(alpha_i * Vol_AFH_i)
                                即盈亏系数加权的积极灵活筹码量
            total_volume_20d: 过去20日总成交量之和

        Returns:
            pd.DataFrame: 修正后的积极灵活筹码因子值
        """
        result = trade_price_vol_afh / total_volume_20d.replace(0, np.nan)
        return result
