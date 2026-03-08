import pandas as pd

from factors.base import BaseFactor


class AttentionMarketVolumeFactor(BaseFactor):
    """基于沪深总成交额的注意力捕捉因子 (Attention Market Volume)。"""

    name = "ATTENTION_MARKET_VOLUME"
    category = "行为金融注意力"
    description = "个股收益对市场总成交额的敏感程度绝对值，衡量注意力捕捉效应"

    def compute(
        self,
        daily_attention_beta: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算基于沪深总成交额的注意力捕捉因子。

        公式: |beta| from r_i = mu + beta*mkt_vol + rho1*Mkt + rho2*SMB
                                  + rho3*HML + rho4*UMD + eps
        因子值为 T 日 EMA。

        Args:
            daily_attention_beta: 预计算的每日 |beta| 值
                (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 注意力捕捉因子的 T 日 EMA
        """
        result = daily_attention_beta.ewm(span=T, min_periods=1).mean()
        return result
