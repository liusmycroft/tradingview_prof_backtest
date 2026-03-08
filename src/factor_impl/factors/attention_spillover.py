import numpy as np
import pandas as pd

from factors.base import BaseFactor


class AttentionSpilloverFactor(BaseFactor):
    """基于异常收益的注意力溢出因子 (Attention Spillover via Abnormal Returns)"""

    name = "SPILL"
    category = "行为金融-投资者注意力"
    description = "相近股票注意力均值与个股注意力之差，衡量注意力溢出程度"

    def compute(
        self,
        daily_attn: pd.DataFrame,
        peer_avg_attn: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算注意力溢出因子。

        预计算逻辑:
          ATTN_{i,t} = (1/m) * sum((r_{i,d} - r_bar_d)^2)  (月度异常收益方差)
          peer_avg_attn = 同行业同市值分组内其他股票的 ATTN 均值

        因子值: SPILL = peer_avg_attn - daily_attn，再取 T 日 EMA。

        Args:
            daily_attn: 个股注意力指标 (index=日期, columns=股票代码)
            peer_avg_attn: 相近股票注意力均值 (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 注意力溢出因子的 T 日 EMA
        """
        spill = peer_avg_attn - daily_attn
        result = spill.ewm(span=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 投资者注意力存在溢出效应。当一只股票关注度较高时，投资者也会注意到
# 与其相似的邻居股票。以异常收益为注意力代理指标，计算相近股票（同行业
# 同市值分组）的注意力均值与个股注意力之差。差值越大，表明未来注意力
# 溢出效应越显著，股票预期会取得正向超额。
