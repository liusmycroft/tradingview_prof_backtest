import numpy as np
import pandas as pd

from factors.base import BaseFactor


class VcdeFactor(BaseFactor):
    """基于筹码分布的V型处置效应因子 (V-shaped Chip Disposition Effect)"""

    name = "VCDE"
    category = "行为金融-处置效应"
    description = "基于筹码分布的V型处置效应，同时刻画盈利和亏损下投资者的卖出意愿"

    def compute(
        self,
        cpgr: pd.DataFrame,
        cplr: pd.DataFrame,
        variant: int = 1,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算V型处置效应因子。

        公式:
          VCDE1 = |CPGR - CPLR|
          VCDE2 = CPGR + 0.23 * CPLR
          VCDE3 = CPGR + CPLR

        Args:
            cpgr: 当日兑现盈利占筹码总盈利比例 (index=日期, columns=股票代码)
            cplr: 当日兑现亏损占筹码总亏损比例 (index=日期, columns=股票代码)
            variant: 因子变体 (1, 2, 3)，默认 1
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: VCDE 因子值的 T 日 EMA
        """
        if variant == 1:
            daily = (cpgr - cplr).abs()
        elif variant == 2:
            daily = cpgr + 0.23 * cplr
        elif variant == 3:
            daily = cpgr + cplr
        else:
            raise ValueError(f"variant must be 1, 2, or 3, got {variant}")

        result = daily.ewm(span=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# V型处置效应因子同时包含了盈利或亏损下投资者的卖出意愿。
# CPGR 为当日兑现盈利占筹码总盈利的比例，CPLR 为当日兑现亏损占筹码
# 总亏损的比例。盈利或亏损幅度越大，V型处置效应越明显，投资者出售意愿越强。
# VCDE1 衡量盈亏兑现差异的绝对值；VCDE2 对亏损兑现赋予较低权重(0.23)；
# VCDE3 为盈亏兑现之和。
