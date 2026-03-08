"""V型处置效应因子 (V-shaped Disposition Effect, VNSP)

VNSP = Gain + sigma * |Loss|，衡量V型处置效应的强度。
"""

import pandas as pd

from factors.base import BaseFactor


class VShapedDispositionFactor(BaseFactor):
    """V型处置效应因子 (VNSP)"""

    name = "VNSP"
    category = "行为金融-处置效应"
    description = "V型处置效应因子：VNSP = Gain + sigma * |Loss|，衡量投资者V型处置效应的强度"

    def compute(
        self,
        daily_gain: pd.DataFrame,
        daily_loss: pd.DataFrame,
        sigma: float = 1.0,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算V型处置效应因子。

        公式: VNSP = EMA_T(Gain + sigma * |Loss|)

        Args:
            daily_gain: 每日已实现盈利比率 (index=日期, columns=股票代码)
            daily_loss: 每日已实现亏损比率 (index=日期, columns=股票代码)，通常为负值
            sigma: 亏损端权重系数，默认 1.0
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日 EMA
        """
        # VNSP = Gain + sigma * |Loss|
        daily_vnsp = daily_gain + sigma * daily_loss.abs()

        # T 日 EMA 平滑
        result = daily_vnsp.ewm(span=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# V型处置效应因子 VNSP 是对传统处置效应的扩展。传统处置效应仅关注
# "卖盈持亏"行为（Gain - Loss），而V型处置效应认为投资者在亏损端
# 也存在"割肉"行为，形成V型反应模式。
#
# VNSP = Gain + sigma * |Loss| 综合了盈利兑现和亏损兑现两端的信息，
# sigma 控制亏损端的权重。因子值越大，说明投资者的处置效应越强烈，
# 无论是卖盈还是割亏都更积极。
