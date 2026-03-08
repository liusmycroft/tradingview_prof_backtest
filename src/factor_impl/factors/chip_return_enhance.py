import pandas as pd

from factors.base import BaseFactor


class ChipReturnEnhanceFactor(BaseFactor):
    """筹码收益增强因子 (Chip Return Enhancement)"""

    name = "CHIP_RETURN_ENHANCE"
    category = "量价因子改进"
    description = "筹码收益调整因子多头端与反转因子空头端的合成，融合动量与反转效应"

    def compute(
        self,
        holding_ret_adj: pd.DataFrame,
        ret20: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算筹码收益增强因子。

        公式: (1 - ret20) * ret20 + (1 - ret20) * holding_ret_adj

        Args:
            holding_ret_adj: 筹码收益调整因子 (index=日期, columns=股票代码)
                holding_ret * sign(mkt_holding_ret)
            ret20: 过去20日收益率 (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 筹码收益增强因子值
        """
        result = (1 - ret20) * ret20 + (1 - ret20) * holding_ret_adj
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 筹码收益增强因子将筹码收益调整因子的多头端与反转因子 ret20 的空头端
# 进行合成。筹码收益调整因子优点在于能够对多头组有较好区分，反转因子
# 优点在于对于空头组具有显著的区分效果，两者合成后兼具动量与反转效应。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.chip_return_enhance import ChipReturnEnhanceFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=30)
#   stocks = ["000001.SZ", "000002.SZ"]
#   holding_ret_adj = pd.DataFrame(
#       np.random.randn(30, 2) * 0.05, index=dates, columns=stocks,
#   )
#   ret20 = pd.DataFrame(
#       np.random.randn(30, 2) * 0.1, index=dates, columns=stocks,
#   )
#
#   factor = ChipReturnEnhanceFactor()
#   result = factor.compute(holding_ret_adj=holding_ret_adj, ret20=ret20)
#   print(result.tail())
