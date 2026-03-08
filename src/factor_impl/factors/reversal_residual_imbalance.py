"""反转残差非孤立成交不平衡因子 (Reversal-Residual Non-Isolated Trade Imbalance)

对反转残差进行非孤立成交量加权，衡量剔除孤立成交后的买卖不平衡。
"""

import pandas as pd

from factors.base import BaseFactor


class ReversalResidualImbalanceFactor(BaseFactor):
    """反转残差非孤立成交不平衡因子"""

    name = "REVERSAL_RESIDUAL_IMBALANCE"
    category = "高频成交分布"
    description = "反转残差非孤立成交不平衡因子：剔除孤立成交后的买卖不平衡与反转残差的乘积，取T日滚动均值"

    def compute(
        self,
        daily_reversal_residual: pd.DataFrame,
        daily_non_isolated_imbalance: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算反转残差非孤立成交不平衡因子。

        公式: factor = (1/T) * sum(reversal_residual * non_isolated_imbalance)

        Args:
            daily_reversal_residual: 每日反转残差 (index=日期, columns=股票代码)
            daily_non_isolated_imbalance: 每日非孤立成交不平衡
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        # 日度因子值 = 反转残差 * 非孤立成交不平衡
        daily_factor = daily_reversal_residual * daily_non_isolated_imbalance

        # T 日滚动均值
        result = daily_factor.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 反转残差非孤立成交不平衡因子结合了两个维度的信息：
# 1. 反转残差：剥离市场和行业因素后的纯粹反转信号
# 2. 非孤立成交不平衡：剔除孤立成交（单笔独立成交）后的买卖不平衡
#
# 孤立成交通常是噪声交易，剔除后的不平衡更能反映知情交易者的方向。
# 将其与反转残差结合，可以识别出由知情交易驱动的反转机会。
