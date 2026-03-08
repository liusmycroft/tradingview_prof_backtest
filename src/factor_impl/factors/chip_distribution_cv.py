import pandas as pd

from factors.base import BaseFactor


class ChipDistributionCVFactor(BaseFactor):
    """筹码分布变异系数因子"""

    name = "CHIP_DISTRIBUTION_CV"
    category = "行为金融-筹码分布"
    description = "筹码分布的变异系数（标准差/均值），衡量筹码分布的离散程度"

    def compute(
        self,
        chip_std: pd.DataFrame,
        chip_mean: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算筹码分布变异系数因子。

        公式:
            daily_cv = chip_std / chip_mean
            factor = rolling_mean(daily_cv, T)

        chip_std: 筹码分布价格的标准差（以成交量为权重）
        chip_mean: 筹码分布价格的均值（以成交量为权重）

        Args:
            chip_std: 每日筹码分布的标准差 (index=日期, columns=股票代码)
            chip_mean: 每日筹码分布的均值 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 筹码分布变异系数的 T 日滚动均值
        """
        daily_cv = chip_std / chip_mean
        result = daily_cv.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 筹码分布变异系数衡量筹码在不同价位上的分散程度。
# 变异系数 = 标准差 / 均值，消除了价格水平的影响。
# CV 越大，说明筹码分布越分散，持有者成本差异越大，
# 可能导致更大的多空分歧和价格波动。
# CV 越小，说明筹码集中在某一价位附近，持有者成本趋同。
