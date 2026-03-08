import pandas as pd

from factors.base import BaseFactor


class ChipDistributionKurtosisFactor(BaseFactor):
    """筹码分布峰度因子"""

    name = "CHIP_DISTRIBUTION_KURTOSIS"
    category = "行为金融-筹码分布"
    description = "筹码分布的加权峰度，衡量筹码在平均值处峰值的高低程度"

    def compute(
        self,
        daily_chip_kurtosis: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算筹码分布峰度因子。

        日内计算逻辑（预计算阶段）：
        1. p_i = vol_i / sum(vol_i)  筹码加权权重
        2. mean = sum(price_i * p_i)  加权平均成本
        3. std = sqrt(sum((price_i - mean)^2 * p_i))  加权标准差
        4. kurtosis = sum(((price_i - mean) / std)^4 * p_i)

        本方法对预计算的日度因子取 T 日 EMA。

        Args:
            daily_chip_kurtosis: 预计算的每日筹码分布峰度，
                index=日期, columns=股票代码。
            T: EMA 窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        result = daily_chip_kurtosis.ewm(span=T, min_periods=1).mean()
        return result


class ChipDistributionSkewnessFactor(BaseFactor):
    """筹码分布偏度因子"""

    name = "CHIP_DISTRIBUTION_SKEWNESS"
    category = "行为金融-筹码分布"
    description = "筹码分布的加权偏度，衡量筹码分布的偏斜程度和方向"

    def compute(
        self,
        daily_chip_skewness: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算筹码分布偏度因子。

        日内计算逻辑（预计算阶段）：
        1. p_i = vol_i / sum(vol_i)  筹码加权权重
        2. mean = sum(price_i * p_i)  加权平均成本
        3. std = sqrt(sum((price_i - mean)^2 * p_i))  加权标准差
        4. skewness = sum(((price_i - mean) / std)^3 * p_i)

        本方法对预计算的日度因子取 T 日 EMA。

        Args:
            daily_chip_skewness: 预计算的每日筹码分布偏度，
                index=日期, columns=股票代码。
            T: EMA 窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        result = daily_chip_skewness.ewm(span=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 筹码分布偏度因子衡量筹码分布偏斜程度和偏斜方向。
# 筹码分布峰度因子衡量筹码分布在平均值处峰值的高低程度。
# 经测试，筹码分布峰度因子有较好的选股能力，与未来收益负相关。
