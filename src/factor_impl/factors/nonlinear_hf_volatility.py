import numpy as np
import pandas as pd

from factors.base import BaseFactor


class NonlinearHFVolatilityFactor(BaseFactor):
    """非线性高频波动率因子 (Nonlinear High-Frequency Volatility)"""

    name = "NONLINEAR_HF_VOLATILITY"
    category = "高频波动跳跃"
    description = "非线性化高频波动率与新特异率的乘积，改进特异波动率因子的头部区分能力"

    def compute(
        self,
        idio_ratio: pd.DataFrame,
        hf_std: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算非线性高频波动率因子。

        公式:
          1. 新特异率 = 1 - R^2 (来自五因子回归)
          2. norm_i = (std_i - min(std)) / (max(std) - min(std))  (横截面归一化)
          3. std_nonlinear = exp(norm)
          4. 特异波动率 = sqrt(新特异率) * std_nonlinear

        Args:
            idio_ratio: 新特异率 (1-R^2)，index=日期, columns=股票代码，值域[0,1]
            hf_std: 高频波动率(日内分钟收益率标准差)，index=日期, columns=股票代码

        Returns:
            pd.DataFrame: 非线性高频波动率因子值
        """
        # 横截面归一化
        row_min = hf_std.min(axis=1)
        row_max = hf_std.max(axis=1)
        denom = row_max - row_min
        denom = denom.replace(0, np.nan)

        norm = hf_std.sub(row_min, axis=0).div(denom, axis=0)

        # 非线性变换
        std_nonlinear = np.exp(norm)

        # 最终因子
        result = np.sqrt(idio_ratio) * std_nonlinear
        return result
