import numpy as np
import pandas as pd

from factors.base import BaseFactor


class VolumeCoeffVariationFactor(BaseFactor):
    """交易量变异系数因子 (Volume Coefficient of Variation)"""

    name = "VOLUME_COEFF_VARIATION"
    category = "高频因子-资金流类"
    description = "成交额序列的标准差与均值之比，衡量市场信息不对称性"

    def compute(
        self,
        minute_amount: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算交易量变异系数因子。

        公式:
            VCV = std(Amount) / mean(Amount)

        Args:
            minute_amount: 分钟成交额序列的日度聚合
                          每行为一天，每列为一只股票
                          值为当日分钟成交额的标准差/均值（预计算）
                          或直接传入日度成交额序列做滚动计算

        Returns:
            pd.DataFrame: 交易量变异系数因子值
        """
        # 如果传入的是日度成交额，做滚动计算
        T = kwargs.get("T", 5)
        rolling_std = minute_amount.rolling(window=T, min_periods=T).std()
        rolling_mean = minute_amount.rolling(window=T, min_periods=T).mean()
        result = rolling_std / rolling_mean.replace(0, np.nan)
        return result
