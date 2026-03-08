import pandas as pd

from factors.base import BaseFactor


class BTypeVolumeDistFactor(BaseFactor):
    """b型成交量分布因子 (B-Type Volume Distribution)。"""

    name = "B_TYPE_VOLUME_DIST"
    category = "高频成交分布"
    description = "成交量支撑区域上限与日内最低价的差异，识别b型成交量分布形态"

    def compute(
        self,
        daily_vsa_high2min: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算b型成交量分布因子。

        公式: vsa_high2min = VSA_High - Low
              VSA_High 为成交量支撑区域上限价格
        因子值为 T 日滚动均值。

        Args:
            daily_vsa_high2min: 预计算的每日 VSA_High 与最低价差异
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_vsa_high2min.rolling(window=T, min_periods=T).mean()
        return result
