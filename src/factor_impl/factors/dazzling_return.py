import numpy as np
import pandas as pd

from factors.base import BaseFactor


class DazzlingReturnFactor(BaseFactor):
    """耀眼收益率因子 (Dazzling Return)"""

    name = "DAZZLING_RETURN"
    category = "高频因子-动量反转类"
    description = "成交量激增时刻的分钟收益率均值，衡量量增引起的价格变动幅度"

    def compute(
        self,
        daily_dazzling_return: pd.DataFrame,
        cross_section_mean: pd.DataFrame = None,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算耀眼收益率因子。

        公式:
            1. 日内成交量差值 > 均值+1倍标准差 的时刻为"成交量激增时刻"
            2. 激增时刻的分钟收益率均值 = 日耀眼收益率
            3. 适度日耀眼收益率 = |日耀眼收益率 - 横截面均值|
            4. 月耀眼收益率 = mean(适度日耀眼收益率, T日) 与 std 等权合成

        Args:
            daily_dazzling_return: 预计算的日耀眼收益率 (index=日期, columns=股票代码)
            cross_section_mean: 横截面均值 (可选，若不提供则自动计算)
            T: 月度合成窗口，默认 20

        Returns:
            pd.DataFrame: 月耀眼收益率因子值
        """
        if cross_section_mean is None:
            cross_section_mean = daily_dazzling_return.mean(axis=1)

        moderate = daily_dazzling_return.sub(cross_section_mean, axis=0).abs()

        rolling_mean = moderate.rolling(window=T, min_periods=1).mean()
        rolling_std = moderate.rolling(window=T, min_periods=1).std().fillna(0)

        result = 0.5 * rolling_mean + 0.5 * rolling_std
        return result
