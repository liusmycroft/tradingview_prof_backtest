"""偏锋涨跌幅因子 (Skewed Momentum Return)

基于日内动量时刻的偏离度加总，再取 20 日标准差。
"""

import pandas as pd

from factors.base import BaseFactor


class SkewedMomentumReturnFactor(BaseFactor):
    """偏锋涨跌幅因子"""

    name = "SKEWED_MOMENTUM_RETURN"
    category = "高频动量反转"
    description = "偏锋涨跌幅，动量时刻偏离度加总的滚动标准差，衡量日内过度反应程度"

    def compute(
        self,
        daily_deviation_sum: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算偏锋涨跌幅因子。

        Args:
            daily_deviation_sum: 预计算的每日动量时刻偏离度加总
                (index=日期, columns=股票代码)。
                每日值 = 当日所有"动量时刻"的 1 分钟偏离度之和。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，过去 T 日的标准差。
        """
        result = daily_deviation_sum.rolling(window=T, min_periods=T).std()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 偏锋涨跌幅因子通过以下步骤构造：
# 1. 每分钟计算市场上涨/下跌股票收益率均值作为动量基准；
# 2. 每只股票每分钟与同方向动量基准之差为偏离度；
# 3. 偏离度偏度 > 0 的时刻为"动量时刻"，加总偏离度得日度因子；
# 4. 取过去 20 日标准差得到最终因子。
#
# 因子值越高，日内过度反应越剧烈，未来预期收益越低。
