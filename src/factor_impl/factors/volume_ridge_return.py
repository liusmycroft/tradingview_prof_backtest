"""量岭分钟收益因子 (Volume Ridge Minute Return)

基于成交量的峰/岭/谷分类，计算"量岭"时点的累计收益。
"""

import pandas as pd

from factors.base import BaseFactor


class VolumeRidgeReturnFactor(BaseFactor):
    """量岭分钟收益因子"""

    name = "VOLUME_RIDGE_RETURN"
    category = "高频量价相关性"
    description = "量岭分钟收益，连续喷发成交量时点的累计收益滚动均值"

    def compute(
        self,
        daily_ridge_return: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算量岭分钟收益因子。

        Args:
            daily_ridge_return: 预计算的每日量岭累计收益
                (index=日期, columns=股票代码)。
                每日值 = 当日所有"量岭"时点的分钟收益之和。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，T 日滚动均值。
        """
        result = daily_ridge_return.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 量岭分钟收益因子基于成交量的峰/岭/谷分类：
# - 喷发成交量：高于过去 20 日同时点均值 + 1 倍标准差
# - 量岭：连续的喷发成交量（前后 1 分钟也存在喷发成交量）
# - 量峰：孤立的喷发成交量
# - 量谷：温和成交量
#
# "量岭"对应连续性大额交易，更符合个人投资者的跟随交易特征。
# 因子值越高，个人投资者交易越可能过度反应，未来收益越低。
