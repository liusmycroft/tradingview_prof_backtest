import pandas as pd

from .base import BaseFactor


class LargeVolPriceCorrFactor(BaseFactor):
    """大成交量价量相关性因子"""

    name = "LARGE_VOL_PRICE_CORR"
    category = "高频量价相关性"
    description = "大成交量时段的价量相关性，大单成交更能代表主力行为"

    def compute(
        self,
        daily_large_vol_corr: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算大成交量价量相关性因子。

        公式:
            corr_VP_{D,i} = corr(v_t, p_t), t in {大成交量对应的分钟区间}
            大成交量 = 分钟成交量排名前 1/3 的区间
            因子 = T 日滚动均值

        Args:
            daily_large_vol_corr: 预计算的每日大成交量价量相关性
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 大成交量价量相关性因子值，T 日滚动均值
        """
        result = daily_large_vol_corr.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 在传统价量相关性因子计算逻辑的基础上加入了成交量大小的信息。
# 将日内分钟成交量排名前 1/3 定为"大成交量"，仅对这些时段计算
# 成交量与价格的相关性。大单成交更能代表主力行为，蕴含的信息更多。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.large_vol_price_corr import LargeVolPriceCorrFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_large_vol_corr = pd.DataFrame(
#       np.random.uniform(-0.8, 0.8, (30, 2)),
#       index=dates, columns=stocks,
#   )
#
#   factor = LargeVolPriceCorrFactor()
#   result = factor.compute(daily_large_vol_corr=daily_large_vol_corr, T=20)
#   print(result.tail())
