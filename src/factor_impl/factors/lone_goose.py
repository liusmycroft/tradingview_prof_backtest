import numpy as np
import pandas as pd

from .base import BaseFactor


class LoneGooseFactor(BaseFactor):
    """「孤雁出群」因子 (Lone Goose Factor)。"""

    name = "LONE_GOOSE"
    category = "高频量价"
    description = "孤雁出群因子，基于非分化时刻成交量相关性的均值与标准差等权组合"

    def compute(
        self,
        daily_lone_goose: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算孤雁出群因子。

        Args:
            daily_lone_goose: 预计算的每日孤雁因子值（日内非分化时刻成交量
                              与其他股票的平均绝对相关系数），
                              index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期，columns=股票代码。
                          滚动 T 日均值与标准差的等权组合。
        """
        # 滚动均值
        rolling_mean = daily_lone_goose.rolling(window=T, min_periods=T).mean()

        # 滚动标准差
        rolling_std = daily_lone_goose.rolling(window=T, min_periods=T).std(ddof=1)

        # 等权组合
        result = 0.5 * rolling_mean + 0.5 * rolling_std

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 「孤雁出群」因子捕捉的是个股在非分化时刻（即市场整体走势趋同的时段）
# 成交量与其他股票成交量的相关性。核心逻辑：
#   1. 在日内非分化时刻，计算个股成交量与其他股票成交量的平均绝对相关系数。
#   2. 对过去 T 天的日度因子值取滚动均值和标准差。
#   3. 等权组合均值和标准差，综合反映该股票的"孤立"程度。
#
# 经济直觉：在市场趋同时段，如果某只股票的成交量模式与其他股票显著不同
# （相关性低），说明该股票可能存在独立的信息驱动交易，即"孤雁出群"。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.lone_goose import LoneGooseFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=25)
#   daily_lone_goose = pd.DataFrame(
#       np.random.rand(25, 3) * 0.5,
#       index=dates,
#       columns=["000001.SZ", "600000.SH", "000002.SZ"],
#   )
#
#   factor = LoneGooseFactor()
#   result = factor.compute(daily_lone_goose=daily_lone_goose, T=20)
#   print(result)
