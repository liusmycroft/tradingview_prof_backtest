"""虹吸效应因子 (Siphon Effect)

衡量资金从未被炒作的股票被"虹吸"到被炒作的股票中，
待炒作结束后资金"回流"带动反弹的效应。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class SiphonEffectFactor(BaseFactor):
    """虹吸效应因子"""

    name = "SIPHON_EFFECT"
    category = "高频资金流"
    description = "虹吸回流综合效应与虹吸效应之差的均值与标准差等权合成，衡量资金虹吸与回流"

    def compute(
        self,
        daily_net_siphon: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算虹吸效应因子。

        日内计算逻辑（已预计算为 daily_net_siphon）:
          1. AM = 个股t分钟成交量 / 个股t-1分钟成交量
          2. RAM = AM if 个股收益率 > TA加权平均收益率 else 0
          3. 市场炒作热度 = TA加权的个股每分钟RAM之和
          4. 热度时间 = 炒作热度最大的23分钟
          5. 虹吸效应 = corr(个股主卖占比, 市场主卖占比) 在热度时间
          6. 虹吸回流综合效应 = corr(个股主卖占比, 市场主卖占比) 在热度时间+回流时刻
          7. 净虹吸效应 = 虹吸回流综合效应 - 虹吸效应

        因子值 = mean(净虹吸效应, T日) + std(净虹吸效应, T日) 等权合成。

        Args:
            daily_net_siphon: 预计算的每日净虹吸效应 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 虹吸效应因子值
        """
        roll_mean = daily_net_siphon.rolling(window=T, min_periods=1).mean()
        roll_std = daily_net_siphon.rolling(window=T, min_periods=1).std(ddof=1)
        # 等权合成
        result = 0.5 * roll_mean + 0.5 * roll_std
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# "虹吸效应"衡量资金从没被炒作的股票被"虹吸"到被炒作的股票中，
# 待炒作结束之后，资金"回流"到没被炒作的股票中，带动这部分股票
# 价格产生反弹。如果资金被"虹吸"当天就产生了资金的回流，那么
# 股票日后反弹的空间就会比较小，是一个反向因子。
