import numpy as np
import pandas as pd

from .base import BaseFactor


class AskDepthFactor(BaseFactor):
    """卖盘深度因子 (Ask Depth)。"""

    name = "ASK_DEPTH"
    category = "高频流动性"
    description = "卖盘深度因子，基于卖盘挂单量与价差的比值衡量流动性深度"

    def compute(
        self,
        daily_ask_depth: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算卖盘深度因子。

        Args:
            daily_ask_depth: 预计算的每日平均卖盘深度，
                             index=日期，columns=股票代码。
            T: 指数移动平均窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期，columns=股票代码。
                          T 日指数移动平均。
        """
        # 指数移动平均，span=T
        result = daily_ask_depth.ewm(span=T, min_periods=T).mean()

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 卖盘深度因子衡量的是卖方挂单的流动性深度。核心公式：
#   ask_depth = sum(|av_i * mid_price / (a_i - last_mid + epsilon)|)
# 其中：
#   - av_i: 第 i 档卖盘挂单量
#   - a_i: 第 i 档卖盘价格
#   - mid_price: 中间价
#   - last_mid: 上一时刻中间价
#   - epsilon: 防止除零的小常数
#
# 该因子通过 EMA 平滑日度卖盘深度，反映中期流动性供给水平。
# 卖盘深度越大，说明卖方流动性越充裕，股票越不容易被大单推高。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.ask_depth import AskDepthFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=25)
#   daily_ask_depth = pd.DataFrame(
#       np.random.rand(25, 2) * 1e6,
#       index=dates,
#       columns=["000001.SZ", "600000.SH"],
#   )
#
#   factor = AskDepthFactor()
#   result = factor.compute(daily_ask_depth=daily_ask_depth, T=20)
#   print(result)
