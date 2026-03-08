import pandas as pd

from factors.base import BaseFactor


class VolumeLongBigSelectFactor(BaseFactor):
    """基于大小单和成交时长的精选复合交易占比因子 (VolumeLongBigSelect)"""

    name = "VOLUME_LONG_BIG_SELECT"
    category = "高频资金流"
    description = "基于大小单和成交时长的精选复合交易占比，衡量知情交易者的交易行为"

    def compute(
        self,
        daily_volume_long_big_select: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算精选复合交易占比因子。

        公式:
            VolumeLongBigSelect =
              (大买&漫长的大卖 + 漫长的大买&非漫长的大卖) / 总成交量
            - (非漫长的非大买&非漫长的大卖 + 非漫长的大买&非漫长的非大卖) / 总成交量

        本方法接收预计算的每日因子值，输出滚动 T 日均值。

        Args:
            daily_volume_long_big_select: 预计算的每日因子值 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 滚动 T 日均值
        """
        result = daily_volume_long_big_select.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 精选复合交易占比因子同时考虑了逐笔委托单的"大小单"和"成交时长"属性，
# 是对成交单更细致的划分。基于细致划分下的因子表现，选取有效性较高的
# 成交单类型进行复合，进而得到表现更优的复合因子。
#
# 大单判断标准：成交量大于前10%分位点的委买/卖单。
# 漫长订单判断标准：成交时长大于前10%分位点的委买/卖单。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.volume_long_big_select import VolumeLongBigSelectFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=30)
#   stocks = ["000001.SZ", "000002.SZ"]
#   daily = pd.DataFrame(
#       np.random.randn(30, 2) * 0.01, index=dates, columns=stocks,
#   )
#
#   factor = VolumeLongBigSelectFactor()
#   result = factor.compute(daily_volume_long_big_select=daily, T=20)
#   print(result.tail())
