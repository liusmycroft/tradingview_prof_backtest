import pandas as pd

from .base import BaseFactor


class NetSupportVolumeFactor(BaseFactor):
    """净支撑成交量因子"""

    name = "NET_SUPPORT_VOLUME"
    category = "量价因子改进"
    description = "净支撑成交量因子，衡量支撑价格不破位的力量强度"

    def compute(
        self,
        daily_net_support_volume: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算净支撑成交量因子。

        公式:
            支撑成交量 = sum(vol_t, close_t < mean(close))
            阻力成交量 = sum(vol_t, close_t > mean(close))
            净支撑成交量 = (支撑成交量 - 阻力成交量) / 流通股本
            因子 = T 日滚动均值

        Args:
            daily_net_support_volume: 预计算的每日净支撑成交量
                (已除以流通股本) (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 净支撑成交量因子值，T 日滚动均值
        """
        result = daily_net_support_volume.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 每个交易日，将分钟收盘价低于日内均价的分钟成交量加总得到支撑成交量，
# 高于均价的加总得到阻力成交量。净支撑成交量 = (支撑 - 阻力) / 流通股本。
#
# 因子值较大表明支撑价格不破位的力量较强，后市上涨概率较高；
# 因子值较小表明阻碍价格上涨的力量较强，后市相对偏空。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.net_support_volume import NetSupportVolumeFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_net_support_volume = pd.DataFrame(
#       np.random.randn(30, 2) * 0.01,
#       index=dates, columns=stocks,
#   )
#
#   factor = NetSupportVolumeFactor()
#   result = factor.compute(daily_net_support_volume=daily_net_support_volume, T=20)
#   print(result.tail())
