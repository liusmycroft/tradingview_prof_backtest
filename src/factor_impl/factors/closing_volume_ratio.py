import pandas as pd

from factors.base import BaseFactor


class ClosingVolumeRatioFactor(BaseFactor):
    """尾盘成交占比因子 (Closing Period Trading Ratio)"""

    name = "CLOSING_VOLUME_RATIO"
    category = "高频成交分布"
    description = "尾盘成交占比：尾盘30分钟成交量占全天成交量的比例，取T日滚动均值"

    def compute(
        self,
        closing_volume: pd.DataFrame,
        daily_volume: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算尾盘成交占比因子。

        公式: (1/T) * sum(closing_30min_volume / daily_volume)

        Args:
            closing_volume: 每日尾盘30分钟成交量 (index=日期, columns=股票代码)
            daily_volume: 每日全天成交量 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        # 日度尾盘占比
        daily_ratio = closing_volume / daily_volume

        # T 日滚动均值
        result = daily_ratio.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 尾盘成交占比衡量收盘前30分钟的成交集中度。尾盘是机构调仓、
# 指数基金再平衡的高频时段，尾盘成交占比偏高可能暗示机构资金
# 的参与度较高。该因子取 T 日均值以平滑日间波动。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.closing_volume_ratio import ClosingVolumeRatioFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   closing_volume = pd.DataFrame(200.0, index=dates, columns=stocks)
#   daily_volume   = pd.DataFrame(1000.0, index=dates, columns=stocks)
#
#   factor = ClosingVolumeRatioFactor()
#   result = factor.compute(closing_volume=closing_volume, daily_volume=daily_volume, T=20)
#   print(result.tail())
