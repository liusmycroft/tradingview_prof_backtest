import pandas as pd

from factors.base import BaseFactor


class HCVOLFactor(BaseFactor):
    """买入浮亏占比因子 (HCVOL)"""

    name = "HCVOL"
    category = "高频成交分布"
    description = "高于收盘价买入的成交量占比，衡量投资者买入情绪与惜售心理"

    def compute(
        self,
        daily_hcvol: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算买入浮亏占比因子。

        日内计算逻辑（预计算阶段）：
        HCVOL = sum(volume_buy_i * I(P_buy_i > close)) / total_volume
        即买入价高于收盘价的买单成交量占总成交量的比例。

        本方法对预计算的日度因子取 T 日滚动均值。

        Args:
            daily_hcvol: 预计算的每日买入浮亏占比，
                index=日期, columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        result = daily_hcvol.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 买入浮亏占比因子统计了高于收盘价买入的成交量占比，与未来收益正相关。
# 因子取值越高，投资者当天买入情绪较高，浮亏占比较低。
# 根据遗憾规避理论，投资者存在惜售心理，未来抛压较低，
# 有着更高的预期收益。
