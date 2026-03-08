import pandas as pd

from factors.base import BaseFactor


class PTypeVolumeFactor(BaseFactor):
    """P型成交量分布因子 (P-Type Volume Distribution)"""

    name = "P_TYPE_VOLUME"
    category = "高频成交分布"
    description = "P型成交量分布：VSA低点在日内价格区间中的相对位置，取T日滚动均值"

    def compute(
        self,
        vsa_low: pd.DataFrame,
        daily_high: pd.DataFrame,
        daily_low: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算P型成交量分布因子。

        公式: (1/T) * sum( (VSA_Low - Daily_Low) / (Daily_High - Daily_Low) )

        Args:
            vsa_low: VSA分析中的低点价格 (index=日期, columns=股票代码)
            daily_high: 每日最高价 (index=日期, columns=股票代码)
            daily_low: 每日最低价 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        price_range = daily_high - daily_low

        # 归一化位置
        daily_position = (vsa_low - daily_low) / price_range

        # T 日滚动均值
        result = daily_position.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# P型成交量分布因子衡量成交量集中区域（VSA低点）在日内价格区间中的
# 相对位置。(VSA_Low - Daily_Low) / (Daily_High - Daily_Low) 将
# VSA低点归一化到 [0, 1] 区间。值接近 0 表示成交集中在低价区域，
# 值接近 1 表示成交集中在高价区域。T日均值平滑短期波动。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.p_type_volume import PTypeVolumeFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   daily_high = pd.DataFrame(11.0, index=dates, columns=stocks)
#   daily_low  = pd.DataFrame(9.0,  index=dates, columns=stocks)
#   vsa_low    = pd.DataFrame(9.5,  index=dates, columns=stocks)
#
#   factor = PTypeVolumeFactor()
#   result = factor.compute(vsa_low=vsa_low, daily_high=daily_high, daily_low=daily_low, T=20)
#   print(result.tail())
