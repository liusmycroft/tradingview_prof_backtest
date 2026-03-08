import pandas as pd

from .base import BaseFactor


class AttentionDecayPanicFactor(BaseFactor):
    """注意力衰减-惊恐因子"""

    name = "ATTENTION_DECAY_PANIC"
    category = "高频动量反转"
    description = "注意力衰减-惊恐因子，对惊恐度做时间衰减后加权调整收益率的均值与波动等权合成"

    def compute(
        self,
        daily_decay_panic: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算注意力衰减-惊恐因子。

        公式:
            惊恐度 = abs(stock_ret - mkt_ret) / (abs(stock_ret) + abs(mkt_ret) + 0.1)
            衰减惊恐度_t = 惊恐度_t - (惊恐度_{t-1} + 惊恐度_{t-2}) / 2
            加权调整收益率 = max(衰减惊恐度, 0) * 日收益率
            惊恐收益 = mean(加权调整收益率, T日)
            惊恐波动 = std(加权调整收益率, T日)
            因子 = (惊恐收益 + 惊恐波动) / 2

        Args:
            daily_decay_panic: 预计算的每日加权调整收益率
                (衰减惊恐度 * 日收益率，负衰减惊恐度已置为 NaN)
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 注意力衰减-惊恐因子值
        """
        panic_ret = daily_decay_panic.rolling(window=T, min_periods=T).mean()
        panic_vol = daily_decay_panic.rolling(window=T, min_periods=T).std()
        result = (panic_ret + panic_vol) / 2
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 注意力衰减-惊恐因子对原始惊恐度做了时间衰减，考虑了投资者对短暂
# 连续显著收益的注意力会随时间减弱，过度反应也会逐渐缓解。
#
# 衰减惊恐度 = 当日惊恐度 - 前两日惊恐度均值，负值置为空值。
# 将衰减惊恐度与日收益率相乘得到加权调整收益率，再计算 T 日均值
# （惊恐收益）和标准差（惊恐波动），等权合成为最终因子。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.attention_decay_panic import AttentionDecayPanicFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_decay_panic = pd.DataFrame(
#       np.random.randn(30, 2) * 0.01,
#       index=dates, columns=stocks,
#   )
#
#   factor = AttentionDecayPanicFactor()
#   result = factor.compute(daily_decay_panic=daily_decay_panic, T=20)
#   print(result.tail())
