import numpy as np
import pandas as pd

from factors.base import BaseFactor


class RetailPanicFactor(BaseFactor):
    """个人投资者交易比-惊恐因子 (Retail Trader Ratio Panic Factor)"""

    name = "RETAIL_PANIC"
    category = "高频动量反转"
    description = "惊恐度×收益率×个人投资者交易比的均值与标准差等权合成"

    def compute(
        self,
        daily_return: pd.DataFrame,
        daily_market_return: pd.Series,
        daily_retail_ratio: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算个人投资者交易比-惊恐因子。

        公式:
            惊恐度 = |r_i - r_m| / (|r_i| + |r_m| + 0.1)
            加权调整收益率 = 惊恐度 * r_i * 个人投资者交易比
            惊恐收益 = rolling_mean(加权调整收益率, T)
            惊恐波动 = rolling_std(加权调整收益率, T)
            因子 = 0.5 * 惊恐收益 + 0.5 * 惊恐波动

        Args:
            daily_return: 个股日收益率 (index=日期, columns=股票代码)
            daily_market_return: 市场日收益率 (index=日期)
            daily_retail_ratio: 个人投资者交易比 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 惊恐因子值
        """
        # 惊恐度
        abs_stock = daily_return.abs()
        abs_market = daily_market_return.abs()
        deviation = daily_return.sub(daily_market_return, axis=0).abs()
        base = abs_stock.add(abs_market, axis=0) + 0.1
        panic_degree = deviation / base

        # 加权调整收益率
        weighted_ret = panic_degree * daily_return * daily_retail_ratio

        # 惊恐收益和惊恐波动
        panic_return = weighted_ret.rolling(window=T, min_periods=T).mean()
        panic_vol = weighted_ret.rolling(window=T, min_periods=T).std()

        # 等权合成
        result = 0.5 * panic_return + 0.5 * panic_vol
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 个人投资者交易比-惊恐因子在原始惊恐因子上新增了个人投资者交易比权重。
# 惊恐度衡量个股收益率对市场收益率的偏离水平，偏离越大惊恐度越高。
#
# 小市值股票的个人投资者交易占比相对更高，更易受情绪影响而出现反应过度。
# 因子综合了惊恐收益（均值）和惊恐波动（标准差），等权合成。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.retail_panic import RetailPanicFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#   np.random.seed(42)
#   daily_return = pd.DataFrame(np.random.randn(30, 2) * 0.02,
#       index=dates, columns=stocks)
#   daily_market_return = pd.Series(np.random.randn(30) * 0.01, index=dates)
#   daily_retail_ratio = pd.DataFrame(np.random.uniform(0.1, 0.5, (30, 2)),
#       index=dates, columns=stocks)
#
#   factor = RetailPanicFactor()
#   result = factor.compute(daily_return=daily_return,
#       daily_market_return=daily_market_return,
#       daily_retail_ratio=daily_retail_ratio, T=20)
#   print(result.tail())
