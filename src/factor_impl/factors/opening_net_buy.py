import pandas as pd

from factors.base import BaseFactor


class OpeningNetBuyFactor(BaseFactor):
    """开盘后净主买强度因子 (Opening Net Main Buy Strength)"""

    name = "OPENING_NET_BUY"
    category = "高频资金流"
    description = "开盘后净主买强度：开盘30分钟内主动买入与卖出之差的信噪比滚动均值"

    def compute(
        self,
        net_buy_mean: pd.DataFrame,
        net_buy_std: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算开盘后净主买强度因子。

        公式: (1/T) * sum(mean(net_buy) / std(net_buy)) over T days

        Args:
            net_buy_mean: 每日开盘30分钟内分钟级净主买金额的均值
                          (index=日期, columns=股票代码)
            net_buy_std: 每日开盘30分钟内分钟级净主买金额的标准差
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值的信噪比
        """
        # 日度信噪比: mean / std
        daily_snr = net_buy_mean / net_buy_std

        # T 日滚动均值
        result = daily_snr.rolling(window=T, min_periods=1).mean()

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 开盘后净主买强度衡量开盘30分钟内主动买入与主动卖出之差（净主买金额）的
# 信噪比。net_buy = active_buy - active_sell，对每日开盘30分钟内的分钟级
# 净主买金额取均值和标准差，均值/标准差即为当日信噪比，再取T日滚动均值
# 以平滑短期波动。
#
# 信噪比越高，说明开盘阶段主力资金净买入越稳定、方向性越强，
# 通常预示着短期内股价上涨的概率较大。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.opening_net_buy import OpeningNetBuyFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   net_buy_mean = pd.DataFrame(100.0, index=dates, columns=stocks)
#   net_buy_std  = pd.DataFrame(50.0,  index=dates, columns=stocks)
#
#   factor = OpeningNetBuyFactor()
#   result = factor.compute(net_buy_mean=net_buy_mean, net_buy_std=net_buy_std, T=20)
#   print(result.tail())
