import pandas as pd

from factors.base import BaseFactor


class SuperLargeBuyFactor(BaseFactor):
    """超大单买入占比因子 (Super Large Order Buy Ratio)"""

    name = "SUPER_LARGE_BUY"
    category = "高频资金流"
    description = "超大单买入占比：超大单买入金额占超大单总成交金额的比例，取T日滚动均值"

    def compute(
        self,
        super_big_buy: pd.DataFrame,
        super_big_sell: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算超大单买入占比因子。

        公式: (1/T) * sum( SuperBigBuy / (SuperBigBuy + SuperBigSell) )

        Args:
            super_big_buy: 每日超大单买入金额 (index=日期, columns=股票代码)
            super_big_sell: 每日超大单卖出金额 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        # 日度买入占比
        total = super_big_buy + super_big_sell
        daily_ratio = super_big_buy / total

        # T 日滚动均值
        result = daily_ratio.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 超大单买入占比衡量超大单（通常为机构资金）在买卖方向上的倾向性。
# 当该比例持续偏高时，说明机构资金以买入为主，可能预示短期股价上涨。
# 取 T 日滚动均值以平滑日间波动，捕捉中期资金流向趋势。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.super_large_buy import SuperLargeBuyFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   super_big_buy  = pd.DataFrame(100.0, index=dates, columns=stocks)
#   super_big_sell = pd.DataFrame(80.0,  index=dates, columns=stocks)
#
#   factor = SuperLargeBuyFactor()
#   result = factor.compute(super_big_buy=super_big_buy, super_big_sell=super_big_sell, T=20)
#   print(result.tail())
