import pandas as pd

from factors.base import BaseFactor


class OpeningBuyIntentionFactor(BaseFactor):
    """开盘后买入意愿占比因子 (Opening Buy Intention Ratio)"""

    name = "OPENING_BUY_INTENTION"
    category = "高频资金流"
    description = "开盘后买入意愿占比：买入意愿金额占总成交金额的比例，取T日滚动均值"

    def compute(
        self,
        buy_intention: pd.DataFrame,
        amount: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算开盘后买入意愿占比因子。

        公式: (1/T) * sum(buy_intention / amount)
              其中 buy_intention = net_main_buy - net_commission_change

        Args:
            buy_intention: 每日买入意愿金额 (index=日期, columns=股票代码)
            amount: 每日总成交金额 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        # 日度买入意愿占比
        daily_ratio = buy_intention / amount

        # T 日滚动均值
        result = daily_ratio.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 开盘后买入意愿占比衡量开盘阶段主力资金的真实买入意愿强度。
# buy_intention = net_main_buy - net_commission_change，即净主买金额
# 减去委托变化量，剔除了挂单但未成交的虚假买入信号。
# 该比例越高，说明主力资金的真实买入意愿越强。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.opening_buy_intention import OpeningBuyIntentionFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   buy_intention = pd.DataFrame(50.0, index=dates, columns=stocks)
#   amount        = pd.DataFrame(500.0, index=dates, columns=stocks)
#
#   factor = OpeningBuyIntentionFactor()
#   result = factor.compute(buy_intention=buy_intention, amount=amount, T=20)
#   print(result.tail())
