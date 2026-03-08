import pandas as pd

from factors.base import BaseFactor


class OpeningCommissionRatioFactor(BaseFactor):
    """开盘后净委买增额占比因子 (Opening Net Commission Buy Ratio)"""

    name = "OPENING_COMMISSION_RATIO"
    category = "高频资金流"
    description = "开盘后30分钟净委买增额占成交额比例的滚动均值，衡量早盘资金流入强度"

    def compute(
        self,
        net_commission_increase: pd.DataFrame,
        amount: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算开盘后净委买增额占比因子。

        公式: factor = (1/T) * sum(net_commission_increase / amount) for 9:30-10:00

        Args:
            net_commission_increase: 开盘后30分钟净委买增额 (index=日期, columns=股票代码)
            amount: 开盘后30分钟成交额 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 净委买增额占比的 T 日滚动均值
        """
        # 避免除零
        safe_amount = amount.replace(0, float("nan"))
        daily_ratio = net_commission_increase / safe_amount
        result = daily_ratio.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 开盘后净委买增额占比因子关注每日 9:30-10:00 这一时间段内，
# 净委买增额（买方委托增量 - 卖方委托增量）占成交额的比例。
#
# 开盘后30分钟是市场信息消化最密集的时段，机构投资者往往在此时段
# 集中下单。净委买增额占比高说明买方力量强，可能预示短期上涨。
# 因子取 T 日滚动均值以平滑日间波动。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.opening_commission_ratio import OpeningCommissionRatioFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   net_comm = pd.DataFrame(
#       np.random.randn(30, 2) * 1e6,
#       index=dates, columns=stocks,
#   )
#   amount = pd.DataFrame(
#       np.random.rand(30, 2) * 1e7 + 1e6,
#       index=dates, columns=stocks,
#   )
#
#   factor = OpeningCommissionRatioFactor()
#   result = factor.compute(
#       net_commission_increase=net_comm, amount=amount, T=20,
#   )
#   print(result.tail())
