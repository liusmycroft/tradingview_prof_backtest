import pandas as pd

from factors.base import BaseFactor


class NetCommissionBuyFactor(BaseFactor):
    """净委买变化率因子 (Net Commission Buy Change Rate)"""

    name = "NET_COMMISSION_BUY"
    category = "高频资金流"
    description = "净委买变化率：委买变化与委卖变化之差占流通股本的滚动均值"

    def compute(
        self,
        net_commission_change: pd.DataFrame,
        float_shares: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算净委买变化率因子。

        公式: daily_rate = net_commission_change / float_shares
              factor = rolling T-day mean of daily_rate

        Args:
            net_commission_change: 每日净委买变化量 (sum(bid_change) - sum(ask_change))
                                   (index=日期, columns=股票代码)
            float_shares: 流通股本 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 净委买变化率因子值
        """
        # 日度净委买变化率
        daily_rate = net_commission_change / float_shares

        # T 日滚动均值
        result = daily_rate.rolling(window=T, min_periods=1).mean()

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 净委买变化率衡量委托买入与委托卖出变化量之差相对于流通股本的比率。
# net_commission_change = sum(bid_change) - sum(ask_change)
# daily_rate = net_commission_change / float_shares
#
# 该因子反映了市场挂单意愿的变化方向：
#   - 正值表示买方挂单增加快于卖方，买方力量增强。
#   - 负值表示卖方挂单增加快于买方，卖方压力增大。
# 取 T 日滚动均值以平滑日间波动，捕捉中期趋势。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.net_commission_buy import NetCommissionBuyFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   net_change   = pd.DataFrame(1000.0, index=dates, columns=stocks)
#   float_shares = pd.DataFrame(1e8,    index=dates, columns=stocks)
#
#   factor = NetCommissionBuyFactor()
#   result = factor.compute(
#       net_commission_change=net_change, float_shares=float_shares, T=20
#   )
#   print(result.tail())
