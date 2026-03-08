import pandas as pd

from factors.base import BaseFactor


class PostOpenNetBuyFactor(BaseFactor):
    """开盘后净主买占比因子"""

    name = "POST_OPEN_NET_BUY"
    category = "高频资金流"
    description = "开盘后一段时间内净主动买入金额占总成交金额的比例，衡量开盘阶段资金方向"

    def compute(
        self,
        post_open_net_buy_amount: pd.DataFrame,
        post_open_total_amount: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算开盘后净主买占比因子。

        公式:
            daily_ratio = post_open_net_buy_amount / post_open_total_amount
            factor = rolling_mean(daily_ratio, T)

        Args:
            post_open_net_buy_amount: 每日开盘后时段净主动买入金额
                (主动买入 - 主动卖出), (index=日期, columns=股票代码)
            post_open_total_amount: 每日开盘后时段总成交金额
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 开盘后净主买占比的 T 日滚动均值
        """
        daily_ratio = post_open_net_buy_amount / post_open_total_amount
        result = daily_ratio.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 开盘后净主买占比衡量开盘后一段时间内（如前30分钟）主力资金的
# 净买入方向和强度。开盘阶段通常包含隔夜信息的集中释放，
# 净主买占比越高，说明主力资金在开盘阶段倾向于买入，
# 反映了对股票的看多情绪。
# 因子取 T 日滚动均值以平滑日间波动。
