"""开盘后大单净买入强度因子 (Post-Open Large Order Net Buy Intensity)

衡量开盘后一段时间内大单净买入金额占总成交金额的比例。
"""

import pandas as pd

from factors.base import BaseFactor


class PostOpenLargeBuyFactor(BaseFactor):
    """开盘后大单净买入强度因子"""

    name = "POST_OPEN_LARGE_BUY"
    category = "高频资金流"
    description = "开盘后大单净买入强度：开盘后大单净买入金额占总成交金额的比例，取T日滚动均值"

    def compute(
        self,
        daily_large_net_buy: pd.DataFrame,
        daily_total_amount: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算开盘后大单净买入强度因子。

        公式: factor = (1/T) * sum(large_net_buy / total_amount)

        Args:
            daily_large_net_buy: 每日开盘后大单净买入金额
                (index=日期, columns=股票代码)
            daily_total_amount: 每日开盘后总成交金额
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值，T日滚动均值
        """
        # 日度净买入强度
        daily_intensity = daily_large_net_buy / daily_total_amount

        # T 日滚动均值
        result = daily_intensity.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 开盘后大单净买入强度衡量开盘后一段时间内（通常为前30分钟）大单资金
# 的净流入方向和强度。开盘阶段是信息交易最活跃的时段，大单通常代表
# 机构或知情交易者的操作。净买入强度越高，说明知情资金看多意愿越强。
