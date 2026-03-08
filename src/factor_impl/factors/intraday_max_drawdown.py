import pandas as pd

from factors.base import BaseFactor


class IntradayMaxDrawdownFactor(BaseFactor):
    """日内最大回撤 — Intraday Maximum Drawdown"""

    name = "INTRADAY_MAX_DRAWDOWN"
    category = "高频动量反转"
    description = "日内价格从峰值到谷值的最大回撤幅度，衡量日内极端下行风险"

    def compute(
        self,
        daily_max_drawdown: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算日内最大回撤因子。

        公式: MDD_t = max_{s<u}((P_s - P_u) / P_s)，日内分钟级价格序列
        因子值为 T 日均值。

        Args:
            daily_max_drawdown: 预计算的每日日内最大回撤 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 日内最大回撤的 T 日均值
        """
        result = daily_max_drawdown.rolling(window=T, min_periods=1).mean()
        return result
