import pandas as pd

from factors.base import BaseFactor


class MTEFactor(BaseFactor):
    """主力交易情绪 (MTE) — Main Force Trading Emotion = RankCorr(order_amount, price)"""

    name = "MTE"
    category = "高频量价相关性"
    description = "主力委托金额与价格的秩相关系数，衡量主力资金的交易情绪方向"

    def compute(
        self,
        daily_mte: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算主力交易情绪因子。

        公式: MTE = RankCorr(order_amount, price)，日内分钟级秩相关
        因子值为 T 日 EMA。

        Args:
            daily_mte: 预计算的每日主力委托金额与价格的秩相关系数 (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 主力交易情绪的 T 日 EMA
        """
        result = daily_mte.ewm(span=T, min_periods=1).mean()
        return result
