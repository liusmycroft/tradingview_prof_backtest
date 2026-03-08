import numpy as np
import pandas as pd

from factors.base import BaseFactor


class ConsistentBuyTradeFactor(BaseFactor):
    """一致买入交易因子 (Consistent Buy Trade - PCV)"""

    name = "CONSISTENT_BUY_TRADE"
    category = "高频成交分布"
    description = "一致买入交易因子，衡量上涨实体K线成交量占比，捕捉集体一致交易行为"

    def compute(
        self,
        daily_consistent_buy_ratio: pd.DataFrame,
        d: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算一致买入交易因子。

        日内预计算逻辑:
          1. 对每根5分钟K线判断是否为实体K线:
             |close - open| <= alpha * |high - low|
          2. 筛选上涨的实体K线(close > open)
          3. daily_consistent_buy_ratio = ConsistentVolume_rise / Volume

        因子值为 d 日移动平均。

        Args:
            daily_consistent_buy_ratio: 预计算的每日一致买入成交量占比
                (index=日期, columns=股票代码)
            d: 移动平均周期，默认 20

        Returns:
            pd.DataFrame: d 日滚动均值
        """
        result = daily_consistent_buy_ratio.rolling(window=d, min_periods=1).mean()
        return result

    @staticmethod
    def compute_daily(
        bar_open: pd.Series,
        bar_close: pd.Series,
        bar_high: pd.Series,
        bar_low: pd.Series,
        bar_volume: pd.Series,
        alpha: float = 0.5,
    ) -> float:
        """从单日5分钟K线数据计算一致买入成交量占比。

        Args:
            bar_open: 5分钟K线开盘价
            bar_close: 5分钟K线收盘价
            bar_high: 5分钟K线最高价
            bar_low: 5分钟K线最低价
            bar_volume: 5分钟K线成交量
            alpha: 一致参数，默认0.5

        Returns:
            float: ConsistentVolume_rise / Volume
        """
        body = (bar_close - bar_open).abs()
        shadow = (bar_high - bar_low).abs()

        # 实体K线: body > alpha * shadow (即上下引线短)
        is_solid = body > alpha * shadow

        # 上涨的实体K线
        is_rise = bar_close > bar_open
        consistent_rise = is_solid & is_rise

        total_volume = bar_volume.sum()
        if total_volume == 0:
            return np.nan

        consistent_volume = bar_volume[consistent_rise].sum()
        return consistent_volume / total_volume
