import numpy as np
import pandas as pd

from factors.base import BaseFactor


class DailyReturnAttentionFactor(BaseFactor):
    """基于日收益率的注意力因子 (Daily Return Attention)"""

    name = "DAILY_RETURN_ATTENTION"
    category = "行为金融因子-注意力"
    description = "当前自然年因子值减去历史因子值，基于收益率偏差的注意力度量"

    def compute(
        self,
        stock_return: pd.DataFrame,
        market_return: pd.DataFrame,
        T: int = 250,
        **kwargs,
    ) -> pd.DataFrame:
        """计算基于日收益率的注意力因子。

        公式:
            原始值 = (1/T) * sum((r_i - R_m)^2)
            处理后因子 = 当前自然年原始值 - 历史原始值

        Args:
            stock_return: 个股日收益率 (index=日期, columns=股票代码)
            market_return: 市场基准收益率 (index=日期, columns=["market"]
                          或与 stock_return 同结构)
            T: 计算窗口（一个自然年交易日数），默认 250

        Returns:
            pd.DataFrame: 注意力因子值
        """
        if market_return.shape[1] == 1:
            mkt_col = market_return.columns[0]
            mkt = market_return[mkt_col]
        else:
            mkt = market_return.mean(axis=1)

        diff_sq = stock_return.sub(mkt, axis=0) ** 2
        raw_factor = diff_sq.rolling(window=T, min_periods=T).mean()

        # 当前自然年 vs 历史: 用滚动窗口近似
        # 历史均值用更长窗口
        hist_window = kwargs.get("hist_window", T * 2)
        hist_factor = raw_factor.rolling(window=hist_window, min_periods=T).mean()

        result = raw_factor - hist_factor
        return result
