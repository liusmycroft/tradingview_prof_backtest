import numpy as np
import pandas as pd

from factors.base import BaseFactor


class SentimentSpilloverFactor(BaseFactor):
    """情绪溢出因子 (Sentiment Spillover via Adjacent Stock Turnover, RNBR_tov)"""

    name = "RNBR_TOV"
    category = "高频成交分布"
    description = "相邻股票换手率对自身换手率回归的残差，衡量情绪溢出效应"

    def compute(
        self,
        turnover: pd.DataFrame,
        nbr_turnover: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算情绪溢出因子。

        对每个截面日期做截面回归:
          turnover_i = alpha + beta * NBR_tov_i + eps_i
        残差 eps_i 即为情绪溢出因子 RNBR_tov。

        Args:
            turnover: 个股过去T日日均换手率 (index=日期, columns=股票代码)
            nbr_turnover: 相邻n只股票过去T日日均换手率的等权均值
                          (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 截面回归残差（情绪溢出因子）
        """
        dates = turnover.index
        stocks = turnover.columns
        result = np.full((len(dates), len(stocks)), np.nan)

        for t in range(len(dates)):
            y = turnover.values[t, :]
            x = nbr_turnover.values[t, :]

            mask = ~(np.isnan(y) | np.isnan(x))
            if mask.sum() < 3:
                continue

            y_clean = y[mask]
            x_clean = x[mask]

            X = np.column_stack([np.ones(mask.sum()), x_clean])
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y_clean, rcond=None)
                fitted = X @ beta
                residuals = y_clean - fitted
                result[t, mask] = residuals
            except np.linalg.LinAlgError:
                continue

        return pd.DataFrame(result, index=dates, columns=stocks)


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 由于投资者注意力有限，行情软件通常按代码顺序展示股票信息，投资者会
# 较轻易地关注到与焦点股票相邻的其它股票，出现"注意力溢出"。焦点股票
# 与相邻股票之间会存在价格与交易情绪的转移。"情绪溢出"对应股票近期
# 引起大量关注时的强烈交易情绪溢出到相邻股票的情况，与未来收益正相关。
