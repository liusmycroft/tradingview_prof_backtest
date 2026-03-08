"""
相似低波因子 (Similar Low Volatility Factor)

通过寻找历史上与当前走势相似的时间窗口，计算这些窗口后续超额收益的波动率倒数。
"""

import numpy as np
import pandas as pd
from factors.base import BaseFactor


class SimilarLowVolFactor(BaseFactor):
    """相似低波因子

    公式: 1 / std(ER_tau)，其中ER_tau为历史相似窗口后续的超额收益。
    """

    name = "SIMILAR_LOW_VOL"
    category = "量价改进"
    description = "相似低波因子，基于历史相似走势后续超额收益波动率的倒数"

    def compute(
        self,
        close: pd.DataFrame,
        excess_returns: pd.DataFrame,
        lookback: int = 120,
        RW: int = 6,
        threshold: float = 0.4,
        holding_time: int = 5,
    ) -> pd.DataFrame:
        """计算相似低波因子。

        Args:
            close: 收盘价，index为日期，columns为股票代码。
            excess_returns: 超额收益率，index为日期，columns为股票代码。
            lookback: 历史回看窗口长度，默认120。
            RW: 相似度比较的窗口长度，默认6。
            threshold: 相关系数阈值，默认0.4。
            holding_time: 持有期天数，默认5。

        Returns:
            pd.DataFrame: 相似低波因子值。
        """
        # 计算收益率
        returns = close.pct_change()
        result = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        for col in close.columns:
            ret_series = returns[col].values
            er_series = excess_returns[col].values

            for i in range(lookback + RW, len(close)):
                # 当前RW日收益率模式
                current_pattern = ret_series[i - RW : i]
                if np.any(np.isnan(current_pattern)):
                    continue

                # 在历史窗口中寻找相似模式
                similar_ers = []
                start_idx = i - lookback
                for j in range(start_idx, i - RW - holding_time):
                    hist_pattern = ret_series[j : j + RW]
                    if np.any(np.isnan(hist_pattern)) or np.std(hist_pattern) == 0:
                        continue

                    # 计算相关系数
                    if np.std(current_pattern) == 0:
                        continue
                    corr = np.corrcoef(current_pattern, hist_pattern)[0, 1]

                    if corr > threshold:
                        # 收集该窗口后续holding_time天的超额收益
                        future_er = er_series[j + RW : j + RW + holding_time]
                        if len(future_er) == holding_time and not np.any(np.isnan(future_er)):
                            similar_ers.append(np.sum(future_er))

                # 因子值 = 1 / std(超额收益)
                if len(similar_ers) >= 2:
                    std_val = np.std(similar_ers, ddof=1)
                    if std_val > 0:
                        result.iloc[i, result.columns.get_loc(col)] = 1.0 / std_val

        return result


# 使用示例
if __name__ == "__main__":
    dates = pd.date_range("2024-01-01", periods=150)
    stocks = ["000001"]
    close = pd.DataFrame(
        np.cumsum(np.random.randn(150, 1) * 0.02, axis=0) + 10,
        index=dates,
        columns=stocks,
    )
    excess_returns = pd.DataFrame(
        np.random.randn(150, 1) * 0.01, index=dates, columns=stocks
    )

    factor = SimilarLowVolFactor()
    print(factor)
    print(factor.compute(close=close, excess_returns=excess_returns))
