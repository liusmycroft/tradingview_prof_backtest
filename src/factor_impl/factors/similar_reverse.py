import numpy as np
import pandas as pd

from .base import BaseFactor


class SimilarReverseFactor(BaseFactor):
    """相似反转因子 (Similar Reverse)。"""

    name = "SIMILAR_REVERSE"
    category = "量价改进"
    description = "相似反转因子，基于历史相似走势后的累计超额收益加权均值取反"

    def compute(
        self,
        close: pd.DataFrame,
        excess_return: pd.DataFrame,
        lookback: int = 120,
        rw: int = 6,
        holding_time: int = 6,
        threshold: float = 0.4,
        half_life: int = 60,
    ) -> pd.DataFrame:
        """计算相似反转因子。

        Args:
            close: 收盘价，index=日期，columns=股票代码。
            excess_return: 每日超额收益率，index=日期，columns=股票代码。
            lookback: 历史回看窗口长度，默认 120。
            rw: 相似走势匹配的序列长度，默认 6。
            holding_time: 持仓时间，默认 6。
            threshold: 相关系数阈值，默认 0.4。
            half_life: 指数衰减半衰期，默认 60。

        Returns:
            pd.DataFrame: 因子值，index=日期，columns=股票代码。
        """
        dates = close.index
        stocks = close.columns
        n_dates = len(dates)
        n_stocks = len(stocks)
        result = np.full((n_dates, n_stocks), np.nan)

        lam = np.log(2) / half_life

        for s in range(n_stocks):
            price = close.iloc[:, s].values
            er = excess_return.iloc[:, s].values

            for t in range(lookback + rw - 1, n_dates):
                current_seq = price[t - rw + 1 : t + 1]
                if np.any(np.isnan(current_seq)):
                    continue

                similar_ers = []
                decay_indices = []

                start = t - lookback
                end = t - rw - holding_time + 1

                if start < 0 or end <= start:
                    continue

                for tau in range(start, end):
                    hist_seq = price[tau : tau + rw]
                    if np.any(np.isnan(hist_seq)):
                        continue

                    if np.std(current_seq) == 0 or np.std(hist_seq) == 0:
                        continue

                    corr = np.corrcoef(current_seq, hist_seq)[0, 1]
                    if np.isnan(corr):
                        continue

                    if abs(corr) >= threshold:
                        # 计算匹配后持仓期的累计超额收益
                        er_start = tau + rw
                        er_end = tau + rw + holding_time
                        if er_end > t + 1:
                            continue
                        er_slice = er[er_start:er_end]
                        if np.any(np.isnan(er_slice)):
                            continue
                        cum_er = np.prod(1 + er_slice) - 1
                        similar_ers.append(cum_er)
                        decay_indices.append(t - tau)

                if len(similar_ers) > 0:
                    similar_ers = np.array(similar_ers)
                    decay_indices = np.array(decay_indices)
                    weights = np.exp(-lam * decay_indices)
                    weights = weights / weights.sum()
                    result[t, s] = -np.sum(weights * similar_ers)

        return pd.DataFrame(result, index=dates, columns=stocks)


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 相似反转因子通过在历史收盘价序列中寻找与当前走势相似的片段，
# 计算这些相似走势之后的累计超额收益的指数衰减加权均值，并取反。
# 核心逻辑：
#   1. 在过去 lookback 天内，用长度为 rw 的滑动窗口匹配与当前走势
#      皮尔逊相关系数超过 threshold 的历史片段。
#   2. 对每个匹配片段，计算其后 holding_time 天的累计超额收益。
#   3. 用指数衰减权重加权平均这些超额收益，取反得到因子值。
#
# 经济直觉：历史中出现相似走势后股票一段时间内的平均超额收益率越高，
# 未来持有同样时间的超额收益率就越低（反转效应）。
