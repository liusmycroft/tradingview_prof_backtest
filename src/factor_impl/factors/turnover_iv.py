"""特质换手波动率因子 (Idiosyncratic Turnover Volatility, TurnoverIV)

通过PCA提取换手率的共同成分，计算残差（特质部分）的波动率。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class TurnoverIVFactor(BaseFactor):
    """特质换手波动率因子 (TurnoverIV)"""

    name = "TURNOVER_IV"
    category = "高频波动跳跃"
    description = "特质换手波动率：通过PCA剥离换手率共同成分后，残差的滚动标准差"

    def compute(
        self,
        turnover: pd.DataFrame,
        T: int = 20,
        n_components: int = 3,
        **kwargs,
    ) -> pd.DataFrame:
        """计算特质换手波动率因子。

        步骤:
        1. 对换手率做滚动窗口PCA，提取前n_components个主成分
        2. 计算残差（特质换手率）
        3. 对残差取滚动标准差

        Args:
            turnover: 换手率 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20
            n_components: PCA主成分数量，默认 3

        Returns:
            pd.DataFrame: 因子值，特质换手率的滚动标准差
        """
        dates = turnover.index
        stocks = turnover.columns
        n_dates, n_stocks = turnover.shape

        residual = pd.DataFrame(np.nan, index=dates, columns=stocks)

        for t in range(T - 1, n_dates):
            window = turnover.iloc[t - T + 1 : t + 1]
            valid_cols = window.columns[window.notna().all()]

            if len(valid_cols) < max(n_components + 1, 3):
                continue

            mat = window[valid_cols].values  # (T, n_valid)
            # 去均值
            mean_vec = mat.mean(axis=0)
            centered = mat - mean_vec

            # SVD 提取主成分
            try:
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                continue

            k = min(n_components, len(S))
            # 投影到前k个主成分并重建
            reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
            resid = centered - reconstructed

            # 取最后一行（当日）的残差
            residual.loc[dates[t], valid_cols] = resid[-1, :]

        # 对残差取滚动标准差
        result = residual.rolling(window=T, min_periods=T).std(ddof=1)
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 特质换手波动率通过PCA剥离换手率中的市场共同成分（如市场情绪、行业
# 轮动等系统性因素），提取个股特有的换手率波动。特质换手波动率越高，
# 说明个股的交易活跃度波动越大，可能反映了信息不对称或投机行为。
