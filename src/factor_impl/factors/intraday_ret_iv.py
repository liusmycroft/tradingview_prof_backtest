"""特质日内收益波动率因子 (Intraday Return Idiosyncratic Volatility)

通过PCA提取日内收益率的共同成分，计算残差的标准差。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class IntradayRetIVFactor(BaseFactor):
    """特质日内收益波动率因子"""

    name = "INTRADAY_RET_IV"
    category = "高频波动跳跃"
    description = "特质日内收益波动率：PCA剥离日内收益率共同成分后残差的标准差"

    def compute(
        self,
        intraday_ret: pd.DataFrame,
        T: int = 20,
        n_components: int = 1,
        var_threshold: float = 0.2,
        **kwargs,
    ) -> pd.DataFrame:
        """计算特质日内收益波动率因子。

        步骤:
        1. 截面上对所有股票过去T天的日内收益率做PCA，提取第一主成分F1
        2. 对每只股票做时序回归: IntradayRet = alpha + beta*F1 + eps
        3. std(eps) 即为特质日内收益波动率

        Args:
            intraday_ret: 日内收益率 Close/Open - 1 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20
            n_components: PCA主成分数量，默认 1
            var_threshold: 第一主成分解释力度阈值，低于此值取前2个，默认 0.2

        Returns:
            pd.DataFrame: 特质日内收益波动率
        """
        dates = intraday_ret.index
        stocks = intraday_ret.columns
        n_dates = len(dates)

        result = pd.DataFrame(np.nan, index=dates, columns=stocks)

        for t in range(T - 1, n_dates):
            window = intraday_ret.iloc[t - T + 1: t + 1]
            valid_cols = window.columns[window.notna().all()]

            if len(valid_cols) < 3:
                continue

            mat = window[valid_cols].values  # (T, n_valid)
            mean_vec = mat.mean(axis=0)
            centered = mat - mean_vec

            try:
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                continue

            # 判断第一主成分解释力度
            total_var = np.sum(S ** 2)
            if total_var == 0:
                continue

            explained_ratio_1 = S[0] ** 2 / total_var
            k = n_components
            if explained_ratio_1 < var_threshold and len(S) >= 2:
                k = 2

            k = min(k, len(S))

            # 提取主成分得分 F: (T, k)
            F = U[:, :k] * S[:k]

            # 对每只有效股票做回归
            X = np.column_stack([np.ones(T), F])  # (T, 1+k)

            for j, col in enumerate(valid_cols):
                y = centered[:, j]
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    residuals = y - X @ beta
                    result.loc[dates[t], col] = np.std(residuals, ddof=1)
                except np.linalg.LinAlgError:
                    continue

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 通过PCA对资产日内收益率序列降维提取共同波动，无需显式借助多因子模型。
# 系统性地解决传统多因子模型遗漏变量、估计误差的问题。
# 特质波动率越高，说明个股日内收益中不可被市场共同因素解释的部分越大。
