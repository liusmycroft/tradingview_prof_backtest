"""特质隔夜收益波动率因子 (Idiosyncratic Overnight Return Volatility via PCA)

截面上对所有股票过去T天的隔夜收益率序列进行PCA，提取主成分作为隐式因子，
对每只股票做时序回归，残差序列的标准差即为特质波动率。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class OvernightRetIVFactor(BaseFactor):
    name = "OvernightRetIV"
    category = "高频波动跳跃"
    description = "特质隔夜收益波动率：基于PCA提取隐式因子后的残差标准差"

    def compute(
        self,
        overnight_ret: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算特质隔夜收益波动率因子。

        步骤:
        1. 滚动 T 日窗口，对截面隔夜收益率做 PCA 提取第一主成分
        2. 对每只股票做时序回归: OvernightRet = alpha + beta * F1 + epsilon
        3. 残差标准差即为特质波动率

        Args:
            overnight_ret: 隔夜收益率 = Open_t / Close_{t-1} - 1
                          (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 特质隔夜收益波动率因子值
        """
        result = pd.DataFrame(np.nan, index=overnight_ret.index, columns=overnight_ret.columns)

        for i in range(T - 1, len(overnight_ret)):
            window = overnight_ret.iloc[i - T + 1: i + 1]  # (T, n_stocks)

            # 去除全 NaN 列
            valid_cols = window.columns[window.notna().all()]
            if len(valid_cols) < 3:
                continue

            mat = window[valid_cols].values.astype(np.float64)  # (T, n_valid)

            # 去均值
            mat_centered = mat - mat.mean(axis=0, keepdims=True)

            # SVD 提取主成分
            try:
                U, S, Vt = np.linalg.svd(mat_centered, full_matrices=False)
            except np.linalg.LinAlgError:
                continue

            total_var = (S ** 2).sum()
            if total_var == 0:
                continue

            # 判断是否需要两个主成分
            explained_ratio_1 = S[0] ** 2 / total_var
            if explained_ratio_1 >= 0.2:
                n_components = 1
            else:
                n_components = min(2, len(S))

            # 提取因子得分 F: (T, n_components)
            F = U[:, :n_components] * S[:n_components]

            # 对每只有效股票做时序回归
            X = np.column_stack([np.ones(T), F])  # (T, 1+n_components)
            for col in valid_cols:
                y = mat_centered[:, list(valid_cols).index(col)]
                try:
                    beta, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                fitted = X @ beta
                epsilon = y - fitted
                result.loc[result.index[i], col] = np.std(epsilon, ddof=1)

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 计算隐式因子框架下的特质波动率时，直接通过主成分分析对资产收益序列
# 降维来提取共同波动，而无需显式借助多因子模型估计共同风险，系统性地
# 解决传统多因子模型遗漏变量、估计误差的问题。此处提取的是隔夜收益率
# 序列的共同波动，残差标准差即为特质波动率。特质波动率越高，说明该
# 股票的隔夜收益中不可被市场共同因子解释的部分越大。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.overnight_ret_iv import OvernightRetIVFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
#
#   np.random.seed(42)
#   overnight_ret = pd.DataFrame(
#       np.random.randn(30, 4) * 0.01,
#       index=dates, columns=stocks,
#   )
#
#   factor = OvernightRetIVFactor()
#   result = factor.compute(overnight_ret=overnight_ret, T=20)
#   print(result.tail())
