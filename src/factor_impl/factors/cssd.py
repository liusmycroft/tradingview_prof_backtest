import numpy as np
import pandas as pd

from .base import BaseFactor


class CSSDFactor(BaseFactor):
    """CSSD 模型因子 (Cross-Sectional Standard Deviation - Herd Behavior)。"""

    name = "CSSD"
    category = "行为金融"
    description = "截面收益标准差因子，基于CSSD模型检测羊群效应并计算个股贡献度"

    def compute(
        self,
        stock_returns: pd.DataFrame,
        market_return: pd.Series,
        threshold_quantile: float = 0.05,
    ) -> dict:
        """计算 CSSD 模型。

        Args:
            stock_returns: 个股收益率，index=日期，columns=股票代码。
            market_return: 市场收益率，index=日期。
            threshold_quantile: 极端收益的分位数阈值，默认 0.05。

        Returns:
            dict: 包含以下键：
                - "cssd": pd.Series，每日 CSSD 值。
                - "beta1": float，上行极端日虚拟变量系数。
                - "beta2": float，下行极端日虚拟变量系数。
                - "contribution": pd.DataFrame，每只股票对 CSSD 的贡献度。
        """
        N = stock_returns.shape[1]

        # 计算每日 CSSD = sqrt(sum((r_i - r_m)^2) / (N-1))
        deviations = stock_returns.sub(market_return, axis=0)
        cssd = np.sqrt((deviations ** 2).sum(axis=1) / (N - 1))

        # 构建极端日虚拟变量
        upper_threshold = market_return.quantile(1 - threshold_quantile)
        lower_threshold = market_return.quantile(threshold_quantile)

        D_u = (market_return >= upper_threshold).astype(float)
        D_d = (market_return <= lower_threshold).astype(float)

        # OLS 回归: CSSD = alpha + beta1 * D_u + beta2 * D_d + epsilon
        X = pd.DataFrame({"const": 1.0, "D_u": D_u, "D_d": D_d}, index=cssd.index)
        # 使用最小二乘法
        X_arr = X.values
        y_arr = cssd.values

        # 过滤 NaN
        valid = ~np.isnan(y_arr)
        X_valid = X_arr[valid]
        y_valid = y_arr[valid]

        beta = np.linalg.lstsq(X_valid, y_valid, rcond=None)[0]
        alpha, beta1, beta2 = beta[0], beta[1], beta[2]

        # 个股对 CSSD 的贡献度: (r_i - r_m)^2 / (N-1) 的时序均值
        contribution = (deviations ** 2).div(N - 1).mean(axis=0)

        return {
            "cssd": cssd,
            "beta1": float(beta1),
            "beta2": float(beta2),
            "alpha": float(alpha),
            "contribution": contribution,
        }


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# CSSD（Cross-Sectional Standard Deviation）模型用于检测市场中的羊群效应。
# 核心逻辑：
#   1. 计算每日截面收益标准差 CSSD = sqrt(sum((r_i - r_m)^2) / (N-1))。
#   2. 构建极端市场日的虚拟变量 D_u（上行极端）和 D_d（下行极端）。
#   3. 回归 CSSD = alpha + beta1*D_u + beta2*D_d + epsilon。
#   4. 如果 beta1 或 beta2 显著为负，说明在极端行情中个股收益趋于一致，
#      即存在羊群效应。
#
# 个股层面：通过计算每只股票对 CSSD 的贡献度，可以识别哪些股票更容易
# 受到羊群效应的影响。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.cssd import CSSDFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=100)
#   stock_returns = pd.DataFrame(
#       np.random.randn(100, 10) * 0.02,
#       index=dates,
#       columns=[f"S{i}" for i in range(10)],
#   )
#   market_return = stock_returns.mean(axis=1)
#
#   factor = CSSDFactor()
#   result = factor.compute(
#       stock_returns=stock_returns,
#       market_return=market_return,
#       threshold_quantile=0.05,
#   )
#   print("beta1 (上行极端):", result["beta1"])
#   print("beta2 (下行极端):", result["beta2"])
#   print("个股贡献度:", result["contribution"])
