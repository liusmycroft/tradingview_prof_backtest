import numpy as np
import pandas as pd

from factors.base import BaseFactor


class CSADFactor(BaseFactor):
    """CSAD模型因子 (Cross-Sectional Absolute Deviation)"""

    name = "CSAD"
    category = "行为金融"
    description = "横截面绝对偏差模型的二次项系数，衡量市场羊群效应强度"

    def compute(
        self,
        stock_returns: pd.DataFrame,
        market_return: pd.Series,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 CSAD 因子。

        公式:
            CSAD_t = mean(|r_i,t - R_m,t|)
            回归: CSAD_t = alpha + beta1*|R_m,t| + beta2*R_m,t^2 + epsilon
            因子值为滚动窗口内的 beta2。

        Args:
            stock_returns: 个股日收益率 (index=日期, columns=股票代码)
            market_return: 市场日收益率 (index=日期)
            T: 滚动回归窗口天数，默认 20

        Returns:
            pd.DataFrame: 每日的 beta2 值 (index=日期, columns=["beta2"])
        """
        # 计算每日 CSAD
        csad = stock_returns.sub(market_return, axis=0).abs().mean(axis=1)

        abs_rm = market_return.abs()
        rm_sq = market_return ** 2

        dates = csad.index
        num_dates = len(dates)
        beta2_vals = np.full(num_dates, np.nan)

        for t in range(T - 1, num_dates):
            y = csad.iloc[t - T + 1 : t + 1].values
            x1 = abs_rm.iloc[t - T + 1 : t + 1].values
            x2 = rm_sq.iloc[t - T + 1 : t + 1].values

            # 检查 NaN
            mask = ~(np.isnan(y) | np.isnan(x1) | np.isnan(x2))
            if mask.sum() < 3:
                continue

            y_clean = y[mask]
            X = np.column_stack([
                np.ones(mask.sum()),
                x1[mask],
                x2[mask],
            ])

            # OLS: beta = (X'X)^{-1} X'y
            try:
                beta = np.linalg.lstsq(X, y_clean, rcond=None)[0]
                beta2_vals[t] = beta[2]
            except np.linalg.LinAlgError:
                continue

        result = pd.DataFrame({"beta2": beta2_vals}, index=dates)
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# CSAD (Cross-Sectional Absolute Deviation) 模型用于检测市场中的
# 羊群效应 (Herding Behavior)。
#
# 核心思想：在理性资产定价模型下，CSAD 与 |R_m| 应呈线性正相关。
# 但如果存在羊群效应，投资者会模仿市场整体行为，导致个股收益率
# 向市场收益率靠拢，CSAD 在市场大幅波动时反而下降。
#
# 通过回归 CSAD = alpha + beta1*|R_m| + beta2*R_m^2，
# 如果 beta2 显著为负，则表明存在羊群效应。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.csad import CSADFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ", "600000.SH"]
#
#   np.random.seed(42)
#   stock_returns = pd.DataFrame(
#       np.random.normal(0, 0.02, (30, 3)),
#       index=dates, columns=stocks,
#   )
#   market_return = stock_returns.mean(axis=1)
#
#   factor = CSADFactor()
#   result = factor.compute(
#       stock_returns=stock_returns,
#       market_return=market_return,
#       T=20,
#   )
#   print(result.tail())
