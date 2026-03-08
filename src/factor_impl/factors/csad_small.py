import numpy as np
import pandas as pd

from factors.base import BaseFactor


class CSADSmallFactor(BaseFactor):
    """CSAD模型-小型板块因子 (CSAD Small Sector Model)"""

    name = "CSAD_SMALL"
    category = "行为金融"
    description = "CSAD小型板块因子：构建10只股票的微型板块，计算截面绝对偏差并标准化，取负T日均值"

    def compute(
        self,
        returns: pd.DataFrame,
        T_corr: int = 120,
        T_factor: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算CSAD小型板块因子。

        步骤:
        1. 对每只股票，基于过去 T_corr 日收益率相关性找到最相关的 10 只股票构成板块
        2. 计算板块内截面绝对偏差 CSAD
        3. 对 CSAD 做截面标准化
        4. 取负的 T_factor 日滚动均值

        Args:
            returns: 个股日收益率 (index=日期, columns=股票代码)
            T_corr: 计算相关性的回溯窗口，默认 120
            T_factor: 因子滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 因子值 (index=日期, columns=股票代码)
        """
        stocks = returns.columns.tolist()
        n_stocks = len(stocks)
        n_days = len(returns)

        result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        for t in range(T_corr + T_factor - 1, n_days):
            # 用 [t - T_corr - T_factor + 1, t - T_factor + 1) 的窗口计算相关性
            corr_start = t - T_corr - T_factor + 1
            corr_end = t - T_factor + 1
            if corr_start < 0:
                continue

            corr_window = returns.iloc[corr_start:corr_end]
            corr_matrix = corr_window.corr()

            # 对 [t - T_factor + 1, t + 1) 的窗口计算 CSAD
            factor_window = returns.iloc[t - T_factor + 1 : t + 1]

            csad_values = []
            for col in stocks:
                # 找到与 col 相关性最高的 10 只股票（不含自身）
                if col not in corr_matrix.columns:
                    csad_values.append(np.nan)
                    continue

                corr_row = corr_matrix[col].drop(col, errors="ignore")
                corr_row = corr_row.dropna()

                n_peers = min(10, len(corr_row))
                if n_peers < 2:
                    csad_values.append(np.nan)
                    continue

                top_peers = corr_row.nlargest(n_peers).index.tolist()
                sector_stocks = [col] + top_peers

                # 板块内 CSAD: 每日截面绝对偏差的均值
                sector_returns = factor_window[sector_stocks]
                sector_mean = sector_returns.mean(axis=1)
                csad_daily = sector_returns.sub(sector_mean, axis=0).abs().mean(axis=1)
                csad_values.append(csad_daily.mean())

            csad_series = pd.Series(csad_values, index=stocks)

            # 截面标准化
            cs_mean = csad_series.mean()
            cs_std = csad_series.std()
            if cs_std > 0:
                standardized = (csad_series - cs_mean) / cs_std
            else:
                standardized = csad_series * 0.0

            # 取负值
            result.iloc[t] = -standardized.values

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# CSAD (Cross-Sectional Absolute Deviation) 衡量板块内个股收益率的分散程度。
# 对每只股票构建一个由相关性最高的10只股票组成的"微型板块"，计算板块内
# 的 CSAD。低 CSAD 意味着板块内股票走势趋同（羊群效应），取负值后
# 高因子值对应强羊群效应，可能预示未来反转。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.csad_small import CSADSmallFactor
#
#   dates = pd.date_range("2024-01-01", periods=160, freq="B")
#   stocks = ["S" + str(i) for i in range(20)]
#
#   np.random.seed(42)
#   returns = pd.DataFrame(np.random.randn(160, 20) * 0.02, index=dates, columns=stocks)
#
#   factor = CSADSmallFactor()
#   result = factor.compute(returns=returns, T_corr=120, T_factor=20)
#   print(result.tail())
