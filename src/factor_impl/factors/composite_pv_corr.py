import numpy as np
import pandas as pd

from factors.base import BaseFactor


class CompositePVCorrFactor(BaseFactor):
    """价量相关性因子 (Composite Price-Volume Correlation - cpv)"""

    name = "CPV"
    category = "高频量价相关性"
    description = "综合价量相关性均值、波动、趋势，剔除反转因子后的复合因子"

    def compute(
        self,
        pv_corr_avg: pd.DataFrame,
        pv_corr_std: pd.DataFrame,
        pv_corr_trend: pd.DataFrame,
        ret_20d: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算价量相关性因子。

        公式:
            1. 对 pv_corr_avg 和 pv_corr_std 分别截面回归剔除 ret_20d，取残差
            2. 残差各自截面标准化后等权相加 -> PV_corr_deRet20
            3. PV_corr_deRet20 截面标准化 + pv_corr_trend 截面标准化 -> cpv

        Args:
            pv_corr_avg: 价量相关性均值 (index=日期, columns=股票代码)
            pv_corr_std: 价量相关性波动 (index=日期, columns=股票代码)
            pv_corr_trend: 价量相关性趋势 (index=日期, columns=股票代码)
            ret_20d: 过去20日收益率 (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: cpv因子值
        """
        dates = pv_corr_avg.index
        stocks = pv_corr_avg.columns

        def _cross_sectional_neutralize(y_df: pd.DataFrame, x_df: pd.DataFrame) -> pd.DataFrame:
            """截面回归剔除x的影响，返回残差。"""
            residuals = pd.DataFrame(np.nan, index=dates, columns=stocks)
            for i, date in enumerate(dates):
                y = y_df.iloc[i].values.astype(float)
                x = x_df.iloc[i].values.astype(float)
                valid = ~(np.isnan(y) | np.isnan(x))
                if valid.sum() < 3:
                    continue
                X = np.column_stack([np.ones(valid.sum()), x[valid]])
                try:
                    beta = np.linalg.lstsq(X, y[valid], rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue
                fitted = np.full(len(y), np.nan)
                fitted[valid] = X @ beta
                res = y - fitted
                residuals.iloc[i] = res
            return residuals

        def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
            """截面标准化。"""
            mean = df.mean(axis=1)
            std = df.std(axis=1)
            std = std.replace(0, np.nan)
            return df.sub(mean, axis=0).div(std, axis=0)

        # 剔除反转因子
        avg_resid = _cross_sectional_neutralize(pv_corr_avg, ret_20d)
        std_resid = _cross_sectional_neutralize(pv_corr_std, ret_20d)

        # 截面标准化后等权相加
        pv_corr_deRet20 = _cross_sectional_zscore(avg_resid) + _cross_sectional_zscore(std_resid)

        # 最终合成
        cpv = _cross_sectional_zscore(pv_corr_deRet20) + _cross_sectional_zscore(pv_corr_trend)

        return cpv


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 价量相关性因子综合了价量相关性平均强度、波动性、趋势性三方面信息，
# 对传统反转因子进行了修正。先将均值和波动剔除过去20日收益率的影响，
# 再与趋势信息合成，具备良好的选股能力。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.composite_pv_corr import CompositePVCorrFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ", "600000.SH"]
#   np.random.seed(42)
#   pv_corr_avg = pd.DataFrame(np.random.randn(30, 3) * 0.3,
#       index=dates, columns=stocks)
#   pv_corr_std = pd.DataFrame(np.random.uniform(0, 0.5, (30, 3)),
#       index=dates, columns=stocks)
#   pv_corr_trend = pd.DataFrame(np.random.randn(30, 3) * 0.1,
#       index=dates, columns=stocks)
#   ret_20d = pd.DataFrame(np.random.randn(30, 3) * 0.05,
#       index=dates, columns=stocks)
#
#   factor = CompositePVCorrFactor()
#   result = factor.compute(pv_corr_avg=pv_corr_avg, pv_corr_std=pv_corr_std,
#       pv_corr_trend=pv_corr_trend, ret_20d=ret_20d)
#   print(result.tail())
