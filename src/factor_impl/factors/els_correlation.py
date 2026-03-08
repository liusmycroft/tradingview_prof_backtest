import pandas as pd
from scipy.stats import spearmanr

from .base import BaseFactor


class ELSCorrelationFactor(BaseFactor):
    """超大单与小单同步相关性因子 (Extra-Large and Small Order Synchronous Correlation)。"""

    name = "ELSCorrelation"
    category = "资金流"
    description = "超大单与小单净流入的Spearman秩相关系数，衡量大小资金的同步性"

    def compute(
        self,
        el_net_inflow: pd.DataFrame,
        s_net_inflow: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算超大单-小单同步相关性因子。

        Args:
            el_net_inflow: 超大单(>100万)每日净流入，index=日期，columns=股票代码。
            s_net_inflow: 小单(<4万)每日净流入，index=日期，columns=股票代码。
            T: 回看窗口天数，默认 20。

        Returns:
            pd.DataFrame: Spearman 秩相关系数，index=日期，columns=股票代码。
        """
        result = pd.DataFrame(index=el_net_inflow.index, columns=el_net_inflow.columns, dtype=float)

        for col in el_net_inflow.columns:
            el_series = el_net_inflow[col]
            s_series = s_net_inflow[col]

            for i in range(len(el_series)):
                if i < T - 1:
                    continue

                el_window = el_series.iloc[i - T + 1 : i + 1]
                s_window = s_series.iloc[i - T + 1 : i + 1]

                # 窗口内存在 NaN 或方差为零则无法计算相关性
                if el_window.isna().any() or s_window.isna().any():
                    continue

                if el_window.nunique() == 1 or s_window.nunique() == 1:
                    continue

                corr, _ = spearmanr(el_window.values, s_window.values)
                result.loc[result.index[i], col] = corr

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# ELSCorrelation（超大单-小单同步相关性）衡量的是超大单资金（单笔成交金额
# 大于 100 万元）与小单资金（单笔成交金额小于 4 万元）在过去 T 个交易日内
# 净流入方向的一致程度。具体做法是：
#   1. 取过去 T 天的超大单每日净流入序列 EL_t 和小单每日净流入序列 S_t；
#   2. 计算两者的 Spearman 秩相关系数 RankCorr(EL_t, S_t)。
#
# 【经济学直觉】
# 正常市场中，机构资金（超大单）与散户资金（小单）的交易方向往往存在分歧：
# 机构买入时散户倾向卖出，反之亦然，因此秩相关系数通常为负或接近零。
# 当该因子显著为正时，说明大小资金罕见地同向操作，可能暗示：
#   - 市场处于极端情绪（恐慌性抛售或一致性追涨）；
#   - 存在信息驱动的单边行情。
# 当该因子显著为负时，说明机构与散户方向高度对立，通常对应筹码交换活跃期。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.els_correlation import ELSCorrelationFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=25)
#   el_net_inflow = pd.DataFrame(
#       {
#           "000001.SZ": [100, -50, 200, -100, 150] * 5,
#           "600000.SH": [80, -30, 120, -60, 90] * 5,
#       },
#       index=dates,
#       dtype=float,
#   )
#   s_net_inflow = pd.DataFrame(
#       {
#           "000001.SZ": [-20, 10, -40, 25, -30] * 5,
#           "600000.SH": [15, -8, 30, -12, 20] * 5,
#       },
#       index=dates,
#       dtype=float,
#   )
#
#   factor = ELSCorrelationFactor()
#   result = factor.compute(el_net_inflow, s_net_inflow, T=20)
#   print(result.dropna())
