import pandas as pd
from scipy.stats import spearmanr

from factors.base import BaseFactor


class SmallOrderLagCorrFactor(BaseFactor):
    """小单和小单的错位相关性因子 (Small Order Misalignment Correlation)"""

    name = "SMALL_ORDER_LAG_CORR"
    category = "高频资金流"
    description = "小单净流入与其滞后一期的Spearman秩相关系数滚动值"

    def compute(
        self,
        s_net_inflow: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算小单错位相关性因子。

        公式: RankCorr(S_t, S_{t+1}) — 小单净流入与其1日滞后的
              Spearman秩相关系数，在 T 日滚动窗口内计算。

        Args:
            s_net_inflow: 小单净流入金额 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 滚动秩相关系数
        """
        result = pd.DataFrame(index=s_net_inflow.index, columns=s_net_inflow.columns, dtype=float)

        for col in s_net_inflow.columns:
            series = s_net_inflow[col]
            lagged = series.shift(1)

            for i in range(len(series)):
                start = max(0, i - T + 1)
                window_orig = series.iloc[start:i + 1]
                window_lag = lagged.iloc[start:i + 1]

                # 去除 NaN 配对
                valid = window_orig.notna() & window_lag.notna()
                if valid.sum() < 3:
                    result.iloc[i, result.columns.get_loc(col)] = float("nan")
                else:
                    corr, _ = spearmanr(window_orig[valid], window_lag[valid])
                    result.iloc[i, result.columns.get_loc(col)] = corr

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 小单错位相关性衡量小单（散户）净流入在时间序列上的自相关结构。
# 通过计算 S_t 与 S_{t+1}（即当日与次日小单净流入）在滚动窗口内的
# Spearman 秩相关系数，捕捉散户资金流的持续性或反转特征。
#
# 正相关表示散户资金流具有动量特征（今天买明天继续买），
# 负相关表示散户资金流具有反转特征（今天买明天倾向卖）。
# 该因子可用于判断散户行为模式，辅助构建反转或动量策略。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.small_order_lag_corr import SmallOrderLagCorrFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   s_net_inflow = pd.DataFrame(
#       np.random.randn(30, 2) * 1e6,
#       index=dates, columns=stocks,
#   )
#
#   factor = SmallOrderLagCorrFactor()
#   result = factor.compute(s_net_inflow=s_net_inflow, T=20)
#   print(result.tail())
