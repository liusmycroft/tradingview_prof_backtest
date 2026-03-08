import pandas as pd

from factors.base import BaseFactor


class MPBFactor(BaseFactor):
    """市价偏离度因子 (Market Price Bias)"""

    name = "MPB"
    category = "高频流动性"
    description = "平均交易价格与平均委托挂单价格的差值，衡量买卖压力方向"

    def compute(
        self,
        daily_mpb: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算市价偏离度因子。

        日内计算逻辑（预计算阶段）：
        1. TP_t = T_t / V_t  (V_t != 0 时), 否则 TP_t = TP_{t-1}
        2. MP_t = (M_t + M_{t-1}) / 2, 其中 M_t = (P_t^B + P_t^A) / 2
        3. MPB_t = avg(TP_t) - avg(MP_t)  日内均值

        本方法对预计算的日度因子取 T 日 EMA。

        Args:
            daily_mpb: 预计算的每日市价偏离度，
                index=日期, columns=股票代码。
            T: EMA 窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        result = daily_mpb.ewm(span=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 市价偏离度为平均交易价格和平均委托挂单价格的差值。
# 当 MPB 为正，交易均价更接近卖一价，为卖方发起的交易，卖压大，
# 未来价格更趋向于下行；反之亦然。
