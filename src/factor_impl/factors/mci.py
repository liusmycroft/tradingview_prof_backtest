import pandas as pd

from factors.base import BaseFactor


class MCIFactor(BaseFactor):
    """边际交易成本因子 (Marginal Cost of Immediacy)"""

    name = "MCI"
    category = "高频流动性"
    description = "边际交易成本，衡量市价交易的单位金额冲击成本"

    def compute(
        self,
        daily_mci: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 MCI 因子。

        日内计算逻辑（预计算阶段）：
        1. VWAP_A = sum(P_A,i * Q_A,i) / sum(Q_A,i)  卖单五档加权均价
        2. M = (P_A,1 + P_B,1) / 2  买一卖一中间价
        3. VWAPM_A = (VWAP_A - M) / M  市价交易成本
        4. DolVol_A = sum(P_A,i * Q_A,i)  卖单报单总金额
        5. MCI_A = VWAPM_A / DolVol_A  边际交易成本 (bps/万元)

        本方法对预计算的日度 MCI 取 T 日 EMA。

        Args:
            daily_mci: 预计算的每日 MCI 值，index=日期, columns=股票代码。
            T: EMA 窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        result = daily_mci.ewm(span=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# MCI 衡量了市场的流动性。买单 MCI_B 衡量卖方发起市价交易的费用，
# MCI_B 较大时卖方要付出较大费用卖出，买压更强，较难下跌，
# 与未来短期收益正相关。卖单 MCI_A 衡量买方发起市价交易的费用。
