import pandas as pd

from factors.base import BaseFactor


class SmartMoneyFactor(BaseFactor):
    """聪明钱因子 (Smart Money Factor)"""

    name = "SMART_MONEY"
    category = "高频量价相关性"
    description = "聪明钱 VWAP 与全市场 VWAP 之比，跟踪机构交易行为"

    def compute(
        self,
        daily_vwap_smart: pd.DataFrame,
        daily_vwap_all: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算聪明钱因子。

        日内计算逻辑（预计算阶段）：
        1. S_t = |R_t| / V_t^0.25  计算每分钟指标
        2. 按 S_t 从大到小排序，取成交量累积占比 20% 的分钟为聪明钱交易
        3. VWAP_smart = 聪明钱交易的成交量加权平均价
        4. VWAP_all = 所有交易的成交量加权平均价

        本方法计算 VWAP_smart / VWAP_all 并取 T 日 EMA。

        Args:
            daily_vwap_smart: 预计算的每日聪明钱 VWAP，
                index=日期, columns=股票代码。
            daily_vwap_all: 预计算的每日全市场 VWAP，
                index=日期, columns=股票代码。
            T: EMA 窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期, columns=股票代码。
        """
        ratio = daily_vwap_smart / daily_vwap_all
        result = ratio.ewm(span=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 聪明钱因子通过识别机构参与交易的多寡来构建选股因子。
# 聪明钱的交易基于"单笔订单数量更大、订单报价更为激进"的特征识别。
# S_t = |R_t| / V_t^0.25 越大，说明该分钟以较小成交量推动了较大价格变动，
# 更可能是知情交易者的行为。
