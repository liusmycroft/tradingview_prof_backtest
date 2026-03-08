import pandas as pd

from factors.base import BaseFactor


class CapitalLossRealizationFactor(BaseFactor):
    """已实现亏损比率因子 (Capital Loss Realization Ratio, CPLR)"""

    name = "CPLR"
    category = "行为金融-处置效应"
    description = "已实现亏损比率，衡量投资者实际卖出亏损股票的倾向程度"

    def compute(
        self,
        realized_loss: pd.DataFrame,
        paper_loss: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算已实现亏损比率因子。

        公式:
            daily_CPLR = realized_loss / (realized_loss + paper_loss)
            CPLR = rolling_mean(daily_CPLR, T)

        Args:
            realized_loss: 每日已实现亏损金额 (绝对值, >=0)
                (index=日期, columns=股票代码)
            paper_loss: 每日账面亏损金额 (绝对值, >=0)
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 已实现亏损比率的 T 日滚动均值
        """
        daily_cplr = realized_loss / (realized_loss + paper_loss)
        result = daily_cplr.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 已实现亏损比率 CPLR 衡量投资者实际卖出亏损头寸的倾向。
# CPLR = 已实现亏损 / (已实现亏损 + 账面亏损)
# 根据处置效应理论，投资者倾向于持有亏损股票而卖出盈利股票。
# CPLR 越高，说明投资者越愿意割肉止损，可能预示着恐慌性抛售
# 或理性止损行为，与未来收益的关系取决于市场环境。
