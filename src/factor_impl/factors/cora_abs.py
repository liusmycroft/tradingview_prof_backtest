import pandas as pd

from factors.base import BaseFactor


class CoraAbsFactor(BaseFactor):
    """绝对收益与滞后成交量相关性因子 (CORA_A)"""

    name = "CORA_A"
    category = "高频量价相关性"
    description = "日内绝对收益率与滞后一期成交额的相关性，捕捉量在价先的异常交易行为"

    def compute(
        self,
        daily_cora_a: pd.DataFrame,
        T: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """计算绝对收益与滞后成交量相关性因子。

        日内计算逻辑（已预计算为 daily_cora_a）:
          CORA_A = corr(|Ret_t|, Amount_{t-1}), t in [1, 240]

        因子值为 T 日滚动均值。

        Args:
            daily_cora_a: 预计算的每日 CORA_A 值 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 10

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_cora_a.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 该因子衡量了前一分钟量与后一分钟价的协同运动情况，捕捉"量在价先"
# 的异常交易行为。取值越大，表示信息的提前泄露程度越高，聪明钱在抢跑
# 交易，与未来收益负相关。
