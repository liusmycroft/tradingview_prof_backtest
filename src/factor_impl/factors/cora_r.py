"""滞后绝对收益与成交量相关性因子 (CORA_R)

衡量前一分钟价与后一分钟量的协同运动，捕捉"价在量先"的异常交易行为。
"""

import pandas as pd

from factors.base import BaseFactor


class CoraRFactor(BaseFactor):
    """滞后绝对收益与成交量相关性因子"""

    name = "CORA_R"
    category = "高频量价相关性"
    description = "日内滞后一期绝对收益率与成交额的相关性，捕捉价在量先的异常交易行为"

    def compute(
        self,
        daily_cora_r: pd.DataFrame,
        T: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """计算滞后绝对收益与成交量相关性因子。

        日内计算逻辑（已预计算为 daily_cora_r）:
          CORA_R = corr(|Ret_{t-1}|, Amount_t), t in [1, 240]

        因子值为 T 日滚动均值。

        Args:
            daily_cora_r: 预计算的每日 CORA_R 值 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 10

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_cora_r.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 该因子衡量了前一分钟价与后一分钟量的协同运动情况，捕捉"价在量先"
# 的异常交易行为，与未来收益负相关。取值越大，表示市场交易情绪受股价
# 涨跌的影响程度越高，聪明钱在利用波动频繁交易，次月负向Alpha越显著。
#
# 与 CORA_A（绝对收益与滞后成交量相关性）互为镜像：
#   CORA_A = corr(|Ret_t|, Amount_{t-1})  量在价先
#   CORA_R = corr(|Ret_{t-1}|, Amount_t)  价在量先
