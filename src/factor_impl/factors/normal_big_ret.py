import numpy as np
import pandas as pd

from factors.base import BaseFactor


class NormalBigRetFactor(BaseFactor):
    """剔除超大单影响后的大单涨跌幅因子 (Normal Big Order Return)"""

    name = "NORMAL_BIG_RET"
    category = "高频动量反转"
    description = "剔除超大单后的普通大单对数价格变动之和，衡量普通大单推动的涨跌幅"

    def compute(
        self,
        daily_normal_big_ret: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算剔除超大单影响后的大单涨跌幅因子。

        日内计算逻辑（已预计算为 daily_normal_big_ret）:
          1. 计算每笔成交的对数价格变动 delta_ln_P
          2. 将订单成交金额占当天总成交额 > 1% 的定义为超大单
          3. 将订单成交金额在前30%分位数的定义为大单
          4. NormalBigRet = sum(delta_ln_P) for 大单中剔除超大单的部分

        因子值为 T 日滚动求和。

        Args:
            daily_normal_big_ret: 预计算的每日普通大单涨跌幅
                                  (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动累计普通大单涨跌幅
        """
        result = daily_normal_big_ret.rolling(window=T, min_periods=1).sum()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 超大单涨跌幅因子具有负向Alpha，剔除超大单后的普通大单涨跌幅的正向
# Alpha会在原始因子基础上得到加强。普通大单指订单成交金额在前30%分位数
# 但不超过当天总成交额1%的订单。因子取T日滚动求和以累积信号。
