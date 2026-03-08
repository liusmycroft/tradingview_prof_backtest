import numpy as np
import pandas as pd

from factors.base import BaseFactor


class ActiveBuySpecificityFactor(BaseFactor):
    """主买成交特异性因子 (Active Buy Transaction Specificity)"""

    name = "ACTIVE_BUY_SPECIFICITY"
    category = "高频资金流"
    description = "修正主买时刻出现次数的截面标准化均值，衡量持续且特异性的主买资金流入"

    def compute(
        self,
        daily_active_buy_count: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算主买成交特异性因子。

        日内计算逻辑（已预计算为 daily_active_buy_count）:
          1. 计算每分钟主买成交量占全市场主买成交总量的比例
          2. 确定主买时刻：占比 > 当日80%分位数
          3. 修正主买时刻：要求主买量 > 前后一分钟
          4. 剔除开盘前15分钟和收盘前15分钟，统计修正主买时刻出现次数

        因子值 = 截面标准化后的 T 日均值。

        Args:
            daily_active_buy_count: 预计算的每日修正主买时刻次数
                                    (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 截面标准化后的 T 日均值
        """
        # 截面标准化: (x - mean) / std
        cs_mean = daily_active_buy_count.mean(axis=1)
        cs_std = daily_active_buy_count.std(axis=1)
        cs_std = cs_std.replace(0, np.nan)

        standardized = daily_active_buy_count.sub(cs_mean, axis=0).div(cs_std, axis=0)

        # T 日滚动均值
        result = standardized.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 主买成交特异性因子刻画了持续且特异性的主买资金流入。"特异性"指个股
# 相对于市场主买成交量的比例在截面上损益相消；"持续性"指修正后的主买
# 时刻在全天分布比较分散，代表股票当前需求持续存在，未来上涨概率更大。
