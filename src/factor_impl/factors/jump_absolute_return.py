import pandas as pd

from factors.base import BaseFactor


class JumpAbsoluteReturnFactor(BaseFactor):
    """累计跳跃绝对收益因子 (Jump Absolute Return, JAR)"""

    name = "JAR"
    category = "高频动量反转"
    description = "累计跳跃绝对收益的T日滚动求和，衡量跳跃成分对总收益的累计贡献"

    def compute(
        self,
        daily_jump_abs_return: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算累计跳跃绝对收益因子。

        对每个交易日，计算日内所有跳跃收益的绝对值之和。
        因子为 T 日滚动求和。

        公式: JAR = rolling_sum(daily_jump_abs_return, T)

        Args:
            daily_jump_abs_return: 预计算的每日跳跃绝对收益之和
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 累计跳跃绝对收益的 T 日滚动求和
        """
        result = daily_jump_abs_return.rolling(window=T, min_periods=1).sum()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 累计跳跃绝对收益 JAR 衡量过去 T 日内跳跃成分对价格变动的累计贡献。
# JAR 越大，说明股票在过去一段时间内经历了更多或更大幅度的跳跃，
# 价格变动中非连续成分占比较高。
# 跳跃通常与信息冲击相关，高 JAR 的股票可能面临更大的信息不确定性，
# 在动量反转策略中具有重要的区分作用。
