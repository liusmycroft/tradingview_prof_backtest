import pandas as pd

from factors.base import BaseFactor


class AvgPositiveJumpReturnFactor(BaseFactor):
    """平均日内正跳跌收益因子"""

    name = "AVG_POSITIVE_JUMP_RETURN"
    category = "高频波动跳跃"
    description = "日内正向跳跃收益的均值的T日滚动均值，衡量正向跳跃的平均幅度"

    def compute(
        self,
        daily_avg_pos_jump_ret: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算平均日内正跳跌收益因子。

        对每个交易日，识别日内正向跳跃（基于BV检验等方法），
        计算正向跳跃收益的均值。因子为 T 日滚动均值。

        Args:
            daily_avg_pos_jump_ret: 预计算的每日正向跳跃收益均值
                (index=日期, columns=股票代码)，无正跳跃的日期为 0 或 NaN
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 平均日内正跳跌收益的 T 日滚动均值
        """
        result = daily_avg_pos_jump_ret.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 平均日内正跳跌收益衡量股票日内正向跳跃的平均幅度。
# 跳跃通常由突发信息（如重大新闻、大单冲击）引起。
# 正向跳跃收益越大，说明股票受到的正面冲击越强烈，
# 可能反映了知情交易者的买入行为或利好信息的释放。
# 因子取 T 日滚动均值以平滑日间波动。
