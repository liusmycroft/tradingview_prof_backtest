import pandas as pd

from factors.base import BaseFactor


class IntradayJumpDropFactor(BaseFactor):
    """日内跳跌度因子 (Intraday Jump Drop Degree)。"""

    name = "INTRADAY_JUMP_DROP"
    category = "高频波动跳跃"
    description = "基于单复利差的日内跳跌度，衡量股价跳跃性涨跌程度"

    def compute(
        self,
        daily_jump_drop_mean: pd.DataFrame,
        daily_jump_drop_std: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算日内跳跌度因子。

        公式: 跳跌度 = (单利-复利)*2 - 复利^2
              月度因子 = mean(T日均值) + mean(T日标准差) 等权合并
        因子值为 T 日均值与标准差的等权合并。

        Args:
            daily_jump_drop_mean: 预计算的每日跳跌度均值 (index=日期, columns=股票代码)
            daily_jump_drop_std: 预计算的每日跳跌度标准差 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: 跳跌度因子值
        """
        roll_mean = daily_jump_drop_mean.rolling(window=T, min_periods=T).mean()
        roll_std = daily_jump_drop_std.rolling(window=T, min_periods=T).mean()
        result = 0.5 * roll_mean + 0.5 * roll_std
        return result
