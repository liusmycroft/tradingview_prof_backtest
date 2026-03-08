import pandas as pd

from factors.base import BaseFactor


class NoonAncientTreeFactor(BaseFactor):
    """"午蔽古木"因子 (Noon Ancient Tree Factor)。"""

    name = "NOON_ANCIENT_TREE"
    category = "高频量价"
    description = "截距项t值绝对值经F值修正后的因子，衡量噪声与信息对价格的影响"

    def compute(
        self,
        daily_noon_ancient_tree: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算"午蔽古木"因子。

        公式:
            1. 回归 ret_t = a0 + a1*voldiff_t + ... + a6*voldiff_{t-5}
            2. 取截距项 |t值| 为 abst_intercept
            3. 若 F_all < 截面均值，则 abst_intercept *= -1
            4. 因子值为 T 日滚动均值

        Args:
            daily_noon_ancient_tree: 预计算的每日"午蔽古木"因子值
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_noon_ancient_tree.rolling(window=T, min_periods=T).mean()
        return result
