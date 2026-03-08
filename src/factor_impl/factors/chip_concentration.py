"""
筹码集中度因子 (Chip Concentration Factor)

衡量筹码分布的集中程度，使用95%分位价格与5%分位价格的归一化差值。
"""

import pandas as pd
from factors.base import BaseFactor


class ChipConcentrationFactor(BaseFactor):
    """筹码集中度因子

    公式: chip_distri = 2 * (price95 - price05) / (price95 + price05)
    """

    name = "CHIP_CONCENTRATION"
    category = "行为金融"
    description = "筹码集中度因子，衡量筹码分布的集中程度"

    def compute(self, price95: pd.DataFrame, price05: pd.DataFrame) -> pd.DataFrame:
        """计算筹码集中度因子。

        Args:
            price95: 95%分位价格，index为日期，columns为股票代码。
            price05: 5%分位价格，index为日期，columns为股票代码。

        Returns:
            pd.DataFrame: 筹码集中度因子值。
        """
        # 归一化差值法
        result = 2 * (price95 - price05) / (price95 + price05)
        return result


# 使用示例
if __name__ == "__main__":
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=5)
    stocks = ["000001", "000002", "000003"]
    price95 = pd.DataFrame(np.random.uniform(10, 20, (5, 3)), index=dates, columns=stocks)
    price05 = pd.DataFrame(np.random.uniform(5, 10, (5, 3)), index=dates, columns=stocks)

    factor = ChipConcentrationFactor()
    print(factor)
    print(factor.compute(price95=price95, price05=price05))
