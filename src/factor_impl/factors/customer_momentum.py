"""最大客户动量因子 (Largest Customer Momentum, cmom)

cmom_i = mom_{j_max_sales}
即股票 i 的最大客户 j 的动量。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class CustomerMomentumFactor(BaseFactor):
    """最大客户动量因子 (cmom)"""

    name = "CMOM"
    category = "图谱网络-动量溢出"
    description = "最大客户动量，股票最大客户的收益动量，捕捉供应链动量溢出效应"

    def compute(
        self,
        returns: pd.DataFrame,
        largest_customer: pd.Series,
        **kwargs,
    ) -> pd.DataFrame:
        """计算最大客户动量因子。

        Args:
            returns: 股票收益率 (index=日期, columns=股票代码)。
                     可以是 1 月/6 月/12 月动量。
            largest_customer: 每只股票的最大客户映射
                (index=股票代码, values=最大客户股票代码)。

        Returns:
            pd.DataFrame: 因子值 (index=日期, columns=股票代码)。
        """
        stocks = returns.columns
        result = pd.DataFrame(np.nan, index=returns.index, columns=stocks)

        for stock in stocks:
            if stock not in largest_customer.index:
                continue
            customer = largest_customer[stock]
            if pd.isna(customer):
                continue
            if customer in returns.columns:
                result[stock] = returns[customer]

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 最大客户动量因子利用供应链关系，将股票最大客户的收益动量作为该股票的
# 因子值。核心逻辑是供应链中的动量溢出主要来源于最大客户。
#
# 当最大客户表现强劲时，供应商往往也会受益于需求增长，但由于信息传导
# 存在时滞，供应商的股价反应可能滞后。
