import pandas as pd

from factors.base import BaseFactor


class ForeignOpsFactor(BaseFactor):
    """海外运营信息因子 (Foreign Operations Information)"""

    name = "FOREIGN_OPS"
    category = "动量溢出"
    description = "海外运营信息：基于海外营收占比与对应国家行业收益率的加权求和，衡量海外动量溢出"

    def compute(
        self,
        foreign_sales_ratio: pd.DataFrame,
        country_industry_return: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        """计算海外运营信息因子。

        公式: InfoProxy_i = sum(f_i^c * R_j^c) for c != domestic

        Args:
            foreign_sales_ratio: 各股票在各国的海外营收占比
                                 (index=股票代码, columns=国家代码)
            country_industry_return: 各国对应行业收益率
                                     (index=国家代码, values=收益率)
                                     可以是 pd.Series 或单列 pd.DataFrame

        Returns:
            pd.Series: 因子值 (index=股票代码)
        """
        # 确保 country_industry_return 是 Series
        if isinstance(country_industry_return, pd.DataFrame):
            if country_industry_return.shape[1] == 1:
                ret_series = country_industry_return.iloc[:, 0]
            else:
                ret_series = country_industry_return.stack()
        else:
            ret_series = country_industry_return

        # 对齐国家维度
        common_countries = foreign_sales_ratio.columns.intersection(ret_series.index)

        if len(common_countries) == 0:
            return pd.Series(0.0, index=foreign_sales_ratio.index)

        aligned_ratio = foreign_sales_ratio[common_countries]
        aligned_return = ret_series[common_countries]

        # 加权求和: InfoProxy_i = sum(f_i^c * R_j^c)
        result = aligned_ratio.mul(aligned_return, axis=1).sum(axis=1)

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 海外运营信息因子利用公司在不同国家的营收占比作为权重，加权求和
# 对应国家的行业收益率。该因子捕捉海外市场动量向国内股票的溢出效应。
# 如果一家公司在某国有较高的营收占比，而该国对应行业近期表现强劲，
# 则该公司可能受益于海外动量溢出。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.foreign_ops import ForeignOpsFactor
#
#   stocks = ["000001.SZ", "000002.SZ", "000003.SZ"]
#   countries = ["US", "JP", "DE"]
#
#   foreign_sales_ratio = pd.DataFrame(
#       [[0.3, 0.2, 0.1], [0.0, 0.5, 0.1], [0.4, 0.0, 0.2]],
#       index=stocks, columns=countries
#   )
#   country_industry_return = pd.Series([0.05, -0.02, 0.03], index=countries)
#
#   factor = ForeignOpsFactor()
#   result = factor.compute(
#       foreign_sales_ratio=foreign_sales_ratio,
#       country_industry_return=country_industry_return
#   )
#   print(result)
