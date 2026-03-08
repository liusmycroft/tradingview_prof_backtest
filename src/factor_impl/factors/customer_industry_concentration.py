import pandas as pd

from factors.base import BaseFactor


class CustomerIndustryConcentrationFactor(BaseFactor):
    """客户行业集中度因子 (Customer Industry Concentration - CH)。"""

    name = "CUSTOMER_INDUSTRY_CONCENTRATION"
    category = "图谱网络"
    description = "客户所属行业赫芬达尔指数的加权平均，反映客户行业竞争格局"

    def compute(
        self,
        daily_customer_industry_conc: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算客户行业集中度因子。

        公式: CH_i = sum(w_ij * H_j)
              H_j = sum(s_ij^2) 为赫芬达尔指数
        因子值为 T 日 EMA。

        Args:
            daily_customer_industry_conc: 预计算的每日客户行业集中度
                (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 客户行业集中度的 T 日 EMA
        """
        result = daily_customer_industry_conc.ewm(span=T, min_periods=1).mean()
        return result
