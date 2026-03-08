import pandas as pd

from factors.base import BaseFactor


class NewsNetworkLeadReturnFactor(BaseFactor):
    """新闻网络领先收益因子 (News Network Lead Return)"""

    name = "NEWS_NETWORK_LEAD_RETURN"
    category = "图谱网络-动量溢出"
    description = "基于新闻共现网络的领先收益复合因子，综合同行业动量与跨行业反转效应"

    def compute(
        self,
        lead_return_pos_same: pd.DataFrame,
        lead_return_neg_same: pd.DataFrame,
        lead_return_pos_diff: pd.DataFrame,
        lead_return_neg_diff: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算新闻网络领先收益复合因子。

        公式: LR_agg = LR+(same) + LR-(same) + LR-(diff) - LR+(diff)

        同行业 lead 股票的正负收益均有动量效应（同向），
        跨行业 lead 股票的正收益有反转效应（反向）。

        Args:
            lead_return_pos_same: 同行业正向领先收益 (index=日期, columns=股票代码)
            lead_return_neg_same: 同行业负向领先收益 (index=日期, columns=股票代码)
            lead_return_pos_diff: 跨行业正向领先收益 (index=日期, columns=股票代码)
            lead_return_neg_diff: 跨行业负向领先收益 (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 复合领先收益因子值
        """
        result = (
            lead_return_pos_same
            + lead_return_neg_same
            + lead_return_neg_diff
            - lead_return_pos_diff
        )
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 新闻网络领先收益因子基于新闻标题与正文中股票的共现关系构建有向图，
# 标题中的股票为 lead，正文中的股票为 follower。将 lead 股票的收益
# 按邻接矩阵加权汇总，即得到 follower 股票的领先收益。
#
# 复合因子利用了两个经验规律：
# 1. 同行业 lead 股票的收益（无论正负）对 follower 有动量效应；
# 2. 跨行业 lead 股票的正收益对 follower 有反转效应。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.news_network_lead_return import NewsNetworkLeadReturnFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   lr_ps = pd.DataFrame(np.random.rand(30, 2) * 0.05, index=dates, columns=stocks)
#   lr_ns = pd.DataFrame(np.random.rand(30, 2) * 0.05, index=dates, columns=stocks)
#   lr_pd = pd.DataFrame(np.random.rand(30, 2) * 0.05, index=dates, columns=stocks)
#   lr_nd = pd.DataFrame(np.random.rand(30, 2) * 0.05, index=dates, columns=stocks)
#
#   factor = NewsNetworkLeadReturnFactor()
#   result = factor.compute(
#       lead_return_pos_same=lr_ps, lead_return_neg_same=lr_ns,
#       lead_return_pos_diff=lr_pd, lead_return_neg_diff=lr_nd,
#   )
#   print(result.tail())
