import numpy as np
import pandas as pd

from factors.base import BaseFactor


class IndustryMomentumFactor(BaseFactor):
    """行业动量溢出效应 (Industry Momentum Spillover) 因子。"""

    name = "INDUSTRY_MOMENTUM"
    category = "动量溢出"
    description = "行业内其他股票市值加权收益率，衡量行业动量溢出效应"

    def compute(
        self,
        returns: pd.Series,
        market_cap: pd.Series,
        industry: pd.Series,
        **kwargs,
    ) -> pd.Series:
        """计算行业动量溢出因子。

        Args:
            returns: 股票收益率，index=股票代码。
            market_cap: 市值，index=股票代码。
            industry: 行业标签，index=股票代码。

        Returns:
            pd.Series: 行业动量溢出值，index=股票代码。
        """
        stocks = returns.index
        result = pd.Series(np.nan, index=stocks)

        for stock in stocks:
            ind = industry[stock]
            # 同行业其他股票
            peers = industry[industry == ind].index
            peers = peers[peers != stock]

            if len(peers) == 0:
                result[stock] = np.nan
                continue

            peer_caps = market_cap[peers]
            total_cap = peer_caps.sum()

            if total_cap == 0:
                result[stock] = np.nan
                continue

            weights = peer_caps / total_cap
            result[stock] = (weights * returns[peers]).sum()

        return result


# ==============================================================================
# 核心思想与原理说明
# ==============================================================================
#
# 行业动量溢出效应因子的核心思想：
#
# 1. 信息在同行业股票间存在传导效应。当行业内某些股票率先反映信息时，
#    其他股票可能滞后跟随，形成"动量溢出"。
#
# 2. 对于股票 i，计算同行业内其他股票的市值加权收益率：
#    RET_IND_i = sum(w_j * r_j)，其中 w_j = MV_j / sum(MV_k, k!=i)
#
# 3. 该因子值高意味着同行业其他股票近期表现好，由于溢出效应，
#    股票 i 未来可能也会有正向表现。
#
# ==============================================================================
# 简单用法示例
# ==============================================================================
#
# import pandas as pd
# from factors.industry_momentum import IndustryMomentumFactor
#
# stocks = ["A", "B", "C", "D"]
# returns = pd.Series([0.02, 0.03, -0.01, 0.05], index=stocks)
# market_cap = pd.Series([100, 200, 150, 300], index=stocks)
# industry = pd.Series(["银行", "银行", "科技", "科技"], index=stocks)
#
# factor = IndustryMomentumFactor()
# result = factor.compute(returns=returns, market_cap=market_cap, industry=industry)
# print(result)
