import pandas as pd

from factors.base import BaseFactor


class GeoMomentumFactor(BaseFactor):
    """地理动量溢出效应因子 (Geographic Momentum Spillover)"""

    name = "GEO_MOMENTUM"
    category = "动量溢出"
    description = "地理动量溢出效应：同地区其他公司市值加权收益率"

    def compute(
        self,
        returns: pd.Series,
        market_cap: pd.Series,
        region: pd.Series,
        **kwargs,
    ) -> pd.Series:
        """计算地理动量溢出因子。

        公式: RET_GEO_i = sum(geo_weight_ij * Ret_j) for j in same region, j != i
        geo_weight_ij = MV_j / sum(MV_k for k != i in same region)

        Args:
            returns: 股票收益率 (index=股票代码)
            market_cap: 总市值 (index=股票代码)
            region: 所属地区 (index=股票代码)

        Returns:
            pd.Series: 地理动量溢出因子值 (index=股票代码)
        """
        result = pd.Series(index=returns.index, dtype=float)

        for stock in returns.index:
            stock_region = region[stock]
            # 同地区其他股票
            peers = region[(region == stock_region) & (region.index != stock)].index

            if len(peers) == 0:
                result[stock] = float("nan")
                continue

            peer_caps = market_cap[peers]
            total_cap = peer_caps.sum()

            if total_cap == 0:
                result[stock] = float("nan")
                continue

            weights = peer_caps / total_cap
            result[stock] = (weights * returns[peers]).sum()

        result.name = self.name
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 地理动量溢出效应 (GEO_MOMENTUM) 捕捉同一地理区域内公司之间的收益率
# 溢出效应。对于公司 i，计算同地区其他公司 j 的市值加权收益率作为因子值。
#
# 核心假设是：同一地区的公司共享相似的经济环境、政策影响和产业链关系，
# 因此一家公司的收益率变动可能预示着同地区其他公司的未来表现。
#
# 该因子可用于：
#   - 动量策略增强：利用地理维度的信息溢出捕捉额外收益。
#   - 区域轮动：识别区域性的投资机会。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.geo_momentum import GeoMomentumFactor
#
#   stocks = ["A", "B", "C", "D"]
#   returns    = pd.Series([0.02, 0.03, -0.01, 0.01], index=stocks)
#   market_cap = pd.Series([1e9, 2e9, 1.5e9, 0.5e9], index=stocks)
#   region     = pd.Series(["北京", "北京", "上海", "上海"], index=stocks)
#
#   factor = GeoMomentumFactor()
#   result = factor.compute(returns=returns, market_cap=market_cap, region=region)
#   print(result)
