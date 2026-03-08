"""筹码分布成本重心因子 (Chip Distribution Price Center - CKDP)

衡量筹码成本重心在最高与最低筹码价格之间的相对位置，反映市场筹码的集中程度与阶段。
"""

import pandas as pd

from .base import BaseFactor


class CKDPFactor(BaseFactor):
    name = "ckdp"
    category = "chip"
    description = "筹码分布成本重心因子：(均价 - 最低筹码价) / (最高筹码价 - 最低筹码价)"

    def compute(
        self,
        chip_mean: pd.DataFrame,
        chip_highest: pd.DataFrame,
        chip_lowest: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 CKDP 因子。

        Args:
            chip_mean: 成交量加权平均成本，index=日期, columns=股票代码
            chip_highest: 最高筹码价格
            chip_lowest: 最低筹码价格

        Returns:
            CKDP 因子值 (0~1)，index=日期, columns=股票代码
        """
        spread = chip_highest - chip_lowest
        ckdp = (chip_mean - chip_lowest) / spread

        return ckdp


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【核心思想】
# 筹码分布成本重心（CKDP）将成交量加权均价在最高与最低筹码价格之间做
# 归一化处理，得到一个 0~1 之间的指标，用于刻画当前市场筹码的成本重心
# 所处的相对位置。
#
# - 接近 0：成本重心偏低，多数筹码集中在低价区域，通常对应吸筹/建仓阶段
# - 接近 1：成本重心偏高，多数筹码集中在高价区域，通常对应派发/出货阶段
# - 接近 0.5：筹码分布较为均衡，市场处于多空平衡状态
#
# 该因子可与换手率、筹码集中度等指标配合使用，辅助判断主力行为和市场阶段。
#
# 【计算公式】
# CKDP_t = (mean_t - Chip_Lowest_t) / (Chip_Highest_t - Chip_Lowest_t)
#
# 其中 mean_t = sum(price_{i,t} * p_{i,t})，p_{i,t} 为各价位筹码占比。
#
# 【使用示例】
#
# import pandas as pd
# from factors.ckdp import CKDPFactor
#
# chip_mean = pd.read_csv("chip_mean.csv", index_col=0, parse_dates=True)
# chip_highest = pd.read_csv("chip_highest.csv", index_col=0, parse_dates=True)
# chip_lowest = pd.read_csv("chip_lowest.csv", index_col=0, parse_dates=True)
#
# factor = CKDPFactor()
# result = factor.compute(
#     chip_mean=chip_mean,
#     chip_highest=chip_highest,
#     chip_lowest=chip_lowest,
# )
# print(result.tail())
