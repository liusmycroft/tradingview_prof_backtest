import numpy as np
import pandas as pd
from math import gamma

from .base import BaseFactor

# 标准正态分布的 2/3 阶绝对矩: mu_{2/3} = 2^{1/3} * Gamma(5/6) / Gamma(1/2)
MU_2_3 = 2 ** (1 / 3) * gamma(5 / 6) / gamma(1 / 2)


class RVJNFactor(BaseFactor):
    """下行跳跃波动率因子 (Downside Jump Volatility)。"""

    name = "RVJN"
    category = "波动率"
    description = "下行跳跃波动率，捕捉负向跳跃对已实现波动率的贡献"

    def compute(
        self,
        rs_negative: pd.DataFrame,
        iv_hat: pd.DataFrame,
    ) -> pd.DataFrame:
        """计算 RVJN 因子。

        Args:
            rs_negative: 下行已实现半方差，index=日期，columns=股票代码。
                         RS^- = sum(r_i^2) for r_i < 0。
            iv_hat: 积分方差估计量，index=日期，columns=股票代码。

        Returns:
            pd.DataFrame: RVJN 因子值，index=日期，columns=股票代码。
        """
        rvjn = (rs_negative - 0.5 * iv_hat).clip(lower=0)
        return rvjn

    @staticmethod
    def compute_from_intraday(intraday_returns_df: pd.DataFrame, k: int = 3) -> dict:
        """从日内收益率计算 rs_negative 和 iv_hat（单只股票）。

        Args:
            intraday_returns_df: 日内对数收益率，index=日期，columns=日内时段编号
                                 （如 0..47 表示 5 分钟 bar），values=对数收益率。
            k: 积分方差估计的滑动窗口长度，默认 3。

        Returns:
            dict: 包含 "rs_negative" 和 "iv_hat" 两个 pd.Series（index=日期）。
        """
        # --- RS^- : 下行已实现半方差 ---
        neg_mask = intraday_returns_df < 0
        rs_negative = (intraday_returns_df.where(neg_mask, 0.0) ** 2).sum(axis=1)

        # --- IV_hat : 积分方差估计 ---
        # 对每一天（每一行），用 k 个连续收益率的 |r|^{2/3} 乘积求和，
        # 再乘以 mu_{2/3}^{-k} 的修正系数。
        # IV_hat_t = mu_{2/3}^{-k} * sum_{i=k}^{n} prod_{j=0}^{k-1} |r_{t,i-j}|^{2/3}
        abs_r_23 = intraday_returns_df.abs() ** (2 / 3)  # |r|^{2/3}

        n_cols = abs_r_23.shape[1]
        if n_cols < k:
            iv_hat = pd.Series(np.nan, index=intraday_returns_df.index)
        else:
            # 滑动窗口乘积：对列方向做 rolling product
            # 逐行计算效率更高的向量化方式：shift 后逐元素相乘
            product = abs_r_23.values.copy()
            # product 初始化为 |r_i|^{2/3}，然后依次乘上 shift 1, 2, ..., k-1
            cols = abs_r_23.values
            running = np.ones_like(cols)
            for j in range(k):
                shifted = np.roll(cols, j, axis=1)
                shifted[:, :j] = np.nan  # 前 j 列无效
                running = running * shifted

            # running[:, i] = prod_{j=0}^{k-1} |r_{t, i-j}|^{2/3}，前 k-1 列为 NaN
            # 对每行求和（忽略 NaN）
            row_sums = np.nansum(running[:, k - 1 :], axis=1)
            iv_hat = pd.Series(
                MU_2_3 ** (-k) * row_sums,
                index=intraday_returns_df.index,
            )

        return {"rs_negative": rs_negative, "iv_hat": iv_hat}


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# RVJN（Downside Jump Volatility，下行跳跃波动率）用于度量资产价格中由负向
# 跳跃引起的波动成分。其核心思路是：
#   1. 已实现半方差 RS^- 捕捉了所有负收益对总波动的贡献（包括连续扩散部分
#      和跳跃部分）。
#   2. 积分方差 IV_hat 是对连续扩散方差的稳健估计（基于多幂次变差，对跳跃
#      具有鲁棒性）。
#   3. RS^- - 0.5 * IV_hat 剥离了连续扩散中属于负方向的一半，剩余部分即为
#      负向跳跃波动的度量。取 max(., 0) 保证非负。
#
# 该因子在实证中常被用于：
#   - 横截面选股：RVJN 较高的股票往往具有更大的尾部风险溢价。
#   - 风险管理：识别具有显著下行跳跃风险的标的。
#   - 波动率建模：作为 HAR-RV 类模型的附加解释变量。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.rvjn import RVJNFactor
#
#   # 方式一：直接传入预计算的 rs_negative 和 iv_hat
#   dates = pd.bdate_range("2025-01-01", periods=5)
#   rs_neg = pd.DataFrame(
#       {"000001.SZ": [0.003, 0.005, 0.002, 0.004, 0.006],
#        "600000.SH": [0.002, 0.003, 0.001, 0.002, 0.004]},
#       index=dates,
#   )
#   iv_hat = pd.DataFrame(
#       {"000001.SZ": [0.004, 0.008, 0.005, 0.006, 0.010],
#        "600000.SH": [0.003, 0.005, 0.003, 0.003, 0.006]},
#       index=dates,
#   )
#   factor = RVJNFactor()
#   rvjn = factor.compute(rs_negative=rs_neg, iv_hat=iv_hat)
#   print(rvjn)
#
#   # 方式二：从日内收益率计算（单只股票）
#   import numpy as np
#   intraday = pd.DataFrame(
#       np.random.randn(5, 48) * 0.001,
#       index=dates,
#       columns=range(48),
#   )
#   result = RVJNFactor.compute_from_intraday(intraday)
#   print(result["rs_negative"])
#   print(result["iv_hat"])
