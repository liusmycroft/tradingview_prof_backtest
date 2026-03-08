"""残差反转因子 (Residual Reversal Factor)

通过回归剥离资金流强度对收益率的解释部分，提取残差作为纯粹的反转信号。
"""

import numpy as np
import pandas as pd

from .base import BaseFactor


class ResidualReversalFactor(BaseFactor):
    name = "residual_reversal"
    category = "reversal"
    description = "残差反转因子：对20日收益率关于资金流强度做截面回归，取残差"

    def compute(
        self,
        close: pd.DataFrame,
        buy_amount: pd.DataFrame,
        sell_amount: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算残差反转因子。

        Args:
            close: 收盘价，index=日期, columns=股票代码
            buy_amount: 买入金额（大单/小单）
            sell_amount: 卖出金额（大单/小单）
            T: 资金流强度回看窗口，默认20

        Returns:
            残差反转因子值，index=日期, columns=股票代码
        """
        # 20日收益率
        ret20 = close / close.shift(T) - 1

        # 资金流强度 S_t = sum(buy - sell) / sum(|buy - sell|)
        net = buy_amount - sell_amount
        rolling_net = net.rolling(window=T, min_periods=T).sum()
        rolling_abs = net.abs().rolling(window=T, min_periods=T).sum()
        strength = rolling_net / rolling_abs

        # 逐日截面回归，取残差
        residual = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        for date in close.index:
            y = ret20.loc[date]
            x = strength.loc[date]

            # 对齐并去除 NaN / Inf
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                continue

            xv = x[mask].values.astype(np.float64)
            yv = y[mask].values.astype(np.float64)

            # 如果 x 方差为零（常数资金流），回归无意义，残差即 y - mean(y)
            if np.std(xv) == 0:
                residual.loc[date, mask] = yv - np.mean(yv)
                continue

            # OLS: y = a + b*x  =>  b = cov(x,y)/var(x), a = mean(y) - b*mean(x)
            b, a = np.polyfit(xv, yv, 1)
            fitted = a + b * xv
            residual.loc[date, mask] = yv - fitted

        return residual


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【核心思想】
# 传统反转因子（如20日收益率）包含两部分信息：
#   1. 由资金流驱动的价格变动（可被资金流强度解释的部分）
#   2. 其他因素驱动的价格变动（残差部分）
#
# 资金流驱动的价格变动往往具有持续性（趋势），而非资金流驱动的部分
# 更可能发生反转。因此，通过截面回归剥离资金流的影响后，残差部分
# 是更纯粹的反转信号，选股效果优于原始反转因子。
#
# 【计算步骤】
# 1. 计算每只股票过去20日的收益率 Ret20
# 2. 计算每只股票过去20日的资金流强度 S（买卖净额/买卖绝对额之和）
# 3. 每个截面日期，对所有股票做回归：Ret20 = a + b * S + epsilon
# 4. 残差 epsilon 即为残差反转因子
#
# 【使用示例】
#
# import pandas as pd
# from factors.residual_reversal import ResidualReversalFactor
#
# # 假设已有数据
# close = pd.read_csv("close.csv", index_col=0, parse_dates=True)
# buy = pd.read_csv("buy_small.csv", index_col=0, parse_dates=True)
# sell = pd.read_csv("sell_small.csv", index_col=0, parse_dates=True)
#
# factor = ResidualReversalFactor()
# result = factor.compute(close=close, buy_amount=buy, sell_amount=sell, T=20)
# print(result.tail())
