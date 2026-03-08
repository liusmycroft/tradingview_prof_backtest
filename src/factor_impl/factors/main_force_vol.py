import numpy as np
import pandas as pd

from .base import BaseFactor


class MainForceVolFactor(BaseFactor):
    """主力波动率因子 (Main Force Volatility)。"""

    name = "MAIN_FORCE_VOL"
    category = "高频波动"
    description = "主力波动率因子，基于放量上涨/下跌等技术形态收益率的绝对值调整后波动率等权合成"

    def compute(
        self,
        vol_up_ret: pd.DataFrame,
        cont_up_ret: pd.DataFrame,
        vol_down_ret: pd.DataFrame,
        cont_down_ret: pd.DataFrame,
        T: int = 20,
    ) -> pd.DataFrame:
        """计算主力波动率因子。

        Args:
            vol_up_ret: 放量上涨日收益率，index=日期，columns=股票代码。
            cont_up_ret: 放量持续上涨日收益率，index=日期，columns=股票代码。
            vol_down_ret: 放量下跌日收益率，index=日期，columns=股票代码。
            cont_down_ret: 放量持续下跌日收益率，index=日期，columns=股票代码。
            T: 滚动窗口天数，默认 20。

        Returns:
            pd.DataFrame: 因子值，index=日期，columns=股票代码。
        """
        components = [vol_up_ret, cont_up_ret, vol_down_ret, cont_down_ret]
        vol_factors = []

        for comp in components:
            # 截面标准化
            cross_mean = comp.mean(axis=1)
            cross_std = comp.std(axis=1, ddof=1)
            standardized = comp.sub(cross_mean, axis=0).div(cross_std, axis=0)

            # 取绝对值
            abs_std = standardized.abs()

            # 再次截面标准化
            cross_mean2 = abs_std.mean(axis=1)
            cross_std2 = abs_std.std(axis=1, ddof=1)
            abs_standardized = abs_std.sub(cross_mean2, axis=0).div(cross_std2, axis=0)

            # 滚动 T 日标准差
            vol = abs_standardized.rolling(window=T, min_periods=T).std(ddof=1)
            vol_factors.append(vol)

        # 等权合成
        result = sum(vol_factors) / len(vol_factors)

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 主力波动率因子通过识别日内"放量上涨"、"放量持续上涨"、"放量下跌"、
# "放量持续下跌"四种技术形态，计算每种形态对应的收益率波动率，
# 再等权合成。核心逻辑：
#   1. 对每种形态的日收益率做截面标准化，取绝对值，再做截面标准化。
#   2. 计算过去 T 日的标准差作为该形态的波动率。
#   3. 四种形态波动率等权合成。
#
# 经济直觉：主力波动率刻画了主力资金行为对股价影响的波动程度，
# 波动程度越大，越有可能导致股价过度反应，未来收益率更低。
