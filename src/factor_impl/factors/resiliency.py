"""高频弹性因子 (High-Frequency Resiliency Factor)

衡量暂时价格从信息优势交易者驱动的价格影响恢复到基本价格的速率。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class ResiliencyFactor(BaseFactor):
    """高频弹性因子"""

    name = "RESILIENCY"
    category = "高频流动性"
    description = "高频弹性因子：通过频谱分析衡量暂时价格恢复到基本价格的速度"

    def compute(
        self,
        transitory_price: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算高频弹性因子。

        步骤:
        1. 对暂时价格 z_t 做离散傅里叶变换得到频谱 Z_k
        2. 归一化频谱幅度 |Z_k_bar| = |Z_k / D|
        3. 弹性 = mean(2 * |Z_k_bar| * f_k) for k=1..D/2

        Args:
            transitory_price: HP滤波后的暂时价格序列
                              MultiIndex (stock, time) 或预计算的每日弹性值
                              (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 弹性因子值
        """
        # 预计算模式
        if not isinstance(transitory_price.index, pd.MultiIndex):
            return transitory_price.copy()

        stocks = transitory_price.columns
        dates_level = transitory_price.index.get_level_values(0).unique()
        result = pd.DataFrame(np.nan, index=dates_level, columns=stocks)

        for date in dates_level:
            z_data = transitory_price.loc[date]

            for stock in stocks:
                z = z_data[stock].values.astype(float)
                valid = ~np.isnan(z)
                if valid.sum() < 4:
                    continue

                z_v = z[valid]
                D = len(z_v)

                # DFT
                Z = np.fft.fft(z_v)
                Z_bar = Z / D

                half = D // 2
                if half < 1:
                    continue

                # 频率 f_k = k / D
                resiliency = 0.0
                for k in range(1, half + 1):
                    amp = np.abs(Z_bar[k])
                    f_k = k / D
                    resiliency += 2 * amp * f_k

                resiliency /= half
                result.loc[date, stock] = resiliency

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 弹性描述为价格从信息优势交易者驱动的暂时价格影响恢复到其基本价格
# 的速率。通过频域的频谱分析得到暂时价格的距离和恢复时间，然后用
# 距离除以恢复时间来计算速度。与未来收益负相关，因子值越高，
# 说明股价从先前暂时价格影响中恢复越快，流动性越高。
