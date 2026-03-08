"""结构化反转因子 (Structured Reversal Factor)

将时间段按成交量大小分为动量时间段和反转时间段，分别构建反转因子后合成。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class StructuredReversalFactor(BaseFactor):
    """结构化反转因子"""

    name = "STRUCTURED_REVERSAL"
    category = "高频动量反转"
    description = "结构化反转因子：以成交量阈值区分动量/反转时段，分别加权后合成"

    def compute(
        self,
        log_returns: pd.DataFrame,
        volumes: pd.DataFrame,
        quantile_threshold: float = 0.1,
        **kwargs,
    ) -> pd.DataFrame:
        """计算结构化反转因子。

        步骤:
        1. 按成交量排序，<=10%分位为动量时段，>10%为反转时段
        2. 动量时段：以成交量倒数为权重计算加权收益
        3. 反转时段：以成交量为权重计算加权收益
        4. Rev_struct = Rev_rev - Rev_mom

        Args:
            log_returns: 对数收益率 log(Close_t/Close_{t-1})
                         (index=时间段, columns=股票代码)
            volumes: 成交量 (index=时间段, columns=股票代码)
            quantile_threshold: 大小成交量划分阈值，默认 0.1

        Returns:
            pd.DataFrame: 结构化反转因子值 (单行 DataFrame)
        """
        stocks = log_returns.columns
        result_dict = {}

        for stock in stocks:
            ret = log_returns[stock].values.astype(float)
            vol = volumes[stock].values.astype(float)

            valid = ~(np.isnan(ret) | np.isnan(vol) | (vol <= 0))
            if valid.sum() < 5:
                result_dict[stock] = np.nan
                continue

            ret_v = ret[valid]
            vol_v = vol[valid]

            # 按成交量排序，划分动量/反转时段
            threshold = np.quantile(vol_v, quantile_threshold)
            mom_mask = vol_v <= threshold
            rev_mask = vol_v > threshold

            # 动量时段：成交量倒数为权重
            if mom_mask.sum() > 0:
                w_mom = 1.0 / vol_v[mom_mask]
                w_mom = w_mom / w_mom.sum()
                rev_mom = np.sum(w_mom * ret_v[mom_mask])
            else:
                rev_mom = 0.0

            # 反转时段：成交量为权重
            if rev_mask.sum() > 0:
                w_rev = vol_v[rev_mask]
                w_rev = w_rev / w_rev.sum()
                rev_rev = np.sum(w_rev * ret_v[rev_mask])
            else:
                rev_rev = 0.0

            result_dict[stock] = rev_rev - rev_mom

        result = pd.DataFrame(result_dict, index=[log_returns.index[-1]])
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 在好坏信息发生初期，多空双方几乎不存在博弈，价格确定性强，成交量低，
# 动量效应强；随着时间推移，进入新的博弈阶段，成交量恢复，反转效应加强。
# 结构化反转因子改进了高频反转因子头部分组收益的线性单调性。
