"""基于市场波动率的注意力捕捉因子 (Attention Capture via Market Volatility)

以行业截面收益标准差为市场关注代理指标，回归得到个股对市场关注的敏感程度。
"""

import numpy as np
import pandas as pd

from factors.base import BaseFactor


class AttentionCaptureVolFactor(BaseFactor):
    """基于市场波动率的注意力捕捉因子"""

    name = "ATTENTION_CAPTURE_VOL"
    category = "行为金融投资者注意力"
    description = "基于市场波动率的注意力捕捉：个股收益对行业截面波动率的敏感程度"

    def compute(
        self,
        stock_returns: pd.DataFrame,
        industry_vol: pd.Series,
        mkt: pd.Series,
        smb: pd.Series,
        hml: pd.Series,
        umd: pd.Series,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算基于市场波动率的注意力捕捉因子。

        对每只股票做月度时序回归:
            r_i = mu + beta * indus_vol + rho1*Mkt + rho2*SMB + rho3*HML + rho4*UMD + eps
        因子值 = |beta|

        Args:
            stock_returns: 个股日度收益 (index=日期, columns=股票代码)
            industry_vol: 行业截面收益标准差 (index=日期)
            mkt: 市场因子收益 (index=日期)
            smb: 规模因子收益 (index=日期)
            hml: 价值因子收益 (index=日期)
            umd: 动量因子收益 (index=日期)
            T: 回归窗口天数，默认 20

        Returns:
            pd.DataFrame: |beta| 因子值
        """
        dates = stock_returns.index
        stocks = stock_returns.columns
        n_dates = len(dates)

        result = pd.DataFrame(np.nan, index=dates, columns=stocks)

        # 构建因子矩阵
        X_full = np.column_stack([
            np.ones(n_dates),
            industry_vol.reindex(dates).values,
            mkt.reindex(dates).values,
            smb.reindex(dates).values,
            hml.reindex(dates).values,
            umd.reindex(dates).values,
        ])

        for t in range(T - 1, n_dates):
            X_win = X_full[t - T + 1: t + 1]

            if np.any(np.isnan(X_win)):
                continue

            for stock in stocks:
                y_win = stock_returns[stock].iloc[t - T + 1: t + 1].values.astype(float)

                if np.any(np.isnan(y_win)):
                    continue

                try:
                    beta = np.linalg.lstsq(X_win, y_win, rcond=None)[0]
                    # beta[1] 是 industry_vol 的系数
                    result.loc[dates[t], stock] = np.abs(beta[1])
                except np.linalg.LinAlgError:
                    continue

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 不同股票对同一市场关注度事件的反映程度往往不同。捕捉能力越强
# （敏感程度越高），表明个股更能吸引市场注意，从而导致买盘压力增大，
# 股价高估。以市场波动率作为市场关注度事件的代理变量。
