import numpy as np
import pandas as pd

from factors.base import BaseFactor


class AttentionTurnoverFactor(BaseFactor):
    """基于换手率的注意力捕捉因子 (Attention Capture via Turnover)"""

    name = "ATTENTION_TURNOVER"
    category = "行为金融-投资者注意力"
    description = "个股收益对行业平均换手率的回归系数绝对值，衡量股票对市场关注的敏感程度"

    def compute(
        self,
        returns: pd.DataFrame,
        industry_turnover: pd.DataFrame,
        mkt: pd.DataFrame,
        smb: pd.DataFrame,
        hml: pd.DataFrame,
        umd: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算基于换手率的注意力捕捉因子。

        对每只股票在过去 T 日内做时序回归:
          r_{i,d} = mu + beta * indus_turn_d + rho1 * Mkt_d + rho2 * SMB_d
                    + rho3 * HML_d + rho4 * UMD_d + eps
        因子值 = |beta|

        Args:
            returns: 个股日收益率 (index=日期, columns=股票代码)
            industry_turnover: 个股所属行业的日均换手率 (index=日期, columns=股票代码)
            mkt: 市场因子日收益 (index=日期, columns=['MKT'] 或单列)
            smb: SMB因子日收益
            hml: HML因子日收益
            umd: UMD因子日收益
            T: 回归窗口天数，默认 20

        Returns:
            pd.DataFrame: |beta| 因子值
        """
        dates = returns.index
        stocks = returns.columns
        n_dates = len(dates)

        # 将因子收益对齐为 numpy 数组
        mkt_vals = mkt.values.ravel() if mkt.ndim == 2 else mkt.values
        smb_vals = smb.values.ravel() if smb.ndim == 2 else smb.values
        hml_vals = hml.values.ravel() if hml.ndim == 2 else hml.values
        umd_vals = umd.values.ravel() if umd.ndim == 2 else umd.values

        result = np.full((n_dates, len(stocks)), np.nan)

        for t in range(T - 1, n_dates):
            start = t - T + 1
            end = t + 1

            m = mkt_vals[start:end]
            s = smb_vals[start:end]
            h = hml_vals[start:end]
            u = umd_vals[start:end]

            # 构建公共 X 矩阵 (截距 + indus_turn + 4因子)
            # indus_turn 因股票而异，需逐股票回归
            for j, stock in enumerate(stocks):
                y = returns.values[start:end, j]
                it = industry_turnover.values[start:end, j]

                mask = ~(np.isnan(y) | np.isnan(it) | np.isnan(m) |
                         np.isnan(s) | np.isnan(h) | np.isnan(u))
                if mask.sum() < 6:  # 至少需要 6 个观测 (6个参数)
                    continue

                X = np.column_stack([
                    np.ones(mask.sum()),
                    it[mask],
                    m[mask],
                    s[mask],
                    h[mask],
                    u[mask],
                ])
                y_clean = y[mask]

                try:
                    beta, _, _, _ = np.linalg.lstsq(X, y_clean, rcond=None)
                    result[t, j] = abs(beta[1])  # beta on indus_turn
                except np.linalg.LinAlgError:
                    continue

        return pd.DataFrame(result, index=dates, columns=stocks)


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 不同股票对同一市场关注度事件的反映程度不同。以行业平均换手率作为市场
# 关注度代理指标，通过个股收益对其的回归系数绝对值来衡量股票对市场关注
# 的敏感程度。捕捉能力越强（|beta|越大），表明个股更能吸引市场注意，
# 从而导致买盘压力增大，股价可能高估。
