import numpy as np
import pandas as pd

from factors.base import BaseFactor


class ResidualFlowStrengthFactor(BaseFactor):
    """残差资金流强度因子 (Residual Flow Strength)"""

    name = "RESIDUAL_FLOW_STRENGTH"
    category = "高频因子-资金流类"
    description = "剥离涨跌幅影响后的资金流强度，衡量同等涨跌幅下资金流的选股能力"

    def compute(
        self,
        buy_amount: pd.DataFrame,
        sell_amount: pd.DataFrame,
        ret20: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算残差资金流强度因子。

        公式:
            S_t = sum(buy - sell) / sum(|buy - sell|)
            S_t = a + b * Ret20 + epsilon
            因子 = epsilon

        Args:
            buy_amount: 买入金额 (index=日期, columns=股票代码)
            sell_amount: 卖出金额
            ret20: 过去20日涨跌幅
            T: 滚动窗口，默认 20

        Returns:
            pd.DataFrame: 残差资金流强度因子值
        """
        net = buy_amount - sell_amount
        numerator = net.rolling(window=T, min_periods=T).sum()
        denominator = net.abs().rolling(window=T, min_periods=T).sum()
        strength = numerator / denominator.replace(0, np.nan)

        result = pd.DataFrame(np.nan, index=buy_amount.index, columns=buy_amount.columns)

        for i in range(len(strength)):
            s_row = strength.iloc[i]
            r_row = ret20.iloc[i]
            valid = s_row.notna() & r_row.notna()
            if valid.sum() < 3:
                continue
            s_vals = s_row[valid].values.astype(float)
            r_vals = r_row[valid].values.astype(float)

            # OLS: S = a + b * Ret20 + eps
            X = np.column_stack([np.ones(len(r_vals)), r_vals])
            try:
                beta = np.linalg.lstsq(X, s_vals, rcond=None)[0]
                residuals = s_vals - X @ beta
                result.iloc[i, result.columns.isin(s_row[valid].index)] = residuals
            except np.linalg.LinAlgError:
                continue

        return result
