import numpy as np
import pandas as pd

from factors.base import BaseFactor


class PriceImpactBiasFactor(BaseFactor):
    """价格冲击偏差因子 (Price Impact Bias)"""

    name = "PRICE_IMPACT_BIAS"
    category = "高频因子-资金流类"
    description = "上涨与下跌方向价格冲击系数的相对偏差，衡量股票涨跌难易程度"

    def compute(
        self,
        bar_return: pd.DataFrame,
        bar_money_flow_ratio: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算价格冲击偏差因子。

        公式:
            return_i = gamma_up * I_i * MF_i + gamma_down * (1-I_i) * MF_i
            gamma_bias = (gamma_up - gamma_down) / ((gamma_up + gamma_down) / 2)

        Args:
            bar_return: 5分钟收益率 (index=bar编号, columns=股票代码)
            bar_money_flow_ratio: 主动净买入占比 MF_i = MoneyFlow/Amount

        Returns:
            pd.DataFrame: 价格冲击偏差因子值 (单行)
        """
        result_dict = {}

        for col in bar_return.columns:
            ret = bar_return[col].values.astype(float)
            mf = bar_money_flow_ratio[col].values.astype(float)

            valid = ~(np.isnan(ret) | np.isnan(mf))
            ret_v = ret[valid]
            mf_v = mf[valid]

            indicator = (mf_v > 0).astype(float)
            x_up = indicator * mf_v
            x_down = (1 - indicator) * mf_v

            # 回归: ret = gamma_up * x_up + gamma_down * x_down
            X = np.column_stack([x_up, x_down])
            if X.shape[0] < 3 or np.all(X == 0):
                result_dict[col] = np.nan
                continue

            try:
                beta = np.linalg.lstsq(X, ret_v, rcond=None)[0]
                gamma_up, gamma_down = beta[0], beta[1]
                avg = (gamma_up + gamma_down) / 2
                if avg == 0:
                    result_dict[col] = np.nan
                else:
                    result_dict[col] = (gamma_up - gamma_down) / avg
            except np.linalg.LinAlgError:
                result_dict[col] = np.nan

        return pd.DataFrame([result_dict])
