import numpy as np
import pandas as pd

from factors.base import BaseFactor


class CorrectedNetInflowFactor(BaseFactor):
    """修正的净流入率因子 (Corrected Net Inflow Ratio - CNIR)"""

    name = "CNIR"
    category = "高频资金流"
    description = "通过回归剥离涨跌幅影响后的主力资金净流入占比"

    def compute(
        self,
        daily_buy_amount: pd.DataFrame,
        daily_sell_amount: pd.DataFrame,
        daily_return: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算修正的净流入率因子。

        公式:
            ln(B_P/S_P) = alpha + beta * Ret + epsilon
            B_hat = e^eps / (1 + e^eps) * (B_P + S_P)
            S_hat = 1 / (1 + e^eps) * (B_P + S_P)
            CNIR = sum(B_hat - S_hat) / sum(B_hat + S_hat)  (T日滚动)

        Args:
            daily_buy_amount: 每日主力买入金额 B_P (index=日期, columns=股票代码)
            daily_sell_amount: 每日主力卖出金额 S_P (index=日期, columns=股票代码)
            daily_return: 每日涨跌幅 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: CNIR因子值
        """
        dates = daily_buy_amount.index
        stocks = daily_buy_amount.columns
        num_dates = len(dates)
        num_stocks = len(stocks)

        buy_vals = daily_buy_amount.values
        sell_vals = daily_sell_amount.values
        ret_vals = daily_return.values

        result = np.full((num_dates, num_stocks), np.nan)

        for t in range(T - 1, num_dates):
            b_win = buy_vals[t - T + 1 : t + 1]
            s_win = sell_vals[t - T + 1 : t + 1]
            r_win = ret_vals[t - T + 1 : t + 1]

            for col in range(num_stocks):
                b = b_win[:, col]
                s = s_win[:, col]
                r = r_win[:, col]

                # 需要 B > 0 且 S > 0
                valid = (b > 0) & (s > 0) & ~np.isnan(r)
                if valid.sum() < 3:
                    continue

                b_v = b[valid]
                s_v = s[valid]
                r_v = r[valid]

                y = np.log(b_v / s_v)
                X = np.column_stack([np.ones(len(r_v)), r_v])

                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue

                # 对整个窗口计算残差和修正值
                total = b_win[:, col] + s_win[:, col]
                log_ratio = np.where(
                    (b_win[:, col] > 0) & (s_win[:, col] > 0),
                    np.log(np.where(b_win[:, col] > 0, b_win[:, col], 1)
                           / np.where(s_win[:, col] > 0, s_win[:, col], 1)),
                    np.nan,
                )
                eps = log_ratio - beta[0] - beta[1] * r_win[:, col]

                valid_all = ~np.isnan(eps) & (total > 0)
                if valid_all.sum() == 0:
                    continue

                e_eps = np.exp(eps[valid_all])
                b_hat = e_eps / (1 + e_eps) * total[valid_all]
                s_hat = 1 / (1 + e_eps) * total[valid_all]

                denom = np.sum(b_hat + s_hat)
                if denom == 0:
                    continue
                result[t, col] = np.sum(b_hat - s_hat) / denom

        return pd.DataFrame(result, index=dates, columns=stocks)


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 修正的净流入率因子通过截面回归剥离涨跌幅对资金流的影响。
# 大单资金流与涨跌幅正相关源自主力资金买卖不平衡，通过回归残差
# 可以提纯资金流的Alpha信息。CNIR > 0 表示修正后主力净流入，
# CNIR < 0 表示修正后主力净流出。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.corrected_net_inflow import CorrectedNetInflowFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#   np.random.seed(42)
#   daily_buy = pd.DataFrame(np.random.uniform(1e6, 1e8, (30, 2)),
#       index=dates, columns=stocks)
#   daily_sell = pd.DataFrame(np.random.uniform(1e6, 1e8, (30, 2)),
#       index=dates, columns=stocks)
#   daily_ret = pd.DataFrame(np.random.randn(30, 2) * 0.02,
#       index=dates, columns=stocks)
#
#   factor = CorrectedNetInflowFactor()
#   result = factor.compute(daily_buy_amount=daily_buy,
#       daily_sell_amount=daily_sell, daily_return=daily_ret, T=20)
#   print(result.tail())
