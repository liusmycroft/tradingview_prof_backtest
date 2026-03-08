import numpy as np
import pandas as pd

from factors.base import BaseFactor


class APMFactor(BaseFactor):
    """APM 因子 (Afternoon-to-Morning Premium)"""

    name = "APM"
    category = "高频因子-动量反转类"
    description = "上午与下午价格行为差异程度的统计量，剥离动量影响后的残差"

    def compute(
        self,
        stock_am_ret: pd.DataFrame,
        stock_pm_ret: pd.DataFrame,
        index_am_ret: pd.DataFrame,
        index_pm_ret: pd.DataFrame,
        ret20: pd.DataFrame,
        N: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 APM 因子。

        公式:
            1. 将上午/下午 (r, R) 合并回归: r = alpha + beta*R + eps
            2. delta_t = eps_am - eps_pm
            3. stat = mean(delta) / (std(delta) / sqrt(N))
            4. stat 对 Ret20 横截面回归取残差

        Args:
            stock_am_ret: 股票上午收益率 (index=日期, columns=股票代码)
            stock_pm_ret: 股票下午收益率
            index_am_ret: 指数上午收益率 (index=日期, columns=["index"])
            index_pm_ret: 指数下午收益率
            ret20: 过去20日收益率
            N: 回溯天数，默认 20

        Returns:
            pd.DataFrame: APM 因子值
        """
        stocks = stock_am_ret.columns.tolist()
        result = pd.DataFrame(np.nan, index=stock_am_ret.index, columns=stocks)

        idx_col = index_am_ret.columns[0]

        for i in range(N - 1, len(stock_am_ret)):
            stat_vals = {}
            for col in stocks:
                am_r = stock_am_ret[col].iloc[i - N + 1 : i + 1].values
                pm_r = stock_pm_ret[col].iloc[i - N + 1 : i + 1].values
                am_R = index_am_ret[idx_col].iloc[i - N + 1 : i + 1].values
                pm_R = index_pm_ret[idx_col].iloc[i - N + 1 : i + 1].values

                r_all = np.concatenate([am_r, pm_r])
                R_all = np.concatenate([am_R, pm_R])

                valid = ~(np.isnan(r_all) | np.isnan(R_all))
                if valid.sum() < 5:
                    continue

                X = np.column_stack([np.ones(valid.sum()), R_all[valid]])
                y = r_all[valid]
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue

                eps_all = r_all - (beta[0] + beta[1] * R_all)
                eps_am = eps_all[:N]
                eps_pm = eps_all[N:]
                delta = eps_am - eps_pm

                valid_d = ~np.isnan(delta)
                if valid_d.sum() < 3:
                    continue

                d = delta[valid_d]
                mu = np.mean(d)
                sigma = np.std(d, ddof=1)
                if sigma == 0:
                    continue
                stat_vals[col] = mu / (sigma / np.sqrt(len(d)))

            if len(stat_vals) < 3:
                continue

            # 横截面回归: stat = b * Ret20 + eps
            stat_s = pd.Series(stat_vals)
            ret_row = ret20.iloc[i].reindex(stat_s.index)
            valid = stat_s.notna() & ret_row.notna()
            if valid.sum() < 3:
                continue

            X = np.column_stack([np.ones(valid.sum()), ret_row[valid].values.astype(float)])
            y = stat_s[valid].values.astype(float)
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ beta
                for idx_j, col in enumerate(stat_s[valid].index):
                    result.loc[result.index[i], col] = residuals[idx_j]
            except np.linalg.LinAlgError:
                continue

        return result
