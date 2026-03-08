import numpy as np
import pandas as pd

from factors.base import BaseFactor


class JumpNonPositiveMomentumFactor(BaseFactor):
    """跳跃关联非正跳跃相对动量因子"""

    name = "JUMP_NON_POSITIVE_MOMENTUM"
    category = "图谱网络-动量溢出"
    description = "以跳跃关联度加权关联股票的负跳跃和非跳跃收益残差之和"

    def compute(
        self,
        jump_corr: pd.DataFrame,
        neg_jump_ret: pd.DataFrame,
        no_jump_ret: pd.DataFrame,
        ret_20d: pd.Series,
        threshold: float = 0.5,
        **kwargs,
    ) -> pd.DataFrame:
        """计算跳跃关联非正跳跃相对动量因子。

        公式:
            Peer_Negjump_Ret_i = sum(Corr_ij * NegJump_Ret_j) / sum(Corr_ij)
            Peer_Nojump_Ret_i = sum(Corr_ij * NoJump_Ret_j) / sum(Corr_ij)
            对 Ret 分别回归 Peer_Negjump_Ret 和 Peer_Nojump_Ret 取残差
            因子 = epsilon_1 + epsilon_2

        Args:
            jump_corr: 跳跃关联度矩阵 (N x N DataFrame)
            neg_jump_ret: 负跳跃收益 (Series, index=股票代码)
            no_jump_ret: 非跳跃收益 (Series, index=股票代码)
            ret_20d: 过去20日收益率 (Series, index=股票代码)
            threshold: 剔除关联度最低的比例，默认 0.5

        Returns:
            pd.DataFrame: 因子值 (index=股票代码, columns=["factor"])
        """
        stocks = jump_corr.index.tolist()
        N = len(stocks)
        corr_mat = jump_corr.values.astype(float)

        # 稀释处理：剔除关联度最低的 threshold 比例
        flat = corr_mat[corr_mat > 0]
        if len(flat) > 0:
            cutoff = np.quantile(flat, threshold)
            corr_mat[corr_mat < cutoff] = 0

        peer_neg = np.full(N, np.nan)
        peer_no = np.full(N, np.nan)

        neg_vals = neg_jump_ret.reindex(stocks).values.astype(float)
        no_vals = no_jump_ret.reindex(stocks).values.astype(float)

        for i in range(N):
            weights = corr_mat[i].copy()
            weights[i] = 0
            total_w = np.nansum(weights)
            if total_w == 0:
                continue
            peer_neg[i] = np.nansum(weights * neg_vals) / total_w
            peer_no[i] = np.nansum(weights * no_vals) / total_w

        ret_vals = ret_20d.reindex(stocks).values.astype(float)
        result_vals = np.full(N, np.nan)

        for peer, label in [(peer_neg, "neg"), (peer_no, "no")]:
            valid = ~(np.isnan(ret_vals) | np.isnan(peer))
            if valid.sum() < 3:
                continue
            X = np.column_stack([np.ones(valid.sum()), peer[valid]])
            y = ret_vals[valid]
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                resid = y - X @ beta
                full_resid = np.full(N, np.nan)
                full_resid[valid] = resid
                if label == "neg":
                    result_vals = full_resid
                else:
                    result_vals = np.where(
                        np.isnan(result_vals) | np.isnan(full_resid),
                        np.nan,
                        result_vals + full_resid,
                    )
            except np.linalg.LinAlgError:
                continue

        return pd.DataFrame(result_vals, index=stocks, columns=["factor"])
