import numpy as np
import pandas as pd

from factors.base import BaseFactor


class SimilarityMomentumFactor(BaseFactor):
    """公司特征相似度动量因子 (Firm Characteristic Similarity Momentum - SIM)"""

    name = "SIM_MOMENTUM"
    category = "图谱网络-动量溢出"
    description = "基于公司特征欧氏距离选取最相似股票，市值加权超额收益均值"

    def compute(
        self,
        price: pd.DataFrame,
        log_mcap: pd.DataFrame,
        bm: pd.DataFrame,
        op: pd.DataFrame,
        inv: pd.DataFrame,
        excess_return: pd.DataFrame,
        mcap: pd.DataFrame,
        K: int = 50,
        **kwargs,
    ) -> pd.DataFrame:
        """计算公司特征相似度动量因子。

        公式:
            D_ij = sqrt((Prc_i-Prc_j)^2 + (SIZE_i-SIZE_j)^2 + (BM_i-BM_j)^2
                        + (OP_i-OP_j)^2 + (INV_i-INV_j)^2)
            SIM_i = sum(w_k * R_excess_k) / sum(w_k), k in N(i) (最近K只)

        Args:
            price: 月末收盘价 (index=日期, columns=股票代码)
            log_mcap: 对数市值 (index=日期, columns=股票代码)
            bm: 账面市值比 (index=日期, columns=股票代码)
            op: 经营利润/净资产 (index=日期, columns=股票代码)
            inv: 总资产同比增速 (index=日期, columns=股票代码)
            excess_return: 过去一个月超额收益率 (index=日期, columns=股票代码)
            mcap: 市值权重 (index=日期, columns=股票代码)
            K: 最相似股票数量，默认 50

        Returns:
            pd.DataFrame: SIM因子值
        """
        dates = price.index
        stocks = price.columns
        n_stocks = len(stocks)

        result = pd.DataFrame(np.nan, index=dates, columns=stocks)

        for t_idx in range(len(dates)):
            # 提取当期截面数据
            feats = np.column_stack([
                price.iloc[t_idx].values,
                log_mcap.iloc[t_idx].values,
                bm.iloc[t_idx].values,
                op.iloc[t_idx].values,
                inv.iloc[t_idx].values,
            ]).astype(float)

            ret = excess_return.iloc[t_idx].values.astype(float)
            w = mcap.iloc[t_idx].values.astype(float)

            # 标准化特征（截面）
            for c in range(feats.shape[1]):
                col_data = feats[:, c]
                valid = ~np.isnan(col_data)
                if valid.sum() < 2:
                    continue
                m = np.nanmean(col_data)
                s = np.nanstd(col_data)
                if s > 0:
                    feats[:, c] = (col_data - m) / s

            for i in range(n_stocks):
                fi = feats[i]
                if np.any(np.isnan(fi)):
                    continue

                # 计算与所有其他股票的距离
                diffs = feats - fi
                # 跳过含NaN的行
                valid_mask = ~np.isnan(diffs).any(axis=1)
                valid_mask[i] = False  # 排除自身

                if valid_mask.sum() < 1:
                    continue

                distances = np.sqrt(np.sum(diffs ** 2, axis=1))
                distances[~valid_mask] = np.inf

                # 选取最近K只
                k_actual = min(K, valid_mask.sum())
                nearest_idx = np.argpartition(distances, k_actual)[:k_actual]

                # 过滤有效的收益和权重
                r_k = ret[nearest_idx]
                w_k = w[nearest_idx]
                valid_rw = ~(np.isnan(r_k) | np.isnan(w_k)) & (w_k > 0)

                if valid_rw.sum() == 0:
                    continue

                result.iloc[t_idx, i] = np.sum(w_k[valid_rw] * r_k[valid_rw]) / np.sum(w_k[valid_rw])

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 公司特征相似度动量因子利用公司基本面特征（收盘价、市值、账面市值比、
# 经营利润率、资产增速）计算股票间的欧氏距离，选取最相似的K只股票，
# 以市值加权计算其超额收益均值。
#
# 具有相似基本面特征的公司之间有着相似的预期收益，可以利用相似公司
# 的历史收益来预测自身的未来收益。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.similarity_momentum import SimilarityMomentumFactor
#
#   dates = pd.date_range("2024-01-01", periods=3, freq="M")
#   stocks = ["A", "B", "C", "D", "E"]
#   np.random.seed(42)
#   n = len(dates); m = len(stocks)
#   factor = SimilarityMomentumFactor()
#   result = factor.compute(
#       price=pd.DataFrame(np.random.uniform(5, 50, (n, m)), index=dates, columns=stocks),
#       log_mcap=pd.DataFrame(np.random.uniform(20, 25, (n, m)), index=dates, columns=stocks),
#       bm=pd.DataFrame(np.random.uniform(0.3, 2, (n, m)), index=dates, columns=stocks),
#       op=pd.DataFrame(np.random.uniform(0, 0.3, (n, m)), index=dates, columns=stocks),
#       inv=pd.DataFrame(np.random.uniform(-0.1, 0.3, (n, m)), index=dates, columns=stocks),
#       excess_return=pd.DataFrame(np.random.randn(n, m) * 0.05, index=dates, columns=stocks),
#       mcap=pd.DataFrame(np.random.uniform(1e9, 1e11, (n, m)), index=dates, columns=stocks),
#       K=2,
#   )
#   print(result)
