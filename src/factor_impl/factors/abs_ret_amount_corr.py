import pandas as pd

from .base import BaseFactor


class AbsRetAmountCorrFactor(BaseFactor):
    """绝对收益与成交量相关性因子 (CORA)"""

    name = "CORA"
    category = "高频量价相关性"
    description = "corr(|Ret|, Amount)，衡量单位成交额对股价推动的关系，与未来收益负相关"

    def compute(
        self,
        daily_cora: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算绝对收益与成交量相关性因子。

        公式:
            CORA_d = corr(|Ret_t|, Amount_t), t in [1, 240]  (日内分钟)
            因子 = T 日滚动均值

        Args:
            daily_cora: 预计算的每日 corr(|Ret|, Amount)
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: CORA 因子值，T 日滚动均值
        """
        result = daily_cora.rolling(window=T, min_periods=T).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# CORA 代表单位成交额对股价推动的关系。日内分钟级别计算绝对收益率与
# 成交额的相关性，正相关表明放量伴随价格剧烈波动。
#
# 因子值过高表明股票投机属性强，资金对价格扰动多，短期交易过热，
# 月度上往往具有负向 Alpha。取 T 日滚动均值平滑日间波动。
#
# 【使用示例】
#
#   import pandas as pd
#   import numpy as np
#   from factors.abs_ret_amount_corr import AbsRetAmountCorrFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   np.random.seed(42)
#   daily_cora = pd.DataFrame(
#       np.random.uniform(-0.5, 0.8, (30, 2)),
#       index=dates, columns=stocks,
#   )
#
#   factor = AbsRetAmountCorrFactor()
#   result = factor.compute(daily_cora=daily_cora, T=20)
#   print(result.tail())
