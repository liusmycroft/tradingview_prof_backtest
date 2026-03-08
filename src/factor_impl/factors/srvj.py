import pandas as pd

from factors.base import BaseFactor


class SRVJFactor(BaseFactor):
    """上下行跳跃波动的不对称性因子 (Jump Volatility Asymmetry - SRVJ)"""

    name = "SRVJ"
    category = "高频波动"
    description = "上下行跳跃波动不对称性：上行跳跃波动减去下行跳跃波动"

    def compute(
        self,
        rvjp: pd.DataFrame,
        rvjn: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 SRVJ 因子。

        公式: SRVJ = RVJP - RVJN

        Args:
            rvjp: 上行跳跃波动率 (index=日期, columns=股票代码)
            rvjn: 下行跳跃波动率 (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: SRVJ 因子值
        """
        return rvjp - rvjn


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# SRVJ（Signed Jump Variation，上下行跳跃波动不对称性）衡量资产价格中
# 正向跳跃与负向跳跃的不对称程度。
#   - RVJP: 上行跳跃波动率，捕捉正向跳跃对已实现波动率的贡献。
#   - RVJN: 下行跳跃波动率，捕捉负向跳跃对已实现波动率的贡献。
#   - SRVJ = RVJP - RVJN
#
# SRVJ > 0 表示正向跳跃占主导，股价有向上突破的倾向；
# SRVJ < 0 表示负向跳跃占主导，股价有向下突破的风险。
#
# 该因子在实证中常被用于：
#   - 横截面选股：SRVJ 较高的股票可能具有正向动量。
#   - 波动率建模：作为 HAR-RV 类模型的附加解释变量。
#   - 尾部风险度量：识别跳跃方向的不对称性。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.srvj import SRVJFactor
#
#   dates = pd.bdate_range("2025-01-01", periods=5)
#   rvjp = pd.DataFrame(
#       {"000001.SZ": [0.003, 0.005, 0.002, 0.004, 0.006],
#        "600000.SH": [0.002, 0.003, 0.001, 0.002, 0.004]},
#       index=dates,
#   )
#   rvjn = pd.DataFrame(
#       {"000001.SZ": [0.001, 0.002, 0.003, 0.001, 0.002],
#        "600000.SH": [0.001, 0.001, 0.002, 0.003, 0.001]},
#       index=dates,
#   )
#   factor = SRVJFactor()
#   result = factor.compute(rvjp=rvjp, rvjn=rvjn)
#   print(result)
