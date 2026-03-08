import pandas as pd

from factors.base import BaseFactor


class JumpDegreeCentralityFactor(BaseFactor):
    """基于跳跃频率关联的点度中心性因子"""

    name = "JUMP_DEGREE_CENTRALITY"
    category = "图谱网络-动量溢出"
    description = "基于跳跃频率关联构建网络后的点度中心性，衡量股票在跳跃传染网络中的核心程度"

    def compute(
        self,
        daily_degree_centrality: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算基于跳跃频率关联的点度中心性因子。

        构建方法:
        1. 对每只股票计算日内跳跃发生的指示变量（基于BV检验等）
        2. 在滚动窗口内计算股票两两之间跳跃指示变量的相关系数
        3. 以相关系数超过阈值为边构建网络
        4. 计算每只股票的点度中心性 (degree centrality)

        Args:
            daily_degree_centrality: 预计算的每日点度中心性
                (index=日期, columns=股票代码)
            T: EMA 窗口天数，默认 20

        Returns:
            pd.DataFrame: 点度中心性的 T 日 EMA
        """
        result = daily_degree_centrality.ewm(span=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 基于跳跃频率关联的点度中心性衡量股票在跳跃传染网络中的核心程度。
# 点度中心性越高，说明该股票的跳跃行为与更多其他股票相关联，
# 在信息传导和风险传染中扮演更重要的角色。
# 因子取 T 日 EMA 以平滑日间波动。
