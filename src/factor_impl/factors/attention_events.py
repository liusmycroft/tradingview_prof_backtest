import pandas as pd

from factors.base import BaseFactor


class AttentionEventsFactor(BaseFactor):
    """关注事件因子 (Attention Events)"""

    name = "ATTENTION_EVENTS"
    category = "行为金融"
    description = "每月涨停、跌停事件次数之和，衡量股票受关注程度"

    def compute(
        self,
        limit_up_count: pd.DataFrame,
        limit_down_count: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """计算关注事件因子。

        公式: ATTENTION_EVENTS = limit_up_count + limit_down_count

        Args:
            limit_up_count: 涨停次数 (index=日期, columns=股票代码)
            limit_down_count: 跌停次数 (index=日期, columns=股票代码)

        Returns:
            pd.DataFrame: 关注事件总次数
        """
        result = limit_up_count.add(limit_down_count, fill_value=0)
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 关注事件因子统计每月涨停、跌停事件的总次数（也可包含龙虎榜上榜次数）。
# 涨停和跌停是市场中极端价格变动事件，容易吸引投资者关注。
# 事件次数越多，说明该股票在市场中的关注度越高，可能存在行为金融
# 中的"注意力效应"——散户倾向于买入近期引起关注的股票。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.attention_events import AttentionEventsFactor
#
#   dates = pd.date_range("2024-01-01", periods=12, freq="ME")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   limit_up = pd.DataFrame(
#       [[2, 1], [0, 3], [1, 0], [0, 0], [1, 2], [0, 1],
#        [3, 0], [0, 1], [2, 2], [1, 0], [0, 1], [1, 1]],
#       index=dates, columns=stocks,
#   )
#   limit_down = pd.DataFrame(
#       [[1, 0], [1, 1], [0, 2], [0, 0], [0, 1], [1, 0],
#        [0, 2], [1, 0], [1, 1], [0, 1], [1, 0], [0, 0]],
#       index=dates, columns=stocks,
#   )
#
#   factor = AttentionEventsFactor()
#   result = factor.compute(limit_up_count=limit_up, limit_down_count=limit_down)
#   print(result)
