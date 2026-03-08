import pandas as pd

from factors.base import BaseFactor


class InstTradeHeatFactor(BaseFactor):
    """机构交易热度因子 (Institutional Trading Activity)"""

    name = "INST_TRADE_HEAT"
    category = "投资者注意力"
    description = "机构交易热度：机构资金流入流出占市值的滚动均值"

    def compute(
        self,
        inflow_inst: pd.DataFrame,
        outflow_inst: pd.DataFrame,
        cap: pd.DataFrame,
        m: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算机构交易热度因子。

        公式: TURN_INST = (1/m) * sum((INFLOW_INST + OUTFLOW_INST) / CAP) over m days

        Args:
            inflow_inst: 机构买入金额 (index=日期, columns=股票代码)
            outflow_inst: 机构卖出金额
            cap: 自由流通市值
            m: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: TURN_INST 因子值
        """
        # 日度机构换手率
        daily_turnover = (inflow_inst + outflow_inst) / cap

        # m 日滚动均值
        result = daily_turnover.rolling(window=m, min_periods=1).mean()

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# 机构交易热度 (TURN_INST) 衡量机构投资者在某只股票上的交易活跃程度。
# 机构交易通常定义为单笔成交金额大于 100 万元的交易。将每日机构买入金额
# 与卖出金额之和除以自由流通市值，得到日度机构换手率，再取过去 m 个交易日
# 的滚动均值，平滑短期波动。
#
# 机构交易占比高的股票通常定价效率较高，信息反映更充分。
# 该因子可用于：
#   - 识别机构关注度高的标的，辅助构建动量策略。
#   - 与散户交易热度对比，分析市场参与者结构。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.inst_trade_heat import InstTradeHeatFactor
#
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   inflow  = pd.DataFrame(1e6, index=dates, columns=stocks)
#   outflow = pd.DataFrame(8e5, index=dates, columns=stocks)
#   cap     = pd.DataFrame(1e9, index=dates, columns=stocks)
#
#   factor = InstTradeHeatFactor()
#   result = factor.compute(inflow_inst=inflow, outflow_inst=outflow, cap=cap, m=20)
#   print(result.tail())
