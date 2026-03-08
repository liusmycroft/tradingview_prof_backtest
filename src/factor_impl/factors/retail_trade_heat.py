import pandas as pd

from factors.base import BaseFactor


class RetailTradeHeatFactor(BaseFactor):
    """个人投资者交易热度因子 (TURN_RETAIL)"""

    name = "TURN_RETAIL"
    category = "资金流向"
    description = "个人投资者交易热度：散户资金流入流出占自由流通市值的滚动均值"

    def compute(
        self,
        inflow_retail: pd.DataFrame,
        outflow_retail: pd.DataFrame,
        cap: pd.DataFrame,
        m: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算 TURN_RETAIL 因子。

        TURN_RETAIL = (1/m) * sum_{t=1}^{m} (INFLOW_RETAIL + OUTFLOW_RETAIL) / CAP

        Args:
            inflow_retail: 散户买入金额 (index=日期, columns=股票代码)
            outflow_retail: 散户卖出金额
            cap: 自由流通市值
            m: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: TURN_RETAIL 因子值
        """
        # 日度散户换手率
        daily_turnover = (inflow_retail + outflow_retail) / cap

        # m 日滚动均值
        result = daily_turnover.rolling(window=m, min_periods=1).mean()

        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子逻辑】
# 个人投资者交易热度 (TURN_RETAIL) 衡量散户投资者在某只股票上的交易活跃程度。
# 散户交易通常定义为单笔成交金额小于 4 万元的交易。将每日散户买入金额与
# 卖出金额之和除以自由流通市值，得到日度散户换手率，再取过去 m 个交易日
# 的滚动均值，平滑短期波动。
#
# 【选股含义】
# 散户交易占比高的股票往往定价效率较低，存在更多的错误定价机会。
# 该因子可用于构建反转类或情绪类策略：散户交易热度过高的股票，
# 短期内可能面临回调压力；反之，散户关注度低的股票可能被低估。
#
# 【使用示例】
#
#   import pandas as pd
#   from factors.retail_trade_heat import RetailTradeHeatFactor
#
#   # 构造示例数据
#   dates = pd.date_range("2024-01-01", periods=30, freq="B")
#   stocks = ["000001.SZ", "000002.SZ"]
#
#   inflow  = pd.DataFrame(100.0, index=dates, columns=stocks)
#   outflow = pd.DataFrame(80.0,  index=dates, columns=stocks)
#   cap     = pd.DataFrame(1e6,   index=dates, columns=stocks)
#
#   factor = RetailTradeHeatFactor()
#   result = factor.compute(inflow_retail=inflow, outflow_retail=outflow, cap=cap, m=20)
#   print(result.tail())
