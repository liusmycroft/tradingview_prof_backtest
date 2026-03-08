import pandas as pd

from factors.base import BaseFactor


class LeadingVolumeAnomalyAtMinFactor(BaseFactor):
    """个股收益最低时的领先成交量异动因子 (AMT_MIN)"""

    name = "AMT_MIN"
    category = "高频动量反转"
    description = "个股收益最低分钟的前一分钟成交量占比之和与均值之比，衡量下跌时的成交量异动"

    def compute(
        self,
        daily_amt_min: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算个股收益最低时的领先成交量异动因子。

        AMT_MIN 的日度值由分钟级数据预计算：
        - 找到日内收益率最低的 M 个分钟
        - 计算这些分钟前一分钟的成交额占比之和
        - 除以日内平均成交额占比

        此处接收预计算的日度 AMT_MIN 值，取 T 日滚动均值。

        Args:
            daily_amt_min: 预计算的每日 AMT_MIN 值 (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: AMT_MIN 的 T 日滚动均值
        """
        result = daily_amt_min.rolling(window=T, min_periods=1).mean()
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# AMT_MIN 衡量了股价大幅下跌时的成交量在全市场成交量中的占比情况。
# 若个股的股价下跌没有伴随市场成交量异动，而是由自身成交量异动造成的
# （因子值越大，占比越大），未在成交量上与市场形成有效共振关系，
# 则容易受到自身交易过热的反噬，未来收益表现较弱。
