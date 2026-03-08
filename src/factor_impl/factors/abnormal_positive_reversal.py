import pandas as pd

from factors.base import BaseFactor


class AbnormalPositiveReversalFactor(BaseFactor):
    """正向日内逆转的异常频率因子 (Abnormal Positive Reversal, AB_PR)"""

    name = "AB_PR"
    category = "高频收益分布"
    description = "正向日内逆转的异常频率，即正向逆转频率减去其长期均值，衡量近期逆转行为的异常程度"

    def compute(
        self,
        ret_co: pd.DataFrame,
        ret_oc: pd.DataFrame,
        T_short: int = 20,
        T_long: int = 60,
        **kwargs,
    ) -> pd.DataFrame:
        """计算正向日内逆转的异常频率因子。

        公式:
            indicator = I{ret_co < 0} * I{ret_oc > 0}
            PR_short = rolling_mean(indicator, T_short)
            PR_long  = rolling_mean(indicator, T_long)
            AB_PR = PR_short - PR_long

        Args:
            ret_co: 隔夜收益率 (close-to-open), index=日期, columns=股票代码
            ret_oc: 日内收益率 (open-to-close), index=日期, columns=股票代码
            T_short: 短期滚动窗口天数，默认 20
            T_long: 长期滚动窗口天数，默认 60

        Returns:
            pd.DataFrame: 正向日内逆转的异常频率
        """
        indicator = ((ret_co < 0) & (ret_oc > 0)).astype(float)

        # 保持原始 NaN
        mask = ret_co.isna() | ret_oc.isna()
        indicator[mask] = float("nan")

        pr_short = indicator.rolling(window=T_short, min_periods=T_short).mean()
        pr_long = indicator.rolling(window=T_long, min_periods=T_long).mean()

        result = pr_short - pr_long
        return result


# ============================================================================
# 核心思想与原理说明
# ============================================================================
#
# 【因子含义】
# AB_PR 衡量正向日内逆转频率相对于其长期水平的异常偏离。
# 正向日内逆转指隔夜收益为负但日内收益为正的情况。
# 当短期逆转频率显著高于长期均值时，说明近期市场对隔夜利空的
# 日内修正行为异常频繁，可能预示着过度修正，未来收益偏低。
