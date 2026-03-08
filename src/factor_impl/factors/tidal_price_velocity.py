import numpy as np
import pandas as pd

from factors.base import BaseFactor


class TidalPriceVelocityFactor(BaseFactor):
    """成交量潮汐的价格变动速率因子 (Tidal Price Velocity)"""

    name = "TIDAL_PRICE_VELOCITY"
    category = "高频量价相关性"
    description = "成交量潮汐过程中的价格变动速率，衡量投资者出售或购买意愿的强烈程度"

    def compute(
        self,
        daily_tidal_velocity: pd.DataFrame,
        T: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """计算成交量潮汐的价格变动速率因子。

        日内预计算逻辑:
          1. 计算每分钟的领域成交量(前后4分钟共9分钟之和)
          2. 找到领域成交量最高点(顶峰时刻t)
          3. 顶峰前的领域成交量最低点为涨潮时刻m
          4. 顶峰后的领域成交量最低点为退潮时刻n
          5. 全潮汐价格变动速率 = (C_n - C_m) / (C_m / (n - m))

        Args:
            daily_tidal_velocity: 预计算的每日潮汐价格变动速率
                (index=日期, columns=股票代码)
            T: 滚动窗口天数，默认 20

        Returns:
            pd.DataFrame: T 日滚动均值
        """
        result = daily_tidal_velocity.rolling(window=T, min_periods=1).mean()
        return result

    @staticmethod
    def compute_daily(minute_volume: pd.Series, minute_close: pd.Series,
                      neighborhood: int = 4) -> float:
        """从单日分钟数据计算潮汐价格变动速率。

        Args:
            minute_volume: 分钟成交量 (index=分钟编号, 剔除开盘收盘)
            minute_close: 分钟收盘价
            neighborhood: 领域半径，默认4

        Returns:
            float: 全潮汐价格变动速率
        """
        n = len(minute_volume)
        if n < 2 * neighborhood + 3:
            return np.nan

        # 计算领域成交量
        vol_arr = minute_volume.values.astype(float)
        domain_vol = np.convolve(vol_arr, np.ones(2 * neighborhood + 1), mode='same')

        # 顶峰时刻
        peak_idx = np.argmax(domain_vol)

        # 涨潮时刻: peak之前的最低点
        if peak_idx <= 0:
            return np.nan
        m_idx = np.argmin(domain_vol[:peak_idx])

        # 退潮时刻: peak之后的最低点
        if peak_idx >= n - 1:
            return np.nan
        n_idx = peak_idx + 1 + np.argmin(domain_vol[peak_idx + 1:])

        close_arr = minute_close.values.astype(float)
        c_m = close_arr[m_idx]
        c_n = close_arr[n_idx]
        duration = n_idx - m_idx

        if duration == 0 or c_m == 0:
            return np.nan

        velocity = (c_n - c_m) / (c_m / duration)
        return velocity
