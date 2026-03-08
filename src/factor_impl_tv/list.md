# 因子 Pine Script 实现进度

## 统计

| 分类 | 数量 | 说明 |
|------|------|------|
| FEASIBLE (已实现) | 24 | 仅需日线 OHLCV，已全部转换为 Pine Script v6 strategy |
| PARTIAL (可近似) | 157 | 原始依赖分钟级预计算数据或截面操作，可用日线近似但不精确 |
| NOT_FEASIBLE (无法实现) | 115 | 需要订单簿/筹码分布/供应链/分析师等 TradingView 不提供的数据 |

---

## FEASIBLE — 已完成 (24/24)

| # | 文件名 | 因子代码 | 说明 | 状态 |
|---|--------|----------|------|------|
| 1 | abnormal_negative_reversal.pine | AB_NR | 反向日内逆转异常频率，当月频率/过去12月均值 | ✅ |
| 2 | abnormal_positive_reversal.pine | AB_PR | 正向日内逆转异常频率，短期频率-长期频率 | ✅ |
| 3 | abnormal_return_avg.pine | ABNRETAVG | 异常收益率平方的滚动均值 | ✅ |
| 4 | abnormal_volume_daily.pine | ABNVOLD | 日成交量/年均成交量的滚动最大值 | ✅ |
| 5 | abnretd.pine | ABNRETD | 异常收益绝对值的滚动最大值 | ✅ |
| 6 | attention_abnormal_return.pine | ATTENTION_ABNORMAL_RETURN | 异常收益绝对值的滚动均值 | ✅ |
| 7 | attention_events.pine | ATTENTION_EVENTS | 涨停跌停事件滚动计数 | ✅ |
| 8 | corrected_amplitude.pine | CORRECTED_AMPLITUDE | 跳空方向修正的日内振幅 | ✅ |
| 9 | corrected_intraday_reversal.pine | CORRECTED_INTRADAY_REVERSAL | 波动率修正的日内反转 | ✅ |
| 10 | csad_small.pine | CSAD_SMALL | CSAD小型板块因子(截面近似) | ✅ |
| 11 | cssd.pine | CSSD | 截面收益标准差(截面近似) | ✅ |
| 12 | daily_return_attention.pine | DAILY_RETURN_ATTENTION | 收益率偏差注意力度量 | ✅ |
| 13 | ideal_amplitude.pine | IDEAL_AMPLITUDE | 高价日与低价日振幅之差 | ✅ |
| 14 | negative_reversal_freq.pine | NEGATIVE_REVERSAL_FREQ | 隔夜负+日内负的频率 | ✅ |
| 15 | overnight_gap.pine | OVERNIGHT_GAP | 隔夜对数收益率绝对值滚动求和 | ✅ |
| 16 | positive_intraday_reversal_freq.pine | POSITIVE_INTRADAY_REVERSAL_FREQ | 隔夜负+日内正的频率 | ✅ |
| 17 | shadow_close_std.pine | SHADOW_CLOSE_STD | 影线/收盘价比率的标准差 | ✅ |
| 18 | similar_low_vol.pine | SIMILAR_LOW_VOL | 相似低波因子(简化版) | ✅ |
| 19 | similar_reverse.pine | SIMILAR_REVERSE | 相似反转因子(简化版) | ✅ |
| 20 | trend_clarity.pine | TREND_CLARITY | 价格对时间回归R² | ✅ |
| 21 | trend_clarity_momentum.pine | TREND_CLARITY_MOMENTUM | 动量与趋势清晰度匹配度 | ✅ |
| 22 | weighted_lower_shadow_freq.pine | WEIGHTED_LOWER_SHADOW_FREQ | 衰减加权下影线频率 | ✅ |
| 23 | weighted_profit_freq.pine | WEIGHTED_PROFIT_FREQ | 衰减加权盈利频率 | ✅ |
| 24 | weighted_upper_shadow_freq.pine | WEIGHTED_UPPER_SHADOW_FREQ | 衰减加权上影线频率 | ✅ |

### 实现说明

- 需要市场基准的因子(ABNRETAVG, ABNRETD, ATTENTION_ABNORMAL_RETURN等)通过 `input.symbol` + `request.security` 获取基准指数数据
- 原始因子中涉及截面操作的(CSAD_SMALL, CSSD, CORRECTED_AMPLITUDE, CORRECTED_INTRADAY_REVERSAL)在代码中用时间序列均值近似，并标注了说明
- SIMILAR_LOW_VOL 和 SIMILAR_REVERSE 原始因子需要历史模式匹配，Pine Script中做了简化近似

---

## PARTIAL — 已完成 (157/157)

这些因子的原始实现依赖分钟级预计算数据或截面操作，已全部用日线近似实现。近似方式：
- 分钟级数据 → 用日线 OHLCV 近似
- 截面操作 → 用时间序列统计近似
- 预计算特征 → 用可获取的日线数据重新构造

### 按数据依赖分类

#### 分钟级预计算数据 (约120个)
abs_ret_amount_corr, acma, amount_entropy, amt_max, atd_price_lowest, avg_outflow_per_trade_ratio, avg_positive_jump_return, b_type_volume_dist, buy_illiquidity, buy_order_concentration, capital_loss_realization, cgo, closing_return, closing_volume_ratio, confidence_normal_active_buy, consistent_trading_volume, cora_abs, cora_r, corrected_net_inflow, corrected_overnight_reversal, dazzling_vol, downside_rv, drop_moment_atd, early_late_composite_ratio, end_of_day_amount, extreme_follow_ratio, flow_following, fog_amount_ratio, fog_volume_ratio, follow_leader, fuzzy_corr, gain_selling_tendency, golden_ratio_reversal, hcvol, hf_downside_vol_ratio, hf_idio_vol, high_low_price_composite, high_low_spread_intensity, high_low_vol_select, ideal_reversal, illiq, improved_large_ratio, intraday_jump_drop, intraday_max_drawdown, intraday_ret_vol_ratio, intraday_snr, jump_absolute_return, jump_arrival, jump_non_positive_momentum, k_line_pattern, lag_abs_ret_adj_amount_corr, large_jump_asymmetry, large_order_return, large_vol_price_corr, lcps, lcvol, leading_volume_anomaly_at_min, lone_goose, loss_selling, main_force_strength, main_force_vol, max_rise, minute_amount_variance, minute_ideal_amplitude, morning_inflow_stability, morning_mist, mte, naive_active_ratio, net_support_volume, p_type_volume, peak_climber, peak_interval_kurtosis, peak_ridge_corr, peak_ridge_ratio, price_impact_bias, price_resiliency, prsi, pvol_ret_corr, realized_bipower_variation, realized_jump_volatility, realized_kurtosis, realized_skewness, residual_flow_strength, resiliency, reversal_residual_imbalance, ridge_gap_skew, rsj, rtv, rvjn, rvljp, salience_return, sell_illiquidity, siphon_effect, skewed_momentum_return, small_downward_jump_vol, smart_money, snr_enhanced_reversal, srvj, structured_reversal, tidal_price_velocity, time_weighted_rpp, trend_capital_net_support, trend_capital_vwap, trend_ratio, tsrjv, uniform_active_ratio, upside_realized_vol, upside_vol_ratio, v_shaped_disposition, valley_ridge_ratio, valley_weighted_price_quantile, vcvar, vol_panic, volume_coeff_variation, volume_entropy, volume_kurtosis, volume_peak_count, volume_peak_minutes, volume_ratio, volume_ridge_relative_vwap, volume_ridge_return, volume_surge_vol, volume_valley_price, vsa_close_diff, weighted_close_ratio, weighted_skewness

#### 分钟级 + 截面操作 (约25个)
apm, attention_capture_vol, attention_limit, attention_market_volume, co_min, composite_pv_corr, dazzling_return, drop_time_centroid, extreme_return_reversal, intraday_amplitude_cut, intraday_ret_iv, multi_layer_snr, night_frost, nonlinear_hf_volatility, noon_ancient_tree, patv, peak_price_quantile, pv_corr_trend, time_centroid, volume_surge_vol, weighted_liquidity_premium

#### 截面操作 (约12个)
csad, id_mag, idio_turnover_vol, industry_momentum, liquidity_premium_ew, overnight_ret_iv, residual_reversal, sentiment_spillover, turnover_iv, consistent_buy_trade, consistent_sell_trade

---

## NOT_FEASIBLE — 无法实现 (115个)

### 按缺失数据类型分类

| 数据类型 | 数量 | 代表因子 |
|----------|------|----------|
| 筹码分布数据 | 15 | chip_concentration, chip_turnover, locked_chips_ratio... |
| 主买主卖分类 | 14 | active_buy_ratio, informed_buy_ratio, large_buy_ratio... |
| 订单簿/委托数据 | 14 | bid_ask_spread, effective_spread, order_imbalance... |
| 供应链/业务关联 | 14 | linkage, supply_chain_position, customer_momentum... |
| 网络/图数据 | 10 | network_centrality, katz_centrality, bottleneck_company... |
| 逐笔成交数据 | 13 | dpin, vpin, toxic_liquidity, mci... |
| 投资者资金流 | 8 | herding_behavior, lsv_herding, retail_herd... |
| 基本面/另类数据 | 8 | complex_momentum, foreign_ops, geo_momentum... |
| 新闻/舆情数据 | 5 | attention_decay_panic, attention_spillover, mrr... |
| 分析师数据 | 3 | analyst_anchoring_bias, analyst_co_coverage... |
| 集合竞价数据 | 5 | cancel_rate, volume_proportion_composite... |
| 订单簿+成交关联 | 4 | order_trade_correlation, soir... |
