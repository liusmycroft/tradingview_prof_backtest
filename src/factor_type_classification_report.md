# 因子类型归类报告

## 1. 说明

本报告基于 `src/factor_classification_report.txt` 中出现的全部 296 个因子进行重新归类。

本次分类口径不是按 `FEASIBLE / PARTIAL / NOT_FEASIBLE` 的可实现性来分，而是按因子的**主导经济机制/交易机制**来分。这样做的目的，是为了后续更自然地做风格组合、策略拼接和风险分散。

需要说明两点：

- 很多因子天然具有跨风格属性，例如既带有“资金流”属性，也带有“反转”属性。为避免重复统计，本报告只将每个因子归入一个最主要的大类。
- 小类是“尝试性细分”，重点是帮助后续做组合设计，不追求学术上完全唯一、绝对严格的边界。

## 2. 一级分类总览

| 一级大类 | 核心含义 | 二级小类示例 |
|---|---|---|
| 趋势 / 动量 / 反转 | 刻画价格延续、时段路径与均值回归 | 隔夜-日内反转、趋势清晰度、路径/形态 |
| 波动 / 跳跃 / 尾部风险 | 刻画波动水平、跳跃冲击和分布尾部 | 上下行波动、跳跃风险、偏度峰度 |
| 成交量 / 价量结构 | 刻画成交在时间、价格区间和峰谷间的分布 | 异常放量、价量相关、峰谷结构 |
| 流动性 / 微观结构 / 知情交易 | 刻画价差、深度、冲击成本与交易毒性 | 价差深度、盘口失衡、知情交易 |
| 资金流 / 主力 / 订单方向 | 刻画主动买卖、大单行为和趋势资金 | 主买主卖、大单主力、支撑资金 |
| 注意力 / 情绪 / 羊群 | 刻画市场关注、从众、惊恐和模糊性厌恶 | 注意力、羊群跟风、惊恐/信噪比 |
| 行为金融 / 筹码 / 处置效应 | 刻画盈亏参照、筹码分布与交易心理 | 处置效应、筹码结构、博弈心理 |
| 关联网络 / 行业溢出 | 刻画行业、供应链、新闻和分析师关联传播 | 行业联动、供应链网络、外部信息关联 |

## 3. 详细分类

### A. 趋势 / 动量 / 反转

#### A1. 隔夜-日内反转

核心特征：利用隔夜收益、日内收益、极端收益或结构化分段来捕捉短期过度反应后的均值回归。

因子：
`AB_NR`, `AB_PR`, `NEGATIVE_REVERSAL_FREQ`, `POSITIVE_INTRADAY_REVERSAL_FREQ`, `CORRECTED_INTRADAY_REVERSAL`, `CORRECTED_OVERNIGHT_REVERSAL`, `GOLDEN_RATIO_REVERSAL`, `EXTREME_RETURN_REVERSAL`, `IDEAL_REVERSAL`, `SNR_ENHANCED_REVERSAL`, `STRUCTURED_REVERSAL`, `residual_reversal`, `REVERSAL_RESIDUAL_IMBALANCE`, `SIMILAR_REVERSE`

#### A2. 趋势延续与方向清晰度

核心特征：识别价格是否沿单一方向稳定推进，而不是高噪声震荡。

因子：
`TREND_CLARITY`, `TREND_CLARITY_MOMENTUM`, `TREND_RATIO`, `CLOSING_RETURN`, `SKEWED_MOMENTUM_RETURN`

#### A3. 时段路径与形态结构

核心特征：从日内时段差异、时间重心、K 线形态等角度描述趋势路径。

因子：
`APM`, `DROP_TIME_CENTROID`, `TIME_CENTROID`, `TIME_WEIGHTED_RPP`, `K_LINE_PATTERN`

### B. 波动 / 跳跃 / 尾部风险

#### B1. 波动水平与方向性波动

核心特征：关注波动率的强弱、上下行方向差异，以及剥离共性后的特异波动。

因子：
`CORRECTED_AMPLITUDE`, `IDEAL_AMPLITUDE`, `INTRADAY_AMPLITUDE_CUT`, `MINUTE_IDEAL_AMPLITUDE`, `DOWNSIDE_RV`, `HF_DOWNSIDE_VOL_RATIO`, `HF_IDIO_VOL`, `INTRADAY_RET_IV`, `INTRADAY_RET_VOL_RATIO`, `NONLINEAR_HF_VOLATILITY`, `OvernightRetIV`, `RBV`, `RTV`, `TURNOVER_IV`, `RS_PLUS`, `UPSIDE_VOL_RATIO`, `VCVaR`, `IDIO_TURNOVER_VOL`, `SIMILAR_LOW_VOL`

#### B2. 跳跃冲击与极端风险

核心特征：捕捉价格跳跃、隔夜冲击、极端上涨或回撤带来的非连续风险。

因子：
`AVG_POSITIVE_JUMP_RETURN`, `INTRADAY_JUMP_DROP`, `JAR`, `JUMP_ARRIVAL`, `LARGE_JUMP_ASYMMETRY`, `MAX_RISE`, `OVERNIGHT_GAP`, `REALIZED_JUMP_VOLATILITY`, `RVJN`, `RVLJP`, `SMALL_DOWNWARD_JUMP_VOL`, `SRVJ`, `TSRJV`, `INTRADAY_MAX_DRAWDOWN`

#### B3. 分布形态与尾部不对称

核心特征：通过偏度、峰度、上下行不对称等统计量刻画收益分布尾部。

因子：
`PEAK_CLIMBER`, `REALIZED_KURTOSIS`, `REALIZED_SKEWNESS`, `RSJ`, `WEIGHTED_SKEWNESS`

### C. 成交量 / 价量结构

#### C1. 异常成交量与时段分布

核心特征：看成交量是否异常放大、是否在某些时段聚集、是否具有持续性与不均匀性。

因子：
`ABNVOLD`, `AMOUNT_ENTROPY`, `AMT_MAX`, `CLOSING_VOLUME_RATIO`, `EARLY_LATE_COMPOSITE_RATIO`, `END_OF_DAY_AMOUNT`, `PATV`, `VMA`, `VOLUME_COEFF_VARIATION`, `VOLUME_ENTROPY`, `VOLUME_KURTOSIS`, `VOLUME_PEAK_COUNT`, `VOLUME_PEAK_MINUTES`, `VR`

#### C2. 价量相关与领先滞后

核心特征：刻画收益与成交量/成交额的同步、滞后、趋势性联动。

因子：
`CORA`, `CPV`, `CORA_A`, `CORA_R`, `DAZZLING_RETURN`, `LAG_ABS_RET_ADJ_AMOUNT_CORR`, `LARGE_VOL_PRICE_CORR`, `PV_CORR_TREND`, `PVOL_RET_CORR`, `WEIGHTED_CLOSE_RATIO`

#### C3. 高低位、峰谷与 VSA 结构

核心特征：从价格区间位置、峰谷分布、量岭量谷和支撑阻力的角度重建日内成交结构。

因子：
`ATD_PRICE_LOWEST`, `ATD_PRICE_LOWEST_SELL`, `B_TYPE_VOLUME_DIST`, `DAZZLING_VOL`, `DROP_MOMENT_ATD`, `DROP_SELL_ATD`, `HIGH_LOW_PRICE_COMPOSITE_RATIO`, `HIGH_LOW_VOL_SELECT`, `AMT_MIN`, `P_TYPE_VOLUME`, `PEAK_INTERVAL_KURTOSIS`, `PEAK_PRICE_QUANTILE`, `PEAK_RIDGE_CORR`, `PEAK_RIDGE_RATIO`, `RIDGE_GAP_SKEW`, `SATD_PRICE_VOL_CORR`, `SATD_VolumeHigh`, `TIDAL_PRICE_VELOCITY`, `VALLEY_RIDGE_RATIO`, `VALLEY_WEIGHTED_PRICE_QUANTILE`, `VOLUME_RIDGE_RELATIVE_VWAP`, `VOLUME_RIDGE_RETURN`, `VOLUME_SURGE_VOL`, `VOLUME_VALLEY_PRICE`, `VSA_CLOSE_DIFF`

### D. 流动性 / 微观结构 / 知情交易

#### D1. 价差、深度与冲击成本

核心特征：衡量交易是否容易成交、冲击成本是否高、盘口能否承接交易。

因子：
`ASK_DEPTH`, `AVG_ORDER_BOOK_DEPTH`, `BID_ASK_SPREAD`, `BID_DEPTH`, `BUY_ILLIQUIDITY`, `CORRECTED_AMBIGUITY_SPREAD`, `CPQSI`, `EFFECTIVE_DEPTH`, `EFFECTIVE_SPREAD`, `HLI`, `illiq`, `LIQUIDITY_PREMIUM_EW`, `MCI`, `MPC`, `MLQS`, `MPB`, `PRICE_IMPACT_BIAS`, `PRICE_RESILIENCY`, `PRSI`, `RESILIENCY`, `SELL_ILLIQUIDITY`, `WEIGHTED_LIQUIDITY_PREMIUM`

#### D2. 盘口失衡、委托结构与竞价行为

核心特征：关注委托簿中的买卖倾斜、净委买变化与集合竞价信号。

因子：
`CANCEL_RATE`, `MOFI`, `NET_COMMISSION_BUY`, `OBCVP`, `OPENING_BUY_INTENTION`, `OPENING_COMMISSION_RATIO`, `ORDER_IMBALANCE`, `ORDER_TRADE_CORRELATION`, `PIR`, `SOIR`, `TOXIC_LIQUIDITY`

#### D3. 知情交易与逐笔交易毒性

核心特征：通过订单切分、成交额分布和交易方向不平衡来识别信息优势资金。

因子：
`DPIN`, `DPIN_SMALL`, `LARGE_ORDER_DPIN`, `PEAK_VOL_SELL_AMT`, `QUA`, `SINGLE_AMOUNT_ENTROPY`, `VPIN`, `VOLUME_LONG_BIG_SELECT`, `VWPIN`

### E. 资金流 / 主力 / 订单方向

#### E1. 主买主卖与主动资金方向

核心特征：识别买卖方向是否偏向主动买入、主动卖出，或特定时段净买入是否占优。

因子：
`ACTIVE_BUY_RATIO`, `ACTIVE_BUY_SPECIFICITY`, `CONFIDENCE_NORMAL_ACTIVE_BUY`, `INFORMED_BUY_RATIO`, `INFORMED_SELL_RATIO`, `NAIVE_ACTIVE_RATIO`, `OPENING_NET_BUY`, `POST_OPEN_NET_BUY`, `POST_OPEN_LARGE_BUY`, `SIDEWAYS_BUY_ATD`, `SMALL_BUY_ACTIVE`, `T_DIST_ACTIVE_BUY`, `UNIFORM_ACTIVE_RATIO`

#### E2. 大单、主力与聪明钱

核心特征：刻画大单推进、主力活跃、聪明钱偏好以及不同资金体量间的力量变化。

因子：
`AVG_OUTFLOW_PER_TRADE_RATIO`, `BUY_ORDER_CONCENTRATION`, `CORRECTED_NET_INFLOW`, `IMPROVED_LARGE_RATIO`, `IMPROVED_LARGE_RATIO_ACTIVE`, `INST_TRADE_HEAT`, `LARGE_BUY_RATIO`, `LARGE_ORDER_RETURN`, `LCPS`, `LCVOL`, `MAIN_FORCE_STRENGTH`, `MAIN_FORCE_VOL`, `MORNING_INFLOW_STABILITY`, `MTE`, `NORMAL_BIG_RET`, `RESIDUAL_FLOW_STRENGTH`, `SMART_MONEY`, `SUPER_BIG_RET`, `SUPER_LARGE_BUY`, `TURN_RETAIL`

#### E3. 趋势资金与支撑力量

核心特征：观察价格背后是否存在持续承接、虹吸回流或趋势性支撑资金。

因子：
`NET_SUPPORT_VOLUME`, `SIPHON_EFFECT`, `TREND_CAPITAL_NET_SUPPORT`, `TREND_CAPITAL_VWAP`

### F. 注意力 / 情绪 / 羊群

#### F1. 注意力捕捉与事件冲击

核心特征：衡量个股是否因异常收益、事件、市场波动或舆情而被集中关注。

因子：
`ABNRETAVG`, `ABNRETD`, `ATTENTION_ABNORMAL_RETURN`, `ATTENTION_CAPTURE_VOL`, `ATTENTION_DECAY_PANIC`, `ATTENTION_EVENTS`, `ATTENTION_LIMIT`, `ATTENTION_MARKET_VOLUME`, `SPILL`, `ATTENTION_TURNOVER`, `DAILY_RETURN_ATTENTION`, `WEIGHTED_PROFIT_FREQ`

#### F2. 羊群、跟风与一致行为

核心特征：识别从众交易、抱团交易、跟随交易以及一致性买卖。

因子：
`ACMA`, `CONSISTENT_BUY_TRADE`, `CONSISTENT_SELL_TRADE`, `CSAD`, `CSAD_SMALL`, `CSSD`, `EXTREME_FOLLOW_RATIO`, `FLOW_FOLLOWING`, `FOLLOW_LEADER`, `HERDING_BEHAVIOR`, `LONE_GOOSE`, `LSV_HERDING`, `RETAIL_HERD`, `SLSV_HERDING`, `TCV`

#### F3. 惊恐、模糊性厌恶与信号质量

核心特征：刻画恐慌情绪、对模糊信息的厌恶，以及价格信号相对于噪声的清晰程度。

因子：
`FOG_AMOUNT_RATIO`, `FOG_VOLUME_RATIO`, `FUZZY_CORR`, `ID_MAG`, `INTRADAY_SNR`, `MORNING_MIST`, `MULTI_LAYER_SNR`, `NIGHT_FROST`, `NOON_ANCIENT_TREE`, `RETAIL_PANIC`, `SALIENCE_RETURN`, `VOL_PANIC`

### G. 行为金融 / 筹码 / 处置效应

#### G1. 盈亏参照与处置效应

核心特征：基于盈利/亏损状态、历史成本和前景理论刻画投资者“止盈拖损”行为。

因子：
`CPLR`, `CGO`, `GAIN_SELLING_TENDENCY`, `LOSS_SELLING`, `PROSPECT_TK`, `VNSP`

#### G2. 筹码分布与成本结构

核心特征：关注筹码集中度、成本重心、锁定筹码和换手结构。

因子：
`ACTIVE_CHIP_RATIO`, `afh_close`, `CHIP_CONCENTRATION`, `CHIP_DEVIATION_RATE`, `CHIP_DISTRIBUTION_CV`, `CHIP_DISTRIBUTION_KURTOSIS`, `CHIP_TURNOVER`, `ckdp`, `LOCKED_CHIPS_RATIO`, `PTR`, `TLRatio`, `UTD`

#### G3. 筹码行为与博弈心理

核心特征：观察盈利/亏损筹码的行为差异、上下影线、博弈力量和筹码增强效应。

因子：
`CHIP_DISPOSITION_EFFECT`, `CHIP_PNL_ADJUSTED_AFH`, `CHIP_RETURN_ENHANCE`, `ELSCorrelation`, `STREN`, `HCVOL`, `SHADOW_CLOSE_STD`, `VCDE`, `WEIGHTED_LOWER_SHADOW_FREQ`, `WEIGHTED_UPPER_SHADOW_FREQ`

### H. 关联网络 / 行业溢出

#### H1. 行业、相邻股票与相似公司联动

核心特征：用行业、相邻股票、相似公司或市场邻居来解释收益传播。

因子：
`CO_MIN`, `COMPLEX_MOMENTUM`, `GEO_MOMENTUM`, `INDUSTRY_MOMENTUM`, `JUMP_LINKAGE_RELATIVE_MOMENTUM`, `JUMP_NON_POSITIVE_MOMENTUM`, `LINKAGE`, `MODIFIED_BUSINESS_LINKAGE`, `NRBR_RET`, `RNBR_TOV`, `SIM_MOMENTUM`

#### H2. 供应链与网络中心性

核心特征：利用供应链拓扑、跳跃网络与空间网络中的中心地位刻画信息扩散。

因子：
`BIZ_COMPLEXITY_LINKAGE`, `BOTTLENECK_COMPANY`, `CUSTOMER_CENTRALITY_CHANGE`, `CUSTOMER_CLOSENESS`, `CUSTOMER_IMPORTANCE`, `CUSTOMER_INDUSTRY_CONCENTRATION`, `CMOM`, `CUSTOMER_MOMENTUM_ID`, `JUMP_DEGREE_CENTRALITY`, `KATZ_CENTRALITY`, `NETWORK_CENTRALITY`, `SCC`, `SMM`, `SUPPLIER_CONCENTRATION`, `SUPPLY_CENTRALITY_CHANGE`, `SUPPLY_CHAIN_DEGREE`, `SUPPLY_CHAIN_POSITION`, `TCC`, `UpstreamMomentum`, `UPSTREAM_TRANSMISSION`

#### H3. 分析师、新闻、搜索与海外关联

核心特征：借助分析师覆盖、新闻共现、搜索关联、论坛讨论和海外敞口构造外部联动。

因子：
`CAF_EP`, `ANALYST_CO_COVERAGE_INDIRECT_MOMENTUM`, `FOREIGN_OPS`, `IMPROVED_ANALYST_COVERAGE_MOMENTUM`, `MRR`, `NEWS_NETWORK_LEAD_RETURN`, `SEARCH_RATIO_DOWNSTREAM`

## 4. 归类后的几点结论

### 4.1 这批因子的主体风格非常偏“高频微观结构 + 行为金融”

从全量分布看，`成交量/价量结构`、`流动性/微观结构/知情交易`、`资金流/主力/订单方向` 三大类占比明显更高，说明这份因子库本质上不是传统低频基本面因子库，而是更偏交易行为与市场微观结构的阿尔法库。

### 4.2 “注意力/情绪/羊群”和“行为金融/筹码/处置效应”是很强的补充层

这两类因子不一定直接决定方向，但很适合作为：

- 趋势策略的拥挤度过滤器
- 反转策略的过度反应确认器
- 风险控制中的情绪温度计

### 4.3 “关联网络 / 行业溢出”更适合做主题轮动和联动传播

这类因子单独使用时容易受数据口径影响，但和趋势、资金流、注意力类因子结合时，往往能把“为什么会继续走”这件事解释得更完整。

### 4.4 当前可落地策略应优先使用 FEASIBLE + PARTIAL 因子

虽然本报告做了全量风格归类，但若考虑 TradingView / Pine Script 或现有数据栈的真实可落地性，优先级应当是：

1. `FEASIBLE` 作为核心信号
2. `PARTIAL` 作为确认层或风格增强层
3. `NOT_FEASIBLE` 作为未来扩展方向，而不是当前回测主干

## 5. 建议的后续用法

- 先从每个一级大类里各选 1 到 2 个最稳定的因子，做低相关组合。
- 策略设计时避免同类因子堆叠过多，例如同时堆很多“成交量峰谷类”因子，容易导致看起来是多风格，实际上仍是单一交易结构暴露。
- 如果后续要做正式组合优化，建议先按本报告的大类做组内去相关，再做组间配比。
