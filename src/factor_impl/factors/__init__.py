from factors.base import BaseFactor
from factors.abnormal_negative_reversal import AbnormalNegativeReversalFactor
from factors.abnormal_positive_reversal import AbnormalPositiveReversalFactor
from factors.abnormal_return_avg import AbnormalReturnAvgFactor
from factors.abnormal_volume_daily import AbnormalVolumeDailyFactor
from factors.abnretd import ABNRETDFactor
from factors.abs_ret_amount_corr import AbsRetAmountCorrFactor
from factors.acma import ACMAFactor
from factors.active_buy_ratio import ActiveBuyRatioFactor
from factors.active_buy_specificity import ActiveBuySpecificityFactor
from factors.active_chip_ratio import ActiveChipRatioFactor
from factors.afh_close import AFHCloseFactor
from factors.amount_entropy import AmountEntropyFactor
from factors.amt_max import AmtMaxFactor
from factors.analyst_anchoring_bias import AnalystAnchoringBiasFactor
from factors.analyst_co_coverage_indirect_momentum import AnalystCoCoverageIndirectMomentumFactor
from factors.apm import APMFactor
from factors.ask_depth import AskDepthFactor
from factors.atd_price_lowest import ATDPriceLowestFactor
from factors.atd_price_lowest_sell import ATDPriceLowestSellFactor
from factors.attention_abnormal_return import AttentionAbnormalReturnFactor
from factors.attention_capture_vol import AttentionCaptureVolFactor
from factors.attention_decay_panic import AttentionDecayPanicFactor
from factors.attention_events import AttentionEventsFactor
from factors.attention_limit import AttentionLimitFactor
from factors.attention_market_volume import AttentionMarketVolumeFactor
from factors.attention_spillover import AttentionSpilloverFactor
from factors.attention_turnover import AttentionTurnoverFactor
from factors.avg_order_book_depth import AvgOrderBookDepthFactor
from factors.avg_outflow_per_trade_ratio import AvgOutflowPerTradeRatioFactor
from factors.avg_positive_jump_return import AvgPositiveJumpReturnFactor
from factors.b_type_volume_dist import BTypeVolumeDistFactor
from factors.bid_ask_price_ratio import BidAskPriceRatioFactor
from factors.bid_ask_spread import BidAskSpreadFactor
from factors.bid_depth import BidDepthFactor
from factors.bottleneck_company import BottleneckCompanyFactor
from factors.business_complexity_linkage import BusinessComplexityLinkageFactor
from factors.buy_illiquidity import BuyIlliquidityFactor
from factors.buy_order_concentration import BuyOrderConcentrationFactor
from factors.cancel_rate import CancelRateFactor
from factors.capital_loss_realization import CapitalLossRealizationFactor
from factors.cgo import CGOFactor
from factors.chip_concentration import ChipConcentrationFactor
from factors.chip_deviation_rate import ChipDeviationRateFactor
from factors.chip_disposition_effect import ChipDispositionEffectFactor
from factors.chip_distribution_cv import ChipDistributionCVFactor
from factors.chip_distribution_shape import ChipDistributionKurtosisFactor
from factors.chip_pnl_adjusted_afh import ChipPnlAdjustedAFHFactor
from factors.chip_return_enhance import ChipReturnEnhanceFactor
from factors.chip_turnover import ChipTurnoverFactor
from factors.ckdp import CKDPFactor
from factors.closing_return import ClosingReturnFactor
from factors.closing_volume_ratio import ClosingVolumeRatioFactor
from factors.co_min import COMinFactor
from factors.complex_momentum import ComplexMomentumFactor
from factors.composite_pv_corr import CompositePVCorrFactor
from factors.confidence_normal_active_buy import ConfidenceNormalActiveBuyFactor
from factors.consistent_buy_trade import ConsistentBuyTradeFactor
from factors.consistent_sell_trade import ConsistentSellTradeFactor
from factors.consistent_trading_volume import ConsistentTradingVolumeFactor
from factors.cora_abs import CoraAbsFactor
from factors.cora_r import CoraRFactor
from factors.corrected_ambiguity_spread import CorrectedAmbiguitySpreadFactor
from factors.corrected_amplitude import CorrectedAmplitudeFactor
from factors.corrected_intraday_reversal import CorrectedIntradayReversalFactor
from factors.corrected_net_inflow import CorrectedNetInflowFactor
from factors.corrected_overnight_reversal import CorrectedOvernightReversalFactor
from factors.cpqsi import CPQSIFactor
from factors.csad import CSADFactor
from factors.csad_small import CSADSmallFactor
from factors.cssd import CSSDFactor
from factors.customer_centrality_change import CustomerCentralityChangeFactor
from factors.customer_closeness import CustomerClosenessFactor
from factors.customer_importance import CustomerImportanceFactor
from factors.customer_industry_concentration import CustomerIndustryConcentrationFactor
from factors.customer_momentum import CustomerMomentumFactor
from factors.customer_momentum_id import CustomerMomentumIDFactor
from factors.daily_return_attention import DailyReturnAttentionFactor
from factors.dazzling_return import DazzlingReturnFactor
from factors.dazzling_vol import DazzlingVolFactor
from factors.downside_rv import DownsideRVFactor
from factors.dpin import DpinFactor
from factors.dpin_small import DpinSmallFactor
from factors.drop_moment_atd import DropMomentATDFactor
from factors.drop_sell_atd import DropSellATDFactor
from factors.drop_time_centroid import DropTimeCentroidFactor
from factors.early_late_composite_ratio import EarlyLateCompositeRatioFactor
from factors.effective_depth import EffectiveDepthFactor
from factors.effective_spread import EffectiveSpreadFactor
from factors.els_correlation import ELSCorrelationFactor
from factors.end_of_day_amount import EndOfDayAmountFactor
from factors.extreme_follow_ratio import ExtremeFollowRatioFactor
from factors.extreme_return_reversal import ExtremeReturnReversalFactor
from factors.flow_following import FlowFollowingFactor
from factors.fog_amount_ratio import FogAmountRatioFactor
from factors.fog_volume_ratio import FogVolumeRatioFactor
from factors.follow_leader import FollowLeaderFactor
from factors.foreign_ops import ForeignOpsFactor
from factors.fuzzy_corr import FuzzyCorrFactor
from factors.gain_selling_tendency import GainSellingTendencyFactor
from factors.gaming_factor import GamingFactor
from factors.geo_momentum import GeoMomentumFactor
from factors.golden_ratio_reversal import GoldenRatioReversalFactor
from factors.hcvol import HCVOLFactor
from factors.herding_behavior import HerdingBehaviorFactor
from factors.hf_downside_vol_ratio import HFDownsideVolRatioFactor
from factors.hf_idio_vol import HfIdioVolFactor
from factors.high_low_price_composite import HighLowPriceCompositeRatioFactor
from factors.high_low_spread_intensity import HighLowSpreadIntensityFactor
from factors.high_low_vol_select import HighLowVolSelectFactor
from factors.id_mag import IDMagFactor
from factors.ideal_amplitude import IdealAmplitudeFactor
from factors.ideal_reversal import IdealReversalFactor
from factors.idio_turnover_vol import IdioTurnoverVolFactor
from factors.illiq import ILLIQFactor
from factors.improved_analyst_coverage_momentum import ImprovedAnalystCoverageMomentumFactor
from factors.improved_large_ratio import ImprovedLargeRatioFactor
from factors.improved_large_ratio_active import ImprovedLargeRatioActiveFactor
from factors.industry_momentum import IndustryMomentumFactor
from factors.informed_buy_ratio import InformedBuyRatioFactor
from factors.informed_sell_ratio import InformedSellRatioFactor
from factors.inst_trade_heat import InstTradeHeatFactor
from factors.intraday_amplitude_cut import IntradayAmplitudeCutFactor
from factors.intraday_jump_drop import IntradayJumpDropFactor
from factors.intraday_max_drawdown import IntradayMaxDrawdownFactor
from factors.intraday_ret_iv import IntradayRetIVFactor
from factors.intraday_ret_vol_ratio import IntradayReturnVolRatioFactor
from factors.intraday_snr import IntradaySNRFactor
from factors.jump_absolute_return import JumpAbsoluteReturnFactor
from factors.jump_arrival import JumpArrivalFactor
from factors.jump_degree_centrality import JumpDegreeCentralityFactor
from factors.jump_linkage_relative_momentum import JumpLinkageRelativeMomentumFactor
from factors.jump_non_positive_momentum import JumpNonPositiveMomentumFactor
from factors.k_line_pattern import KLinePatternFactor
from factors.katz_centrality import KatzCentralityFactor
from factors.lag_abs_ret_adj_amount_corr import LagAbsRetAdjAmountCorrFactor
from factors.large_buy_ratio import LargeBuyRatioFactor
from factors.large_jump_asymmetry import LargeJumpAsymmetryFactor
from factors.large_order_dpin import LargeOrderDpinFactor
from factors.large_order_return import LargeOrderReturnFactor
from factors.large_vol_price_corr import LargeVolPriceCorrFactor
from factors.lcps import LCPSFactor
from factors.lcvol import LCVOLFactor
from factors.leading_volume_anomaly_at_min import LeadingVolumeAnomalyAtMinFactor
from factors.linkage import LinkageFactor
from factors.liquidity_premium_ew import LiquidityPremiumEWFactor
from factors.locked_chips_ratio import LockedChipsRatioFactor
from factors.lone_goose import LoneGooseFactor
from factors.loss_selling import LossSellingFactor
from factors.lsv_herding import LSVHerdingFactor
from factors.main_force_strength import MainForceStrengthFactor
from factors.main_force_vol import MainForceVolFactor
from factors.max_rise import MaxRiseFactor
from factors.mci import MCIFactor
from factors.mid_price_change import MidPriceChangeFactor
from factors.minute_amount_variance import MinuteAmountVarianceFactor
from factors.minute_ideal_amplitude import MinuteIdealAmplitudeFactor
from factors.mlqs import MLQSFactor
from factors.modified_business_linkage import ModifiedBusinessLinkageFactor
from factors.mofi import MOFIFactor
from factors.morning_inflow_stability import MorningInflowStabilityFactor
from factors.morning_mist import MorningMistFactor
from factors.mpb import MPBFactor
from factors.mrr import MRRFactor
from factors.mte import MTEFactor
from factors.multi_layer_snr import MultiLayerSNRFactor
from factors.naive_active_ratio import NaiveActiveRatioFactor
from factors.negative_reversal_freq import NegativeReversalFreqFactor
from factors.net_commission_buy import NetCommissionBuyFactor
from factors.net_support_volume import NetSupportVolumeFactor
from factors.network_centrality import NetworkCentralityFactor
from factors.news_network_lead_return import NewsNetworkLeadReturnFactor
from factors.night_frost import NightFrostFactor
from factors.nonlinear_hf_volatility import NonlinearHFVolatilityFactor
from factors.noon_ancient_tree import NoonAncientTreeFactor
from factors.normal_big_ret import NormalBigRetFactor
from factors.nrbr_ret import NRBRRetFactor
from factors.opening_buy_intention import OpeningBuyIntentionFactor
from factors.opening_commission_ratio import OpeningCommissionRatioFactor
from factors.opening_net_buy import OpeningNetBuyFactor
from factors.order_imbalance import OrderImbalanceFactor
from factors.order_trade_correlation import OrderTradeCorrelationFactor
from factors.overnight_gap import OvernightGapFactor
from factors.overnight_ret_iv import OvernightRetIVFactor
from factors.p_type_volume import PTypeVolumeFactor
from factors.patv import PATVFactor
from factors.peak_climber import PeakClimberFactor
from factors.peak_interval_kurtosis import PeakIntervalKurtosisFactor
from factors.peak_price_quantile import PeakPriceQuantileFactor
from factors.peak_ridge_corr import PeakRidgeCorrFactor
from factors.peak_ridge_ratio import PeakRidgeRatioFactor
from factors.peak_vol_sell_amt import PeakVolSellAmtFactor
from factors.positive_intraday_reversal_freq import PositiveIntradayReversalFreqFactor
from factors.post_open_large_buy import PostOpenLargeBuyFactor
from factors.post_open_net_buy import PostOpenNetBuyFactor
from factors.price_impact_bias import PriceImpactBiasFactor
from factors.price_resiliency import PriceResiliencyFactor
from factors.prospect_tk import ProspectTKFactor
from factors.prsi import PRSIFactor
from factors.ptr import PTRFactor
from factors.pv_corr_trend import PVCorrTrendFactor
from factors.pvol_ret_corr import PvolRetCorrFactor
from factors.qua import QUAFactor
from factors.realized_bipower_variation import RealizedBipowerVariationFactor
from factors.realized_jump_volatility import RealizedJumpVolatilityFactor
from factors.realized_kurtosis import RealizedKurtosisFactor
from factors.realized_skewness import RealizedSkewnessFactor
from factors.residual_flow_strength import ResidualFlowStrengthFactor
from factors.residual_reversal import ResidualReversalFactor
from factors.resiliency import ResiliencyFactor
from factors.retail_herd import RetailHerdFactor
from factors.retail_panic import RetailPanicFactor
from factors.retail_trade_heat import RetailTradeHeatFactor
from factors.reversal_residual_imbalance import ReversalResidualImbalanceFactor
from factors.ridge_gap_skew import RidgeGapSkewFactor
from factors.rsj import RSJFactor
from factors.rtv import RTVFactor
from factors.rvjn import RVJNFactor
from factors.rvljp import RVLJPFactor
from factors.salience_return import SalienceReturnFactor
from factors.satd_price_vol_corr import SatdPriceVolCorrFactor
from factors.satd_volume_high import SATDVolumeHighFactor
from factors.scc import SCCFactor
from factors.search_ratio_downstream import SearchRatioDownstreamFactor
from factors.sell_illiquidity import SellIlliquidityFactor
from factors.sentiment_spillover import SentimentSpilloverFactor
from factors.shadow_close_std import ShadowCloseStdFactor
from factors.sideways_buy_atd import SidewaysBuyATDFactor
from factors.similar_low_vol import SimilarLowVolFactor
from factors.similar_reverse import SimilarReverseFactor
from factors.similarity_momentum import SimilarityMomentumFactor
from factors.single_amount_entropy import SingleAmountEntropyFactor
from factors.siphon_effect import SiphonEffectFactor
from factors.skewed_momentum_return import SkewedMomentumReturnFactor
from factors.slsv_herding import SLSVHerdingFactor
from factors.small_buy_active import SmallBuyActiveFactor
from factors.small_downward_jump_vol import SmallDownwardJumpVolFactor
from factors.small_order_lag_corr import SmallOrderLagCorrFactor
from factors.smart_money import SmartMoneyFactor
from factors.smm import SMMFactor
from factors.snr_enhanced_reversal import SNREnhancedReversalFactor
from factors.soir import SOIRFactor
from factors.srvj import SRVJFactor
from factors.structured_reversal import StructuredReversalFactor
from factors.super_big_ret import SuperBigRetFactor
from factors.super_large_buy import SuperLargeBuyFactor
from factors.supplier_concentration import SupplierConcentrationFactor
from factors.supply_centrality_change import SupplyCentralityChangeFactor
from factors.supply_chain_degree import SupplyChainDegreeFactor
from factors.supply_chain_position import SupplyChainPositionFactor
from factors.t_dist_active_buy import TDistActiveBuyFactor
from factors.tcc import TCCFactor
from factors.tidal_price_velocity import TidalPriceVelocityFactor
from factors.time_centroid import TimeCentroidFactor
from factors.time_weighted_rpp import TimeWeightedRelativePricePositionFactor
from factors.tl_ratio import TLRatioFactor
from factors.toxic_liquidity import ToxicLiquidityFactor
from factors.trend_capital_net_support import TrendCapitalNetSupportFactor
from factors.trend_capital_vwap import TrendCapitalVWAPFactor
from factors.trend_clarity import TrendClarityFactor
from factors.trend_clarity_momentum import TrendClarityMomentumFactor
from factors.trend_ratio import TrendRatioFactor
from factors.tsrjv import TSRJVFactor
from factors.turnover_iv import TurnoverIVFactor
from factors.uniform_active_ratio import UniformActiveRatioFactor
from factors.upside_realized_vol import UpsideRealizedVolFactor
from factors.upside_vol_ratio import UpsideVolRatioFactor
from factors.upstream_momentum import UpstreamMomentumFactor
from factors.upstream_transmission import UpstreamTransmissionFactor
from factors.utd import UTDFactor
from factors.v_shaped_disposition import VShapedDispositionFactor
from factors.valley_ridge_ratio import ValleyRidgeRatioFactor
from factors.valley_weighted_price_quantile import ValleyWeightedPriceQuantileFactor
from factors.vcde import VcdeFactor
from factors.vcvar import VCVaRFactor
from factors.vol_panic import VolPanicFactor
from factors.volume_coeff_variation import VolumeCoeffVariationFactor
from factors.volume_entropy import VolumeEntropyFactor
from factors.volume_kurtosis import VolumeKurtosisFactor
from factors.volume_long_big_select import VolumeLongBigSelectFactor
from factors.volume_peak_count import VolumePeakCountFactor
from factors.volume_peak_minutes import VolumePeakMinutesFactor
from factors.volume_proportion_composite import VolumeProportionCompositeFactor
from factors.volume_ratio import VolumeRatioFactor
from factors.volume_ridge_relative_vwap import VolumeRidgeRelativeVWAPFactor
from factors.volume_ridge_return import VolumeRidgeReturnFactor
from factors.volume_surge_vol import VolumeSurgeVolFactor
from factors.volume_valley_price import VolumeValleyPriceFactor
from factors.vpin import VPINFactor
from factors.vsa_close_diff import VSACloseDiffFactor
from factors.vwpin import VWPINFactor
from factors.weighted_close_ratio import WeightedCloseRatioFactor
from factors.weighted_liquidity_premium import WeightedLiquidityPremiumFactor
from factors.weighted_lower_shadow_freq import WeightedLowerShadowFreqFactor
from factors.weighted_profit_freq import WeightedProfitFreqFactor
from factors.weighted_skewness import WeightedSkewnessFactor
from factors.weighted_upper_shadow_freq import WeightedUpperShadowFreqFactor

__all__ = [
    "BaseFactor",
    "AbnormalNegativeReversalFactor",
    "AbnormalPositiveReversalFactor",
    "AbnormalReturnAvgFactor",
    "AbnormalVolumeDailyFactor",
    "ABNRETDFactor",
    "AbsRetAmountCorrFactor",
    "ACMAFactor",
    "ActiveBuyRatioFactor",
    "ActiveBuySpecificityFactor",
    "ActiveChipRatioFactor",
    "AFHCloseFactor",
    "AmountEntropyFactor",
    "AmtMaxFactor",
    "AnalystAnchoringBiasFactor",
    "AnalystCoCoverageIndirectMomentumFactor",
    "APMFactor",
    "AskDepthFactor",
    "ATDPriceLowestFactor",
    "ATDPriceLowestSellFactor",
    "AttentionAbnormalReturnFactor",
    "AttentionCaptureVolFactor",
    "AttentionDecayPanicFactor",
    "AttentionEventsFactor",
    "AttentionLimitFactor",
    "AttentionMarketVolumeFactor",
    "AttentionSpilloverFactor",
    "AttentionTurnoverFactor",
    "AvgOrderBookDepthFactor",
    "AvgOutflowPerTradeRatioFactor",
    "AvgPositiveJumpReturnFactor",
    "BTypeVolumeDistFactor",
    "BidAskPriceRatioFactor",
    "BidAskSpreadFactor",
    "BidDepthFactor",
    "BottleneckCompanyFactor",
    "BusinessComplexityLinkageFactor",
    "BuyIlliquidityFactor",
    "BuyOrderConcentrationFactor",
    "CancelRateFactor",
    "CapitalLossRealizationFactor",
    "CGOFactor",
    "ChipConcentrationFactor",
    "ChipDeviationRateFactor",
    "ChipDispositionEffectFactor",
    "ChipDistributionCVFactor",
    "ChipDistributionKurtosisFactor",
    "ChipPnlAdjustedAFHFactor",
    "ChipReturnEnhanceFactor",
    "ChipTurnoverFactor",
    "CKDPFactor",
    "ClosingReturnFactor",
    "ClosingVolumeRatioFactor",
    "COMinFactor",
    "ComplexMomentumFactor",
    "CompositePVCorrFactor",
    "ConfidenceNormalActiveBuyFactor",
    "ConsistentBuyTradeFactor",
    "ConsistentSellTradeFactor",
    "ConsistentTradingVolumeFactor",
    "CoraAbsFactor",
    "CoraRFactor",
    "CorrectedAmbiguitySpreadFactor",
    "CorrectedAmplitudeFactor",
    "CorrectedIntradayReversalFactor",
    "CorrectedNetInflowFactor",
    "CorrectedOvernightReversalFactor",
    "CPQSIFactor",
    "CSADFactor",
    "CSADSmallFactor",
    "CSSDFactor",
    "CustomerCentralityChangeFactor",
    "CustomerClosenessFactor",
    "CustomerImportanceFactor",
    "CustomerIndustryConcentrationFactor",
    "CustomerMomentumFactor",
    "CustomerMomentumIDFactor",
    "DailyReturnAttentionFactor",
    "DazzlingReturnFactor",
    "DazzlingVolFactor",
    "DownsideRVFactor",
    "DpinFactor",
    "DpinSmallFactor",
    "DropMomentATDFactor",
    "DropSellATDFactor",
    "DropTimeCentroidFactor",
    "EarlyLateCompositeRatioFactor",
    "EffectiveDepthFactor",
    "EffectiveSpreadFactor",
    "ELSCorrelationFactor",
    "EndOfDayAmountFactor",
    "ExtremeFollowRatioFactor",
    "ExtremeReturnReversalFactor",
    "FlowFollowingFactor",
    "FogAmountRatioFactor",
    "FogVolumeRatioFactor",
    "FollowLeaderFactor",
    "ForeignOpsFactor",
    "FuzzyCorrFactor",
    "GainSellingTendencyFactor",
    "GamingFactor",
    "GeoMomentumFactor",
    "GoldenRatioReversalFactor",
    "HCVOLFactor",
    "HerdingBehaviorFactor",
    "HFDownsideVolRatioFactor",
    "HfIdioVolFactor",
    "HighLowPriceCompositeRatioFactor",
    "HighLowSpreadIntensityFactor",
    "HighLowVolSelectFactor",
    "IDMagFactor",
    "IdealAmplitudeFactor",
    "IdealReversalFactor",
    "IdioTurnoverVolFactor",
    "ILLIQFactor",
    "ImprovedAnalystCoverageMomentumFactor",
    "ImprovedLargeRatioFactor",
    "ImprovedLargeRatioActiveFactor",
    "IndustryMomentumFactor",
    "InformedBuyRatioFactor",
    "InformedSellRatioFactor",
    "InstTradeHeatFactor",
    "IntradayAmplitudeCutFactor",
    "IntradayJumpDropFactor",
    "IntradayMaxDrawdownFactor",
    "IntradayRetIVFactor",
    "IntradayReturnVolRatioFactor",
    "IntradaySNRFactor",
    "JumpAbsoluteReturnFactor",
    "JumpArrivalFactor",
    "JumpDegreeCentralityFactor",
    "JumpLinkageRelativeMomentumFactor",
    "JumpNonPositiveMomentumFactor",
    "KLinePatternFactor",
    "KatzCentralityFactor",
    "LagAbsRetAdjAmountCorrFactor",
    "LargeBuyRatioFactor",
    "LargeJumpAsymmetryFactor",
    "LargeOrderDpinFactor",
    "LargeOrderReturnFactor",
    "LargeVolPriceCorrFactor",
    "LCPSFactor",
    "LCVOLFactor",
    "LeadingVolumeAnomalyAtMinFactor",
    "LinkageFactor",
    "LiquidityPremiumEWFactor",
    "LockedChipsRatioFactor",
    "LoneGooseFactor",
    "LossSellingFactor",
    "LSVHerdingFactor",
    "MainForceStrengthFactor",
    "MainForceVolFactor",
    "MaxRiseFactor",
    "MCIFactor",
    "MidPriceChangeFactor",
    "MinuteAmountVarianceFactor",
    "MinuteIdealAmplitudeFactor",
    "MLQSFactor",
    "ModifiedBusinessLinkageFactor",
    "MOFIFactor",
    "MorningInflowStabilityFactor",
    "MorningMistFactor",
    "MPBFactor",
    "MRRFactor",
    "MTEFactor",
    "MultiLayerSNRFactor",
    "NaiveActiveRatioFactor",
    "NegativeReversalFreqFactor",
    "NetCommissionBuyFactor",
    "NetSupportVolumeFactor",
    "NetworkCentralityFactor",
    "NewsNetworkLeadReturnFactor",
    "NightFrostFactor",
    "NonlinearHFVolatilityFactor",
    "NoonAncientTreeFactor",
    "NormalBigRetFactor",
    "NRBRRetFactor",
    "OpeningBuyIntentionFactor",
    "OpeningCommissionRatioFactor",
    "OpeningNetBuyFactor",
    "OrderImbalanceFactor",
    "OrderTradeCorrelationFactor",
    "OvernightGapFactor",
    "OvernightRetIVFactor",
    "PTypeVolumeFactor",
    "PATVFactor",
    "PeakClimberFactor",
    "PeakIntervalKurtosisFactor",
    "PeakPriceQuantileFactor",
    "PeakRidgeCorrFactor",
    "PeakRidgeRatioFactor",
    "PeakVolSellAmtFactor",
    "PositiveIntradayReversalFreqFactor",
    "PostOpenLargeBuyFactor",
    "PostOpenNetBuyFactor",
    "PriceImpactBiasFactor",
    "PriceResiliencyFactor",
    "ProspectTKFactor",
    "PRSIFactor",
    "PTRFactor",
    "PVCorrTrendFactor",
    "PvolRetCorrFactor",
    "QUAFactor",
    "RealizedBipowerVariationFactor",
    "RealizedJumpVolatilityFactor",
    "RealizedKurtosisFactor",
    "RealizedSkewnessFactor",
    "ResidualFlowStrengthFactor",
    "ResidualReversalFactor",
    "ResiliencyFactor",
    "RetailHerdFactor",
    "RetailPanicFactor",
    "RetailTradeHeatFactor",
    "ReversalResidualImbalanceFactor",
    "RidgeGapSkewFactor",
    "RSJFactor",
    "RTVFactor",
    "RVJNFactor",
    "RVLJPFactor",
    "SalienceReturnFactor",
    "SatdPriceVolCorrFactor",
    "SATDVolumeHighFactor",
    "SCCFactor",
    "SearchRatioDownstreamFactor",
    "SellIlliquidityFactor",
    "SentimentSpilloverFactor",
    "ShadowCloseStdFactor",
    "SidewaysBuyATDFactor",
    "SimilarLowVolFactor",
    "SimilarReverseFactor",
    "SimilarityMomentumFactor",
    "SingleAmountEntropyFactor",
    "SiphonEffectFactor",
    "SkewedMomentumReturnFactor",
    "SLSVHerdingFactor",
    "SmallBuyActiveFactor",
    "SmallDownwardJumpVolFactor",
    "SmallOrderLagCorrFactor",
    "SmartMoneyFactor",
    "SMMFactor",
    "SNREnhancedReversalFactor",
    "SOIRFactor",
    "SRVJFactor",
    "StructuredReversalFactor",
    "SuperBigRetFactor",
    "SuperLargeBuyFactor",
    "SupplierConcentrationFactor",
    "SupplyCentralityChangeFactor",
    "SupplyChainDegreeFactor",
    "SupplyChainPositionFactor",
    "TDistActiveBuyFactor",
    "TCCFactor",
    "TidalPriceVelocityFactor",
    "TimeCentroidFactor",
    "TimeWeightedRelativePricePositionFactor",
    "TLRatioFactor",
    "ToxicLiquidityFactor",
    "TrendCapitalNetSupportFactor",
    "TrendCapitalVWAPFactor",
    "TrendClarityFactor",
    "TrendClarityMomentumFactor",
    "TrendRatioFactor",
    "TSRJVFactor",
    "TurnoverIVFactor",
    "UniformActiveRatioFactor",
    "UpsideRealizedVolFactor",
    "UpsideVolRatioFactor",
    "UpstreamMomentumFactor",
    "UpstreamTransmissionFactor",
    "UTDFactor",
    "VShapedDispositionFactor",
    "ValleyRidgeRatioFactor",
    "ValleyWeightedPriceQuantileFactor",
    "VcdeFactor",
    "VCVaRFactor",
    "VolPanicFactor",
    "VolumeCoeffVariationFactor",
    "VolumeEntropyFactor",
    "VolumeKurtosisFactor",
    "VolumeLongBigSelectFactor",
    "VolumePeakCountFactor",
    "VolumePeakMinutesFactor",
    "VolumeProportionCompositeFactor",
    "VolumeRatioFactor",
    "VolumeRidgeRelativeVWAPFactor",
    "VolumeRidgeReturnFactor",
    "VolumeSurgeVolFactor",
    "VolumeValleyPriceFactor",
    "VPINFactor",
    "VSACloseDiffFactor",
    "VWPINFactor",
    "WeightedCloseRatioFactor",
    "WeightedLiquidityPremiumFactor",
    "WeightedLowerShadowFreqFactor",
    "WeightedProfitFreqFactor",
    "WeightedSkewnessFactor",
    "WeightedUpperShadowFreqFactor",
]
