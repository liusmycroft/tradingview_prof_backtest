import numpy as np
import pandas as pd
import pytest

from factors.news_network_lead_return import NewsNetworkLeadReturnFactor


@pytest.fixture
def factor():
    return NewsNetworkLeadReturnFactor()


class TestNewsNetworkLeadReturnMetadata:
    def test_name(self, factor):
        assert factor.name == "NEWS_NETWORK_LEAD_RETURN"

    def test_category(self, factor):
        assert factor.category == "图谱网络-动量溢出"

    def test_repr(self, factor):
        assert "NEWS_NETWORK_LEAD_RETURN" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "NEWS_NETWORK_LEAD_RETURN"
        assert meta["category"] == "图谱网络-动量溢出"


class TestNewsNetworkLeadReturnHandCalculated:
    """手算验证 LR_agg = LR+(same) + LR-(same) + LR-(diff) - LR+(diff)"""

    def test_simple_values(self, factor):
        """简单数值验证。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        lr_ps = pd.DataFrame([0.1, 0.2, 0.3], index=dates, columns=stocks)
        lr_ns = pd.DataFrame([0.05, 0.10, 0.15], index=dates, columns=stocks)
        lr_pd = pd.DataFrame([0.08, 0.12, 0.20], index=dates, columns=stocks)
        lr_nd = pd.DataFrame([0.03, 0.06, 0.09], index=dates, columns=stocks)

        result = factor.compute(
            lead_return_pos_same=lr_ps,
            lead_return_neg_same=lr_ns,
            lead_return_pos_diff=lr_pd,
            lead_return_neg_diff=lr_nd,
        )

        # row 0: 0.1 + 0.05 + 0.03 - 0.08 = 0.10
        assert result.iloc[0, 0] == pytest.approx(0.10, rel=1e-10)
        # row 1: 0.2 + 0.10 + 0.06 - 0.12 = 0.24
        assert result.iloc[1, 0] == pytest.approx(0.24, rel=1e-10)
        # row 2: 0.3 + 0.15 + 0.09 - 0.20 = 0.34
        assert result.iloc[2, 0] == pytest.approx(0.34, rel=1e-10)

    def test_zero_inputs(self, factor):
        """全零输入时结果应为零。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A", "B"]
        zeros = pd.DataFrame(0.0, index=dates, columns=stocks)

        result = factor.compute(
            lead_return_pos_same=zeros,
            lead_return_neg_same=zeros,
            lead_return_pos_diff=zeros,
            lead_return_neg_diff=zeros,
        )
        np.testing.assert_array_almost_equal(result.values, 0.0)

    def test_two_stocks_independent(self, factor):
        """两只股票应独立计算。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A", "B"]

        lr_ps = pd.DataFrame({"A": [0.1, 0.2, 0.3], "B": [0.5, 0.6, 0.7]}, index=dates)
        lr_ns = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=dates)
        lr_pd = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=dates)
        lr_nd = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=dates)

        result = factor.compute(
            lead_return_pos_same=lr_ps,
            lead_return_neg_same=lr_ns,
            lead_return_pos_diff=lr_pd,
            lead_return_neg_diff=lr_nd,
        )
        assert result.iloc[0, 0] == pytest.approx(0.1, rel=1e-10)
        assert result.iloc[0, 1] == pytest.approx(0.5, rel=1e-10)

    def test_diff_sector_reversal(self, factor):
        """跨行业正收益应被减去（反转效应）。"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        stocks = ["A"]

        lr_ps = pd.DataFrame([0.0], index=dates, columns=stocks)
        lr_ns = pd.DataFrame([0.0], index=dates, columns=stocks)
        lr_pd = pd.DataFrame([0.5], index=dates, columns=stocks)
        lr_nd = pd.DataFrame([0.0], index=dates, columns=stocks)

        result = factor.compute(
            lead_return_pos_same=lr_ps,
            lead_return_neg_same=lr_ns,
            lead_return_pos_diff=lr_pd,
            lead_return_neg_diff=lr_nd,
        )
        assert result.iloc[0, 0] == pytest.approx(-0.5, rel=1e-10)


class TestNewsNetworkLeadReturnEdgeCases:
    def test_nan_in_input(self, factor):
        """输入含 NaN 时不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        lr_ps = pd.DataFrame([0.1, np.nan, 0.3, 0.4, 0.5], index=dates, columns=stocks)
        lr_ns = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)
        lr_pd = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)
        lr_nd = pd.DataFrame([0.1, 0.2, 0.3, 0.4, 0.5], index=dates, columns=stocks)

        result = factor.compute(
            lead_return_pos_same=lr_ps,
            lead_return_neg_same=lr_ns,
            lead_return_pos_diff=lr_pd,
            lead_return_neg_diff=lr_nd,
        )
        assert isinstance(result, pd.DataFrame)
        assert np.isnan(result.iloc[1, 0])

    def test_all_nan(self, factor):
        """全 NaN 输入时结果应全为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        nans = pd.DataFrame(np.nan, index=dates, columns=stocks)

        result = factor.compute(
            lead_return_pos_same=nans,
            lead_return_neg_same=nans,
            lead_return_pos_diff=nans,
            lead_return_neg_diff=nans,
        )
        assert result.isna().all().all()


class TestNewsNetworkLeadReturnOutputShape:
    def test_output_shape_matches_input(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B", "C"]
        lr_ps = pd.DataFrame(np.random.rand(30, 3), index=dates, columns=stocks)
        lr_ns = pd.DataFrame(np.random.rand(30, 3), index=dates, columns=stocks)
        lr_pd = pd.DataFrame(np.random.rand(30, 3), index=dates, columns=stocks)
        lr_nd = pd.DataFrame(np.random.rand(30, 3), index=dates, columns=stocks)

        result = factor.compute(
            lead_return_pos_same=lr_ps,
            lead_return_neg_same=lr_ns,
            lead_return_pos_diff=lr_pd,
            lead_return_neg_diff=lr_nd,
        )
        assert result.shape == lr_ps.shape
        assert list(result.columns) == list(lr_ps.columns)
        assert list(result.index) == list(lr_ps.index)

    def test_output_is_dataframe(self, factor):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        ones = pd.DataFrame(1.0, index=dates, columns=stocks)

        result = factor.compute(
            lead_return_pos_same=ones,
            lead_return_neg_same=ones,
            lead_return_pos_diff=ones,
            lead_return_neg_diff=ones,
        )
        assert isinstance(result, pd.DataFrame)
