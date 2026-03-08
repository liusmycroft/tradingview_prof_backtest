import numpy as np
import pandas as pd
import pytest

from factors.prospect_tk import ProspectTKFactor, _w_plus, _w_minus


@pytest.fixture
def factor():
    return ProspectTKFactor()


class TestProspectTKMetadata:
    def test_name(self, factor):
        assert factor.name == "PROSPECT_TK"

    def test_category(self, factor):
        assert factor.category == "行为金融"

    def test_repr(self, factor):
        assert "PROSPECT_TK" in repr(factor)

    def test_get_metadata(self, factor):
        meta = factor.get_metadata()
        assert meta["name"] == "PROSPECT_TK"
        assert meta["category"] == "行为金融"


class TestProspectTKHandCalculated:
    """用手算数据验证 TK 公式的正确性。"""

    def test_all_positive_returns(self, factor):
        """所有 PB 变化率为正时，使用 w+ 权重函数。

        N=3, 单只股票, pb_change = [0.01, 0.02, 0.03] (已排序)
        quantiles = [0, 1/3, 2/3, 1]

        v(0.01) = 0.01^0.88
        v(0.02) = 0.02^0.88
        v(0.03) = 0.03^0.88

        TK = v(0.01)*(w+(1/3)-w+(0)) + v(0.02)*(w+(2/3)-w+(1/3)) + v(0.03)*(w+(1)-w+(2/3))
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        pb = pd.DataFrame([0.01, 0.02, 0.03], index=dates, columns=stocks)
        result = factor.compute(pb_change_rates=pb, N=3)

        alpha, gamma = 0.88, 0.61
        r = np.array([0.01, 0.02, 0.03])
        v = r ** alpha
        qs = np.array([0, 1/3, 2/3, 1.0])
        w = np.array([_w_plus(q, gamma) for q in qs])
        expected = sum(v[i] * (w[i+1] - w[i]) for i in range(3))

        assert result.iloc[2, 0] == pytest.approx(expected, rel=1e-8)

    def test_all_negative_returns(self, factor):
        """所有 PB 变化率为负时，使用 w- 权重函数和损失厌恶。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]

        pb = pd.DataFrame([-0.03, -0.02, -0.01], index=dates, columns=stocks)
        result = factor.compute(pb_change_rates=pb, N=3)

        alpha, lam, delta = 0.88, 2.25, 0.69
        r = np.array([-0.03, -0.02, -0.01])
        v = np.array([-lam * ((-x) ** alpha) for x in r])
        qs = np.array([0, 1/3, 2/3, 1.0])
        w = np.array([_w_minus(q, delta) for q in qs])
        expected = sum(v[i] * (w[i+1] - w[i]) for i in range(3))

        assert result.iloc[2, 0] == pytest.approx(expected, rel=1e-8)

    def test_mixed_returns(self, factor):
        """混合正负 PB 变化率。"""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stocks = ["A"]

        pb = pd.DataFrame([-0.02, 0.01, -0.01, 0.03], index=dates, columns=stocks)
        result = factor.compute(pb_change_rates=pb, N=4)

        alpha, lam, gamma, delta = 0.88, 2.25, 0.61, 0.69
        sorted_r = np.sort([-0.02, 0.01, -0.01, 0.03])
        qs = np.arange(5) / 4.0

        tk_val = 0.0
        for i in range(4):
            r = sorted_r[i]
            if r >= 0:
                v = r ** alpha
                w_up = _w_plus(qs[i+1], gamma)
                w_lo = _w_plus(qs[i], gamma)
            else:
                v = -lam * ((-r) ** alpha)
                w_up = _w_minus(qs[i+1], delta)
                w_lo = _w_minus(qs[i], delta)
            tk_val += v * (w_up - w_lo)

        assert result.iloc[3, 0] == pytest.approx(tk_val, rel=1e-8)

    def test_first_N_minus_1_nan(self, factor):
        """前 N-1 行应为 NaN。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        stocks = ["A"]
        pb = pd.DataFrame([0.01, 0.02, -0.01, 0.03, -0.02], index=dates, columns=stocks)
        result = factor.compute(pb_change_rates=pb, N=3)

        assert np.isnan(result.iloc[0, 0])
        assert np.isnan(result.iloc[1, 0])
        assert not np.isnan(result.iloc[2, 0])


class TestProspectTKEdgeCases:
    def test_nan_in_data(self, factor):
        """含 NaN 的数据不应抛异常。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        pb = pd.DataFrame([np.nan, 0.01, 0.02], index=dates, columns=stocks)
        result = factor.compute(pb_change_rates=pb, N=3)
        assert np.isnan(result.iloc[2, 0])

    def test_zero_returns(self, factor):
        """全零变化率时 TK 应为 0。"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["A"]
        pb = pd.DataFrame([0.0, 0.0, 0.0], index=dates, columns=stocks)
        result = factor.compute(pb_change_rates=pb, N=3)
        assert result.iloc[2, 0] == pytest.approx(0.0, abs=1e-15)

    def test_output_shape(self, factor):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        stocks = ["A", "B"]
        pb = pd.DataFrame(
            np.random.uniform(-0.1, 0.1, (30, 2)), index=dates, columns=stocks
        )
        result = factor.compute(pb_change_rates=pb, N=20)
        assert result.shape == pb.shape
        assert list(result.columns) == list(pb.columns)
        assert isinstance(result, pd.DataFrame)


class TestWeightFunctions:
    def test_w_plus_boundary(self):
        assert _w_plus(0.0, 0.61) == 0.0
        assert _w_plus(1.0, 0.61) == 1.0

    def test_w_minus_boundary(self):
        assert _w_minus(0.0, 0.69) == 0.0
        assert _w_minus(1.0, 0.69) == 1.0

    def test_w_plus_monotonic(self):
        vals = [_w_plus(p, 0.61) for p in np.linspace(0, 1, 11)]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1]

    def test_w_minus_monotonic(self):
        vals = [_w_minus(p, 0.69) for p in np.linspace(0, 1, 11)]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1]
