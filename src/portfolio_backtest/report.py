"""Generate HTML report using quantstats."""

import html as html_mod
import tempfile
from pathlib import Path

import pandas as pd

try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False


def generate_html_report(
    returns: pd.Series,
    title: str = "Portfolio Backtest Report",
    benchmark_returns: pd.Series | None = None,
) -> str:
    """Generate a quantstats HTML tearsheet and return as string."""
    if not HAS_QUANTSTATS:
        return _fallback_report(returns, title)

    if returns.empty:
        return "<html><body><h1>No data available</h1></body></html>"

    # work on a copy to avoid mutating caller's data
    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)
    returns.index.name = None

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        tmp_path = f.name

    try:
        qs.reports.html(
            returns,
            benchmark=benchmark_returns,
            title=html_mod.escape(title),
            output=tmp_path,
        )
        result = Path(tmp_path).read_text()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


def _fallback_report(returns: pd.Series, title: str) -> str:
    """Simple HTML report when quantstats is not installed."""
    safe_title = html_mod.escape(title)

    if returns.empty:
        return "<html><body><h1>No data available</h1></body></html>"

    total_return = ((1 + returns).prod() - 1) * 100
    cum = (1 + returns).cumprod()
    max_dd = ((cum / cum.cummax()) - 1).min() * 100

    return f"""<!DOCTYPE html>
<html>
<head><title>{safe_title}</title></head>
<body>
<h1>{safe_title}</h1>
<p>Total Return: {total_return:.2f}%</p>
<p>Max Drawdown: {max_dd:.2f}%</p>
<p><em>Install quantstats for full report: pip install quantstats</em></p>
</body>
</html>"""
