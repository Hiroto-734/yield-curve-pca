"""Tests for the FRED loader.

We don't hit the network in unit tests — only the URL construction
is verified directly. The end-to-end fetch is exercised by the
notebooks and by the live integration check in ``test_loader_live.py``
(skipped by default; run with ``-m live``).
"""

from __future__ import annotations

from yield_curve_pca.data.loader import _build_fred_url
from yield_curve_pca.utils.config import FRED_CSV_ENDPOINT


def test_build_fred_url_full() -> None:
    url = _build_fred_url("DGS10", "2024-01-01", "2024-12-31")
    assert url.startswith(FRED_CSV_ENDPOINT)
    assert "id=DGS10" in url
    assert "cosd=2024-01-01" in url
    assert "coed=2024-12-31" in url


def test_build_fred_url_no_dates() -> None:
    url = _build_fred_url("CPIAUCSL", None, None)
    assert "id=CPIAUCSL" in url
    assert "cosd" not in url
    assert "coed" not in url


def test_build_fred_url_only_start() -> None:
    url = _build_fred_url("PAYEMS", "2020-01-01", None)
    assert "cosd=2020-01-01" in url
    assert "coed" not in url
