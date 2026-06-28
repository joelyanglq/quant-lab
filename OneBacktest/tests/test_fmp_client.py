"""FMPClient 单元测试"""
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.etl.client.fmp import FMPClient, RateLimiter


class TestRateLimiter:
    def test_first_call_no_wait(self):
        limiter = RateLimiter(max_per_minute=60)
        t0 = __import__("time").monotonic()
        limiter.wait()
        elapsed = __import__("time").monotonic() - t0
        assert elapsed < 0.1

    def test_interval_respected(self):
        limiter = RateLimiter(max_per_minute=6000)  # 10ms interval
        limiter.wait()
        t0 = __import__("time").monotonic()
        limiter.wait()
        elapsed = __import__("time").monotonic() - t0
        assert elapsed >= 0.005  # at least ~half the interval


class TestFMPClient:
    def _make_client(self, **kwargs):
        return FMPClient(api_key="test_key", max_per_minute=100000, **kwargs)

    def test_init_defaults(self):
        client = self._make_client()
        assert client.api_key == "test_key"
        assert "financialmodelingprep.com" in client.base_url

    def test_init_custom_base_url(self):
        client = self._make_client(base_url="https://custom.url/api")
        assert client.base_url == "https://custom.url/api"

    @patch("data.etl.client.fmp.requests.Session")
    def test_get_json_success(self, mock_session_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"symbol": "AAPL", "revenue": 100}]

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        client = self._make_client()
        result = client.get_json("/income-statement", symbol="AAPL", period="annual")

        assert result == [{"symbol": "AAPL", "revenue": 100}]
        # 验证 apikey 被注入
        call_args = mock_session.get.call_args
        assert call_args[1]["params"]["apikey"] == "test_key"
        assert call_args[1]["params"]["symbol"] == "AAPL"

    @patch("data.etl.client.fmp.requests.Session")
    def test_get_csv_success(self, mock_session_cls):
        csv_text = "symbol,value\nAAPL,100\nMSFT,200\n"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = csv_text

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        client = self._make_client()
        df = client.get_csv("/ratios-ttm-bulk")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["symbol", "value"]

    @patch("data.etl.client.fmp.requests.Session")
    @patch("data.etl.client.fmp.time.sleep")
    def test_retry_on_429(self, mock_sleep, mock_session_cls):
        mock_resp_429 = MagicMock()
        mock_resp_429.status_code = 429

        mock_resp_200 = MagicMock()
        mock_resp_200.status_code = 200
        mock_resp_200.json.return_value = [{"ok": True}]

        mock_session = MagicMock()
        mock_session.get.side_effect = [mock_resp_429, mock_resp_200]
        mock_session_cls.return_value = mock_session

        client = self._make_client(max_retries=3)
        result = client.get_json("/test")

        assert result == [{"ok": True}]
        assert mock_session.get.call_count == 2

    @patch("data.etl.client.fmp.requests.Session")
    @patch("data.etl.client.fmp.time.sleep")
    def test_retry_exhausted_raises(self, mock_sleep, mock_session_cls):
        mock_resp_500 = MagicMock()
        mock_resp_500.status_code = 500

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp_500
        mock_session_cls.return_value = mock_session

        client = self._make_client(max_retries=2)
        with pytest.raises(Exception):
            client.get_json("/test")

        assert mock_session.get.call_count == 2

    def test_context_manager(self):
        with self._make_client() as client:
            assert client.api_key == "test_key"
