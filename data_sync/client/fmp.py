"""
FMP API 客户端

封装 Financial Modeling Prep API 调用：
- Token bucket rate limiter
- Exponential backoff retry (429/5xx)
- 自动 base URL + apikey 注入
- JSON / CSV 响应解析
- 请求日志
"""
import io
import logging
import time
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.interval = 60.0 / max_per_minute
        self._last_call = 0.0

    def wait(self):
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.interval:
            sleep_time = self.interval - elapsed
            time.sleep(sleep_time)
        self._last_call = time.monotonic()


class FMPClient:
    """
    FMP API 客户端

    Usage:
        client = FMPClient(api_key="xxx")
        data = client.get_json("/income-statement", symbol="AAPL", period="annual")
        df = client.get_csv("/ratios-ttm-bulk")
    """

    BASE_URL = "https://financialmodelingprep.com/stable"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        max_per_minute: int = 3000,
        max_retries: int = 5,
        timeout: int = 60,
    ):
        self.api_key = api_key
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = requests.Session()
        self._limiter = RateLimiter(max_per_minute)

    def _request(self, endpoint: str, params: Dict[str, Any]) -> requests.Response:
        """发送 GET 请求，带 rate limit + retry。"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        params["apikey"] = self.api_key

        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            self._limiter.wait()
            t0 = time.monotonic()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
                elapsed = time.monotonic() - t0
                logger.debug(
                    "%s %s → %d (%.1fs)", "GET", endpoint, resp.status_code, elapsed
                )

                if resp.status_code == 200:
                    return resp

                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = min(2 ** attempt, 30)
                    logger.warning(
                        "HTTP %d on %s, retry %d/%d in %ds",
                        resp.status_code, endpoint, attempt, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    last_exc = requests.HTTPError(
                        f"HTTP {resp.status_code}", response=resp
                    )
                    continue

                resp.raise_for_status()

            except requests.RequestException as e:
                elapsed = time.monotonic() - t0
                # 4xx (非 429) 不重试
                if hasattr(e, 'response') and e.response is not None:
                    if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                        raise
                logger.warning(
                    "Request error on %s: %s (%.1fs), retry %d/%d",
                    endpoint, e, elapsed, attempt, self.max_retries,
                )
                last_exc = e
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 30))

        raise last_exc or RuntimeError(f"Failed after {self.max_retries} retries: {endpoint}")

    def get_json(
        self, endpoint: str, **params: Any
    ) -> Union[List[Dict], Dict]:
        """
        GET 请求，返回 parsed JSON。

        Args:
            endpoint: API 路径 (e.g. "/income-statement")
            **params: 查询参数 (e.g. symbol="AAPL", period="annual")

        Returns:
            JSON list 或 dict
        """
        resp = self._request(endpoint, params)
        return resp.json()

    def get_csv(self, endpoint: str, **params: Any) -> pd.DataFrame:
        """
        GET 请求，解析 CSV 响应为 DataFrame。

        用于 bulk 端点 (e.g. /ratios-ttm-bulk)。

        Args:
            endpoint: API 路径
            **params: 查询参数

        Returns:
            pd.DataFrame
        """
        resp = self._request(endpoint, params)
        return pd.read_csv(io.StringIO(resp.text))

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
