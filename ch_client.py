"""
Companies House client that respects the published rate limit.

The limit is 600 requests per five minute window across all endpoints. This
flow makes two calls per company before the document download, and the
download itself is served from S3 and does not count. So 5,000 companies is
roughly 10,000 calls, or about an hour and a half of steady running.

Two things the previous version got wrong and this one does not: the 429
retry recursed without a bound, and the remaining count was tracked per
process, so running four uvicorn workers quietly quadrupled the assumed
budget.
"""

import os
import threading
import time
from typing import Optional

import httpx

BASE = "https://api.company-information.service.gov.uk"
RESERVE = int(os.getenv("CH_RESERVE", "25"))
MAX_RETRIES = int(os.getenv("CH_MAX_RETRIES", "5"))


class CompaniesHouse:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.remaining: Optional[int] = None
        self.reset_at = 0.0
        self._lock = threading.Lock()

    def _wait(self):
        with self._lock:
            if self.remaining is None or self.remaining > RESERVE:
                return
            sleep_for = max(0.0, self.reset_at - time.time()) + 1
        if sleep_for > 0:
            time.sleep(min(sleep_for, 300))

    def _record(self, response: httpx.Response):
        with self._lock:
            try:
                self.remaining = int(response.headers["x-ratelimit-remaining"])
            except (KeyError, TypeError, ValueError):
                pass
            try:
                self.reset_at = float(response.headers["x-ratelimit-reset"])
            except (KeyError, TypeError, ValueError):
                pass

    def get(self, url: str, accept: Optional[str] = None,
            follow: bool = True) -> httpx.Response:
        if not self.api_key:
            raise RuntimeError("CH_API_KEY is not set")

        for attempt in range(MAX_RETRIES):
            self._wait()
            headers = {"Accept": accept} if accept else {}
            with httpx.Client(timeout=120.0, follow_redirects=False) as client:
                response = client.get(url, headers=headers,
                                      auth=(self.api_key, ""))
            self._record(response)

            if response.status_code == 429:
                wait = max(1.0, self.reset_at - time.time()) + 1
                time.sleep(min(wait, 300))
                continue          # bounded loop, not recursion

            # The document endpoint redirects to a signed S3 URL. The
            # Companies House credentials must not be carried across or S3
            # rejects the request.
            if follow and response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location")
                if location:
                    with httpx.Client(timeout=180.0,
                                      follow_redirects=True) as client:
                        return client.get(location)
            return response

        raise RuntimeError(f"rate limited after {MAX_RETRIES} attempts: {url}")

    def accounts_filings(self, number: str, limit: int = 30):
        response = self.get(f"{BASE}/company/{number}/filing-history"
                            f"?category=accounts&items_per_page={limit}")
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise RuntimeError(
                f"filing history {response.status_code}: {response.text[:200]}")
        items = [
            item for item in response.json().get("items", [])
            if item.get("category") == "accounts"
            and item.get("links", {}).get("document_metadata")
        ]
        items.sort(key=lambda i: i.get("date", ""), reverse=True)
        return items


def normalise(number: str) -> str:
    number = number.strip().upper()
    return number.zfill(8) if number.isdigit() else number
