# SPDX-License-Identifier: MIT
# Copyright (c) 2025–present wiqilee

# fetchers.py — resilient fetchers for NewsAPI, Reddit (snscrape), and CoinGecko
# - Safe retries + backoff
# - Explicit User-Agent
# - Reddit date filtering done in Python (since: in CLI can be flaky)
# - Returns DataFrames compatible with the rest of the pipeline
#   (publishedAt, source, title, description, url, channel, ...)

from __future__ import annotations

import time
import subprocess
import sys
import json
from datetime import datetime, timezone
from typing import Dict, Optional

import requests
import pandas as pd


# -----------------------------
# Shared helpers
# -----------------------------
UA = "CryptoSentimentDashboard/1.0 (+https://github.com/wiqilee)"
DEFAULT_TIMEOUT = 20
RETRY_STATUS = {429, 500, 502, 503, 504}


def _sleep_backoff(attempt: int, base: float = 1.0, cap: float = 8.0):
    """Exponential backoff with cap."""
    delay = min(cap, base * (2 ** (attempt - 1)))
    time.sleep(delay)


def _requests_get(url: str, *, params: dict | None = None, timeout: int = DEFAULT_TIMEOUT, max_retries: int = 3):
    """GET with minimal retries/backoff and a polite User-Agent."""
    attempt = 1
    while True:
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": UA})
        except requests.RequestException:
            if attempt >= max_retries:
                raise
            _sleep_backoff(attempt)
            attempt += 1
            continue

        if r.status_code in RETRY_STATUS and attempt < max_retries:
            _sleep_backoff(attempt)
            attempt += 1
            continue

        return r


def _to_iso_utc(dt: datetime | str) -> str:
    """Return ISO 8601 Z string."""
    if isinstance(dt, str):
        try:
            dtp = pd.to_datetime(dt, utc=True)
        except Exception:
            return dt
        return dtp.strftime("%Y-%m-%dT%H:%M:%SZ")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_ts_utc(x) -> pd.Timestamp:
    """Normalize any datetime-like (naive or tz-aware) to UTC Timestamp."""
    ts = pd.Timestamp(x)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


# -----------------------------
# NewsAPI (headlines)
# -----------------------------
def fetch_newsapi(
    api_key: str,
    query: str,
    from_iso: str | datetime,
    to_iso: str | datetime,
    *,
    language: str = "en",
    page_size: int = 50,
    max_pages: int = 5,
    search_in: str = "title,description"  # keep body out for fewer false positives on the free plan
) -> pd.DataFrame:
    """
    Fetch articles from NewsAPI /v2/everything with polite retries.
    Returns DataFrame with columns: source, author, title, description, url, publishedAt, channel
    """
    url = "https://newsapi.org/v2/everything"
    MAX_TOTAL = 100  # NewsAPI free plan hard cap

    f_iso = _to_iso_utc(from_iso)
    t_iso = _to_iso_utc(to_iso)

    rows = []
    fetched, page = 0, 1

    while fetched < MAX_TOTAL and page <= max_pages:
        size = min(int(page_size), MAX_TOTAL - fetched)
        params = {
            "q": query,
            "from": f_iso,
            "to": t_iso,
            "language": language,
            "sortBy": "publishedAt",
            "searchIn": search_in,
            "pageSize": size,
            "page": page,
            "apiKey": api_key,
        }
        r = _requests_get(url, params=params)
        if r.status_code != 200:
            # 426 = upgrade required (plan); stop silently (like previous behavior)
            if r.status_code == 426:
                break
            raise RuntimeError(f"NewsAPI error {r.status_code}: {r.text}")

        data = r.json()
        arts = data.get("articles", []) or []
        for a in arts:
            rows.append({
                "source": (a.get("source") or {}).get("name"),
                "author": a.get("author"),
                "title": a.get("title") or "",
                "description": a.get("description") or "",
                "url": a.get("url"),
                "publishedAt": a.get("publishedAt"),
                "channel": "newsapi",
            })
        fetched += len(arts)
        if len(arts) < size:
            break
        page += 1
        time.sleep(1.0)  # polite

    df = pd.DataFrame(rows)
    # Light normalization (keep as strings; your loader will cast tz & columns)
    if "publishedAt" in df.columns and not df.empty:
        try:
            df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
    return df


# -----------------------------
# Reddit via snscrape (no API key)
# -----------------------------
def _ensure_snscrape() -> bool:
    try:
        import snscrape  # noqa: F401
        return True
    except Exception:
        return False


def fetch_reddit_snscrape(
    query: str,
    since_dt: datetime,
    *,
    limit: int = 100,
) -> pd.DataFrame:
    """
    Scrape Reddit search results via snscrape CLI.
    NOTE: `since:` in the query string can be flaky; we filter by date in Python.
    Returns columns: source (subreddit), author, title, description(selfText), url, publishedAt, channel
    """
    if not _ensure_snscrape():
        raise RuntimeError("snscrape not installed. Add 'snscrape' to requirements and pip install it.")

    cmd = [
        sys.executable, "-m", "snscrape", "--jsonl", "--max-results", str(limit),
        "reddit-search", query  # keep query clean; we'll filter dates locally
    ]

    since_ts = _to_ts_utc(since_dt)

    # Run CLI
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    rows = []
    try:
        assert p.stdout is not None
        for line in p.stdout:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # parse date & filter
            dt = obj.get("date")
            try:
                dtp = pd.to_datetime(dt, utc=True)  # tz-aware UTC
            except Exception:
                dtp = None
            if dtp is not None and dtp < since_ts:
                continue

            rows.append({
                "source": obj.get("subreddit"),
                "author": obj.get("author"),
                "title": obj.get("title") or "",
                "description": obj.get("selfText") or "",
                "url": obj.get("url"),
                "publishedAt": obj.get("date"),  # keep original ISO; loader will normalize
                "channel": "reddit",
            })
        p.wait()
    finally:
        # ensure process ends
        if p.poll() is None:
            p.terminate()

    # If snscrape errors and no rows, surface stderr for debugging
    if p.returncode not in (0, None) and not rows:
        err = ""
        if p.stderr:
            try:
                err = p.stderr.read()
            except Exception:
                err = ""
        raise RuntimeError(f"snscrape failed (code {p.returncode}). Stderr: {err[:500]}")

    df = pd.DataFrame(rows)
    if "publishedAt" in df.columns and not df.empty:
        try:
            df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
    return df


# -----------------------------
# CoinGecko prices (daily)
# -----------------------------
def fetch_prices_coingecko(id_map: Dict[str, str], days: int) -> pd.DataFrame:
    """
    Fetch daily USD prices from CoinGecko and return a wide DataFrame:
      columns: date, <TICKER>_price, ...
    id_map example: {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
    """
    def one(cg_id: str) -> pd.DataFrame:
        url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
        attempt = 1
        while True:
            r = _requests_get(url, params={"vs_currency": "usd", "days": days})
            if r.status_code == 200:
                break
            if r.status_code in RETRY_STATUS and attempt < 3:
                _sleep_backoff(attempt)
                attempt += 1
                continue
            r.raise_for_status()

        data = r.json().get("prices", []) or []
        df = pd.DataFrame(data, columns=["ts_ms", "price"])
        if df.empty:
            return pd.DataFrame(columns=["date", "price"])
        df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.date
        out = df.groupby("date", as_index=False)["price"].mean()
        return out

    combined: Optional[pd.DataFrame] = None
    for sym, cg_id in id_map.items():
        df = one(cg_id)
        df = df.rename(columns={"price": f"{sym}_price"})
        combined = df if combined is None else combined.merge(df, on="date", how="outer")
        time.sleep(1.0)  # polite to CG

    if combined is None:
        return pd.DataFrame(columns=["date"] + [f"{s}_price" for s in id_map.keys()])
    return combined.sort_values("date").reset_index(drop=True)
